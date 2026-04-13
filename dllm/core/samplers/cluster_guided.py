"""
Cluster-Guided Decoding for Masked Diffusion Language Models.

At each decoding step t the attention map is used to cluster token positions
via spectral clustering.  For each masked position i the hidden states of
already-decoded tokens that belong to the *same cluster* are averaged into
a centroid, which is then passed through the LM head to obtain guidance
logits.  These guidance logits are mixed into the original logits with a
time-decaying weight γ(t) = α·(1 − t/T)^β.

Reference equation numbers follow the technical description in the paper.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ClusterGuidedSamplerConfig(BaseSamplerConfig):
    # -- generation knobs (same as MDLMSamplerConfig) --
    max_new_tokens: int = 128
    max_length: int | None = None
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False

    # -- cluster-guidance knobs --
    # Which transformer layer's attention is used for clustering (-1 = last).
    cluster_attention_layer_idx: int = -1
    # Number of spectral clusters.
    n_clusters: int = 8
    # γ(t) = gamma_alpha · (t/T)^gamma_beta
    gamma_alpha: float = 1.0
    gamma_beta: float = 1.0
    # Minimum number of already-decoded tokens in a cluster before
    # guidance is computed (avoids noise from tiny anchor sets).
    min_anchor_size: int = 1


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class ClusterGuidedSampler(BaseSampler):
    """
    Masked-diffusion sampler with attention-based spectral clustering to
    guide logit predictions for masked positions.
    """

    # ------------------------------------------------------------------
    # γ(t) schedule
    # ------------------------------------------------------------------

    def _gamma(self, t: int, T: int, alpha: float, beta: float) -> float:
        """γ(t) = α · (t/T)^β.  Returns 0 at step 0 (all tokens masked,
        clustering unreliable) and α at the last step (most tokens decoded,
        clustering is best)."""
        return alpha * (t / T) ** beta

    # ------------------------------------------------------------------
    # Spectral clustering
    # ------------------------------------------------------------------

    def _spectral_cluster(
        self,
        attn: torch.Tensor,          # [valid_len, valid_len]  head-averaged attention
        n_clusters: int,
        valid_len: int,
    ) -> np.ndarray:
        """
        Cluster `valid_len` token positions via spectral clustering on the
        attention affinity matrix.

        Steps:
          1. Symmetrise: A = (attn + attn^T) / 2  so the affinity is undirected.
          2. Clamp to [0, 1] and zero the diagonal (self-loops skew the Laplacian).
          3. Run sklearn SpectralClustering with the precomputed affinity matrix.

        Falls back to assigning all tokens to cluster 0 when:
          - sklearn is not installed, or
          - valid_len <= n_clusters (not enough tokens to form distinct clusters).

        Returns an int64 numpy array of shape [valid_len].
        """
        # Graceful fallback when there are not enough tokens to cluster
        k = min(n_clusters, valid_len)
        if k <= 1:
            return np.zeros(valid_len, dtype=np.int64)

        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            return np.zeros(valid_len, dtype=np.int64)

        # Build a symmetric, non-negative affinity matrix on CPU as float64
        A = attn.float()
        A = (A + A.t()) / 2.0          # symmetrise
        A = A.clamp(min=0.0)           # ensure non-negative
        A.fill_diagonal_(0.0)          # remove self-loops
        A_np = A.cpu().numpy().astype(np.float64)

        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
            n_init=10,
        )
        labels = sc.fit_predict(A_np).astype(np.int64)
        return labels

    # ------------------------------------------------------------------
    # Guidance logit computation
    # ------------------------------------------------------------------

    def _compute_guidance_logits(
        self,
        hidden: torch.Tensor,        # [L, d]  last-layer hidden states (one sample)
        logits: torch.Tensor,        # [L, V]  original logits (one sample)
        attn: torch.Tensor,          # [L, L]  head-averaged attention (one sample)
        mask_index: torch.Tensor,    # [L]     bool, True = still masked
        valid_len: int,
        n_clusters: int,
        min_anchor_size: int,
        gamma: float,
    ) -> torch.Tensor:
        """
        Compute and inject guidance logits for all masked positions in one sample.

        Pipeline (per sample):
          1. Spectral-cluster the valid token positions using the attention map.
          2. For each cluster, average the hidden states of already-decoded tokens
             via scatter-add → centroid h̄_c.
          3. Apply the model's final layer norm (if present), then pass all
             centroids through the LM head in a single batched call → g_c.
          4. For every masked position i whose cluster has a valid centroid:
               l̂_i = l_i + γ · g_{C(i)}

        Returns the biased logits tensor (same shape as `logits`).
        """
        if gamma == 0.0:
            return logits

        # ---- Step 1: cluster assignment [valid_len] -------------------------
        cluster_labels_np = self._spectral_cluster(
            attn[:valid_len, :valid_len], n_clusters, valid_len
        )
        labels   = torch.from_numpy(cluster_labels_np).to(hidden.device)  # [valid_len]
        k_actual = int(labels.max().item()) + 1   # number of clusters actually used

        # ---- Step 2: build centroids via scatter-add (no Python loop) -------
        decoded_valid = ~mask_index[:valid_len]   # [valid_len] bool
        masked_valid  =  mask_index[:valid_len]   # [valid_len] bool

        d = hidden.shape[-1]
        centroid_sum  = torch.zeros(k_actual, d, dtype=hidden.dtype,  device=hidden.device)
        anchor_counts = torch.zeros(k_actual,    dtype=torch.long,    device=hidden.device)

        decoded_indices = torch.where(decoded_valid)[0]           # positions of decoded tokens
        if decoded_indices.numel() > 0:
            decoded_labels = labels[decoded_indices]              # their cluster ids  [D]
            decoded_hidden = hidden[:valid_len][decoded_indices]  # their hidden states [D, d]
            # Accumulate sum and count per cluster
            centroid_sum.scatter_add_(
                0,
                decoded_labels.unsqueeze(1).expand(-1, d),
                decoded_hidden,
            )
            anchor_counts.scatter_add_(
                0, decoded_labels, torch.ones_like(decoded_labels)
            )

        # Only use clusters that have at least min_anchor_size decoded tokens
        valid_cluster = anchor_counts >= min_anchor_size           # [k_actual] bool
        if not valid_cluster.any():
            return logits

        safe_counts = anchor_counts.to(centroid_sum.dtype).clamp(min=1.0).unsqueeze(1)  # [k_actual, 1]
        centroids   = centroid_sum / safe_counts                           # [k_actual, d]

        # ---- Step 3: apply output norm + LM head (single batched call) ------
        # Identify masked positions whose cluster has a valid centroid
        masked_indices        = torch.where(masked_valid)[0]       # [N_masked]
        masked_cluster_labels = labels[masked_indices]             # [N_masked]
        has_guidance          = valid_cluster[masked_cluster_labels]  # [N_masked] bool
        guided_pos            = masked_indices[has_guidance]        # [M]
        guided_cluster_ids    = masked_cluster_labels[has_guidance] # [M]

        if guided_pos.numel() == 0:
            return logits

        # One centroid per guided masked position: [M, d]
        centroid_stack = centroids[guided_cluster_ids]

        # NOTE: no explicit norm call here.
        # Both Dream and LLaDA already apply their final layer norm *before*
        # appending the last entry to hidden_states, so hidden_states[-1] is
        # already post-norm and can be passed directly to the LM head.

        # Single batched LM-head call → [M, V]
        guidance = self._apply_lm_head(centroid_stack)

        # ---- Step 4: inject with a single tensor index assignment -----------
        guided_logits = logits.clone()
        guided_logits[guided_pos] = logits[guided_pos] + gamma * guidance

        return guided_logits

    # ------------------------------------------------------------------
    # LM-head accessor
    # ------------------------------------------------------------------

    def _apply_lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project `hidden` [M, d] → logits [M, V] using the model's output
        projection, handling the two cases found in this codebase:

        • Dream  — exposes `self.lm_head` (nn.Linear): call directly.
        • LLaDA  — exposes output projection via `get_output_embeddings()`:
            - weight-tied: returns nn.Embedding → use F.linear(hidden, w)
            - non-tied:    returns nn.Linear   → call directly.

        Falls back to `get_output_embeddings()` for any other architecture.
        """
        m = self.model
        if hasattr(m, "module"):   # unwrap DataParallel / DDP
            m = m.module

        # Fast path: standard HuggingFace lm_head (Dream, Qwen2, Llama, …)
        if hasattr(m, "lm_head"):
            return m.lm_head(hidden)

        # Fallback: use the standard HuggingFace API, then call correctly
        # depending on whether the output layer is Linear or Embedding.
        out_emb = m.get_output_embeddings()
        if out_emb is None:
            raise AttributeError(
                f"{type(m).__name__} has neither `lm_head` nor a registered "
                "output embedding. Override `_apply_lm_head` in a subclass."
            )
        if isinstance(out_emb, torch.nn.Embedding):
            # Weight-tied: the embedding matrix doubles as the output projection.
            # Must use F.linear — calling the Embedding module directly would
            # interpret `hidden` as integer indices.
            return F.linear(hidden, out_emb.weight)
        # nn.Linear or any callable
        return out_emb(hidden)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: ClusterGuidedSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = ClusterGuidedSamplerConfig()

        # ---- pull config (allow kwargs overrides) ----
        steps               = kwargs.get("steps",               config.steps)
        max_new_tokens      = kwargs.get("max_new_tokens",      config.max_new_tokens)
        max_length          = kwargs.get("max_length",          config.max_length)
        block_size          = kwargs.get("block_size",          config.block_size)
        temperature         = kwargs.get("temperature",         config.temperature)
        cfg_scale           = kwargs.get("cfg_scale",           config.cfg_scale)
        cfg_keep_tokens     = kwargs.get("cfg_keep_tokens",     config.cfg_keep_tokens)
        remasking           = kwargs.get("remasking",           config.remasking)
        suppress_tokens     = kwargs.get("suppress_tokens",     config.suppress_tokens)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict         = kwargs.get("return_dict",         config.return_dict)
        right_shift_logits  = kwargs.get("right_shift_logits",  config.right_shift_logits)
        
        cluster_attn_layer  = int(kwargs.get("cluster_attention_layer_idx", config.cluster_attention_layer_idx))
        n_clusters          = int(kwargs.get("n_clusters",          config.n_clusters))
        gamma_alpha         = float(kwargs.get("gamma_alpha",        config.gamma_alpha))
        gamma_beta          = float(kwargs.get("gamma_beta",         config.gamma_beta))
        min_anchor_size     = int(kwargs.get("min_anchor_size",      config.min_anchor_size))

        assert 1 <= block_size
        assert 1 <= steps

        mask_id = self.tokenizer.mask_token_id
        bos_id  = self.tokenizer.bos_token_id
        eos_id  = self.tokenizer.eos_token_id

        # ---- build canvas ----
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p
                for p in inputs
            ]
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            attention_mask[i, : min(pl + max_new_tokens, T)] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if cfg_keep_tokens:
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ---- block loop ----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end   = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    block_mask_index[j, : end - start] = (x[j, start:end] == mask_id)

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                mask_index = x == mask_id  # [B, T]

                # γ(t): t goes from 0..effective_steps-1
                gamma = self._gamma(i, effective_steps, gamma_alpha, gamma_beta)

                # ---- forward pass ----
                need_hidden   = True   # always needed for guidance
                need_attn     = True   # needed for clustering
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    _out = self.model(
                        x_,
                        attention_mask=attention_mask.repeat(2, 1),
                        output_hidden_states=need_hidden,
                        output_attentions=need_attn,
                    )
                    logits, un_logits = torch.chunk(_out.logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    hidden_states = _out.hidden_states[-1][:B]    # [B, T, d]
                    attn_weights  = _out.attentions[cluster_attn_layer][:B]  # [B, H, T, T]
                else:
                    _out = self.model(
                        x,
                        attention_mask=attention_mask,
                        output_hidden_states=need_hidden,
                        output_attentions=need_attn,
                    )
                    logits        = _out.logits                     # [B, T, V]
                    hidden_states = _out.hidden_states[-1]          # [B, T, d]
                    attn_weights  = _out.attentions[cluster_attn_layer]  # [B, H, T, T]

                if suppress_tokens:
                    for tok in suppress_tokens:
                        logits[:, :, tok] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # ---- cluster-guided logit adjustment ----
                # Average attention over heads: [B, H, T, T] -> [B, T, T]
                attn_avg = attn_weights.mean(dim=1)  # [B, T, T]

                for j in range(B):
                    valid_len = int(attention_mask[j].sum().item())
                    logits[j] = self._compute_guidance_logits(
                        hidden     = hidden_states[j],        # [T, d]
                        logits     = logits[j],               # [T, V]
                        attn       = attn_avg[j],             # [T, T]
                        mask_index = mask_index[j],           # [T]
                        valid_len  = valid_len,
                        n_clusters = n_clusters,
                        min_anchor_size = min_anchor_size,
                        gamma      = gamma,
                    )

                # ---- token selection ----
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens:
                    for tok in begin_suppress_tokens:
                        logits[:, :, tok] = -torch.inf

                if remasking == "low_confidence":
                    p_soft = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p_soft, -1, x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=x.device)
                else:
                    raise NotImplementedError(f"remasking={remasking!r} not supported by ClusterGuidedSampler")

                # Restrict to current block
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                x0         = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(B):
                    _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, sel] = True

                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    # ------------------------------------------------------------------
    # infill  (mirrors sample; shares the same guidance logic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: ClusterGuidedSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        raise NotImplementedError(
            "ClusterGuidedSampler.infill will be implemented in a later step."
        )
