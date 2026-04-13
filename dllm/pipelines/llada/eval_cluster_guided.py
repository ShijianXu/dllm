"""
accelerate launch --num_processes 1 \
    dllm/pipelines/llada/eval_cluster_guided.py \
    --tasks "gsm8k_cot" \
    --model "llada_cluster" \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],n_clusters=8,gamma_alpha=1.0,gamma_beta=1.0,min_anchor_size=1"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig
from dllm.core.eval.mdlm import MDLMEvalHarness
from dllm.core.samplers import ClusterGuidedSampler, ClusterGuidedSamplerConfig


@dataclass
class LLaDAClusterEvalSamplerConfig(ClusterGuidedSamplerConfig):
    """Default sampler config for LLaDA cluster-guided eval."""

    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDAClusterEvalConfig(MDLMEvalConfig):
    max_length: int = 4096


@register_model("llada_cluster")
class LLaDAClusterEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: LLaDAClusterEvalConfig | None = None,
        sampler_config: ClusterGuidedSamplerConfig | None = None,
        **kwargs,
    ):
        eval_config = eval_config or LLaDAClusterEvalConfig()
        sampler_config = sampler_config or LLaDAClusterEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=ClusterGuidedSampler,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
