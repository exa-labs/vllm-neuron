# SPDX-License-Identifier: Apache-2.0
"""Registrations that let vLLM resolve nvidia/Nemotron-3-Embed-8B-BF16.

Two gaps block the model from even reaching the Neuron worker:

1. vLLM 0.16 pins ``transformers<5``, which has no ``ministral3`` model type,
   so ``AutoConfig.from_pretrained`` on the checkpoint fails. We register a
   pass-through ``PretrainedConfig`` for ``model_type="ministral3"`` —
   ``PretrainedConfig`` stores unknown kwargs as attributes, which is all
   vLLM's config plumbing needs (hidden_size, heads, layers, vocab, ...).

2. The ``Ministral3Model`` architecture is not in vLLM's model registry. We
   register a stub class carrying the pooling-model interface flags
   (``is_pooling_model``, MEAN pooling, encoder-only attention) so
   ``--runner pooling`` resolves with ``--convert none``. The stub is never
   executed: on Neuron the actual forward runs inside
   ``NemotronEmbedModelRunner`` against the precompiled NxD artifact.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_registered = False


def _make_config_cls():
    from transformers import PretrainedConfig

    class Ministral3Config(PretrainedConfig):
        model_type = "ministral3"

    return Ministral3Config


class NemotronEmbedStubModel(nn.Module):
    """Registry stub for ``Ministral3Model`` (never executed on Neuron)."""

    is_pooling_model = True
    default_seq_pooling_type = "MEAN"
    default_tok_pooling_type = "ALL"
    attn_type = "encoder_only"

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        raise RuntimeError(
            "NemotronEmbedStubModel exists only for registry inspection; "
            "on Neuron the model executes via NemotronEmbedModelRunner"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


def register_nemotron_embed() -> None:
    """Idempotently register the ministral3 config + architecture stub."""
    global _registered
    if _registered:
        return

    from transformers import AutoConfig
    from vllm import ModelRegistry

    try:
        AutoConfig.from_pretrained  # noqa: B018
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "ministral3" not in CONFIG_MAPPING:
            AutoConfig.register("ministral3", _make_config_cls())
            logger.info("Registered ministral3 passthrough config")
    except Exception:
        logger.exception("Failed to register ministral3 config")
        raise

    if "Ministral3Model" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Ministral3Model",
            "vllm_neuron.models.nemotron_embed:NemotronEmbedStubModel",
        )
        logger.info("Registered Ministral3Model pooling stub")

    _registered = True
