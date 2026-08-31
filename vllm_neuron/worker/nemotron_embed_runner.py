# SPDX-License-Identifier: Apache-2.0
"""Pooling/embedding model runner for precompiled Nemotron NxD artifacts.

The stock Neuron runner only implements the decoder generate path
(``pooler_output=[]``), and NxDI has no embedding-task model registry, so
embedding models cannot be served through the normal NxDI loader. This runner
instead executes a self-contained ``nemotron-nxd-tp2-v1`` artifact (torchscript
``NxDModel`` traced with one NEFF per ``(seq_len, batch_size)`` bucket, TP=2
weights inline) produced by the monorepo compile recipe
(``python/services/embed/embed/nemotron_compile_tp2.py``).

Each traced bucket takes ``(input_ids[int32 B,S], attention_mask[int32 B,S])``
and returns L2-normalized masked-mean-pooled embeddings ``[B, 4096]`` in fp32,
so there is no KV cache, no sampling, and every request completes in a single
forward. Requests are grouped by the smallest bucket that fits their prompt
and executed batch-by-batch within one scheduler step.

Enable with::

    --runner pooling --additional-config \
        '{"nemotron_embed_artifact_dir": "/path/with/model.pt+manifest.json"}'
"""

import json
import logging
import os
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class _Bucket:
    seq_len: int
    batch_size: int
    tag: str


class NemotronEmbedModelRunner:
    """Single-forward, KV-cache-free embedding runner for NxD artifacts."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        additional = vllm_config.additional_config or {}
        self.artifact_dir = additional.get("nemotron_embed_artifact_dir")
        if not self.artifact_dir:
            raise ValueError(
                "The pooling runner requires --additional-config "
                '\'{"nemotron_embed_artifact_dir": "..."}\' pointing at a '
                "nemotron-nxd-tp2-v1 artifact directory"
            )
        self.model = None
        self.buckets: list[_Bucket] = []
        self.pad_token_id: int = 11
        self.hidden_size: int = 4096
        # Retained decoder-runner attribute contract used by NeuronWorker.
        self.is_block_kv_layout = False

    def load_model(self) -> None:
        from neuronx_distributed.trace.nxd_model import NxDModel

        with open(os.path.join(self.artifact_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["schema"] == "nemotron-nxd-tp2-v1", manifest["schema"]
        self.buckets = sorted(
            (_Bucket(b["seq_len"], b["batch_size"], b["tag"]) for b in manifest["buckets"]),
            key=lambda b: b.seq_len,
        )
        with open(os.path.join(self.artifact_dir, "config.json")) as f:
            hf_config = json.load(f)
        self.pad_token_id = hf_config.get("pad_token_id", 11)
        self.hidden_size = hf_config.get("hidden_size", 4096)
        model_path = os.path.join(self.artifact_dir, "model.pt")
        logger.info("Loading NxD embedding artifact from %s", model_path)
        self.model = NxDModel.load(model_path)

    def _bucket_for(self, length: int) -> _Bucket:
        for bucket in self.buckets:
            if length <= bucket.seq_len:
                return bucket
        return self.buckets[-1]

    def _forward_bucket(
        self, bucket: _Bucket, prompts: list[list[int]]
    ) -> torch.Tensor:
        """Pad ``prompts`` into the bucket shape and run one traced forward."""
        ids = torch.full(
            (bucket.batch_size, bucket.seq_len), self.pad_token_id, dtype=torch.int32
        )
        mask = torch.zeros((bucket.batch_size, bucket.seq_len), dtype=torch.int32)
        for row, prompt in enumerate(prompts):
            live = min(len(prompt), bucket.seq_len)
            ids[row, :live] = torch.tensor(prompt[:live], dtype=torch.int32)
            mask[row, :live] = 1
        with torch.inference_mode():
            out = self.model(ids, mask, model_name=bucket.tag)
        return out[: len(prompts)].to(torch.float32).cpu()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ) -> ModelRunnerOutput:
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        reqs = scheduler_output.scheduled_new_reqs
        if scheduler_output.scheduled_cached_reqs.req_ids:
            raise RuntimeError(
                "Pooling requests must be scheduled as whole prompts "
                "(chunked prefill is unsupported); got cached requests "
                f"{scheduler_output.scheduled_cached_reqs.req_ids}"
            )

        req_ids = [r.req_id for r in reqs]
        by_bucket: dict[_Bucket, list[int]] = {}
        for i, req in enumerate(reqs):
            by_bucket.setdefault(self._bucket_for(len(req.prompt_token_ids)), []).append(i)

        pooler_output: list[torch.Tensor | None] = [None] * len(reqs)
        for bucket, indices in by_bucket.items():
            for start in range(0, len(indices), bucket.batch_size):
                chunk = indices[start : start + bucket.batch_size]
                embeddings = self._forward_bucket(
                    bucket, [reqs[i].prompt_token_ids for i in chunk]
                )
                for row, i in enumerate(chunk):
                    pooler_output[i] = embeddings[row]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        # Single-forward embedding model: no attention KV cache at all.
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        return None

    def ensure_kv_transfer_shutdown(self) -> None:
        return None

    def _dummy_run(self, *args, **kwargs) -> None:
        return None
