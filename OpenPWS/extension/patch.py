"""Scoped vLLM indexer intervention. Single sequence, eager, no prefix cache only.

No checkpoint/config/package files are changed. Imported via worker_extension_cls.
"""

import hashlib
import os
from pathlib import Path

import torch

from extension.kernels import candidate_scores, dense_scores, select_candidates

MODE = None
ACTIVE = False
ORIGINAL = None
ORIGINAL_INIT = None
SCRATCH = None
STATES = {}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    return dict(
        mode=MODE,
        active=ACTIVE,
        patch_sha256=file_sha(__file__),
        kernels_sha256=file_sha(Path(__file__).with_name("kernels.py")),
        autotune_cache_sha256=file_sha(os.environ["PWS_AUTOTUNE_CONFIG"])
        if os.environ.get("PWS_AUTOTUNE_CONFIG")
        else None,
        layers={
            k: {
                name: s[name]
                for name in [
                    "calls",
                    "prefill_calls",
                    "decode_calls",
                    "resets",
                    "next_pos",
                    "token_rows",
                    "selection_rows",
                    "selection_audits",
                ]
            }
            for k, s in STATES.items()
        },
    )


class AuditWorker:
    def glm_indexer_audit(self):
        return audit()

    def glm_indexer_set_mode(self, mode):
        # Called only between complete llm.generate calls; every new prompt starts at 0.
        global MODE, ACTIVE
        assert mode in ("native", "hisa", "dsa")
        MODE = "direct" if mode == "dsa" else "two_stage"
        ACTIVE = mode != "native"
        return mode


def install(mode):
    global MODE, ORIGINAL, ORIGINAL_INIT
    assert mode in ("two_stage", "direct")
    if MODE is not None:
        assert mode == MODE
        return
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    MODE = mode
    ORIGINAL = SparseAttnIndexerKpool.forward_cuda
    ORIGINAL_INIT = SparseAttnIndexerKpool.__init__
    SparseAttnIndexerKpool.__init__ = initialize
    SparseAttnIndexerKpool.forward_cuda = forward
    import importlib

    warmup = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
    if os.environ.get("PWS_AUTOTUNE_CONFIG"):
        warmup.flashinfer_autotune = load_baseline_autotune
    print(f"GLM indexer ablation installed: {mode}", flush=True)


def load_baseline_autotune(runner):
    # Pin baseline tactics instead of re-benchmarking a synchronized MoE tune.
    from flashinfer.autotuner import AutoTuner

    path = Path(os.environ["PWS_AUTOTUNE_CONFIG"])
    assert AutoTuner.get().load_configs(str(path)), "Incompatible baseline autotune cache"
    print(f"Loaded frozen baseline FlashInfer tactics: {file_sha(path)}", flush=True)


def initialize(self, *args, **kwargs):
    from vllm.config import get_current_vllm_config

    ORIGINAL_INIT(self, *args, **kwargs)
    self._glm_ablation_config = get_current_vllm_config()


def state_for(self, device):
    # vLLM sets its config during construction, not necessarily in profile/forward.
    cfg = self._glm_ablation_config
    assert cfg.scheduler_config.max_num_seqs == 1
    assert cfg.model_config.enforce_eager and not cfg.cache_config.enable_prefix_caching
    assert cfg.speculative_config is None
    assert cfg.parallel_config.prefill_context_parallel_size == 1
    assert cfg.parallel_config.decode_context_parallel_size == 1
    prefix = str(self.k_cache.prefix)
    if prefix not in STATES:
        cap = ((cfg.model_config.max_model_len + 127) // 128) * 128
        STATES[prefix] = dict(
            k=torch.empty((cap, 128), device=device, dtype=torch.float8_e4m3fn),
            ks=torch.empty(cap, device=device, dtype=torch.float32),
            calls=0,
            prefill_calls=0,
            decode_calls=0,
            resets=0,
            next_pos=0,
            token_rows=0,
            selection_rows=0,
            selection_audits=[],
        )
    return STATES[prefix]


def forward(
    self,
    hidden_states,
    q_quant,
    k,
    weights,
    *,
    gate_score=None,
    compress_ape=None,
    index_kpool=1,
    positions=None,
):
    global SCRATCH
    from vllm.forward_context import get_forward_context
    from vllm.models.glm5next.common.sparse_indexer import _fill_causal_indices
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
    from vllm.utils.torch_utils import _resolve_layer_name

    assert index_kpool == 4 and self.topk_tokens == 2048 and not self.use_fp4_cache
    assert not isinstance(q_quant, tuple) and not self.skip_k_cache_insert
    state = state_for(self, hidden_states.device)
    metadata = get_forward_context().attn_metadata
    output = self.topk_indices_buffer
    # The shared scratch is reused across layers and accounted for during profiling.
    if MODE == "two_stage" and (SCRATCH is None or SCRATCH.shape[0] < output.shape[0]):
        SCRATCH = torch.empty((output.shape[0], 8195), dtype=torch.int32, device=output.device)
    if not ACTIVE or not isinstance(metadata, dict):
        # Engine profiling/autotuning uses synthetic (noncontiguous) positions.
        # Stay native until the controller explicitly arms real-request inference.
        # Persistent experimental buffers above are still counted in memory profiling.
        return ORIGINAL(
            self,
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )
    meta = metadata[_resolve_layer_name(self.k_cache.prefix)]
    assert meta.num_decodes + meta.num_prefills == 1, "Sidecar cache requires one active request"
    n = meta.slot_mapping.shape[0]
    assert positions is not None and n > 0
    start, end = (int(x) for x in positions[[0, n - 1]].tolist())
    end += 1
    assert end - start == n, "Noncontiguous token positions"
    if start == 0:
        state["next_pos"] = 0
        state["resets"] += 1
    assert start == state["next_pos"], (
        f"Uncached prefix/reordered request: {start} vs {state['next_pos']}"
    )
    assert end <= state["k"].shape[0]
    raw, scale = fwht128_quant_fp8(k[:n].contiguous())
    state["k"][start:end].copy_(raw)
    state["ks"][start:end].copy_(scale.flatten())
    state["next_pos"] = end
    state["calls"] += 1
    state["prefill_calls"] += int(meta.num_prefills > 0)
    state["decode_calls"] += int(meta.num_decodes > 0)
    state["token_rows"] += n
    candidates = None
    if MODE == "two_stage":
        # Run the untouched native pool selection with 4x the pool budget.
        # It also updates its paged pool/tail caches, including during short prefills.
        old_budget, old_buffer = self.topk_tokens, self.topk_indices_buffer
        try:
            self.topk_tokens = 8192
            self.topk_indices_buffer = SCRATCH
            candidates = ORIGINAL(
                self,
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )[:n]
        finally:
            self.topk_tokens, self.topk_indices_buffer = old_budget, old_buffer
    output[:n].fill_(-1)
    if end <= 2048:
        _fill_causal_indices(output[:n, :2048], positions[:n])
        return output
    # Chunking bounds temporary logits; semantics are identical for prefill/decode.
    for left in range(0, n, 128):
        right = min(left + 128, n)
        pos = positions[left:right].to(torch.int32)
        if MODE == "two_stage":
            ids = candidates[left:right]
            scores = candidate_scores(
                q_quant[left:right],
                state["k"][:end],
                state["ks"][:end],
                weights[left:right],
                ids,
                pos,
            )
            selected = select_candidates(scores, ids)
        else:
            scores = dense_scores(
                q_quant[left:right], state["k"][:end], state["ks"][:end], weights[left:right], pos
            )
            values, indices = torch.topk(scores, 2048, dim=-1, sorted=False)
            selected = torch.where(torch.isfinite(values), indices, -1).to(torch.int32)
        output[left:right, :2048] = selected
    if state["resets"] == 1 and (state["calls"] == 1 or n == 1):
        for row in sorted(set((0, n - 1))):
            query_pos = int(positions[row])
            chosen = output[row, :2048].tolist()
            chosen = [i for i in chosen if i >= 0]
            assert len(chosen) == min(query_pos + 1, 2048)
            assert len(set(chosen)) == len(chosen) and max(chosen) <= query_pos
            if candidates is not None:
                assert set(chosen) <= set(candidates[row].tolist())
            state["selection_audits"].append(
                dict(
                    query_position=query_pos,
                    count=len(chosen),
                    unique_and_causal=True,
                    within_candidate_set=candidates is not None,
                )
            )
    state["selection_rows"] += n
    return output
