"""Exercise actual patched direct forward, cache reset, causality and decode off-model."""

import json
from types import SimpleNamespace as S
from unittest.mock import patch as mockpatch

import torch

from extension import patch as p


def main():
    torch.manual_seed(17)
    cfg = S(
        scheduler_config=S(max_num_seqs=1),
        model_config=S(enforce_eager=True, max_model_len=5000),
        cache_config=S(enable_prefix_caching=False),
        speculative_config=None,
        parallel_config=S(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    op = S(
        _glm_ablation_config=cfg,
        k_cache=S(prefix="runtime_test"),
        topk_tokens=2048,
        use_fp4_cache=False,
        skip_k_cache_insert=False,
        topk_indices_buffer=torch.empty((2304, 2051), device="cuda", dtype=torch.int32),
    )
    q = (torch.randn(2304, 32, 128, device="cuda") * 4).to(torch.float8_e4m3fn)
    k = torch.randn(2304, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2304, 32, device="cuda") * 0.01
    pos = torch.arange(2304, device="cuda")
    meta = S(num_decodes=0, num_prefills=1, slot_mapping=torch.zeros(2304, device="cuda"))
    ctx = S(attn_metadata={"runtime_test": meta})
    p.MODE = "direct"
    p.ACTIVE = True
    with mockpatch("vllm.forward_context.get_forward_context", return_value=ctx):
        a = p.forward(op, k, q, k, w, index_kpool=4, positions=pos).clone()
        # Decode extends the same sidecar by one token.
        meta.num_decodes = 1
        meta.num_prefills = 0
        meta.slot_mapping = torch.zeros(1, device="cuda")
        dec = p.forward(
            op,
            k[:1],
            q[:1],
            k[:1],
            w[:1],
            index_kpool=4,
            positions=torch.tensor([2304], device="cuda"),
        )[:1].clone()
        assert len(set(dec[0, :2048].tolist())) == 2048 and int(dec.max()) <= 2304
        try:
            p.forward(
                op,
                k[:1],
                q[:1],
                k[:1],
                w[:1],
                index_kpool=4,
                positions=torch.tensor([2306], device="cuda"),
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("Missing uncached-prefix guard")
        meta.num_decodes = 0
        meta.num_prefills = 1
        meta.slot_mapping = torch.zeros(2304, device="cuda")
        b = p.forward(op, k, q, k, w, index_kpool=4, positions=pos).clone()
        assert torch.equal(a, b), "Sidecar reset changed selection on identical inputs"
    assert bool((a[:, 2048:] == -1).all())
    for row in [0, 1, 2047, 2048, 2303]:
        ids = a[row][a[row] >= 0]
        assert (
            len(ids) == min(row + 1, 2048)
            and len(ids) == len(ids.unique())
            and int(ids.max()) <= row
        )
    result = dict(
        passed=True,
        prefill=True,
        decode=True,
        reset_bitwise_repeat=True,
        uncached_prefix_rejected=True,
        strict_2048_and_causal=True,
        sampled_selection_audits=p.audit()["layers"]["runtime_test"]["selection_audits"],
    )

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
