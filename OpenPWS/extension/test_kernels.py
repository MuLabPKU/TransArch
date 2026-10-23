"""GPU score/mask/top-k tests against an independent dequantized reference."""

import torch
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

from extension.kernels import candidate_scores, select_candidates


def main():
    torch.manual_seed(1729)
    results = []
    for n, m, c in [(4096, 4, 4099), (8448, 3, 8195), (4101, 1, 4104), (128007, 1, 8195)]:
        q = (torch.randn(m, 32, 128, device="cuda") * 8).to(torch.float8_e4m3fn)
        k = (torch.randn(n, 128, device="cuda") * 8).to(torch.float8_e4m3fn)
        ks = torch.rand(n, device="cuda") * 0.02 + 0.01
        w = torch.randn(m, 32, device="cuda") * 0.01
        pos = torch.tensor(
            [n - 1] if m == 1 else [0, n // 2, n - 2, n - 1][:m], device="cuda", dtype=torch.int32
        )
        ids = torch.stack([torch.randperm(n, device="cuda")[: min(c, n)] for _ in range(m)]).int()
        if c > n:
            ids = torch.cat([ids, torch.full((m, c - n), -1, device="cuda", dtype=torch.int32)], 1)
        got = candidate_scores(q, k, ks, w, ids, pos)
        reference = []
        for i in range(m):
            safe = ids[i].long().clamp_min(0)
            dot = q[i].float() @ k[safe].float().T
            scores = (dot.relu() * w[i, :, None]).sum(0) * ks[safe]
            valid = (ids[i] >= 0) & (ids[i] <= pos[i])
            reference.append(scores.masked_fill(~valid, -torch.inf))
        reference = torch.stack(reference)
        torch.testing.assert_close(got, reference, rtol=3e-4, atol=2e-3)
        selected = select_candidates(got, ids)
        assert selected.shape == (m, 2048)
        for i in range(m):
            chosen = selected[i][selected[i] >= 0]
            assert len(chosen) == len(chosen.unique())
            assert bool((chosen <= pos[i]).all())
            assert set(chosen.tolist()) <= set(ids[i].tolist())
        dense = fp8_fp4_mqa_logits(
            (q, None), (k, ks), w, torch.zeros_like(pos), pos + 1, clean_logits=True
        )
        valid = ids >= 0
        gathered = torch.gather(dense, 1, ids.long().clamp_min(0)).masked_fill(~valid, -torch.inf)
        torch.testing.assert_close(got, gathered, rtol=0, atol=0)
        for i in range(m):
            valid_scores = reference[i][torch.isfinite(reference[i])]
            if len(valid_scores) >= 2048:
                threshold = valid_scores.topk(2048).values[-1]
                chosen = selected[i][selected[i] >= 0].long()
                assert bool((dense[i, chosen] >= threshold - 2e-3).all())
        results.append(
            dict(
                n=n,
                queries=m,
                candidates=c,
                fp32_reference_within_2e_3_and_native_bitwise_match=True,
                causal_unique_subset_and_budget=True,
            )
        )
        print(results[-1], flush=True)


if __name__ == "__main__":
    main()
