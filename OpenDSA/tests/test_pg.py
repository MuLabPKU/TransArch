"""test_pg.py — 2-process smoke test for opendsa.dist collectives.

Run:  torchrun --nproc_per_node=2 tests/test_pg.py
Also runs standalone (world=1, trivial mode):  python tests/test_pg.py
"""
import _path  # noqa: F401
import torch
from opendsa.dist import (init_parallel, cp_size, cp_rank, local_shard,
                          zigzag_local_gpos,
                          all_gather_seq, all_to_all, is_dist,
                          reduce_scatter_tensor, all_gather_into_tensor, cp_group)


def main():
    S = init_parallel()
    dev = f"cuda:{S.local_rank}" if torch.cuda.is_available() else "cpu"
    n = cp_size()
    r = cp_rank()

    # --- local_shard (zigzag) + all_gather_seq round-trip (no grad) ---
    L = 8
    full = torch.arange(L, device=dev).float().unsqueeze(-1)     # [L,1] known
    shard = local_shard(full, dim=0)                            # [L/n,1] zigzag rows
    Lloc = shard.shape[0]
    # this rank's zigzag shard must hold exactly its global-position rows, in order
    gpos = zigzag_local_gpos(L, dev)
    assert torch.equal(shard.squeeze(-1), gpos.float()), f"zigzag shard mismatch rank {r}"
    gathered = all_gather_seq(shard, grad=False)                # reordered to sequential
    assert torch.equal(gathered, full), f"gather mismatch on rank {r}"
    if r == 0:
        print(f"[pg] zigzag local_shard+all_gather round-trip OK (n={n})")

    # --- AllGatherSeq autograd: grad should reduce-scatter back to the shard ---
    x = shard.clone().requires_grad_(True)
    g = all_gather_seq(x, grad=True)                            # [L,1]
    # loss = sum over full seq of g^2 ; dL/dg = 2g ; each rank keeps its slice,
    # but all ranks contribute the same g (identical copies), so backward all-reduces
    # -> local grad = n * 2 * shard
    (g.pow(2).sum()).backward()
    expected = (2.0 * shard) * (n if is_dist() else 1)
    err = (x.grad - expected).abs().max().item()
    assert err < 1e-5, f"AllGatherSeq backward mismatch rank {r}: {err}"
    if r == 0:
        print(f"[pg] AllGatherSeq autograd backward OK (err={err:.1e})")

    # --- all_to_all: rank sends chunk i to rank i ---
    if is_dist():
        a = (torch.full((n, 3), float(r), device=dev))         # [n,3], all = my rank
        b = all_to_all(a)                                      # row i now = value from rank i
        expect = torch.arange(n, device=dev).float().unsqueeze(-1).expand(n, 3)
        assert torch.equal(b, expect), f"all_to_all mismatch rank {r}: {b}"
        if r == 0:
            print(f"[pg] all_to_all OK")

    # --- reduce_scatter_tensor + all_gather_into_tensor round-trip (ZeRO-2 building
    # blocks). Each rank contributes an identical [n*4] full tensor of ones; after
    # reduce_scatter(SUM) each rank should own a [4]-slice of value n. Then
    # all_gather_into_tensor reassembles the full [n*4] with value n everywhere. ---
    if is_dist():
        full_ones = torch.ones(n * 4, device=dev, dtype=torch.float32)
        shard_out = torch.empty(4, device=dev, dtype=torch.float32)
        reduce_scatter_tensor(shard_out, full_ones, group=cp_group())
        expect_shard = torch.full((4,), float(n), device=dev)
        err = (shard_out - expect_shard).abs().max().item()
        assert err < 1e-6, f"reduce_scatter_tensor mismatch rank {r}: {err}"

        gathered = torch.empty(n * 4, device=dev, dtype=torch.float32)
        all_gather_into_tensor(gathered, shard_out, group=cp_group())
        expect_full = torch.full((n * 4,), float(n), device=dev)
        err = (gathered - expect_full).abs().max().item()
        assert err < 1e-6, f"all_gather_into_tensor mismatch rank {r}: {err}"
        if r == 0:
            print(f"[pg] reduce_scatter_tensor + all_gather_into_tensor OK")

    if r == 0:
        print("ALL PG TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        _path.shutdown_dist()
