import torch
import tilelang
from tilelang import language as T
from tilelang.autotuner import *
from tilelang_utils import prepare_ks_ke_from_cu_seqlens_qk
from tilelang.profiler import do_bench
import itertools
import argparse

# ============================================================
# 默认测试参数（与 test_speed.py 一致）
# ============================================================
B = 1
SQ = 2048
SK = 131072
H = 64
INDEX_D = 128
TOPK_HEADS = 64
K_BLOCK_SIZE = 128
BLOCK_TOPK = 128
STAGE_1_TOPK = 8192

WARMUP = 10
REP = 10

global_pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}

# ============================================================
# 模拟数据生成（使用BF16格式，匹配kernel实现）
# ============================================================
def make_test_data():
    torch.manual_seed(42)
    q_bf16 = torch.randn((SQ, H, INDEX_D)).cuda().bfloat16()
    k_bf16 = torch.randn((SK, INDEX_D)).cuda().bfloat16()
    weights = torch.randn((SQ, H)).cuda().bfloat16()

    offsets_q = list(range(0, SQ, SQ // B))[:B] + [SQ]
    offsets_k = list(range(0, SK, SK // B))[:B] + [SK]
    offsets_q = torch.tensor(offsets_q, dtype=torch.int32).cuda()
    offsets_k = torch.tensor(offsets_k, dtype=torch.int32).cuda()

    cu_seqlen_ks, cu_seqlen_ke = prepare_ks_ke_from_cu_seqlens_qk(offsets_q, offsets_k)

    return {
        "q_bf16": q_bf16,
        "k_bf16": k_bf16,
        "weights": weights,
        "cu_seqlen_ks": cu_seqlen_ks,
        "cu_seqlen_ke": cu_seqlen_ke,
    }

# ============================================================
# Kernel 1: misa_mqa_sparse_head_return_logits
# ============================================================
def get_configs_sparse_head():
    params = dict(
        block_N=[64, 128, 256],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )
    return [dict(zip(params, v)) for v in itertools.product(*params.values())]

def bench_sparse_head(data):
    from misa_ops import misa_mqa_sparse_head_return_logits

    q_bf16 = data["q_bf16"][:, :TOPK_HEADS, :].contiguous()
    weights = data["weights"][:, :TOPK_HEADS].contiguous()
    k_bf16 = data["k_bf16"]
    cu_ks = data["cu_seqlen_ks"]
    cu_ke = data["cu_seqlen_ke"]

    seq_len, heads, index_dim = q_bf16.shape
    seq_len_kv = k_bf16.shape[0]

    q_flat = q_bf16.reshape(seq_len * heads, index_dim).contiguous()
    logits = torch.full([seq_len, seq_len_kv], float('-inf'), device=q_bf16.device, dtype=torch.float32)

    best_latency = float('inf')
    best_config = None

    for config in get_configs_sparse_head():
        block_N = config["block_N"]
        num_stages = config["num_stages"]
        threads = config["threads"]

        try:
            kernel = misa_mqa_sparse_head_return_logits(
                heads=heads, index_dim=index_dim,
                block_N=block_N, num_stages=num_stages, threads=threads,
                dtype="bfloat16",
            )

            logits_test = torch.full_like(logits, float('-inf'))

            def run():
                kernel(q_flat, k_bf16, logits_test, weights, cu_ks, cu_ke)

            latency = do_bench(run, warmup=WARMUP, rep=REP)
            print(f"  sparse_head config={config}, latency={latency:.4f} ms")

            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"  sparse_head config={config} FAILED: {e}")

    return best_latency, best_config

# ============================================================
# Kernel 2: block_mean_pooling
# ============================================================
def get_configs_mean_pooling():
    params = dict(
        block_N=[64, 128, 256],
        num_stages=[1, 2],
        threads=[128, 256],
    )
    return [dict(zip(params, v)) for v in itertools.product(*params.values())]

def bench_mean_pooling(data):
    from block_ops import block_mean_pooling

    k_bf16 = data["k_bf16"]
    seq_len_k, dim = k_bf16.shape
    max_num_pooling_blocks = (seq_len_k + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE
    blocked_k = torch.empty((max_num_pooling_blocks, dim), device=k_bf16.device, dtype=torch.float32)

    best_latency = float('inf')
    best_config = None

    for config in get_configs_mean_pooling():
        block_N = config["block_N"]
        num_stages = config["num_stages"]
        threads = config["threads"]

        try:
            kernel = block_mean_pooling(
                pooling_block_size=K_BLOCK_SIZE, dim=dim,
                block_N=block_N, num_stages=num_stages, threads=threads,
                dtype="bfloat16",
            )

            out = torch.empty_like(blocked_k)

            def run():
                kernel(k_bf16, out)

            latency = do_bench(run, warmup=WARMUP, rep=REP)
            print(f"  mean_pooling config={config}, latency={latency:.4f} ms")

            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"  mean_pooling config={config} FAILED: {e}")

    return best_latency, best_config

# ============================================================
# Kernel 3: block_mqa_attn_return_logits
# ============================================================
def get_configs_block_mqa():
    params = dict(
        block_N=[64, 128, 256],
        num_stages=[1, 2, 3],
        threads=[128, 256, 512],
    )
    return [dict(zip(params, v)) for v in itertools.product(*params.values())]

def bench_block_mqa(data):
    from block_ops import block_mqa_attn_return_logits

    q_bf16 = data["q_bf16"]
    k_bf16 = data["k_bf16"]
    weights = data["weights"]
    cu_ks = data["cu_seqlen_ks"]
    cu_ke = data["cu_seqlen_ke"]

    seq_len, heads, index_dim = q_bf16.shape
    seq_len_kv = k_bf16.shape[0]

    max_num_pooling_blocks = (seq_len_kv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE
    blocked_k_bf16 = torch.randn((max_num_pooling_blocks, index_dim), device=q_bf16.device, dtype=torch.bfloat16)

    cu_seqlen_blocked_ks = cu_ks // K_BLOCK_SIZE
    cu_seqlen_blocked_ke = (cu_ke + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    q_flat = q_bf16.reshape(seq_len * heads, index_dim).contiguous()
    seq_len_blocked_kv = blocked_k_bf16.shape[0]
    q_head_score = torch.zeros([seq_len, heads], device=q_bf16.device, dtype=torch.float32)

    best_latency = float('inf')
    best_config = None

    for config in get_configs_block_mqa():
        block_N = config["block_N"]
        num_stages = config["num_stages"]
        threads = config["threads"]

        try:
            kernel = block_mqa_attn_return_logits(
                heads=heads, index_dim=index_dim,
                block_N=block_N, num_stages=num_stages, threads=threads,
                dtype="bfloat16",
            )

            q_score_test = torch.zeros_like(q_head_score)

            def run():
                kernel(q_flat, blocked_k_bf16, q_score_test, weights, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)

            latency = do_bench(run, warmup=WARMUP, rep=REP)
            print(f"  block_mqa config={config}, latency={latency:.4f} ms")

            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"  block_mqa config={config} FAILED: {e}")

    return best_latency, best_config

# ============================================================
# Kernel 4: batch_decode_block_mqa_attn_return_logits
# ============================================================
def get_configs_batch_decode_block_mqa():
    params = dict(
        block_N=[32, 64, 128],
        block_H=[32, 64, 128],
        num_stages=[1, 2, 3],
        threads=[128, 256, 512],
    )
    return [dict(zip(params, v)) for v in itertools.product(*params.values())]

def bench_batch_decode_block_mqa(data):
    from block_ops import batch_decode_block_mqa_attn_return_logits

    q_bf16 = data["q_bf16"]
    k_bf16 = data["k_bf16"]
    weights = data["weights"]
    cu_ks = data["cu_seqlen_ks"]
    cu_ke = data["cu_seqlen_ke"]

    seq_len, heads, index_dim = q_bf16.shape
    seq_len_kv = k_bf16.shape[0]

    # 模拟batch decode场景，q长度为1
    batch = seq_len
    q_decode = q_bf16[:, :1, :, :]  # [seq_len, 1, heads, index_dim]
    weights_decode = weights[:, :1, :]  # [seq_len, 1, heads]

    # 创建block KV
    max_num_pooling_blocks = (seq_len_kv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE
    blocked_k_bf16 = torch.randn((batch, max_num_pooling_blocks, index_dim), device=q_bf16.device, dtype=torch.bfloat16)

    # 计算每个序列的有效block数量
    context_lens = torch.full((batch,), max_num_pooling_blocks, device=q_bf16.device, dtype=torch.int32)

    q_head_score = torch.zeros([batch, heads], device=q_bf16.device, dtype=torch.float32)

    best_latency = float('inf')
    best_config = None

    for config in get_configs_batch_decode_block_mqa():
        block_N = config["block_N"]
        block_H = config["block_H"]
        num_stages = config["num_stages"]
        threads = config["threads"]

        # 确保block_H与heads兼容
        if heads > block_H:
            continue

        try:
            kernel = batch_decode_block_mqa_attn_return_logits(
                heads=heads,
                index_dim=index_dim,
                block_N=block_N,
                block_H=block_H,
                num_stages=num_stages,
                threads=threads,
                dtype="bfloat16",
            )

            q_score_test = torch.zeros_like(q_head_score)

            def run():
                kernel(
                    q_decode.view(batch, heads, index_dim),
                    blocked_k_bf16,
                    q_score_test,
                    weights_decode.view(batch, heads),
                    context_lens,
                )

            latency = do_bench(run, warmup=WARMUP, rep=REP)
            print(f"  batch_decode_block_mqa config={config}, latency={latency:.4f} ms")

            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"  batch_decode_block_mqa config={config} FAILED: {e}")

    return best_latency, best_config

# ============================================================
# Kernel 5: misa_mqa_attn_return_logits (stage2)
# ============================================================
def get_configs_mqa_attn():
    params = dict(
        block_N=[64, 128, 256],
        num_stages=[1, 2, 3],
        threads=[128, 256, 512],
    )
    return [dict(zip(params, v)) for v in itertools.product(*params.values())]

def bench_mqa_attn(data):
    from misa_ops import misa_mqa_attn_return_logits

    q_bf16 = data["q_bf16"]
    k_bf16 = data["k_bf16"]
    weights = data["weights"]
    cu_ks = data["cu_seqlen_ks"]
    cu_ke = data["cu_seqlen_ke"]

    seq_len, heads, index_dim = q_bf16.shape
    seq_len_kv = k_bf16.shape[0]

    q_flat = q_bf16.reshape(seq_len * heads, index_dim).contiguous()

    padded_topk = ((STAGE_1_TOPK + 128 - 1) // 128) * 128

    sparse_topk_indices = torch.randint(
        0, seq_len_kv, (seq_len, padded_topk),
        device=q_bf16.device, dtype=torch.int32,
    )

    padding_size = padded_topk - STAGE_1_TOPK
    if padding_size > 0:
        kv_padded = torch.cat([
            k_bf16,
            torch.zeros(padding_size, index_dim, device=k_bf16.device, dtype=k_bf16.dtype),
        ], dim=0)
    else:
        kv_padded = k_bf16

    logits = torch.empty([seq_len, padded_topk], device=q_bf16.device, dtype=torch.float32)

    best_latency = float('inf')
    best_config = None

    for config in get_configs_mqa_attn():
        block_N = config["block_N"]
        num_stages = config["num_stages"]
        threads = config["threads"]

        try:
            kernel = misa_mqa_attn_return_logits(
                heads=heads, index_dim=index_dim,
                block_N=block_N, num_stages=num_stages, threads=threads,
                dtype="bfloat16",
            )

            logits_test = torch.empty_like(logits)

            def run():
                kernel(q_flat, kv_padded, logits_test, weights, sparse_topk_indices, cu_ks, cu_ke)

            latency = do_bench(run, warmup=WARMUP, rep=REP)
            print(f"  mqa_attn config={config}, latency={latency:.4f} ms")

            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"  mqa_attn config={config} FAILED: {e}")

    return best_latency, best_config

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Autotune tilelang kernels for misa_indexer_topk_reducesum")
    parser.add_argument("--kernel", type=str, default="all",
                        choices=["all", "sparse_head", "mean_pooling", "block_mqa", "batch_decode_block_mqa", "mqa_attn"],
                        help="Which kernel to tune")
    args = parser.parse_args()

    data = make_test_data()

    results = {}

    if args.kernel in ("all", "sparse_head"):
        print("=" * 60)
        print("Tuning: misa_mqa_sparse_head_return_logits")
        print(f"  heads={TOPK_HEADS}, index_dim={INDEX_D}")
        print("=" * 60)
        lat, cfg = bench_sparse_head(data)
        results["misa_mqa_sparse_head_return_logits"] = (lat, cfg)

    if args.kernel in ("all", "mean_pooling"):
        print("=" * 60)
        print("Tuning: block_mean_pooling")
        print(f"  pooling_block_size={K_BLOCK_SIZE}, dim={INDEX_D}")
        print("=" * 60)
        lat, cfg = bench_mean_pooling(data)
        results["block_mean_pooling"] = (lat, cfg)

    if args.kernel in ("all", "block_mqa"):
        print("=" * 60)
        print("Tuning: block_mqa_attn_return_logits")
        print(f"  heads={H}, index_dim={INDEX_D}")
        print("=" * 60)
        lat, cfg = bench_block_mqa(data)
        results["block_mqa_attn_return_logits"] = (lat, cfg)

    # if args.kernel in ("all", "batch_decode_block_mqa"):
    #     print("=" * 60)
    #     print("Tuning: batch_decode_block_mqa_attn_return_logits")
    #     print(f"  heads={H}, index_dim={INDEX_D}")
    #     print("=" * 60)
    #     lat, cfg = bench_batch_decode_block_mqa(data)
    #     results["batch_decode_block_mqa_attn_return_logits"] = (lat, cfg)

    if args.kernel in ("all", "mqa_attn"):
        print("=" * 60)
        print("Tuning: misa_mqa_attn_return_logits (stage2)")
        print(f"  heads={H}, index_dim={INDEX_D}, stage_1_topk={STAGE_1_TOPK}")
        print("=" * 60)
        lat, cfg = bench_mqa_attn(data)
        results["misa_mqa_attn_return_logits"] = (lat, cfg)

    print("\n" + "=" * 60)
    print("AUTOTUNE RESULTS SUMMARY")
    print("=" * 60)
    for name, (lat, cfg) in results.items():
        print(f"  {name}:")
        print(f"    latency = {lat:.4f} ms")
        print(f"    config  = {cfg}")
    print("=" * 60)

if __name__ == "__main__":
    main()
