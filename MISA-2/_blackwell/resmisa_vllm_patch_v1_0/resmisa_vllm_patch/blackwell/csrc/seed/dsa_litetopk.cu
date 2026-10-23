#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include <cuda_fp16.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {
using CandidateValue = uint16_t;

constexpr uint32_t kCandidateIndexBits = 20;

constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload,
                                             const uint32_t kv_index) {
  return (kv_index & kCandidateIndexMask) |
         ((payload >> 16) << kCandidateIndexBits);
}

CUTLASS_DEVICE uint32_t candidate_ordered_fp32_code(const float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

CUTLASS_DEVICE uint32_t candidate_fp24_code(const float value) {
  return candidate_ordered_fp32_code(value) >> 8;
}

CUTLASS_DEVICE void store_candidate(CandidateValue* value_dst,
                                    int32_t* index_dst, const float bq,
                                    const uint32_t kv_index) {
  const uint32_t payload = candidate_fp24_code(bq);
  __stcs(value_dst, static_cast<CandidateValue>(payload));
  __stcs(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

}

#include "../dense_topk_litetopk.cuh"

namespace  {
using CandidateValue = dsa_litetopk::CandidateValue;

static CandidateValue* candidate_data_ptr(torch::Tensor& tensor) {
  return reinterpret_cast<CandidateValue*>(tensor.data_ptr<at::Half>());
}

static void check_candidate_dtype(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.scalar_type() == torch::kHalf,
              "cand_val must use float16 as opaque packed storage");
}

constexpr int kSeedThreads = 256;

constexpr int kSeed12Threads = 256;

template <int kRetainedHead, int BT>
__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride, const int head,
    const int NB, const int K,
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // headroom*span (absolute, resolution-preserving
                           // when NB is scaled up with it): drifted scores
                           // land in real buckets instead of clamping to
                           // bucket 0 where refresh can never resolve them
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket, CandidateValue* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx, int32_t* __restrict__ cand_cnt,
    const int cand_cap, const int physical_index_base,
    int32_t* __restrict__ bcount_out) {
  constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
  static_assert(kRetainedHead == 8192 || kRetainedHead == 12288,
                "production seed supports only the qualified 8K/12K layouts");
  constexpr int kRetainVecs = kRetainedHead / (BT * 4);
  const int row = gridDim.x - 1 - blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const float* srow = slog + (size_t)row * slog_stride;
  extern __shared__ int s_hist[];  // NSUB * NB ints

  // pass 1: min/max of the row's FINITE scores (vectorized). -inf appears
  // when the caller passes clean_logits=True full-row logits (dense-select
  // mode) for the out-of-range causal tail; it must not poison the range.
  __shared__ float s_mx[BT / 32];
  __shared__ float s_mn[BT / 32];
  float mx = -INFINITY, mn = INFINITY;
  const auto acc = [&](const float s) {
    if (isfinite(s)) {
      mx = fmaxf(mx, s);
      mn = fminf(mn, s);
    }
  };
  // The production 8K and 12K specializations retain respectively eight
  // and twelve float4 values per thread. Keep them live across the CTA
  // reduction so histogram construction and emission never reread the
  // materialized prefix logits. Missing tail lanes carry -inf and are
  // ignored by the generic <=8K compatibility path.
  static_assert(BT == 256 || BT == 384 || BT == 512,
                "retained HOT seed requires a qualified CTA size");
  static_assert(BT % (NSUB * 32) == 0,
                "each seed sub-histogram must own whole warps");
  float4 retained[kRetainVecs];
  if (head == kRetainedHead) {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  } else {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      if (j + 3 < head) {
        s4 = *reinterpret_cast<const float4*>(srow + j);
      } else {
        if (j < head) s4.x = srow[j];
        if (j + 1 < head) s4.y = srow[j + 1];
        if (j + 2 < head) s4.z = srow[j + 2];
      }
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
  }
  if (lane == 0) {
    s_mx[tid >> 5] = mx;
    s_mn[tid >> 5] = mn;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int wgi = 1; wgi < BT / 32; ++wgi) {
      s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
      s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
    }
  }
  __syncthreads();
  float o = -s_mx[0];         // min over x = -score
  const float hi = -s_mn[0];  // max over x
  const float span = fmaxf(hi - o, 1e-20f);
  o -= headroom * span;  // forward (above-max) drift headroom
  float inv = (NB - 1) / (span * (1.0f + headroom));
  const float vth = -o * inv;

  // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
  // smem atomic conflicts, vectorized loads.
  for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
  __syncthreads();
  int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
  const auto bucket_of = [&](const float s) -> int {
    // Use the byte-for-byte arithmetic contract consumed by both the
    // seed emitter and the main scan.  Computing (-s - o) * inv as two
    // rounded operations can put a boundary value one bucket below its
    // FMA result: the histogram would then certify K records while the
    // emitter rejects one of them, producing a silent underfill.
    const float bq = fmaf(-s, inv, vth);
    int b = static_cast<int>(bq);
    return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
  };
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const float4 s4 = retained[it];
    if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
    if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
    if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
    if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
  }
  __syncthreads();
  // merge sub-histograms into s_hist[0..NB)
  for (int b = tid; b < NB; b += BT) {
    int c = s_hist[b];
#pragma unroll
    for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
    s_hist[b] = c;
  }
  __syncthreads();
  if (bcount_out != nullptr) {
    // Full-row overwrite of the sample histogram. The ring daemon warm-
    // starts its refresh base from these counts: they are genuine row
    // records in the final (origin, inv) bucket space, so adding them to
    // the daemon's subset cum can only tighten the published edge safely
    // — provided the main scan starts after the sampled prefix (the
    // exact-once contract), or the same records would count twice.
    for (int b = tid; b < NB; b += BT)
      bcount_out[(size_t)row * NB + b] = s_hist[b];
  }
  // Coarse K-th estimate on the single (o, inv) scale built above. There
  // is deliberately NO scale rebuild: th_bucket, origin/inv, the emitted
  // candidates, and the exported bcount histogram must all share one
  // bucket space — the ring warm-start base is only sound under that
  // identity. Headroom above the sample max keeps drifted scores out of
  // bucket 0 where refresh could never resolve them.
  // The production U16 contract uses emit_limit==0 and a single KV split.
  // Its scan covers the complete KV range and initializes the CTA-local
  // histogram itself, so writing Q*NB zeros to global memory is dead work.
  // Find the first histogram prefix that reaches K in parallel.  The old
  // single-thread walk serialized 256 dependent shared-memory loads while
  // the other 1023 threads waited.  NB <= BT gives every bin one owner;
  // the half-open prefix ranges are disjoint, so exactly one thread writes
  // the same threshold as the serial "first cumulative sum >= K" rule.
  __shared__ int s_th;
  __shared__ int s_wsum[BT / 32];
  if (tid == 0) s_th = NB - 1;
  const int h = (tid < NB) ? s_hist[tid] : 0;
  int x = h;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if ((tid & 31) >= off) x += y;
  }
  if ((tid & 31) == 31) s_wsum[tid >> 5] = x;
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int w = 0; w < BT / 32; ++w)
    if (w < (tid >> 5)) base += s_wsum[w];
  const int incl = base + x;
  const int excl = incl - h;
  if (tid < NB && excl < K && K <= incl) s_th = tid;
  __syncthreads();
  if (tid == 0) {
    th_bucket[row] = s_th;
    origin[row] = o;
    inv_delta[row] = inv;
  }
  __syncthreads();
  {
    // Large exact-once mode: the HOT scores retained above are the physical
    // prefix [physical_index_base, physical_index_base + head).  Emit their
    // passing records now, then let the main producer start after `head`.
    //
    // One iteration covers 256 * float4 == 1024 ordered columns.  A warp
    // scan plus eight warp totals gives every thread a deterministic CTA
    // prefix; this reserves no global counter per candidate.  The one CTA
    // owning the row publishes the true (possibly over-cap) total once.
    int emitted_before = 0;
    const float gate_edge = static_cast<float>(s_th + 1);
    const uint64_t row_base = static_cast<uint64_t>(row) * cand_cap;
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j0 = tid * 4 + it * BT * 4;
      const float4 s4 = retained[it];
      const float score[4] = {s4.x, s4.y, s4.z, s4.w};
      float bq[4];
      bool pass[4];
      int local_count = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        bq[k] = fmaf(-score[k], inv, vth);
        pass[k] = j0 + k < head && isfinite(score[k]) &&
                  __float_as_int(bq[k]) < __float_as_int(gate_edge);
        local_count += pass[k] ? 1 : 0;
      }

      int warp_inclusive = local_count;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, off);
        if (lane >= off) warp_inclusive += other;
      }
      if (lane == 31) s_wsum[tid >> 5] = warp_inclusive;
      __syncthreads();

      int warp_before = 0;
#pragma unroll
      for (int w = 0; w < BT / 32; ++w) {
        if (w < (tid >> 5)) warp_before += s_wsum[w];
      }
      const int thread_base =
          emitted_before + warp_before + warp_inclusive - local_count;
      int local_rank = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if (pass[k]) {
          const int out = thread_base + local_rank++;
          if (out < cand_cap) {
            const uint32_t physical_idx =
                static_cast<uint32_t>(physical_index_base + j0 + k);
            dsa_litetopk::store_candidate(cand_val + row_base + out,
                                          cand_idx + row_base + out, bq[k],
                                          physical_idx);
          }
        }
      }

      int block_total = 0;
#pragma unroll
      for (int w = 0; w < BT / 32; ++w) block_total += s_wsum[w];
      emitted_before += block_total;
      // Do not let an early warp overwrite s_wsum for the next retained
      // group while a slower warp still consumes this group's totals.
      __syncthreads();
    }
    if (tid == 0) cand_cnt[row] = emitted_before;
  }
}

void launch_seed_prep(const float* slog, int64_t slog_stride, int Q, int head,
                      int NB, int K, float headroom, float* origin,
                      float* inv_delta, int32_t* th_bucket,
                      CandidateValue* cand_val, int32_t* cand_idx,
                      int32_t* cand_cnt, int cand_cap,
                      int physical_index_base, int32_t* bcount,
                      cudaStream_t stream) {
  const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
  if (head == 12288) {
    {
      seed_prep_kernel<12288, kSeed12Threads>
          <<<Q, kSeed12Threads, seed_smem, stream>>>(
              slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
              th_bucket, cand_val, cand_idx, cand_cnt, cand_cap,
              physical_index_base, bcount);
    }
  } else {
    seed_prep_kernel<8192, kSeedThreads>
        <<<Q, kSeedThreads, seed_smem, stream>>>(
            slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
            th_bucket, cand_val, cand_idx, cand_cnt, cand_cap,
            physical_index_base, bcount);
  }
}

void seed_prep_litetopk_(torch::Tensor slog,
                         torch::Tensor origin, torch::Tensor inv_delta,
                         torch::Tensor th_bucket, torch::Tensor bcount,
                         torch::Tensor cand_val, torch::Tensor cand_idx,
                         torch::Tensor cand_cnt) {
  TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
  TORCH_CHECK(origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda() &&
                  bcount.is_cuda() && cand_val.is_cuda() &&
                  cand_idx.is_cuda() && cand_cnt.is_cuda(),
              "seed prep outputs must be CUDA tensors");
  TORCH_CHECK(slog.device() == origin.device() &&
                  slog.device() == inv_delta.device() &&
                  slog.device() == th_bucket.device() &&
                  slog.device() == bcount.device() &&
                  slog.device() == cand_val.device() &&
                  slog.device() == cand_idx.device() &&
                  slog.device() == cand_cnt.device(),
              "seed prep tensors must be on one CUDA device");
  TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous() &&
                  th_bucket.is_contiguous() && bcount.is_contiguous() &&
                  cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                  cand_cnt.is_contiguous(),
              "seed prep outputs must be contiguous");
  TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
  TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
  const int Q = (int)slog.size(0);
  const int head = (int)slog.size(1);
  constexpr int NB = 256;
  constexpr int K = 2048;
  constexpr int cap = 65536;
  constexpr float headroom = 0.0f;
  TORCH_CHECK(head >= K && (head <= 8192 || head == 12288),
              "production seed prep requires topk <= HOT <= 8192 or HOT=12288");
  TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
  TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
  TORCH_CHECK(
      origin.scalar_type() == torch::kFloat &&
          inv_delta.scalar_type() == torch::kFloat &&
          th_bucket.scalar_type() == torch::kInt &&
          bcount.scalar_type() == torch::kInt &&
          cand_idx.scalar_type() == torch::kInt &&
          cand_cnt.scalar_type() == torch::kInt,
      "seed prep affine outputs must be fp32 and metadata/indices int32");
  TORCH_CHECK(origin.dim() == 1 && origin.numel() >= Q &&
                  inv_delta.dim() == 1 && inv_delta.numel() >= Q &&
                  th_bucket.dim() == 1 && th_bucket.numel() >= Q &&
                  cand_cnt.dim() == 1 && cand_cnt.numel() >= Q,
              "origin/inv_delta/th_bucket/cand_cnt must cover Q rows");
  TORCH_CHECK(cand_val.dim() == 2 && cand_val.size(0) >= Q &&
                  cand_val.size(1) == cap &&
                  cand_idx.sizes() == cand_val.sizes(),
              "cand_val/cand_idx must be [>=Q,cand_cap]");
  check_candidate_dtype(cand_val);
  TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) >= Q && bcount.size(1) == NB,
              "bcount must be [>=Q,num_buckets]");
  TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                  (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
              "slog rows must be 16B aligned");
  const c10::cuda::CUDAGuard device_guard(slog.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int64_t physical_index_base64 = 0;
  TORCH_CHECK(
      physical_index_base64 >= 0 &&
          physical_index_base64 + head <=
              (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
      "HOT physical index range exceeds the packed 20-bit candidate ABI");
  launch_seed_prep(slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
                   static_cast<float>(headroom), origin.data_ptr<float>(),
                   inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
                   candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
                   cand_cnt.data_ptr<int32_t>(), cap,
                   static_cast<int>(physical_index_base64),
                   bcount.data_ptr<int32_t>(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("seed_prep_", &seed_prep_litetopk_);
  m.def("dense_topk_litetopk_", &litetopk_dense::dense_topk_litetopk_);
}
