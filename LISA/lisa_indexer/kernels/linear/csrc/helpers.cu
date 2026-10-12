// Candidate preparation, HOT permutation planning, and carry-vote selection.
// Derived from LiteTopK; see NOTICE.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include "scan.cuh"
#include "dense_topk.cuh"

namespace {
namespace pair_swap_gather {
namespace cg = cooperative_groups;

constexpr int kPlanThreads = 256;

constexpr int kHotSize = 12288;

template <typename HotIndexT>
__global__ __launch_bounds__(kPlanThreads, 1) void cooperative_plan_kernel(
    const HotIndexT* __restrict__ hot, int* __restrict__ hot_epoch,
    int* __restrict__ permutation, int* __restrict__ swap_a,
    int* __restrict__ swap_b, int* __restrict__ counts, int hot_size,
    int window_start, int common_end, int epoch) {
  cg::grid_group grid = cg::this_grid();
  const int i = static_cast<int>(blockIdx.x) * kPlanThreads + threadIdx.x;
  const int lane = threadIdx.x & 31;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counts[1] = 0;
    counts[2] = 0;
  }
  const int old_count = max(0, min(counts[0], hot_size));
  if (i < old_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = a;
    permutation[b] = b;
  }
  if (i < hot_size) {
    const int64_t value = static_cast<int64_t>(hot[i]);
    if (value >= window_start && value < common_end) {
      const int previous =
          atomicExch(hot_epoch + static_cast<int>(value), epoch);
      if (previous == epoch) {
        atomicOr(counts + 3, 1);
      }
    } else {
      atomicOr(counts + 3, 2);
    }
  }
  grid.sync();

  const int64_t hot_value64 = static_cast<int64_t>(hot[i]);
  const bool hot_in_range =
      hot_value64 >= window_start && hot_value64 < common_end;
  const int hot_value =
      hot_in_range ? static_cast<int>(hot_value64) : window_start;
  const int window_value = window_start + i;
  const bool take_a = hot_in_range && hot_value >= window_start + hot_size;
  const bool take_b = hot_epoch[window_value] != epoch;

  const unsigned a_mask = __ballot_sync(0xffffffffu, take_a);
  int a_base = 0;
  if (lane == 0 && a_mask != 0) {
    a_base = atomicAdd(counts + 1, __popc(a_mask));
  }
  a_base = __shfl_sync(0xffffffffu, a_base, 0);
  if (take_a) {
    const int rank = __popc(a_mask & ((1u << lane) - 1u));
    swap_a[a_base + rank] = hot_value;
  }

  const unsigned b_mask = __ballot_sync(0xffffffffu, take_b);
  int b_base = 0;
  if (lane == 0 && b_mask != 0) {
    b_base = atomicAdd(counts + 2, __popc(b_mask));
  }
  b_base = __shfl_sync(0xffffffffu, b_base, 0);
  if (take_b) {
    const int rank = __popc(b_mask & ((1u << lane) - 1u));
    swap_b[b_base + rank] = window_value;
  }
  grid.sync();

  const int pair_count = min(counts[1], counts[2]);
  const int metadata_error = counts[3];
  if (metadata_error != 0 || counts[1] != counts[2]) {
    if (i == 0 && counts[1] != counts[2]) {
      atomicOr(counts + 3, 8);
    }
    asm volatile("trap;");
    return;
  }
  if (i < pair_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = b;
    permutation[b] = a;
  }
  if (i == 0) {
    counts[0] = pair_count;
  }
}

void validate_plan(const torch::Tensor& hot, const torch::Tensor& hot_epoch,
                   const torch::Tensor& permutation,
                   const torch::Tensor& swap_a, const torch::Tensor& swap_b,
                   const torch::Tensor& counts, int64_t window_start,
                   int64_t common_end, int64_t epoch) {
  TORCH_CHECK(hot.is_cuda() && hot_epoch.is_cuda() && permutation.is_cuda() &&
                  swap_a.is_cuda() && swap_b.is_cuda() && counts.is_cuda(),
              "all pair-swap tensors must be CUDA");
  TORCH_CHECK(
      (hot.scalar_type() == torch::kInt || hot.scalar_type() == torch::kLong) &&
          hot_epoch.scalar_type() == torch::kInt &&
          permutation.scalar_type() == torch::kInt &&
          swap_a.scalar_type() == torch::kInt &&
          swap_b.scalar_type() == torch::kInt &&
          counts.scalar_type() == torch::kInt,
      "hot must be int32/int64; pair-swap workspaces must be int32");
  TORCH_CHECK(hot.is_contiguous() && hot_epoch.is_contiguous() &&
                  permutation.is_contiguous() && swap_a.is_contiguous() &&
                  swap_b.is_contiguous() && counts.is_contiguous(),
              "all pair-swap tensors must be contiguous");
  TORCH_CHECK(hot.dim() == 1 && hot_epoch.dim() == 1 &&
                  permutation.dim() == 1 && swap_a.dim() == 1 &&
                  swap_b.dim() == 1 && counts.dim() == 1,
              "all pair-swap tensors must be vectors");
  TORCH_CHECK(hot.device() == hot_epoch.device() &&
                  hot.device() == permutation.device() &&
                  hot.device() == swap_a.device() &&
                  hot.device() == swap_b.device() &&
                  hot.device() == counts.device(),
              "all pair-swap tensors must be on one CUDA device");
  TORCH_CHECK(hot_epoch.numel() == permutation.numel(),
              "epoch and permutation lengths must match");
  TORCH_CHECK(swap_a.numel() >= hot.numel() && swap_b.numel() >= hot.numel(),
              "swap workspaces must hold HOT entries");
  TORCH_CHECK(
      hot.numel() == kHotSize,
      "production pair-swap planner requires exactly 12288 hot indices");
  TORCH_CHECK(counts.numel() >= 4, "counts must hold four int32 values");
  TORCH_CHECK(window_start >= 0 && window_start + hot.numel() <= common_end &&
                  common_end <= permutation.numel() &&
                  permutation.numel() <= std::numeric_limits<int>::max(),
              "expected 0 <= window_start, window_start + HOT <= common_end "
              "<= sequence length");
  TORCH_CHECK(epoch > 0 && epoch <= std::numeric_limits<int>::max(),
              "epoch must be positive int32");
}

template <typename HotIndexT>
void launch_plan_typed(const torch::Tensor& hot, const torch::Tensor& hot_epoch,
                       const torch::Tensor& permutation,
                       const torch::Tensor& swap_a, const torch::Tensor& swap_b,
                       const torch::Tensor& counts, int window_start,
                       int common_end, int epoch, cudaStream_t stream) {
  int hot_size = static_cast<int>(hot.numel());
  const int blocks = (hot_size + kPlanThreads - 1) / kPlanThreads;
  const HotIndexT* hot_ptr = hot.data_ptr<HotIndexT>();
  int* hot_epoch_ptr = hot_epoch.data_ptr<int>();
  int* permutation_ptr = permutation.data_ptr<int>();
  int* swap_a_ptr = swap_a.data_ptr<int>();
  int* swap_b_ptr = swap_b.data_ptr<int>();
  int* counts_ptr = counts.data_ptr<int>();
  void* args[] = {
      &hot_ptr,    &hot_epoch_ptr, &permutation_ptr, &swap_a_ptr, &swap_b_ptr,
      &counts_ptr, &hot_size,      &window_start,    &common_end, &epoch,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(cooperative_plan_kernel<HotIndexT>),
      dim3(blocks), dim3(kPlanThreads), args, 0, stream));
}

void launch_plan(const torch::Tensor& hot, const torch::Tensor& hot_epoch,
                 const torch::Tensor& permutation, const torch::Tensor& swap_a,
                 const torch::Tensor& swap_b, const torch::Tensor& counts,
                 int window_start, int common_end, int epoch,
                 cudaStream_t stream) {
  if (hot.scalar_type() == torch::kLong) {
    launch_plan_typed<int64_t>(hot, hot_epoch, permutation, swap_a, swap_b,
                               counts, window_start, common_end, epoch, stream);
  } else {
    launch_plan_typed<int32_t>(hot, hot_epoch, permutation, swap_a, swap_b,
                               counts, window_start, common_end, epoch, stream);
  }
}

}  // namespace pair_swap_gather

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

template <bool kEmitCandidates, int kRetainedHead, int BT>
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
                "seed supports only the 8K/12K layouts");
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
  // The 8K and 12K specializations retain respectively eight
  // and twelve float4 values per thread. Keep them live across the CTA
  // reduction so histogram construction and emission never reread the
  // materialized prefix logits. Missing tail lanes carry -inf and are
  // ignored by the generic <=8K compatibility path.
  static_assert(BT == 256 || BT == 384 || BT == 512,
                "HOT seed requires a supported CTA size");
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
  // The U16 contract uses emit_limit==0 and a single KV split.
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
  if constexpr (kEmitCandidates) {
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
  } else {
    if (tid == 0) cand_cnt[row] = 0;
  }
}

constexpr int kCarryTileItems = 8192;

constexpr int kCarryMaxItems = 1 << 20;

constexpr int kCarryMaxK = 12288;

constexpr int kCarryMaxVote = 8192;

constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;

constexpr int kCarryThreads = 256;

constexpr int kCarryWarps = kCarryThreads / 32;

constexpr int kCarryPlannerWindow = 8192;

enum CarryStateOffset : int {
  kCarryTicket = 0,
  kCarryThreshold = 1,
  kCarryTieBlock = 2,
  kCarryTieTake = 3,
  kCarryOutK = 4,
  kCarryNumBlocks = 5,
  kCarryBlockOffsets = 6,
};

constexpr int kCarryStateInts = kCarryBlockOffsets + kCarryMaxBlocks + 1;

enum CarryPlannerStateOffset : int {
  kCarryPlannerPairCount = 0,
  kCarryPlannerTicket = 1,
  kCarryPlannerStatus = 2,
  kCarryPlannerSelectedInside = 3,
  kCarryPlannerPreviousWindow = 4,
};

constexpr int kCarryPlannerStateInts = 5;

__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <bool FusePlanner>
__global__ void carry_votes_plan_litetopk_kernel(
    const int32_t* __restrict__ votes, int count, int min_index, int out_k,
    int max_vote, volatile int16_t* __restrict__ partial, int partial_stride,
    int32_t* __restrict__ state, int32_t* __restrict__ permutation,
    int32_t* __restrict__ planner_state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = (FusePlanner ? min_index : 0) + block * kCarryTileItems;
  const int end = min(begin + kCarryTileItems, count);
  const int bins = max_vote + 1;
  const int previous_window_start =
      FusePlanner ? planner_state[kCarryPlannerPreviousWindow] : 0;

  // In the fused path the physical HOT window itself is the compact list of
  // the previous pair swaps. Restore those pairs while K1 is already reading
  // the next vote histogram; no epoch array or planner launch is needed.
  if constexpr (FusePlanner) {
    for (int offset = block * kCarryThreads + tid; offset < kCarryPlannerWindow;
         offset += gridDim.x * kCarryThreads) {
      const int b = previous_window_start + offset;
      const int a = permutation[b];
      if (a != b) {
        permutation[a] = a;
        permutation[b] = b;
      }
    }
  }

  extern __shared__ uint32_t s_freq[];
  __shared__ int s_warp_sum[kCarryWarps];
  __shared__ int s_last;
  __shared__ int s_scan_base;
  __shared__ int s_found;
  __shared__ int s_threshold;
  __shared__ int s_count_gt;
  __shared__ int s_tie_block;
  __shared__ int s_tie_take;
  __shared__ int s_block_count[kCarryMaxBlocks];

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    s_freq[bin] = 0;
  }
  __syncthreads();

  // Zero votes dominate most corpora. Count them in registers and reduce
  // once instead of serializing every zero through one shared atomic.
  int local_zero = 0;
  for (int index = begin + tid; index < end; index += kCarryThreads) {
    if (index < min_index) {
      continue;
    }
    int value = votes[index];
    // The selector emits unique winners per sampled row, so this clamp is
    // unreachable under the public ABI. Keep release builds memory-safe
    // if an upstream invariant is violated.
    value = value < 0 ? 0 : (value > max_vote ? max_vote : value);
    if (value == 0) {
      ++local_zero;
    } else {
      atomicAdd(&s_freq[value], 1u);
    }
  }
  local_zero = carry_warp_sum(local_zero);
  if (lane == 0) {
    s_warp_sum[warp] = local_zero;
  }
  __syncthreads();
  if (warp == 0) {
    int value = lane < kCarryWarps ? s_warp_sum[lane] : 0;
    value = carry_warp_sum(value);
    if (lane == 0) {
      s_freq[0] = static_cast<uint32_t>(value);
    }
  }
  __syncthreads();

  volatile int16_t* block_partial =
      partial + static_cast<size_t>(block) * partial_stride;
  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    // A CTA owns at most 8192 positions, safely inside signed int16.
    block_partial[bin] = static_cast<int16_t>(s_freq[bin]);
  }
  // Every thread publishes its own global stores. A fence in tid0 alone
  // would not release the other 255 writers before the completion ticket.
  __threadfence();
  __syncthreads();

  // CUDA's canonical "last block" reduction pattern. No CTA spins: every
  // non-last block exits, while the last ticket holder sees all partial
  // writes made visible before the atomic increment.
  if (tid == 0) {
    const int old = atomicAdd(&state[kCarryTicket], 1);
    s_last = old == gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) {
    return;
  }

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    int total = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      total += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    s_freq[bin] = static_cast<uint32_t>(total);
  }
  __syncthreads();

  // Descending 256-bin tiles. This is the seed-prep parallel prefix in the
  // opposite direction, extended to the dynamic [0,max_vote] domain.
  if (tid == 0) {
    s_scan_base = 0;
    s_found = 0;
    s_threshold = 0;
    s_count_gt = 0;
  }
  __syncthreads();
  for (int tile = 0; tile < bins; tile += kCarryThreads) {
    const int bin = max_vote - tile - tid;
    const int count_here = bin >= 0 ? static_cast<int>(s_freq[bin]) : 0;
    int inclusive = count_here;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += other;
      }
    }
    if (lane == 31) {
      s_warp_sum[warp] = inclusive;
    }
    __syncthreads();
    int warp_base = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
      if (source_warp < warp) {
        warp_base += s_warp_sum[source_warp];
      }
    }
    const int exclusive = s_scan_base + warp_base + inclusive - count_here;
    const int inclusive_global = exclusive + count_here;
    if (bin >= 0 && exclusive < out_k && out_k <= inclusive_global) {
      s_threshold = bin;
      s_count_gt = exclusive;
      s_found = 1;
    }
    __syncthreads();
    if (s_found) {
      break;
    }
    if (tid == 0) {
      int tile_total = 0;
#pragma unroll
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        tile_total += s_warp_sum[source_warp];
      }
      s_scan_base += tile_total;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int need_equal = out_k - s_count_gt;
    int equal_before = 0;
    s_tie_block = gridDim.x - 1;
    s_tie_take = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      const int equal_here = static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride +
                  s_threshold]);
      if (equal_before < need_equal &&
          need_equal <= equal_before + equal_here) {
        s_tie_block = source_block;
        s_tie_take = need_equal - equal_before;
        break;
      }
      equal_before += equal_here;
    }
  }
  __syncthreads();

  // Compute each block's exact stable-output size. Warps read one partial
  // row at a time so the second partial pass remains coalesced.
  for (int source_block = warp; source_block < gridDim.x;
       source_block += kCarryWarps) {
    int selected = 0;
    for (int bin = s_threshold + 1 + lane; bin < bins; bin += 32) {
      selected += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    selected = carry_warp_sum(selected);
    if (lane == 0) {
      int equal_take = 0;
      if (source_block < s_tie_block) {
        equal_take = static_cast<int>(
            partial[static_cast<size_t>(source_block) * partial_stride +
                    s_threshold]);
      } else if (source_block == s_tie_block) {
        equal_take = s_tie_take;
      }
      s_block_count[source_block] = selected + equal_take;
    }
  }
  __syncthreads();

  // The last K1 CTA already owns the exact selected count for the first
  // aligned 8192-position window. It initializes all K2 planner state, so
  // the fused path needs neither a memset nor a separate init kernel.
  if constexpr (FusePlanner) {
    if (tid == 0) {
      const int selected_inside = s_block_count[0];
      planner_state[kCarryPlannerPairCount] =
          kCarryPlannerWindow - selected_inside;
      planner_state[kCarryPlannerTicket] = 0;
      planner_state[kCarryPlannerStatus] = 0;
      planner_state[kCarryPlannerSelectedInside] = selected_inside;
      planner_state[kCarryPlannerPreviousWindow] = min_index;
    }
  }
  __syncthreads();

  if (tid == 0) {
    int offset = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      state[kCarryBlockOffsets + source_block] = offset;
      offset += s_block_count[source_block];
    }
    state[kCarryBlockOffsets + gridDim.x] = offset;
    state[kCarryThreshold] = s_threshold;
    state[kCarryTieBlock] = s_tie_block;
    state[kCarryTieTake] = s_tie_take;
    state[kCarryOutK] = out_k;
    state[kCarryNumBlocks] = gridDim.x;
    __threadfence();
    atomicExch(&state[kCarryTicket], 0);
  }
}

template <bool FusePlanner>
__global__ void carry_votes_emit_reset_litetopk_kernel(
    int32_t* __restrict__ votes, int count, int min_index, int max_vote,
    int64_t* __restrict__ out_idx, const int32_t* __restrict__ state,
    int32_t* __restrict__ permutation, int32_t* __restrict__ planner_state,
    int window_start, int target_length) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = (FusePlanner ? min_index : 0) + block * kCarryTileItems;
  const int threshold = state[kCarryThreshold];
  const int tie_block = state[kCarryTieBlock];
  const int tie_take = state[kCarryTieTake];
  const int output_base = state[kCarryBlockOffsets + block];

  __shared__ int s_warp_count[kCarryWarps];
  __shared__ int s_warp_prefix[kCarryWarps];
  __shared__ int s_tile_output_base;
  __shared__ int s_tie_seen;
  __shared__ int s_tile_total;
  if (tid == 0) {
    s_tile_output_base = 0;
    s_tie_seen = 0;
  }
  __syncthreads();

  if constexpr (FusePlanner) {
    // The aligned fused grid excludes sink positions. Clear their votes
    // here so that every live vote is reset.
    if (block == 0) {
      for (int index = tid; index < min_index; index += kCarryThreads) {
        votes[index] = 0;
      }
    }
  }

  constexpr unsigned kFullMask = 0xffffffffu;
  const unsigned lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
  for (int tile = 0; tile < kCarryTileItems; tile += kCarryThreads) {
    const int index = begin + tile + tid;
    const bool valid = index < count;
    const int raw_value = valid ? votes[index] : 0;
    const int value =
        raw_value < 0 ? 0 : (raw_value > max_vote ? max_vote : raw_value);
    if (valid) {
      votes[index] = 0;
    }
    const bool eligible = valid && index >= min_index;
    const bool is_equal = eligible && value == threshold;

    bool take_equal = is_equal && block < tie_block;
    if (block == tie_block) {
      const unsigned equal_mask = __ballot_sync(kFullMask, is_equal);
      if (lane == 0) {
        s_warp_count[warp] = __popc(equal_mask);
      }
      __syncthreads();
      if (tid == 0) {
        int prefix = 0;
        for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
          s_warp_prefix[source_warp] = prefix;
          prefix += s_warp_count[source_warp];
        }
        s_tile_total = prefix;
      }
      __syncthreads();
      const int equal_rank =
          s_tie_seen + s_warp_prefix[warp] + __popc(equal_mask & lane_mask);
      take_equal = is_equal && equal_rank < tie_take;
      __syncthreads();
      if (tid == 0) {
        s_tie_seen += s_tile_total;
      }
      __syncthreads();
    }

    const bool selected = eligible && (value > threshold || take_equal);
    const unsigned selected_mask = __ballot_sync(kFullMask, selected);
    if (lane == 0) {
      s_warp_count[warp] = __popc(selected_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix = 0;
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        s_warp_prefix[source_warp] = prefix;
        prefix += s_warp_count[source_warp];
      }
      s_tile_total = prefix;
    }
    __syncthreads();
    const int local_rank =
        s_warp_prefix[warp] + __popc(selected_mask & lane_mask);
    if (selected) {
      out_idx[output_base + s_tile_output_base + local_rank] =
          static_cast<int64_t>(index);
    }
    if constexpr (FusePlanner) {
      if (block == 0 && valid && !selected) {
        // Every non-selected position in the physical HOT window is a
        // hole. Reuse the already-consumed vote prefix as its compact
        // endpoint list; hole_rank needs no extra ballot or atomic.
        const int position = tile + tid;
        const int hole_rank = position - (s_tile_output_base + local_rank);
        reinterpret_cast<volatile int32_t*>(votes)[hole_rank] = index;
      }
    }
    __syncthreads();
    if (tid == 0) {
      s_tile_output_base += s_tile_total;
    }
    __syncthreads();
  }

  if constexpr (FusePlanner) {
    // Small-sequence grids have at most 16 CTAs on B200, so all
    // CTAs are simultaneously resident. Each producer releases its own
    // endpoint stores before publishing a ticket; the final -1 is the
    // acquire/release point for the parallel pair pass.
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int old = atomicAdd(&planner_state[kCarryPlannerTicket], 1);
      if (old == gridDim.x - 1) {
        atomicExch(&planner_state[kCarryPlannerTicket], -1);
      } else {
        while (atomicAdd(&planner_state[kCarryPlannerTicket], 0) != -1) {
          __nanosleep(64);
        }
      }
    }
    __syncthreads();

    const int selected_inside = planner_state[kCarryPlannerSelectedInside];
    const int pair_count = planner_state[kCarryPlannerPairCount];
    for (int rank = block * kCarryThreads + tid; rank < pair_count;
         rank += gridDim.x * kCarryThreads) {
      const int a = static_cast<int>(
          reinterpret_cast<volatile int64_t*>(out_idx)[selected_inside + rank]);
      const int b = reinterpret_cast<volatile int32_t*>(votes)[rank];
      if (a < window_start + kCarryPlannerWindow || a >= count ||
          b < window_start || b >= window_start + kCarryPlannerWindow ||
          b >= target_length) {
        atomicOr(&planner_state[kCarryPlannerStatus], 1);
        asm volatile("trap;");
      } else {
        permutation[a] = b;
        permutation[b] = a;
      }
      votes[rank] = 0;
    }
  }
}

__global__ void map_topk_indices_and_accumulate_votes_litetopk_kernel(
    int32_t* __restrict__ out_idx, const int32_t* __restrict__ index_map,
    const int32_t* __restrict__ status, int32_t* __restrict__ votes,
    int64_t total, int rows, int index_map_size, int topk, int votes_len,
    int vote_recent_rows, const int32_t* __restrict__ cand_cnt,
    int32_t* __restrict__ stat_run_max, int32_t* __restrict__ stat_over,
    int stat_watermark) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vote_begin =
      static_cast<int64_t>(rows - vote_recent_rows) * topk;
  // A bad/underfilled row may contain selector padding rather than K valid
  // physical indices.  Preserve fail-closed behavior before treating the
  // complete output matrix as mappable winners.  The candidate-count
  // telemetry rides the same one-thread-per-row sweep (replaces a 5-launch
  // amax/maximum/gt/sum/add_ tail on the host path).
  int stat_local_max = 0;
  int stat_local_over = 0;
  for (int row = static_cast<int>(global_thread); row < rows;
       row += static_cast<int>(step)) {
    if (status[row] != 0) {
      asm volatile("trap;");
      return;
    }
    if (cand_cnt != nullptr) {
      const int c = cand_cnt[row];
      stat_local_max = c > stat_local_max ? c : stat_local_max;
      stat_local_over += c > stat_watermark ? 1 : 0;
    }
  }
  if (cand_cnt != nullptr) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int m = __shfl_down_sync(0xffffffffu, stat_local_max, off);
      stat_local_max = m > stat_local_max ? m : stat_local_max;
      stat_local_over += __shfl_down_sync(0xffffffffu, stat_local_over, off);
    }
    __shared__ int stat_smax[32], stat_exceed[32];
    const int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
      stat_smax[wid] = stat_local_max;
      stat_exceed[wid] = stat_local_over;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int m = 0, ov = 0;
      const int warps = (blockDim.x + 31) >> 5;
      for (int w = 0; w < warps; ++w) {
        m = stat_smax[w] > m ? stat_smax[w] : m;
        ov += stat_exceed[w];
      }
      if (m > 0) atomicMax(stat_run_max, m);
      if (ov > 0) atomicAdd(stat_over, ov);
    }
  }
  for (int64_t linear = global_thread; linear < total; linear += step) {
    const int32_t physical_idx = out_idx[linear];
    if (static_cast<uint32_t>(physical_idx) >=
        static_cast<uint32_t>(index_map_size)) {
      asm volatile("trap;");
      return;
    }
    const int32_t original_idx = index_map[physical_idx];
    if (static_cast<uint32_t>(original_idx) >=
            static_cast<uint32_t>(index_map_size) ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
      asm volatile("trap;");
      return;
    }
    out_idx[linear] = original_idx;

    if (votes != nullptr && votes_len > 0) {
      // The next chunk is best predicted by the most recent query
      // window.  Mapping already visits every Q*K winner, so voting all
      // winners from the last 1536 rows only adds the atomics; it needs
      // no extra winner read or launch.
      if (linear >= vote_begin) {
        const int32_t vote_idx =
            original_idx < 0
                ? 0
                : (original_idx >= votes_len ? votes_len - 1 : original_idx);
        atomicAdd(votes + vote_idx, 1);
      }
    }
  }
}

void launch_seed_prep(const float* slog, int64_t slog_stride, int Q, int head,
                      int NB, int K, float headroom, float* origin,
                      float* inv_delta, int32_t* th_bucket,
                      CandidateValue* cand_val, int32_t* cand_idx,
                      int32_t* cand_cnt, int cand_cap, int emit_limit,
                      int physical_index_base, int32_t* bcount,
                      cudaStream_t stream) {
  const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
  if (head == 12288) {
    if (emit_limit == head) {
      seed_prep_kernel<true, 12288, kSeed12Threads>
          <<<Q, kSeed12Threads, seed_smem, stream>>>(
              slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
              th_bucket, cand_val, cand_idx, cand_cnt, cand_cap,
              physical_index_base, bcount);
    } else {
      seed_prep_kernel<false, 12288, kSeed12Threads>
          <<<Q, kSeed12Threads, seed_smem, stream>>>(
              slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
              th_bucket, nullptr, nullptr, cand_cnt, cand_cap, 0, bcount);
    }
  } else if (emit_limit == head) {
    seed_prep_kernel<true, 8192, kSeedThreads>
        <<<Q, kSeedThreads, seed_smem, stream>>>(
            slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
            th_bucket, cand_val, cand_idx, cand_cnt, cand_cap,
            physical_index_base, bcount);
  } else {
    seed_prep_kernel<false, 8192, kSeedThreads>
        <<<Q, kSeedThreads, seed_smem, stream>>>(
            slog, slog_stride, head, NB, K, headroom, origin, inv_delta,
            th_bucket, nullptr, nullptr, cand_cnt, cand_cap, 0, bcount);
  }
}

void seed_prep_litetopk_(torch::Tensor slog, int64_t num_buckets64,
                         int64_t topk64, int64_t cand_cap64,
                         int64_t emit_limit64, double headroom,
                         int64_t probe_stride_tok64, int64_t hist_stride64,
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
  const int NB = (int)num_buckets64;
  const int K = (int)topk64;
  const int cap = (int)cand_cap64;
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
  const int emit_limit =
      emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
  TORCH_CHECK(emit_limit == 0 || emit_limit == head,
              "production seed prep requires emit_limit to be 0 or HOT");
  TORCH_CHECK(hist_stride64 == 1,
              "production seed prep requires hist_stride=1");
  const int64_t physical_index_base64 =
      emit_limit == head ? probe_stride_tok64 : 0;
  TORCH_CHECK(
      physical_index_base64 >= 0 &&
          physical_index_base64 + head <=
              (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
      "HOT physical index range exceeds the packed 20-bit candidate ABI");
  launch_seed_prep(slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
                   static_cast<float>(headroom), origin.data_ptr<float>(),
                   inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
                   candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
                   cand_cnt.data_ptr<int32_t>(), cap, emit_limit,
                   static_cast<int>(physical_index_base64),
                   bcount.data_ptr<int32_t>(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool FusePlanner>
void carry_votes_topk_reset_litetopk_impl_(
    torch::Tensor votes, torch::Tensor out_idx, torch::Tensor partial,
    torch::Tensor state, torch::Tensor permutation, torch::Tensor planner_state,
    int64_t k64, int64_t max_vote64, int64_t min_index64,
    int64_t window_start64) {
  TORCH_CHECK(votes.is_cuda() && out_idx.is_cuda() && partial.is_cuda() &&
                  state.is_cuda(),
              "votes/out_idx/partial/state must be CUDA tensors");
  TORCH_CHECK(votes.is_contiguous() && out_idx.is_contiguous() &&
                  partial.is_contiguous() && state.is_contiguous(),
              "votes/out_idx/partial/state must be contiguous");
  TORCH_CHECK(votes.scalar_type() == torch::kInt, "votes must be int32");
  TORCH_CHECK(out_idx.scalar_type() == torch::kLong, "out_idx must be int64");
  TORCH_CHECK(partial.scalar_type() == torch::kShort, "partial must be int16");
  TORCH_CHECK(state.scalar_type() == torch::kInt, "state must be int32");
  TORCH_CHECK(votes.device() == out_idx.device() &&
                  votes.device() == partial.device() &&
                  votes.device() == state.device(),
              "votes/out_idx/partial/state must be on the same CUDA device");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram");
  TORCH_CHECK(out_idx.dim() == 1, "out_idx must be 1-D");
  TORCH_CHECK(partial.dim() == 2, "partial must be [blocks,bins]");
  TORCH_CHECK(state.dim() == 1 && state.numel() >= kCarryStateInts,
              "state is too small for the carry top-k ABI");

  if constexpr (FusePlanner) {
    TORCH_CHECK(permutation.is_cuda() && planner_state.is_cuda(),
                "permutation/planner_state must be CUDA tensors");
    TORCH_CHECK(permutation.is_contiguous() && planner_state.is_contiguous(),
                "permutation/planner_state must be contiguous");
    TORCH_CHECK(permutation.scalar_type() == torch::kInt &&
                    planner_state.scalar_type() == torch::kInt,
                "permutation/planner_state must be int32");
    TORCH_CHECK(permutation.dim() == 1 && planner_state.dim() == 1 &&
                    planner_state.numel() >= kCarryPlannerStateInts,
                "invalid fused carry-planner workspace");
    TORCH_CHECK(votes.device() == permutation.device() &&
                    votes.device() == planner_state.device(),
                "planner workspace must be on the carry CUDA device");
  }

  const int64_t count64 = votes.numel();
  TORCH_CHECK(count64 >= 1 && count64 <= kCarryMaxItems,
              "votes length must be in [1,1048576]");
  TORCH_CHECK(k64 >= 1 && k64 <= kCarryMaxK, "k must be in [1,12288]");
  TORCH_CHECK(max_vote64 >= 1 && max_vote64 <= kCarryMaxVote,
              "max_vote must be in [1,8192]");
  TORCH_CHECK(min_index64 >= 0 && min_index64 < count64,
              "min_index must be in [0,votes.numel())");
  const int count = static_cast<int>(count64);
  const int min_index = static_cast<int>(min_index64);
  const int eligible = count - min_index;
  const int out_k = static_cast<int>(min(k64, static_cast<int64_t>(eligible)));
  const int max_vote = static_cast<int>(max_vote64);
  const int bins = max_vote + 1;
  const int blocks = ((FusePlanner ? eligible : count) + kCarryTileItems - 1) /
                     kCarryTileItems;
  if constexpr (FusePlanner) {
    TORCH_CHECK(out_k == kCarryPlannerWindow,
                "fused carry planner requires k=8192");
    TORCH_CHECK(window_start64 == min_index64,
                "fused carry planner requires window_start=min_index");
    TORCH_CHECK(min_index64 + kCarryPlannerWindow <= count64,
                "fused carry planner window exceeds the vote extent");
    TORCH_CHECK(permutation.numel() >= count64,
                "permutation must cover the complete vote extent");
    TORCH_CHECK(permutation.numel() <= std::numeric_limits<int>::max(),
                "permutation is too large for the int32 planner ABI");
    // The software grid barrier supports only the <=128K exact
    // path: at most 16 of these 256-thread CTAs on fixed B200 hardware.
    TORCH_CHECK(blocks <= 16,
                "fused carry planner is restricted to at most 16 CTAs");
  }
  TORCH_CHECK(out_idx.numel() == out_k,
              "out_idx must have min(k,votes.numel()-min_index) elements");
  TORCH_CHECK(partial.size(0) >= blocks && partial.size(1) >= bins,
              "partial must provide at least [ceil(N/8192),max_vote+1]");

  const int partial_stride = static_cast<int>(partial.size(1));
  const size_t dynamic_smem = static_cast<size_t>(bins) * sizeof(uint32_t);
  const c10::cuda::CUDAGuard device_guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  carry_votes_plan_litetopk_kernel<FusePlanner>
      <<<blocks, kCarryThreads, dynamic_smem, stream>>>(
          votes.data_ptr<int32_t>(), count, min_index, out_k, max_vote,
          partial.data_ptr<int16_t>(), partial_stride,
          state.data_ptr<int32_t>(),
          FusePlanner ? permutation.data_ptr<int32_t>() : nullptr,
          FusePlanner ? planner_state.data_ptr<int32_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  carry_votes_emit_reset_litetopk_kernel<FusePlanner>
      <<<blocks, kCarryThreads, 0, stream>>>(
          votes.data_ptr<int32_t>(), count, min_index, max_vote,
          out_idx.data_ptr<int64_t>(), state.data_ptr<int32_t>(),
          FusePlanner ? permutation.data_ptr<int32_t>() : nullptr,
          FusePlanner ? planner_state.data_ptr<int32_t>() : nullptr,
          static_cast<int>(window_start64),
          FusePlanner ? static_cast<int>(permutation.numel()) : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void carry_votes_topk_reset_litetopk_(torch::Tensor votes,
                                      torch::Tensor out_idx,
                                      torch::Tensor partial,
                                      torch::Tensor state, int64_t k64,
                                      int64_t max_vote64, int64_t min_index64) {
  carry_votes_topk_reset_litetopk_impl_<false>(votes, out_idx, partial, state,
                                               torch::Tensor(), torch::Tensor(),
                                               k64, max_vote64, min_index64, 0);
}

void map_topk_indices_and_accumulate_votes_litetopk_(
    torch::Tensor out_idx, torch::Tensor index_map, torch::Tensor status,
    torch::Tensor votes, int64_t vote_recent_rows64) {
  TORCH_CHECK(out_idx.is_cuda() && index_map.is_cuda() && status.is_cuda() &&
                  votes.is_cuda(),
              "out_idx/index_map/status/votes must be CUDA");
  TORCH_CHECK(out_idx.device() == index_map.device() &&
                  out_idx.device() == status.device() &&
                  out_idx.device() == votes.device(),
              "out_idx/index_map/status/votes must be on one CUDA device");
  TORCH_CHECK(out_idx.is_contiguous() && index_map.is_contiguous() &&
                  status.is_contiguous() && votes.is_contiguous(),
              "out_idx/index_map/status/votes must be contiguous");
  TORCH_CHECK(out_idx.scalar_type() == torch::kInt &&
                  index_map.scalar_type() == torch::kInt &&
                  status.scalar_type() == torch::kInt &&
                  votes.scalar_type() == torch::kInt,
              "out_idx/index_map/status/votes must be int32");
  TORCH_CHECK(out_idx.dim() == 2 && out_idx.numel() > 0,
              "out_idx must be a nonempty [R,K] matrix");
  TORCH_CHECK(index_map.dim() == 1 && index_map.numel() > 0 &&
                  index_map.numel() <=
                      (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
              "index_map must be a nonempty vector with at most 1M entries");
  TORCH_CHECK(status.dim() == 1 && status.numel() == out_idx.size(0),
              "status must contain one value per output row");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram (or empty)");
  TORCH_CHECK(vote_recent_rows64 > 0 && vote_recent_rows64 <= out_idx.size(0),
              "vote_recent_rows must be in [1,R]");

  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const int votes_len = static_cast<int>(votes.numel());
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_topk_indices_and_accumulate_votes_litetopk_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, total,
      static_cast<int>(out_idx.size(0)), static_cast<int>(index_map.numel()),
      static_cast<int>(out_idx.size(1)), votes_len,
      static_cast<int>(vote_recent_rows64), nullptr, nullptr, nullptr, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void map_topk_vote_stats_litetopk_(
    torch::Tensor out_idx, torch::Tensor index_map, torch::Tensor status,
    torch::Tensor votes, int64_t vote_recent_rows64, torch::Tensor cand_cnt,
    torch::Tensor run_max, torch::Tensor over_events, int64_t watermark64) {
  TORCH_CHECK(cand_cnt.is_cuda() && run_max.is_cuda() && over_events.is_cuda(),
              "stats tensors must be CUDA");
  TORCH_CHECK(cand_cnt.is_contiguous() && run_max.is_contiguous() &&
                  over_events.is_contiguous(),
              "stats tensors must be contiguous");
  TORCH_CHECK(cand_cnt.scalar_type() == torch::kInt &&
                  run_max.scalar_type() == torch::kInt &&
                  over_events.scalar_type() == torch::kInt,
              "stats tensors must be int32");
  TORCH_CHECK(cand_cnt.numel() >= out_idx.size(0) && run_max.numel() >= 1 &&
                  over_events.numel() >= 1,
              "stats tensors too small");
  TORCH_CHECK(out_idx.is_cuda() && out_idx.is_contiguous() &&
                  out_idx.scalar_type() == torch::kInt && out_idx.dim() == 2 &&
                  out_idx.numel() > 0,
              "out_idx must be a nonempty contiguous int32 [R,K]");
  TORCH_CHECK(status.numel() == out_idx.size(0) && votes.dim() == 1 &&
                  vote_recent_rows64 > 0 &&
                  vote_recent_rows64 <= out_idx.size(0),
              "map stats: bad status/votes/recent-rows");

  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const int votes_len = static_cast<int>(votes.numel());
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_topk_indices_and_accumulate_votes_litetopk_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, total,
      static_cast<int>(out_idx.size(0)), static_cast<int>(index_map.numel()),
      static_cast<int>(out_idx.size(1)), votes_len,
      static_cast<int>(vote_recent_rows64), cand_cnt.data_ptr<int32_t>(),
      run_max.data_ptr<int32_t>(), over_events.data_ptr<int32_t>(),
      static_cast<int>(watermark64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan_hot_permutation_out", [](const torch::Tensor& hot,
      const torch::Tensor& hot_epoch, const torch::Tensor& permutation,
      const torch::Tensor& swap_a, const torch::Tensor& swap_b,
      const torch::Tensor& counts, int64_t common_end, int64_t epoch) {
    pair_swap_gather::validate_plan(hot, hot_epoch, permutation, swap_a,
                                   swap_b, counts, 0, common_end, epoch);
    const c10::cuda::CUDAGuard guard(hot.device());
    pair_swap_gather::launch_plan(hot, hot_epoch, permutation, swap_a,
        swap_b, counts, 0, static_cast<int>(common_end), static_cast<int>(epoch),
        c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  m.def("dense_topk_litetopk_", &litetopk_dense::dense_topk_litetopk_,
        "Single-launch exact dense prefill top-k", pybind11::arg("logits"),
        pybind11::arg("row_starts"), pybind11::arg("row_ends"),
        pybind11::arg("out"), pybind11::arg("rows"), pybind11::arg("stride0"),
        pybind11::arg("stride1"), pybind11::arg("topk"),
        pybind11::arg("num_init_tokens"), pybind11::arg("num_local_tokens"));
  m.def("seed_prep_litetopk_", &seed_prep_litetopk_,
        "In-place fused sample prep (caller-owned buffers)",
        pybind11::arg("slog"), pybind11::arg("num_buckets"),
        pybind11::arg("topk"), pybind11::arg("cand_cap"),
        pybind11::arg("emit_limit"), pybind11::arg("headroom"),
        pybind11::arg("probe_stride_tok"), pybind11::arg("hist_stride"),
        pybind11::arg("origin"), pybind11::arg("inv_delta"),
        pybind11::arg("th_bucket"), pybind11::arg("bcount"),
        pybind11::arg("cand_val"), pybind11::arg("cand_idx"),
        pybind11::arg("cand_cnt"));
  m.def("carry_votes_topk_reset_", &carry_votes_topk_reset_litetopk_,
        "Deterministic carry-vote top-k with fused histogram reset",
        pybind11::arg("votes"), pybind11::arg("out_idx"),
        pybind11::arg("partial"), pybind11::arg("state"), pybind11::arg("k"),
        pybind11::arg("max_vote"), pybind11::arg("min_index") = 0);
  m.def("map_topk_indices_and_accumulate_votes_litetopk_",
        &map_topk_indices_and_accumulate_votes_litetopk_,
        "Map physical TOPK winners to corpus space and vote every winner "
        "from the most recent query rows in one grid-wide pass",
        pybind11::arg("out_idx"), pybind11::arg("index_map"),
        pybind11::arg("status"), pybind11::arg("votes"),
        pybind11::arg("vote_recent_rows") = 1);
  m.def("map_topk_vote_stats_litetopk_", &map_topk_vote_stats_litetopk_,
        "map/vote pass with fused candidate-count telemetry");
}
