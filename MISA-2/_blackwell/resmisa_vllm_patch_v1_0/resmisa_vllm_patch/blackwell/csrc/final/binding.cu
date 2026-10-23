#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <limits>
#include <cstdint>
using CandidateValue=uint16_t;
static void check_candidate_dtype(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type()==torch::kHalf,"opaque U16 candidate storage");
}
namespace h2048_safe_topk {

constexpr int kBins = 256;
constexpr int kMinCap = 65536;
constexpr int kMaxCap = 1 << 20;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadCertificate = 1u << 4,
  // Recoverable candidate-boundary overflow.
  kBoundaryOverflow = 1u << 5,
};

constexpr uint32_t kNonOverflowStatusMask =
    kBadCount | kNonFinite | kBadPhysical | kBadCertificate;
static_assert(kBoundaryOverflow == 32u);
static_assert((kNonOverflowStatusMask & kBoundaryOverflow) == 0u);

__device__ __forceinline__ uint32_t candidate_score_code(uint16_t value,
                                                         int32_t packed_index) {
  return ((static_cast<uint32_t>(packed_index) >> 20) << 16) |
         static_cast<uint32_t>(value);
}

__device__ __forceinline__ float decode_score_code(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}


__device__ __forceinline__ int coarse_bucket_scaled(uint32_t code) {
  constexpr int bins = kBins * 8;
  const float value = decode_score_code(code);
  return value < 0.0f
             ? 0
             : (value >= static_cast<float>(kBins)
                    ? bins - 1
                    : static_cast<int>(value * static_cast<float>(8)));
}

__device__ __forceinline__ uint32_t fp24_code(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return ordered >> 8;
}

template <int Bins>
__device__ __forceinline__ void find_radix_digit(
    const uint32_t* __restrict__ hist, uint32_t* __restrict__ desired,
    uint32_t* __restrict__ rank, uint32_t* __restrict__ selected_count,
    int shift) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= 32) return;
  constexpr unsigned kFull = 0xffffffffu;
  static_assert(Bins == 256 || Bins == 512 || Bins == 1024 || Bins == 2048 ||
                Bins == 4096);
  constexpr int kGroupBins = Bins / 32;
  constexpr int kItemsPerLane = (kGroupBins + 31) / 32;
  const int lane = tid;
  const int first = lane * kGroupBins;
  uint32_t group_count = 0u;
#pragma unroll
  for (int i = 0; i < kGroupBins; ++i) group_count += hist[first + i];
  uint32_t inclusive = group_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFull, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  const uint32_t target = *rank;
  const unsigned group_mask = __ballot_sync(kFull, inclusive >= target);
  if (target == 0u || group_mask == 0u) return;
  const int winning_group = __ffs(group_mask) - 1;
  const uint32_t group_before =
      __shfl_sync(kFull, inclusive - group_count, winning_group);
  // For 2048/4096 bins the winning coarse group contains 64/128 bins.
  // Give each lane a contiguous 2/4-bin segment, scan segment totals, then
  // let the winning lane locate the exact bin locally.  The same code also
  // handles the 8-bin exact-radix histogram used below.
  const int segment_offset = lane * kItemsPerLane;
  const bool segment_valid = segment_offset < kGroupBins;
  const int segment_first = winning_group * kGroupBins + segment_offset;
  uint32_t segment_count = 0u;
#pragma unroll
  for (int i = 0; i < kItemsPerLane; ++i) {
    if (segment_offset + i < kGroupBins) {
      segment_count += hist[segment_first + i];
    }
  }
  uint32_t segment_inclusive = segment_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFull, segment_inclusive, offset);
    if (lane >= offset) segment_inclusive += other;
  }
  const unsigned segment_mask = __ballot_sync(
      kFull, segment_valid && group_before + segment_inclusive >= target);
  if (segment_mask == 0u) return;
  const int winning_lane = __ffs(segment_mask) - 1;
  const uint32_t segment_before =
      __shfl_sync(kFull, segment_inclusive - segment_count, winning_lane);

  uint32_t local_digit = 0u;
  uint32_t local_before = 0u;
  uint32_t local_count = 0u;
  if (lane == winning_lane) {
    const uint32_t local_target = target - group_before - segment_before;
    uint32_t running = 0u;
    bool found = false;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) {
      const bool valid = segment_offset + i < kGroupBins;
      const uint32_t count = valid ? hist[segment_first + i] : 0u;
      if (!found && valid && running + count >= local_target) {
        local_digit = static_cast<uint32_t>(segment_offset + i);
        local_before = running;
        local_count = count;
        found = true;
      }
      running += count;
    }
  }
  local_digit = __shfl_sync(kFull, local_digit, winning_lane);
  local_before = __shfl_sync(kFull, local_before, winning_lane);
  local_count = __shfl_sync(kFull, local_count, winning_lane);
  if (lane == 0) {
    const uint32_t digit =
        static_cast<uint32_t>(winning_group * kGroupBins) + local_digit;
    *desired |= digit << static_cast<uint32_t>(shift);
    *rank = target - group_before - segment_before - local_before;
    *selected_count = local_count;
  }
}

// Integer-code predicates and scalar coarse histogram.
__global__ __launch_bounds__(256) void coarse_tiering_topk_kernel(
    const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed_indices,
    const int32_t* __restrict__ counts, int32_t* __restrict__ output,
    int32_t* __restrict__ status, int rows,
    int cap, int topk, int sequence_length) {
  constexpr int Threads = 256;
  constexpr int BoundaryCapacity = 512;
  constexpr int kCoarseBins = kBins * 8;
  constexpr unsigned kFull = 0xffffffffu;
  const int logical_block = static_cast<int>(blockIdx.x);
  // Real count argmax is near the end; longest rows launch first.
  const int row = rows - 1 - logical_block;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  if (row < 0 || row >= rows) return;

  __shared__ uint32_t hist[kCoarseBins];
  __shared__ uint32_t boundary_code[BoundaryCapacity];
  // Candidate rows may use the production 196608-record capacity.  A uint16
  // slot silently wrapped beyond the old 65536-record qualification shape.
  __shared__ uint32_t boundary_slot[BoundaryCapacity];
  __shared__ uint32_t s_status;
  __shared__ uint32_t s_desired;
  __shared__ uint32_t s_rank;
  __shared__ uint32_t s_selected_count;
  __shared__ int s_count;
  __shared__ int s_coarse_bucket;
  __shared__ int s_coarse_lt;
  __shared__ int s_coarse_need;
  __shared__ int s_boundary_count;
  __shared__ int s_strict_cursor;
  __shared__ int s_boundary_cursor;
  __shared__ int s_boundary_lt_cursor;

  if (tid == 0) {
    const int raw_count = counts[row];
    s_status = 0u;
    if (raw_count < topk || raw_count > cap || raw_count < 0) {
      s_status |= kBadCount;
    }
    s_count = max(0, min(raw_count, cap));
    s_desired = 0u;
    s_rank = static_cast<uint32_t>(topk);
    s_selected_count = 0u;
    s_coarse_bucket = -1;
    s_coarse_lt = 0;
    s_coarse_need = 0;
    s_boundary_count = 0;
    s_strict_cursor = 0;
    s_boundary_cursor = 0;
    s_boundary_lt_cursor = 0;
    status[row] = 0;
  }
  for (int i = tid; i < kCoarseBins; i += Threads) hist[i] = 0u;
  __syncthreads();

  const int count = s_count;
  const int64_t row_base = static_cast<int64_t>(row) * cap;
  const int64_t out_base = static_cast<int64_t>(row) * topk;
  if (s_status != 0u) {
    for (int i = tid; i < topk; i += Threads) output[out_base + i] = -1;
    if (tid == 0) status[row] = static_cast<int32_t>(s_status);
    return;
  }

  // Coarse certificate for the selected scalar histogram.
  {
    for (int slot = tid; slot < count; slot += Threads) {
      const uint16_t value = values[row_base + slot];
      const int32_t packed_index = packed_indices[row_base + slot];
      const uint32_t physical =
          static_cast<uint32_t>(packed_index) & kPhysicalMask;
      if (physical >= static_cast<uint32_t>(sequence_length)) {
        atomicOr(&s_status, kBadPhysical);
        continue;
      }
      const uint32_t code = candidate_score_code(value, packed_index);
      const float decoded = decode_score_code(code);
      if (!isfinite(decoded)) {
        atomicOr(&s_status, kNonFinite);
        continue;
      }
      atomicAdd(hist + coarse_bucket_scaled(code), 1u);
    }
  }
  __syncthreads();
  if (s_status == 0u) {
    find_radix_digit<kCoarseBins>(hist, &s_desired, &s_rank, &s_selected_count,
                                  0);
  }
  __syncthreads();
  if (tid == 0 && s_status == 0u) {
    s_coarse_bucket = static_cast<int>(s_desired);
    s_coarse_need = static_cast<int>(s_rank);
    s_coarse_lt = topk - s_coarse_need;
    s_boundary_count = static_cast<int>(s_selected_count);
    if (s_coarse_bucket < 0 || s_coarse_bucket >= kCoarseBins ||
        s_coarse_lt < 0 || s_coarse_lt >= topk || s_coarse_need <= 0 ||
        s_coarse_need > s_boundary_count) {
      s_status |= kBadCertificate;
    }
    if (s_boundary_count > BoundaryCapacity) {
      s_status |= kBoundaryOverflow;
    }
  }
  __syncthreads();
  if (s_status != 0u) {
    for (int i = tid; i < topk; i += Threads) output[out_base + i] = -1;
    if (tid == 0) {
      status[row] = static_cast<int32_t>(s_status);

    }
    return;
  }

  // Emit physical IDs below the coarse boundary and gather the boundary.
  const int threshold_bucket = s_coarse_bucket;
  const float threshold_edge =
      static_cast<float>(threshold_bucket) / static_cast<float>(8);
  const float next_threshold_edge =
      static_cast<float>(threshold_bucket + 1) / static_cast<float>(8);
  const uint32_t threshold_code = fp24_code(threshold_edge);
  const uint32_t next_threshold_code = fp24_code(next_threshold_edge);
  const int warp = tid >> 5;
  for (int base = warp * 32; base < count; base += Threads) {
    const int slot = base + lane;
    uint32_t code = 0u;
    int32_t packed_index = 0;
    const bool valid = slot < count;
    if (valid) {
      packed_index = packed_indices[row_base + slot];
      code = candidate_score_code(values[row_base + slot], packed_index);
    }
    const bool is_strict =
        (valid && threshold_bucket > 0 && code < threshold_code);
    const bool is_boundary =
        (valid &&
                     (threshold_bucket == kCoarseBins - 1 ||
                      code < next_threshold_code) &&
                     (threshold_bucket == 0 || code >= threshold_code));
    const unsigned strict_mask = __ballot_sync(kFull, is_strict);
    const unsigned boundary_mask = __ballot_sync(kFull, is_boundary);
    int strict_base = 0;
    int boundary_base = 0;
    if (lane == 0) {
      const int strict_n = __popc(strict_mask);
      const int boundary_n = __popc(boundary_mask);
      if (strict_n) strict_base = atomicAdd(&s_strict_cursor, strict_n);
      if (boundary_n) {
        boundary_base = atomicAdd(&s_boundary_cursor, boundary_n);
      }
    }
    strict_base = __shfl_sync(kFull, strict_base, 0);
    boundary_base = __shfl_sync(kFull, boundary_base, 0);
    const unsigned lane_before =
        lane == 0 ? 0u : ((1u << static_cast<uint32_t>(lane)) - 1u);
    if (is_strict) {
      const int pos = strict_base + __popc(strict_mask & lane_before);
      if (pos < topk) {
        output[out_base + pos] = static_cast<int32_t>(
            static_cast<uint32_t>(packed_index) & kPhysicalMask);
      }
    }
    if (is_boundary) {
      const int pos = boundary_base + __popc(boundary_mask & lane_before);
      if (pos < BoundaryCapacity) {
        boundary_code[pos] = code;
        boundary_slot[pos] = static_cast<uint32_t>(slot);
      }
    }
  }
  __syncthreads();
  if (tid == 0 && (s_strict_cursor != s_coarse_lt ||
                   s_boundary_cursor != s_boundary_count)) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  if (tid == 0) {
    s_desired = 0u;
    s_rank = static_cast<uint32_t>(s_coarse_need);
    s_selected_count = 0u;
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 3; ++pass) {
    for (int i = tid; i < kBins; i += Threads) hist[i] = 0u;
    __syncthreads();
    const uint32_t desired = s_desired;
    for (int j = tid; j < s_boundary_count; j += Threads) {
      const uint32_t code = boundary_code[j];
      bool keep = true;
      if (pass == 1) keep = (code >> 16) == (desired >> 16);
      if (pass == 2) keep = (code >> 8) == (desired >> 8);
      if (keep) {
        atomicAdd(hist + ((code >> (16 - 8 * pass)) & 0xffu), 1u);
      }
    }
    __syncthreads();
    find_radix_digit<kBins>(hist, &s_desired, &s_rank, &s_selected_count,
                            16 - pass * 8);
    __syncthreads();
  }
  if (tid == 0 && (s_rank == 0u || s_rank > s_selected_count ||
                   s_rank > static_cast<uint32_t>(s_coarse_need))) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  const uint32_t exact_pivot = s_desired;
  const int exact_equal_take = static_cast<int>(s_rank);
  const int boundary_strict = s_coarse_need - exact_equal_take;
  for (int j = tid; j < s_boundary_count; j += Threads) {
    const uint32_t code = boundary_code[j];
    bool take = code < exact_pivot;
    int output_pos = -1;
    if (take) {
      output_pos = s_coarse_lt + atomicAdd(&s_boundary_lt_cursor, 1);
    } else if (code == exact_pivot) {
      const uint32_t slot = boundary_slot[j];
      int equal_rank = 0;
#pragma unroll 1
      for (int other = 0; other < s_boundary_count; ++other) {
        equal_rank +=
            boundary_code[other] == exact_pivot && boundary_slot[other] < slot;
      }
      if (equal_rank < exact_equal_take) {
        take = true;
        output_pos = s_coarse_lt + boundary_strict + equal_rank;
      }
    }
    if (take && output_pos >= 0 && output_pos < topk) {
      const int slot = static_cast<int>(boundary_slot[j]);
      output[out_base + output_pos] = static_cast<int32_t>(
          static_cast<uint32_t>(packed_indices[row_base + slot]) &
          kPhysicalMask);
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (s_boundary_lt_cursor != boundary_strict) {
      s_status |= kBadCertificate;
    }
    status[row] = static_cast<int32_t>(s_status);

  }
}

namespace overflow {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRadixBits = 12;
constexpr int kRadixBins = 1 << kRadixBits;
constexpr int kBinsPerThread = kRadixBins / kThreads;
constexpr int kTopK = 2048;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadMapped = 1u << 3,
  kHistogramFailure = 1u << 4,
  kBoundaryOverflow = 1u << 5,
  kCompactFailure = 1u << 6,
};

__device__ __forceinline__ uint32_t candidate_score_code(uint16_t value,
                                                         int32_t packed_index) {
  return ((static_cast<uint32_t>(packed_index) >> 20) << 16) |
         static_cast<uint32_t>(value);
}

__device__ __forceinline__ float decode_candidate_score(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int block_exclusive_sum(int value,
                                                   int* warp_prefix) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  int inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) warp_prefix[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int original = lane < kWarps ? warp_prefix[lane] : 0;
    int warp_inclusive = original;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
      if (lane >= offset) warp_inclusive += other;
    }
    if (lane < kWarps) warp_prefix[lane] = warp_inclusive - original;
  }
  __syncthreads();
  return warp_prefix[warp] + inclusive - value;
}

__device__ __forceinline__ void select_histogram_bin(const int* histogram,
                                                     int target,
                                                     int* warp_prefix,
                                                     int* selected_bin,
                                                     int* selected_count_lt) {
  const int begin = static_cast<int>(threadIdx.x) * kBinsPerThread;
  int segment_sum = 0;
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    segment_sum += histogram[begin + i];
  }
  const int segment_lt = block_exclusive_sum(segment_sum, warp_prefix);
  if (target > segment_lt && target <= segment_lt + segment_sum) {
    int local_lt = 0;
#pragma unroll
    for (int i = 0; i < kBinsPerThread; ++i) {
      const int count = histogram[begin + i];
      if (target <= segment_lt + local_lt + count) {
        *selected_bin = begin + i;
        *selected_count_lt = segment_lt + local_lt;
        break;
      }
      local_lt += count;
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreads) void overflow_exact_topk_kernel(
    const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed_indices,
    const int32_t* __restrict__ counts, int32_t* __restrict__ output,
    int32_t* __restrict__ status, int rows, int cap, int sequence_length,
    int topk_arg = kTopK) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;

  // CTA-uniform and before any barrier: the common no-overflow case only
  // reads one cached status word and retires the block.
  const uint32_t input_status = static_cast<uint32_t>(status[row]);
  if ((input_status & kBoundaryOverflow) == 0u) return;

  __shared__ int histogram[kRadixBins];
  __shared__ int warp_scratch[kWarps];
  __shared__ int selected_bin;
  __shared__ int selected_count_lt;
  __shared__ int first_count_lt;
  __shared__ uint32_t threshold_code;
  __shared__ int warp_lt[kWarps];
  __shared__ int warp_eq[kWarps];
  __shared__ int tile_lt;
  __shared__ int tile_eq;
  __shared__ int base_lt;
  __shared__ int base_eq;
  __shared__ uint32_t block_status;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lane_before =
      lane == 0 ? 0u : ((1u << static_cast<uint32_t>(lane)) - 1u);
  const int64_t candidate_row = static_cast<int64_t>(row) * cap;
  const int64_t output_row = static_cast<int64_t>(row) * topk_arg;
  const int count = counts[row];

  if (tid == 0) {
    // Overflow is recoverable. Any other fast-path error is retained and
    // prevents the slow path from declaring success.
    block_status = input_status & ~kBoundaryOverflow;
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();

  if (count < topk_arg || count > cap) {
    if (tid == 0) block_status |= kBadCount;
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  __syncthreads();

  // Pass 1: all high 12 bits. It also repeats fail-closed ABI validation so
  // the fallback is independently safe from a stale/corrupt fast flag.
  for (int col = tid; col < count; col += kThreads) {
    const int64_t offset = candidate_row + col;
    const int32_t packed = packed_indices[offset];
    const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
    const uint32_t code = candidate_score_code(values[offset], packed);
    if (!isfinite(decode_candidate_score(code))) {
      atomicOr(&block_status, static_cast<uint32_t>(kNonFinite));
    }
    if (physical >= static_cast<uint32_t>(sequence_length)) {
      atomicOr(&block_status, static_cast<uint32_t>(kBadPhysical));
    }
    atomicAdd(histogram + (code >> kRadixBits), 1);
  }
  __syncthreads();

  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  select_histogram_bin(histogram, topk_arg, warp_scratch, &selected_bin,
                       &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 ||
        selected_count_lt >= topk_arg) {
      block_status |= kHistogramFailure;
    } else {
      first_count_lt = selected_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }
  const int high_bin = selected_bin;
  const int remaining_rank = topk_arg - first_count_lt;
  // All warps snapshot pass-one state before thread zero resets it.
  __syncthreads();

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  if (tid == 0) {
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();

  // Pass 2: low 12 bits inside the winning high bucket.
  for (int col = tid; col < count; col += kThreads) {
    const int64_t offset = candidate_row + col;
    const int32_t packed = packed_indices[offset];
    const uint32_t code = candidate_score_code(values[offset], packed);
    if (static_cast<int>(code >> kRadixBits) == high_bin) {
      atomicAdd(histogram + (code & (kRadixBins - 1)), 1);
    }
  }
  __syncthreads();
  select_histogram_bin(histogram, remaining_rank, warp_scratch, &selected_bin,
                       &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 ||
        selected_count_lt >= remaining_rank) {
      block_status |= kHistogramFailure;
    } else {
      threshold_code = (static_cast<uint32_t>(high_bin) << kRadixBits) |
                       static_cast<uint32_t>(selected_bin);
      first_count_lt += selected_count_lt;
      base_lt = 0;
      base_eq = first_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
    __syncthreads();
    if (tid == 0) status[row] = static_cast<int32_t>(block_status);
    return;
  }

  // Pass 3: stable exact compact and fused winner mapping.
  for (int tile = 0; tile < count; tile += kThreads) {
    const int col = tile + tid;
    uint32_t code = 0xffffffffu;
    int32_t packed = 0;
    if (col < count) {
      packed = packed_indices[candidate_row + col];
      code = candidate_score_code(values[candidate_row + col], packed);
    }
    const bool is_lt = col < count && code < threshold_code;
    const bool is_eq = col < count && code == threshold_code;
    const unsigned lt_mask = __ballot_sync(0xffffffffu, is_lt);
    const unsigned eq_mask = __ballot_sync(0xffffffffu, is_eq);
    const int lane_lt = __popc(lt_mask & lane_before);
    const int lane_eq = __popc(eq_mask & lane_before);
    if (lane == 0) {
      warp_lt[warp] = __popc(lt_mask);
      warp_eq[warp] = __popc(eq_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix_lt = 0;
      int prefix_eq = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        const int count_lt = warp_lt[w];
        const int count_eq = warp_eq[w];
        warp_lt[w] = prefix_lt;
        warp_eq[w] = prefix_eq;
        prefix_lt += count_lt;
        prefix_eq += count_eq;
      }
      tile_lt = base_lt;
      tile_eq = base_eq;
      base_lt += prefix_lt;
      base_eq += prefix_eq;
    }
    __syncthreads();

    int output_col = -1;
    if (is_lt) {
      output_col = tile_lt + warp_lt[warp] + lane_lt;
    } else if (is_eq) {
      output_col = tile_eq + warp_eq[warp] + lane_eq;
      if (output_col >= topk_arg) output_col = -1;
    }
    if (output_col >= 0 && output_col < topk_arg) {
      const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
      // Match the production fast-path contract: emit physical winners here.
      // The existing uniform winner-map kernel runs once after both paths.
      output[output_row + output_col] = static_cast<int32_t>(physical);
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (base_lt != first_count_lt || base_lt > topk_arg || base_eq < topk_arg) {
      block_status |= kCompactFailure;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    for (int col = tid; col < topk_arg; col += kThreads) {
      output[output_row + col] = -1;
    }
  }
  __syncthreads();
  if (tid == 0) {
    // Successful completion clears the recoverable overflow bit.
    status[row] = static_cast<int32_t>(block_status);
  }
}

}  // namespace overflow

static_assert(kBoundaryOverflow == overflow::kBoundaryOverflow);
static_assert(kBoundaryOverflow == 32u);

}  // namespace h2048_safe_topk
void h2048_safe_topk_out_litetopk_(torch::Tensor cand_val,
                                   torch::Tensor cand_idx,
                                   torch::Tensor cand_cnt,
                                   torch::Tensor out_idx, torch::Tensor status,
                                   int64_t index_limit64) {
  static_assert(
      20 == 20,
      "h2048 safe selector requires the production 20-bit physical ID ABI");
  static_assert(sizeof(CandidateValue) == sizeof(uint16_t));

  TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() &&
                  out_idx.is_cuda() && status.is_cuda(),
              "h2048 safe selector tensors must be CUDA");
  TORCH_CHECK(cand_val.device() == cand_idx.device() &&
                  cand_val.device() == cand_cnt.device() &&
                  cand_val.device() == out_idx.device() &&
                  cand_val.device() == status.device(),
              "h2048 safe selector tensors must be on one CUDA device");
  TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                  cand_cnt.is_contiguous() && out_idx.is_contiguous() &&
                  status.is_contiguous(),
              "h2048 safe selector tensors must be contiguous");
  check_candidate_dtype(cand_val);
  TORCH_CHECK(cand_idx.scalar_type() == torch::kInt &&
                  cand_cnt.scalar_type() == torch::kInt &&
                  out_idx.scalar_type() == torch::kInt &&
                  status.scalar_type() == torch::kInt,
              "h2048 safe selector metadata and output must be int32");
  TORCH_CHECK(cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
              "h2048 candidate tensors must be [R,CAP]");
  const int64_t rows64 = cand_val.size(0);
  // Capacity floor scales with the selection width: 32x topk matches the
  // historical 65536 floor at K=2048 and admits 16384 at K=512.
  const int64_t min_cap64 =
      std::max<int64_t>(16384, out_idx.dim() == 2 ? 32 * out_idx.size(1)
                                                  : h2048_safe_topk::kMinCap);
  TORCH_CHECK(rows64 > 0 && rows64 <= std::numeric_limits<int>::max() &&
                  cand_val.size(1) >= min_cap64 &&
                  cand_val.size(1) <= h2048_safe_topk::kMaxCap,
              "h2048 safe selector requires [R,CAP] candidates with "
              "max(16384, 32*topk) <= CAP <= 1M");
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == rows64 &&
                  status.dim() == 1 && status.numel() == rows64,
              "h2048 cand_cnt/status must have R elements");
  TORCH_CHECK(out_idx.dim() == 2 && out_idx.size(0) == rows64 &&
                  out_idx.size(1) >= 1 &&
                  out_idx.size(1) <= h2048_safe_topk::overflow::kTopK &&
                  out_idx.size(1) <= cand_val.size(1),
              "h2048 safe selector output must be [R,topk<=2048]");
  TORCH_CHECK(
      index_limit64 > 0 &&
          index_limit64 <= (int64_t{1} << 20),
      "h2048 index_limit must be in [1,1M]");

  const int rows = static_cast<int>(rows64);
  const int cap = static_cast<int>(cand_val.size(1));
  const int topk = static_cast<int>(out_idx.size(1));
  const int index_limit = static_cast<int>(index_limit64);
  const c10::cuda::CUDAGuard device_guard(cand_val.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  h2048_safe_topk::coarse_tiering_topk_kernel
      <<<rows, 256, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(cand_val.data_ptr<at::Half>()),
          cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
          out_idx.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
          rows, cap, topk, index_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // No-overflow rows return before the fallback's first CTA barrier.  An
  // overflow row is selected exactly in physical space and clears bit 32;
  // all non-overflow errors remain nonzero for the following map to trap.
  h2048_safe_topk::overflow::overflow_exact_topk_kernel<<<
      rows, h2048_safe_topk::overflow::kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(cand_val.data_ptr<at::Half>()),
      cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
      out_idx.data_ptr<int32_t>(), status.data_ptr<int32_t>(), rows, cap,
      index_limit, topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("safe_topk_",&h2048_safe_topk_out_litetopk_);}
