/**
 * Radix Top-M selection derived from MiSA.
 * Scores remain FP32. A coarse FP16 bin is only a search accelerator:
 * overflowing bins use full-row FP32 radix selection, never truncated candidates.
 * Contract: finite scores or -inf masks; equal-score boundary IDs are unordered.
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstddef>
#include <cstdint>
#include <climits>
#include <mutex>
#include <unordered_set>

namespace {

constexpr int MaxTopK = 8192;
constexpr int ThreadsPerBlock = 1024;
constexpr size_t DynamicSmem = 24 * 1024 * sizeof(uint32_t);

struct FastTopKParams {
  const float *__restrict__ input;
  int32_t *__restrict__ indices;
  int64_t input_stride;
  int32_t length;
};

__device__ void naive_topk_cuda(const float *input, int32_t *__restrict__ indices, int32_t length,
                                int32_t topk) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < topk; i += ThreadsPerBlock) {
    indices[i] = (i < length && isfinite(input[i])) ? i : -1;
  }
}

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key =
      (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                      : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}


/** Exact slow path: find the full FP32 cutoff in four full-row passes.
 * No candidate buffer and no host synchronization. The last pass leaves the
 * number of ties needed; emit every strict winner plus that many boundary ties.
 */
__device__ void rescan_topm(const float *input, int32_t *indices,
                           int length, int target) {
  __shared__ int hist[2][257];
  __shared__ int threshold;
  __shared__ int above_counter, tie_counter;
  const int tx = threadIdx.x;
  uint32_t prefix = 0, mask = 0;
  int remain = target;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tx <= 256) hist[0][tx] = 0;
    __syncthreads();
    for (int idx = tx; idx < length; idx += ThreadsPerBlock) {
      const uint32_t key = convert_to_uint32(input[idx]);
      if ((key & mask) == prefix)
        atomicAdd(&hist[0][(key >> shift) & 255], 1);
    }
    __syncthreads();
    for (int round = 0; round < 8; ++round) {
      if (tx < 256) {
        const int distance = 1 << round;
        int value = hist[round & 1][tx];
        if (tx + distance < 256) value += hist[round & 1][tx + distance];
        hist[(round & 1) ^ 1][tx] = value;
      }
      __syncthreads();
    }
    if (tx < 256 && hist[0][tx] >= remain && hist[0][tx + 1] < remain)
      threshold = tx;
    __syncthreads();
    const int bin = threshold;
    remain -= hist[0][bin + 1];
    prefix |= static_cast<uint32_t>(bin) << shift;
    mask |= 255u << shift;
    // Every thread must finish reading hist before the next round resets it.
    __syncthreads();
  }
  if (tx == 0) above_counter = tie_counter = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += ThreadsPerBlock) {
    const uint32_t key = convert_to_uint32(input[idx]);
    if (key > prefix) {
      const int pos = atomicAdd(&above_counter, 1);
      indices[pos] = isfinite(input[idx]) ? idx : -1;
    } else if (key == prefix) {
      const int pos = atomicAdd(&tie_counter, 1);
      if (pos < remain)
        indices[target - remain + pos] = isfinite(input[idx]) ? idx : -1;
    }
  }
  __syncthreads();
}

template <bool CacheScores>
__device__ void fast_topk_cuda(const float *__restrict__ input,
                               int32_t *__restrict__ indices, int32_t length,
                               int32_t topk_target) {
  int topk = topk_target;
  constexpr int BlockSize = 1024;
  constexpr int Radix = 256;
  constexpr int SmemInputSize =
      DynamicSmem / (2 * static_cast<int>(sizeof(int)));

  alignas(128) __shared__ int histogram_buffer[2][Radix + 128];
  alignas(128) __shared__ int counter;
  alignas(128) __shared__ int threshold_bin_id;
  alignas(128) __shared__ int num_input[2];

  auto &histogram = histogram_buffer[0];
  extern __shared__ int input_indices[][SmemInputSize];
  const int tx = threadIdx.x;

  if (tx < Radix + 1) {
    histogram[tx] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += BlockSize) {
    const auto bin = convert_to_uint8(input[idx]);
    atomicAdd(&histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == Radix);
      if (C10_LIKELY(tx < Radix)) {
        const int distance = 1 << i;
        const int buffer = i & 1;
        int value = histogram_buffer[buffer][tx];
        if (tx < Radix - distance) {
          value += histogram_buffer[buffer][tx + distance];
        }
        histogram_buffer[buffer ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < Radix && histogram[tx] > topk &&
      histogram[tx + 1] <= topk) {
    threshold_bin_id = tx;
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  int threshold_bin = threshold_bin_id;
  topk -= histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BlockSize) {
      const int bin = static_cast<int>(convert_to_uint8(input[idx]));
      if (bin > threshold_bin) {
        const int position = atomicAdd(&counter, 1);
        if (position < topk_target) {
          indices[position] = isfinite(input[idx]) ? idx : -1;
        }
      }
    }
    __syncthreads();
    return;
  }

  const int coarse_count = histogram[threshold_bin] - histogram[threshold_bin + 1];
  if (coarse_count > SmemInputSize) {
    rescan_topm(input, indices, length, topk_target);
    return;
  }

  __syncthreads();
  if (tx < Radix + 1) {
    histogram[tx] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += BlockSize) {
    const float raw = input[idx];
    const int bin = static_cast<int>(convert_to_uint8(raw));
    if (bin > threshold_bin) {
      const int position = atomicAdd(&counter, 1);
      if (position < topk_target) {
        indices[position] = isfinite(input[idx]) ? idx : -1;
      }
    } else if (bin == threshold_bin) {
      const int position = atomicAdd(&num_input[0], 1);
      if (C10_LIKELY(position < SmemInputSize)) {
        input_indices[0][position] = idx;
        if constexpr (CacheScores)
          input_indices[1][position] = static_cast<int>(convert_to_uint32(raw));
        const int sub_bin = (convert_to_uint32(raw) >> 24) & 0xFF;
        atomicAdd(&histogram[sub_bin], 1);
      }
    }
  }
  __syncthreads();

  if constexpr (CacheScores) {
    // Keep the original coarse-bin capacity and launch memory footprint:
    // buffer 0 stores key IDs, buffer 1 caches the exact ordered FP32 bits.
    // Refine via shared-memory prefix scans instead of gathering global scores
    // through a ping-pong index list on each radix pass.
    uint32_t prefix = 0, mask = 0;
    int remain = topk;
    for (int shift = 24; shift >= 0; shift -= 8) {
      run_cumsum();
      if (tx < Radix && histogram[tx] >= remain &&
          histogram[tx + 1] < remain)
        threshold_bin_id = tx;
      __syncthreads();
      const int bin = threshold_bin_id;
      remain -= histogram[bin + 1];
      prefix |= static_cast<uint32_t>(bin) << shift;
      mask |= 255u << shift;
      __syncthreads();
      if (shift > 0) {
        if (tx < Radix + 1) histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < coarse_count; i += BlockSize) {
          const uint32_t key = static_cast<uint32_t>(input_indices[1][i]);
          if ((key & mask) == prefix)
            atomicAdd(&histogram[(key >> (shift - 8)) & 255], 1);
        }
        __syncthreads();
      }
    }
    if (tx == 0) num_input[1] = 0;
    __syncthreads();
    for (int i = tx; i < coarse_count; i += BlockSize) {
      const uint32_t key = static_cast<uint32_t>(input_indices[1][i]);
      const uint32_t bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
      const int idx = isfinite(__uint_as_float(bits)) ? input_indices[0][i] : -1;
      if (key > prefix) {
        const int position = atomicAdd(&counter, 1);
        if (position < topk_target) indices[position] = idx;
      } else if (key == prefix) {
        const int position = atomicAdd(&num_input[1], 1);
        if (position < remain) indices[topk_target - remain + position] = idx;
      }
    }
    __syncthreads();
    return;
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int last_remain;
    const int read_buffer = round % 2;
    // The full coarse bin fits; every later bin is a subset, so cannot overflow.
    const int active_input = num_input[read_buffer];

    run_cumsum();
    if (tx < Radix && histogram[tx] > topk &&
        histogram[tx + 1] <= topk) {
      threshold_bin_id = tx;
      num_input[read_buffer ^ 1] = 0;
      last_remain = topk - histogram[tx + 1];
    }
    __syncthreads();

    threshold_bin = threshold_bin_id;
    topk -= histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < active_input; i += BlockSize) {
        const int idx = input_indices[read_buffer][i];
        const int offset = 24 - round * 8;
        const int bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const int position = atomicAdd(&counter, 1);
          if (position < topk_target) {
            indices[position] = isfinite(input[idx]) ? idx : -1;
          }
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (tx < Radix + 1) {
      histogram[tx] = 0;
    }
    __syncthreads();
    for (int i = tx; i < active_input; i += BlockSize) {
      const int idx = input_indices[read_buffer][i];
      const float raw = input[idx];
      const int offset = 24 - round * 8;
      const int bin = (convert_to_uint32(raw) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const int position = atomicAdd(&counter, 1);
        if (position < topk_target) {
          indices[position] = isfinite(input[idx]) ? idx : -1;
        }
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const int previous = atomicAdd(&last_remain, -1);
          const int output = topk_target - previous;
          if (previous > 0 && output >= 0 && output < topk_target) {
            indices[output] = isfinite(input[idx]) ? idx : -1;
          }
        } else {
          const int position = atomicAdd(&num_input[read_buffer ^ 1], 1);
          if (C10_LIKELY(position < SmemInputSize)) {
            input_indices[read_buffer ^ 1][position] = idx;
            const int sub_bin =
                (convert_to_uint32(raw) >> (offset - 8)) & 0xFF;
            atomicAdd(&histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}


} // namespace
