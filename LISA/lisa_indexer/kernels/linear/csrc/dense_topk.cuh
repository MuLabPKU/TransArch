#pragma once

// Exact dense-logit top-k replacement for the short-context vLLM prefill
// path. The entry point builds an FP16-coarse histogram and
// performs exact boundary selection in one CTA/launch without global
// metadata. Unusually large/tied boundaries stay on device and refine
// through exact sortable-FP32 radix digits.

#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>

#include <climits>
#include <cstdint>

namespace litetopk_dense {

constexpr int kDenseTopkThreads = 512;
constexpr int kDenseTopkFineBins = 4096;
constexpr int kDenseTopkBoundaryCap = 2048;

// Ascending codes correspond to descending numerical score order. This is
// the same FP16 coarse ordering used by vLLM's prefill top-k.
__device__ __forceinline__ int dense_topk_first_half_bin(float x) {
  // CUDA may canonicalize a negative FP32 NaN to a positive FP16 NaN.
  // Restoring the source sign after conversion is branch-free; for finite
  // values, infinities, and zeros the conversion already has this sign.
  const uint32_t fp32_bits = __float_as_uint(x);
  const __half hx = __float2half(x);
  uint16_t bits = __half_as_ushort(hx) & 0x7fffu;
  bits |= static_cast<uint16_t>((fp32_bits >> 16) & 0x8000u);
  bits = (bits & 0x8000u) ? bits : static_cast<uint16_t>(~bits & 0x7fffu);
  return static_cast<int>(bits >> 4);
}

// Full-width counterpart of dense_topk_first_half_bin. Ascending codes are
// descending IEEE-754 bit order, including signed zero and NaN payloads.
__device__ __forceinline__ uint32_t dense_topk_descending_float_code(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

template <typename Fn>
__device__ __forceinline__ void dense_topk_visit_row(const float* row,
                                                     int begin, int end,
                                                     int stride1, Fn fn) {
  if (stride1 == 1) {
    const int len = end - begin;
    const float* base = row + begin;
    const int misalignment =
        static_cast<int>((reinterpret_cast<uintptr_t>(base) >> 2) & 3u);
    const int prefix = min(len, (4 - misalignment) & 3);
    for (int offset = threadIdx.x; offset < prefix; offset += blockDim.x) {
      fn(base[offset], offset);
    }
    base += prefix;
    const int vector_len = len - prefix;
    const int n4 = vector_len / 4;
    const float4* base4 = reinterpret_cast<const float4*>(base);
    for (int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
      const float4 value = base4[i4];
      const int i = prefix + i4 * 4;
      fn(value.x, i);
      fn(value.y, i + 1);
      fn(value.z, i + 2);
      fn(value.w, i + 3);
    }
    for (int i = prefix + n4 * 4 + threadIdx.x; i < len; i += blockDim.x) {
      fn(base[i - prefix], i);
    }
  } else {
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
      fn(row[static_cast<int64_t>(i) * stride1], i - begin);
    }
  }
}

__device__ __forceinline__ void dense_topk_write_streaming_result(
    int32_t* row_out, const int32_t* selected, int topk, int free_k,
    int prefix_len, int suffix_len, int row_len) {
  for (int out = threadIdx.x; out < topk; out += blockDim.x) {
    if (out < free_k) {
      // Interior indices are relative to the interior start. Convert
      // them back to the row-local contract used by vLLM prefill.
      row_out[out] = selected[out] + prefix_len;
    } else if (out < free_k + prefix_len) {
      row_out[out] = out - free_k;
    } else {
      // Emit local tokens from newest to oldest.
      const int suffix_rank = out - free_k - prefix_len;
      row_out[out] = row_len - 1 - suffix_rank;
    }
  }
}

using DenseTopkFusedScan = cub::BlockScan<int, kDenseTopkThreads>;

struct DenseTopkFusedHistogramPhase {
  int bins[kDenseTopkFineBins];
  DenseTopkFusedScan::TempStorage scan;
};

struct DenseTopkFusedBoundaryPhase {
  uint32_t codes[kDenseTopkBoundaryCap];
  int32_t indices[kDenseTopkBoundaryCap];
};

union DenseTopkFusedPhaseWork {
  DenseTopkFusedHistogramPhase histogram;
  DenseTopkFusedBoundaryPhase boundary;
};

// Single-launch dense top-k. Streaming sink/local positions are appended
// without touching logits; the histogram and exact selector scan only the
// de-duplicated interior for the remaining slots. The coarse histogram and
// selector reuse one CTA and one shared-memory phase slab.
__global__ __launch_bounds__(kDenseTopkThreads) void dense_topk_litetopk_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ row_ends, int32_t* __restrict__ out,
    int stride0, int stride1, int topk, int num_init_tokens,
    int num_local_tokens) {
  __shared__ DenseTopkFusedPhaseWork work;
  __shared__ int32_t selected[kDenseTopkBoundaryCap];
  __shared__ int coarse_threshold;
  __shared__ int coarse_direct;
  __shared__ int prefix_base;
  __shared__ int threshold_found;
  __shared__ int direct_count;
  __shared__ int boundary_count;
  __shared__ int radix_target;
  __shared__ int radix_boundary_count;
  __shared__ uint32_t radix_prefix;
  __shared__ uint32_t radix_mask;
  __shared__ int output_count;
  __shared__ int equal_seen;

  const int row_idx = blockIdx.x;
  const int begin = row_starts[row_idx];
  const int end = row_ends[row_idx];
  const int row_len = end - begin;
  const float* row = logits + static_cast<int64_t>(row_idx) * stride0;
  int32_t* row_out = out + static_cast<int64_t>(row_idx) * topk;

  if (row_len <= topk) {
    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
      row_out[i] = i < row_len ? i : -1;
    }
    return;
  }

  const int prefix_len = min(num_init_tokens, row_len);
  const int suffix_begin = max(prefix_len, row_len - num_local_tokens);
  const int suffix_len = row_len - suffix_begin;
  const int forced_count = prefix_len + suffix_len;
  const int free_k = topk - forced_count;
  const int interior_begin = begin + prefix_len;
  const int interior_end = begin + suffix_begin;

  // Phase 1: histogram only the non-forced interior and locate free_k.
  for (int bin = threadIdx.x; bin < kDenseTopkFineBins; bin += blockDim.x) {
    work.histogram.bins[bin] = 0;
  }
  if (threadIdx.x == 0) {
    coarse_threshold = kDenseTopkFineBins - 1;
    coarse_direct = 0;
    prefix_base = 0;
    threshold_found = 0;
  }
  __syncthreads();

  dense_topk_visit_row(
      row, interior_begin, interior_end, stride1, [&](float value, int) {
        atomicAdd(work.histogram.bins + dense_topk_first_half_bin(value), 1);
      });
  __syncthreads();

#pragma unroll
  for (int round = 0; round < kDenseTopkFineBins / kDenseTopkThreads; ++round) {
    const int bin = round * kDenseTopkThreads + threadIdx.x;
    const int count = work.histogram.bins[bin];
    int exclusive = 0;
    int aggregate = 0;
    DenseTopkFusedScan(work.histogram.scan)
        .ExclusiveSum(count, exclusive, aggregate);
    const int before = prefix_base + exclusive;
    if (before < free_k && before + count >= free_k) {
      if (atomicCAS(&threshold_found, 0, 1) == 0) {
        coarse_threshold = bin;
        coarse_direct = before;
      }
    }
    __syncthreads();
    if (threshold_found) {
      break;
    }
    if (threadIdx.x == 0) {
      prefix_base += aggregate;
    }
    __syncthreads();
  }

  // Phase 2: keep direct winners and only the exact boundary payload.
  if (threadIdx.x == 0) {
    direct_count = 0;
    boundary_count = 0;
  }
  __syncthreads();

  const int threshold = coarse_threshold;
  dense_topk_visit_row(
      row, interior_begin, interior_end, stride1, [&](float value, int index) {
        const int bin = dense_topk_first_half_bin(value);
        if (bin < threshold) {
          const int pos = atomicAdd(&direct_count, 1);
          if (pos < free_k) {
            selected[pos] = index;
          }
        } else if (bin == threshold) {
          const int pos = atomicAdd(&boundary_count, 1);
          if (pos < kDenseTopkBoundaryCap) {
            work.boundary.codes[pos] = dense_topk_descending_float_code(value);
            work.boundary.indices[pos] = index;
          }
        }
      });
  __syncthreads();

  if (boundary_count <= kDenseTopkBoundaryCap) {
    const int base = direct_count;
    for (int i = threadIdx.x; i < boundary_count; i += blockDim.x) {
      const uint32_t key = work.boundary.codes[i];
      int rank = 0;
      for (int j = 0; j < boundary_count; ++j) {
        const uint32_t other = work.boundary.codes[j];
        if (key > other || (key == other && i < j)) {
          ++rank;
        }
      }
      const int dst = base + rank;
      if (dst < free_k) {
        selected[dst] = work.boundary.indices[i];
      }
    }
    __syncthreads();
    dense_topk_write_streaming_result(row_out, selected, topk, free_k,
                                      prefix_len, suffix_len, row_len);
    return;
  }

  // Oversized boundary: refine the selected coarse bin through all exact
  // sortable-FP32 digits without materializing candidates globally.
  if (threadIdx.x == 0) {
    radix_target = free_k - coarse_direct;
    radix_boundary_count = -1;
    radix_prefix = 0;
    radix_mask = 0;
  }
  __syncthreads();

  constexpr int shifts[3] = {21, 10, 0};
  constexpr int digit_masks[3] = {0x7ff, 0x7ff, 0x3ff};
  constexpr int digit_bins[3] = {2048, 2048, 1024};

#pragma unroll
  for (int pass = 0; pass < 3; ++pass) {
    for (int bin = threadIdx.x; bin < kDenseTopkFineBins; bin += blockDim.x) {
      work.histogram.bins[bin] = 0;
    }
    if (threadIdx.x == 0) {
      radix_boundary_count = -1;
    }
    __syncthreads();

    const uint32_t pass_prefix = radix_prefix;
    const uint32_t pass_mask = radix_mask;
    dense_topk_visit_row(
        row, interior_begin, interior_end, stride1, [&](float value, int) {
          if (dense_topk_first_half_bin(value) != threshold) {
            return;
          }
          const uint32_t code = dense_topk_descending_float_code(value);
          if ((code & pass_mask) != pass_prefix) {
            return;
          }
          const int digit =
              static_cast<int>((code >> shifts[pass]) & digit_masks[pass]);
          atomicAdd(work.histogram.bins + digit, 1);
        });
    __syncthreads();

    if (threadIdx.x == 0) {
      int before = 0;
      int chosen = digit_bins[pass] - 1;
      for (int bin = 0; bin < digit_bins[pass]; ++bin) {
        const int count = work.histogram.bins[bin];
        if (before + count >= radix_target) {
          chosen = bin;
          radix_boundary_count = count;
          break;
        }
        before += count;
      }
      radix_target -= before;
      const uint32_t digit_mask = static_cast<uint32_t>(digit_masks[pass])
                                  << shifts[pass];
      radix_prefix |= static_cast<uint32_t>(chosen) << shifts[pass];
      radix_mask |= digit_mask;
    }
    __syncthreads();

    if (radix_boundary_count >= 0 &&
        radix_boundary_count <= kDenseTopkBoundaryCap) {
      if (threadIdx.x == 0) {
        direct_count = coarse_direct;
        boundary_count = 0;
      }
      __syncthreads();

      const uint32_t terminal_prefix = radix_prefix;
      const uint32_t terminal_mask = radix_mask;
      dense_topk_visit_row(
          row, interior_begin, interior_end, stride1,
          [&](float value, int index) {
            if (dense_topk_first_half_bin(value) != threshold) {
              return;
            }
            const uint32_t code = dense_topk_descending_float_code(value);
            const uint32_t masked = code & terminal_mask;
            if (masked < terminal_prefix) {
              const int pos = atomicAdd(&direct_count, 1);
              if (pos < free_k) {
                selected[pos] = index;
              }
            } else if (masked == terminal_prefix) {
              const int pos = atomicAdd(&boundary_count, 1);
              if (pos < kDenseTopkBoundaryCap) {
                work.boundary.codes[pos] = code;
                work.boundary.indices[pos] = index;
              }
            }
          });
      __syncthreads();

      const int base = direct_count;
      for (int i = threadIdx.x; i < boundary_count; i += blockDim.x) {
        const uint32_t key = work.boundary.codes[i];
        int rank = 0;
        for (int j = 0; j < boundary_count; ++j) {
          const uint32_t other = work.boundary.codes[j];
          if (key > other || (key == other && i < j)) {
            ++rank;
          }
        }
        const int dst = base + rank;
        if (dst < free_k) {
          selected[dst] = work.boundary.indices[i];
        }
      }
      __syncthreads();
      dense_topk_write_streaming_result(row_out, selected, topk, free_k,
                                        prefix_len, suffix_len, row_len);
      return;
    }
  }

  // The exact cutoff still contains more than the boundary slab. All
  // remaining values share one code, so any required subset is exact.
  if (threadIdx.x == 0) {
    output_count = coarse_direct;
    equal_seen = 0;
  }
  __syncthreads();

  const uint32_t cutoff = radix_prefix;
  const int tie_take = radix_target;
  dense_topk_visit_row(
      row, interior_begin, interior_end, stride1, [&](float value, int index) {
        if (dense_topk_first_half_bin(value) != threshold) {
          return;
        }
        const uint32_t code = dense_topk_descending_float_code(value);
        if (code < cutoff) {
          const int pos = atomicAdd(&output_count, 1);
          if (pos < free_k) {
            selected[pos] = index;
          }
        } else if (code == cutoff) {
          const int rank = atomicAdd(&equal_seen, 1);
          if (rank < tie_take) {
            const int pos = atomicAdd(&output_count, 1);
            if (pos < free_k) {
              selected[pos] = index;
            }
          }
        }
      });
  __syncthreads();
  dense_topk_write_streaming_result(row_out, selected, topk, free_k, prefix_len,
                                    suffix_len, row_len);
}

inline void dense_topk_check_inputs(const torch::Tensor& logits,
                                    const torch::Tensor& row_starts,
                                    const torch::Tensor& row_ends, int64_t rows,
                                    int64_t stride0, int64_t stride1,
                                    int64_t topk) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat &&
                  logits.dim() == 2,
              "logits must be a 2D CUDA float32 tensor");
  TORCH_CHECK(row_starts.is_cuda() && row_ends.is_cuda() &&
                  row_starts.scalar_type() == torch::kInt &&
                  row_ends.scalar_type() == torch::kInt &&
                  row_starts.is_contiguous() && row_ends.is_contiguous() &&
                  row_starts.dim() == 1 && row_ends.dim() == 1 &&
                  row_starts.numel() >= rows && row_ends.numel() >= rows,
              "row bounds must be contiguous CUDA int32 [>= rows]");
  TORCH_CHECK(row_starts.device() == logits.device() &&
                  row_ends.device() == logits.device(),
              "row bounds must be on the same device as logits");
  TORCH_CHECK(rows > 0 && rows <= logits.size(0) && rows <= INT_MAX,
              "invalid row count");
  TORCH_CHECK(topk > 0 && topk <= kDenseTopkBoundaryCap,
              "topk must be in [1, 2048]");
  TORCH_CHECK(stride0 == logits.stride(0) && stride1 == logits.stride(1) &&
                  stride0 > 0 && stride1 > 0 && stride0 <= INT_MAX &&
                  stride1 <= INT_MAX,
              "explicit positive logits strides must match and fit int32");
}

inline void dense_topk_check_output(const torch::Tensor& logits,
                                    const torch::Tensor& out, int64_t rows,
                                    int64_t topk) {
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kInt &&
          out.is_contiguous() && out.dim() == 2 && out.size(0) >= rows &&
          out.size(1) == topk && out.device() == logits.device(),
      "out must be contiguous CUDA int32 [rows, topk] on logits device");
}

inline void dense_topk_litetopk_(torch::Tensor logits, torch::Tensor row_starts,
                                 torch::Tensor row_ends, torch::Tensor out,
                                 int64_t rows, int64_t stride0, int64_t stride1,
                                 int64_t topk, int64_t num_init_tokens,
                                 int64_t num_local_tokens) {
  dense_topk_check_inputs(logits, row_starts, row_ends, rows, stride0, stride1,
                          topk);
  dense_topk_check_output(logits, out, rows, topk);
  TORCH_CHECK(num_init_tokens >= 0 && num_local_tokens >= 0 &&
                  num_init_tokens + num_local_tokens < topk,
              "streaming token counts must be nonnegative and sum to < topk");
  const c10::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  dense_topk_litetopk_kernel<<<static_cast<int>(rows), kDenseTopkThreads, 0,
                               stream>>>(
      logits.data_ptr<float>(), row_starts.data_ptr<int32_t>(),
      row_ends.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
      static_cast<int>(stride0), static_cast<int>(stride1),
      static_cast<int>(topk), static_cast<int>(num_init_tokens),
      static_cast<int>(num_local_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace litetopk_dense
