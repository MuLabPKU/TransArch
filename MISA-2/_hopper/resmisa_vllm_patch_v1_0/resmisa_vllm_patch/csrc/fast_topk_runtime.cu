/**
 * fast_topk_runtime.cu — runtime-topk version of sgl-kernel's fast_topk.
 *
 * Source algorithm: 8-bit radix-select copied verbatim from
 *   sglang/sgl-kernel/csrc/elementwise/topk.cu (which is in turn adapted
 *   from tilelang's deepseek_v32/topk_selector.py).
 * Difference from upstream: ``topk`` is a runtime arg (not a compile-time
 * constexpr). Constraint: 0 < topk <= MaxTopK (= 8192).
 * Dynamic SMEM = 96KB (2 rounds × 12288 ints/round), requires
 * cudaFuncSetAttribute opt-in (called once at first launch, before graph capture).
 *
 * Loaded JIT via torch.utils.cpp_extension.load() — see fast_topk_runtime.py.
 * Registers torch op ``resmisa_topk_v1::fast_topk_runtime``.
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr int MaxTopK = 8192;
constexpr int kThreadsPerBlock = 1024;
// 96KB dynamic SMEM for large topk support.
// H100/H200 support up to 228KB dynamic SMEM per block.
// cudaFuncSetAttribute is called once at module load time (before any
// CUDA graph capture), so it's safe.
// SMEM_INPUT_SIZE = 96KB / (2 * 4B) = 12288 ints/round.
constexpr size_t kSmem = 24 * 1024 * sizeof(uint32_t);

struct FastTopKParams {
  const float *__restrict__ input;
  int32_t *__restrict__ indices;
  int64_t input_stride;
  int32_t length; // same for every row — caller masks invalid positions to -inf
};

// length <= topk → all valid rows are top; pad with -1 to topk.
__device__ void naive_topk_cuda(int32_t *__restrict__ indice, int32_t length,
                                int32_t topk) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < topk; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_cuda_tl(const float *__restrict__ input,
                                  int *__restrict__ index, int row_start,
                                  int length, int topk_target) {
  // 8-bit radix-select. Assumes length > topk_target (caller guarantees).
  int topk = topk_target;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto &s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8-bit coarse histogram
  if (tx < RADIX + 1)
    s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  // A coarse FP16 bucket may contain arbitrarily many FP32 values. The
  // shared candidate list is only an optimization, never a correctness cap.
  // Detect overflow BEFORE collecting/truncating candidates. Subsequent
  // refinements of a fitting bucket cannot exceed its initial population.
  const int coarse_bucket_count =
      s_histogram[threshold_bin] - s_histogram[threshold_bin + 1];
  if (topk > 0 && coarse_bucket_count > int(SMEM_INPUT_SIZE)) {
    // Preserve the normal fast path below. On overflow, scan the original
    // row at each radix pass, filtering by the fixed coarse bucket and the
    // already chosen FP32-key prefix. No candidate arrays or global scratch
    // are needed; all control flow is uniform within this CTA.
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin =
          static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        if (pos < topk_target)
          index[pos] = idx;
      }
    }
    __syncthreads();

    uint32_t prefix = 0;
    uint32_t prefix_mask = 0;
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
      const int offset = 24 - round * 8;
      if (tx < RADIX + 1)
        s_histogram[tx] = 0;
      __syncthreads();

      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const float value = input[idx + row_start];
        const uint32_t key = convert_to_uint32(value);
        if (static_cast<int>(convert_to_uint8(value)) == threshold_bin &&
            (key & prefix_mask) == prefix) {
          ::atomicAdd(&s_histogram[(key >> offset) & 0xFFu], 1);
        }
      }
      __syncthreads();
      run_cumsum();
      // Invariant: the active prefix contains more than `topk` values.
      // Thus this crossing exists. The chosen bucket plus its strictly
      // higher buckets contains every still-required winner.
      if (tx < RADIX && s_histogram[tx] > topk &&
          s_histogram[tx + 1] <= topk)
        s_threshold_bin_id = tx;
      __syncthreads();
      const int cutoff = s_threshold_bin_id;
      topk -= s_histogram[cutoff + 1];

      // Emit all strictly higher keys before any final equal-key ties.
      // This is a separate pass and barrier: ties cannot steal slots from
      // strict winners, irrespective of atomic arrival order.
      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const float value = input[idx + row_start];
        const uint32_t key = convert_to_uint32(value);
        if (static_cast<int>(convert_to_uint8(value)) == threshold_bin &&
            (key & prefix_mask) == prefix &&
            static_cast<int>((key >> offset) & 0xFFu) > cutoff) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos < topk_target)
            index[pos] = idx;
        }
      }
      __syncthreads();
      if (topk == 0)
        return;

      prefix |= static_cast<uint32_t>(cutoff) << offset;
      prefix_mask |= 0xFFu << offset;
      if (round == 3) {
        // All 32 key bits now match: any `topk` remaining IDs are valid
        // equal-score ties. Each input ID participates once, so no duplicate
        // IDs are emitted. Strict winners occupy an already completed prefix
        // of the output. Atomic order affects only equal-key tie membership.
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
          const float value = input[idx + row_start];
          if (static_cast<int>(convert_to_uint8(value)) == threshold_bin &&
              convert_to_uint32(value) == prefix) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            if (pos < topk_target)
              index[pos] = idx;
          }
        }
        __syncthreads();
        return;
      }
    }
  }

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin =
          static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        if (pos < topk_target)
          index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        if (pos < topk_target)
          index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: 4 rounds of 8-bit radix refinement
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                               ? _raw_num_input
                               : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin =
            (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos < topk_target)
            index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos < topk_target)
            index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0 && (topk_target - pos) >= 0 && (topk_target - pos) < topk_target) {
              index[topk_target - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void topk_kernel_runtime(
    const FastTopKParams params, int32_t topk) {
  const auto &[input, indices, input_stride, length] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto indice = indices + bid * topk;
  const auto score = input + bid * input_stride;
  if (length <= topk) {
    return naive_topk_cuda(indice, length, topk);
  } else {
    return fast_topk_cuda_tl(score, indice, /*row_start=*/0, length, topk);
  }
}

// kSmem (96KB) exceeds the default 48KB per-block dynamic SMEM limit.
// cudaFuncSetAttribute is called once at first kernel launch to opt-in.
// This happens during model warmup, BEFORE any CUDA graph capture begins.

} // namespace

static void ensure_smem_configured() {
  static bool configured = false;
  if (!configured) {
    auto err = cudaFuncSetAttribute(
        topk_kernel_runtime,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmem));
    TORCH_CHECK(err == cudaSuccess, "FastTopK shared-memory opt-in failed: ",
                cudaGetErrorString(err));
    configured = true;
  }
}

void fast_topk_runtime_interface(const at::Tensor &score, at::Tensor &indices) {
  TORCH_CHECK(score.is_cuda(), "score must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(score.device() == indices.device(), "score and indices must share a device");
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1,
              "score [B, L] f32, last-dim contiguous");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be f32");
  TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous(),
              "indices [B, topk] i32 contiguous");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be i32");

  const int64_t B = score.size(0);
  const int64_t L = score.size(1);
  const int64_t topk = indices.size(1);
  TORCH_CHECK(indices.size(0) == B, "indices.size(0) must match score.size(0)");
  TORCH_CHECK(topk > 0 && topk <= MaxTopK, "topk must be in (0, ", MaxTopK,
              "], got ", topk);
  TORCH_CHECK(L >= topk, "score.size(1)=", L, " must be >= topk=", topk);

  FastTopKParams params{
      .input = score.data_ptr<float>(),
      .indices = indices.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
      .length = static_cast<int32_t>(L),
  };

  ensure_smem_configured();

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  topk_kernel_runtime<<<grid, block, kSmem, stream>>>(
      params, static_cast<int32_t>(topk));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "topk_kernel_runtime launch failed");
}

TORCH_LIBRARY(resmisa_topk_v1, m) {
  m.def("fast_topk_runtime(Tensor score, Tensor(a!) indices) -> ()");
}

TORCH_LIBRARY_IMPL(resmisa_topk_v1, CUDA, m) {
  m.impl("fast_topk_runtime", fast_topk_runtime_interface);
}
