// Cooperative HOT planning and in-place gathered-KV swaps.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <limits>
namespace pair_swap_gather {

namespace cg = cooperative_groups;

constexpr int kPlanThreads = 256;
constexpr int kHotSize = 12288;

// Restore the previous epoch's swaps, mark this epoch's HOT set, collect both
// sides of the bijection, and publish the new swaps in one cooperative launch.
// HOT12288 produces exactly 48 resident CTAs on the qualified B200 path.
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


// Each pair joins one prefix hole to one HOT entry outside the prefix. Pairs
// are disjoint, so each 8-thread group can swap both 128-byte keys and their
// FP32 scale bits without atomics, a full-KV copy, or a second paged gather.
__global__ void swap_gathered_kernel(uint8_t* key, float* scale,
                                    const int* a, const int* b,
                                    const int* counts) {
  const int pair = blockIdx.x * 32 + threadIdx.y;
  if (pair >= counts[0]) return;
  const int ia = a[pair], ib = b[pair];
  auto* ka = reinterpret_cast<float4*>(key + static_cast<int64_t>(ia) * 128);
  auto* kb = reinterpret_cast<float4*>(key + static_cast<int64_t>(ib) * 128);
  const float4 va = ka[threadIdx.x], vb = kb[threadIdx.x];
  float sa = 0.0f, sb = 0.0f;
  if (threadIdx.x == 0) { sa = scale[ia]; sb = scale[ib]; }
  ka[threadIdx.x] = vb;
  kb[threadIdx.x] = va;
  if (threadIdx.x == 0) { scale[ia] = sb; scale[ib] = sa; }
}

void validate_key(const torch::Tensor& key, const torch::Tensor& scale,
                  const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& counts) {
  TORCH_CHECK(key.is_cuda() && scale.is_cuda(), "CUDA keys and scales required");
  TORCH_CHECK(key.scalar_type() == torch::kUInt8 &&
              scale.scalar_type() == torch::kFloat32, "FP8 bytes and FP32 scales required");
  TORCH_CHECK(key.dim() == 2 && key.size(1) == 128 &&
              scale.dim() == 1 && scale.numel() == key.size(0), "invalid MISA key/scale shape");
  TORCH_CHECK(key.is_contiguous() && scale.is_contiguous(), "contiguous MISA operands required");
  TORCH_CHECK(key.device() == scale.device() && key.device() == a.device() &&
              key.device() == b.device() && key.device() == counts.device(), "device mismatch");
  TORCH_CHECK(a.scalar_type() == torch::kInt && b.scalar_type() == torch::kInt &&
              counts.scalar_type() == torch::kInt && a.is_contiguous() &&
              b.is_contiguous() && counts.is_contiguous() && a.numel() >= kHotSize &&
              b.numel() >= kHotSize && counts.numel() >= 4, "invalid swap metadata");
}

void swap_gathered_(torch::Tensor key, torch::Tensor scale,
                    const torch::Tensor& a, const torch::Tensor& b,
                    const torch::Tensor& counts) {
  validate_key(key, scale, a, b, counts);
  const c10::cuda::CUDAGuard guard(key.device());
  swap_gathered_kernel<<<(kHotSize + 31) / 32, dim3(8, 32), 0,
                         c10::cuda::getCurrentCUDAStream()>>>(
      key.data_ptr<uint8_t>(), scale.data_ptr<float>(), a.data_ptr<int>(),
      b.data_ptr<int>(), counts.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plan_and_swap_(const torch::Tensor& hot, const torch::Tensor& hot_epoch,
                    const torch::Tensor& permutation, const torch::Tensor& a,
                    const torch::Tensor& b, const torch::Tensor& counts,
                    int64_t window_start, int64_t common_end, int64_t epoch,
                    torch::Tensor key, torch::Tensor scale) {
  validate_plan(hot, hot_epoch, permutation, a, b, counts,
                window_start, common_end, epoch);
  validate_key(key, scale, a, b, counts);
  TORCH_CHECK(key.device() == hot.device() && common_end <= key.size(0) &&
              permutation.numel() >= key.size(0), "invalid common prefix extent");
  const c10::cuda::CUDAGuard guard(key.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_plan(hot, hot_epoch, permutation, a, b, counts,
              static_cast<int>(window_start), static_cast<int>(common_end),
              static_cast<int>(epoch), stream);
  swap_gathered_kernel<<<(kHotSize + 31) / 32, dim3(8, 32), 0, stream>>>(
      key.data_ptr<uint8_t>(), scale.data_ptr<float>(), a.data_ptr<int>(),
      b.data_ptr<int>(), counts.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace pair_swap_gather

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan_and_swap_", &pair_swap_gather::plan_and_swap_);
  m.def("swap_gathered_", &pair_swap_gather::swap_gathered_);
}
