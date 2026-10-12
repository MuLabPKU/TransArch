#include "topm.cuh"

namespace {

template <bool CacheScores>
__global__ __launch_bounds__(ThreadsPerBlock) void counted_finish_kernel(
    const float* scores, const int32_t* candidates, const int32_t* counts,
    const int32_t* status, const int32_t* starts, int32_t* output,
    int64_t capacity, int k) {
  const int64_t row = blockIdx.x;
  const int length = counts[row];
  int32_t* result = output + row * k;
  if (status[row] != 0 || length < 0 || length > capacity) {
    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) result[i] = -1;
    return;
  }
  const float* values = scores + row * capacity;
  if (length <= k) {
    naive_topk_cuda(values, result, length, k);
  } else {
    // Run radix selection with a separate valid length for each row.
    fast_topk_cuda<CacheScores>(values, result, length, k);
  }
  __syncthreads();
  const int32_t start = starts[row];
  for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
    const int slot = result[i];
    const int32_t id = (slot >= 0 && slot < length) ? candidates[row * capacity + slot] : -1;
    result[i] = id >= 0 ? id - start : -1;
  }
}

void ensure_counted_smem(int device) {
  static std::mutex mutex;
  static std::unordered_set<int> configured;
  const std::lock_guard<std::mutex> lock(mutex);
  if (configured.count(device) == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(counted_finish_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicSmem)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(counted_finish_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicSmem)));
    configured.insert(device);
  }
}

void run(const at::Tensor& scores, const at::Tensor& candidates,
         const at::Tensor& counts, const at::Tensor& status,
         const at::Tensor& starts, at::Tensor& output, bool cached) {
  TORCH_CHECK(scores.is_cuda() && scores.dim() == 2 && scores.is_contiguous() &&
              scores.scalar_type() == at::kFloat, "scores must be contiguous CUDA FP32 [Q,cap]");
  const c10::cuda::CUDAGuard guard(scores.device());
  for (const auto* tensor : {&candidates, &counts, &status, &starts,
                            static_cast<const at::Tensor*>(&output)}) {
    TORCH_CHECK(tensor->device() == scores.device() && tensor->is_contiguous() &&
                tensor->scalar_type() == at::kInt, "integer inputs must be contiguous on score device");
  }
  const auto rows = scores.size(0), capacity = scores.size(1);
  TORCH_CHECK(candidates.sizes() == scores.sizes(), "candidate/score shape mismatch");
  for (const auto* tensor : {&counts, &status, &starts}) {
    TORCH_CHECK(tensor->dim() == 1 && tensor->size(0) == rows, "expected int32 [Q]");
  }
  TORCH_CHECK(output.dim() == 2 && output.size(0) == rows, "expected output [Q,K]");
  const auto k = output.size(1);
  TORCH_CHECK(k > 0 && k <= MaxTopK && k <= capacity, "invalid final K");
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX && capacity <= INT32_MAX - ThreadsPerBlock,
              "dimensions exceed supported range");
  ensure_counted_smem(scores.get_device());
  const auto kernel = cached ? counted_finish_kernel<true> : counted_finish_kernel<false>;
  kernel<<<static_cast<uint32_t>(rows), ThreadsPerBlock, DynamicSmem,
                         at::cuda::getCurrentCUDAStream().stream()>>>(
      scores.data_ptr<float>(), candidates.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(), starts.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
      capacity, static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

TORCH_LIBRARY(relu_hisa_counted_finish, m) {
  m.def("run(Tensor scores, Tensor candidates, Tensor counts, Tensor status, Tensor starts, Tensor(a!) output, bool cached=False) -> ()");
}
TORCH_LIBRARY_IMPL(relu_hisa_counted_finish, CUDA, m) {
  m.impl("run", &run);
}
