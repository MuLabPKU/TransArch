#include "finalize.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("finalize_", [](torch::Tensor cv, torch::Tensor ci, torch::Tensor cc,
      torch::Tensor out, torch::Tensor status, torch::Tensor scratch, int64_t n) {
    h2048_safe_topk_out_litetopk_(cv, ci, cc, out, status, scratch, n, false);
  });
}
