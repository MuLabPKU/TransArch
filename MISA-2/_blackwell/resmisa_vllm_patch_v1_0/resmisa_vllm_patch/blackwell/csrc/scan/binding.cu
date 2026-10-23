#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include "misa_native_v19.cuh"
using CandidateValue = dsa_litetopk::CandidateValue;
static CandidateValue* candidate_data_ptr(torch::Tensor& t) {
  return reinterpret_cast<CandidateValue*>(t.data_ptr<at::Half>());
}
static void check_candidate_dtype(const torch::Tensor& t) {
  TORCH_CHECK(t.scalar_type()==torch::kHalf, "opaque U16 candidate storage");
}
static void* driver_handle() {
  static void* h = nullptr;
  if (!h) {
    h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    TORCH_CHECK(h, "failed to load libcuda.so.1");
  }
  return h;
}

static CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt,
                          cuuint32_t rank, void* addr, const cuuint64_t* dims,
                          const cuuint64_t* strides, const cuuint32_t* box,
                          const cuuint32_t* estrides, CUtensorMapInterleave il,
                          CUtensorMapSwizzle sw, CUtensorMapL2promotion l2,
                          CUtensorMapFloatOOBfill oob) {
  using FT =
      CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static FT f = nullptr;
  if (!f) {
    f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
    TORCH_CHECK(f, "failed to load cuTensorMapEncodeTiled");
  }
  return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

static CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer, int smem_inner,
                           int smem_outer, long gmem_outer_stride,
                           int swizzle_mode) {
  if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
  CUtensorMap tm;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                               : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : swizzle_mode == 32
                                   ? CU_TENSOR_MAP_SWIZZLE_32B
                                   : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int HEAD_DIM = 128;
constexpr int NUM_Q_STAGES = 1;  // one q-block per CTA
constexpr int SPEC_THREADS = 128;

template <int kImplHeads, int kImplBlockQ, int NUM_KV_STAGES, int SLOTS, int BLOCK_KV, int MATH_THREADS>
static int compute_smem_bytes_t() {
  const int esz_fp8 = 1, esz_f32 = 4;
  const int smem_q = kImplBlockQ * kImplHeads * HEAD_DIM * esz_fp8;
  const int smem_w = align_up(kImplBlockQ * kImplHeads * esz_f32, 512);
  const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
  const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
  const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 +
                           (MATH_THREADS / 128) * dsa_litetopk::kUmmaStages * 2;
  const int smem_barriers = num_barriers * 8;
  const int smem_slots =
      4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
  constexpr int emit_record_bytes = (int)sizeof(uint32_t);
  const int smem_warpq = (MATH_THREADS / 32) * kImplBlockQ *
                         ((int)sizeof(int32_t) + SLOTS *
                                                     32 * emit_record_bytes);
  const int smem_hist = kImplBlockQ * 256 * (int)sizeof(int32_t);  // per-CTA refresh histogram (NB<=256)
  return NUM_KV_STAGES * sizeof(uint64_t) + NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
         NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks + smem_barriers +
         smem_slots + smem_warpq + smem_hist;
}

template <int kImplHeads, int kImplBlockQ, int NUM_KV_STAGES, int SLOTS, int REGS, int EMIT, int BLOCK_KV, int MATH_THREADS, int MIN_BLOCKS>
static void mqa_logits_dsa_static_hot_litetopk_impl_t(
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scales,
    torch::Tensor weights, torch::Tensor cu_start, torch::Tensor cu_end,
    torch::Tensor origin, torch::Tensor inv_delta, torch::Tensor th_bucket,
    torch::Tensor cand_val, torch::Tensor cand_idx, torch::Tensor cand_cnt,
    torch::Tensor bcount, int kv_splits) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
                  weights.is_cuda() && cu_start.is_cuda() && cu_end.is_cuda() &&
                  origin.is_cuda() && inv_delta.is_cuda() &&
                  th_bucket.is_cuda() && cand_val.is_cuda() &&
                  cand_idx.is_cuda() && cand_cnt.is_cuda() && bcount.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(
      q.device() == kv.device() && q.device() == kv_scales.device() &&
          q.device() == weights.device() && q.device() == cu_start.device() &&
          q.device() == cu_end.device() && q.device() == origin.device() &&
          q.device() == inv_delta.device() &&
          q.device() == th_bucket.device() && q.device() == cand_val.device() &&
          q.device() == cand_idx.device() && q.device() == cand_cnt.device() &&
          q.device() == bcount.device(),
      "all tensors must be on the same CUDA device");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() &&
                  kv_scales.is_contiguous() && weights.is_contiguous() &&
                  cu_start.is_contiguous() && cu_end.is_contiguous() &&
                  origin.is_contiguous() && inv_delta.is_contiguous() &&
                  th_bucket.is_contiguous() && cand_val.is_contiguous() &&
                  cand_idx.is_contiguous() && cand_cnt.is_contiguous() &&
                  bcount.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn &&
                  kv.scalar_type() == torch::kFloat8_e4m3fn,
              "q/kv must be fp8_e4m3fn");
  TORCH_CHECK(kv_scales.scalar_type() == torch::kFloat &&
                  weights.scalar_type() == torch::kFloat &&
                  origin.scalar_type() == torch::kFloat &&
                  inv_delta.scalar_type() == torch::kFloat,
              "kv_scales/weights/origin/inv_delta must be fp32");
  TORCH_CHECK(
      cu_start.scalar_type() == torch::kInt &&
          cu_end.scalar_type() == torch::kInt &&
          th_bucket.scalar_type() == torch::kInt &&
          cand_idx.scalar_type() == torch::kInt &&
          cand_cnt.scalar_type() == torch::kInt &&
          bcount.scalar_type() == torch::kInt,
      "range, threshold, candidate index/count and metadata must be int32");
  check_candidate_dtype(cand_val);

  TORCH_CHECK(q.dim() == 3, "q must be [Q,32,128]");
  TORCH_CHECK(kv.dim() == 2, "kv must be [S,128]");
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  TORCH_CHECK(seq_len > 0 && seq_len_kv > 0, "Q and S must be nonzero");
  TORCH_CHECK(q.size(1) == kImplHeads && q.size(2) == HEAD_DIM &&
                  kv.size(1) == HEAD_DIM,
              "static HOT path requires H in {32,64}, D=128");
  TORCH_CHECK(seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
              "packed candidates support at most 1M KV positions");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == seq_len &&
                  weights.size(1) == kImplHeads,
              "weights must be [Q,32]");
  TORCH_CHECK(cu_start.dim() == 1 && cu_start.numel() == seq_len &&
                  cu_end.dim() == 1 && cu_end.numel() == seq_len,
              "cu_start/cu_end must have Q elements");
  TORCH_CHECK(origin.dim() == 1 && origin.numel() == seq_len &&
                  inv_delta.dim() == 1 && inv_delta.numel() == seq_len &&
                  th_bucket.dim() == 1 && th_bucket.numel() == seq_len,
              "origin/inv_delta/th_bucket must have Q elements");
  TORCH_CHECK(cand_val.dim() == 2 && cand_val.size(0) == seq_len &&
                  cand_idx.sizes() == cand_val.sizes(),
              "cand_val/cand_idx must be [Q,cand_cap]");
  const int cand_cap = static_cast<int>(cand_val.size(1));
  constexpr int num_buckets = 256;
  constexpr int topk = 2048;
  TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
              "static HOT path requires 3 <= num_buckets <= 256");
  TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1,cand_cap]");
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == seq_len,
              "cand_cnt must have Q elements");
  TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) == seq_len &&
                  bcount.size(1) == num_buckets,
              "bcount must be [Q,num_buckets]");

  c10::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int esz_fp8 = 1;
  const int esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  TORCH_CHECK(kv_scales.numel() >= ks_aligned,
              "kv_scales storage is shorter than the aligned KV length");
  auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                      HEAD_DIM, seq_len * kImplHeads, HEAD_DIM,
                      kImplBlockQ * kImplHeads, HEAD_DIM, HEAD_DIM);
  auto tm_kv =
      make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM,
              seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
  auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       esz_f32, ks_aligned, 1, BLOCK_KV, 1, 0, 0);
  auto tm_w =
      make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
              kImplHeads, seq_len, kImplHeads, kImplBlockQ, kImplHeads, 0);

  const int num_q_blocks = (seq_len + kImplBlockQ - 1) / kImplBlockQ;
  
  // Preserve the established launch allocation and shared-memory layout.
  // The trailing legacy scratch is reserved but no longer addressed.
  constexpr int kRingScratchBytes =
      2 * (MATH_THREADS / 32) * kImplBlockQ * static_cast<int>(sizeof(uint32_t)) +
      3 * kImplBlockQ * static_cast<int>(sizeof(int32_t)) +
      (MATH_THREADS / 32) * kImplBlockQ * 32;
  // Fixed bucket-gate/no-hist ring scan.
  const int smem = compute_smem_bytes_t<kImplHeads, kImplBlockQ, NUM_KV_STAGES, SLOTS, BLOCK_KV, MATH_THREADS>() +
                   kRingScratchBytes;
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
      kImplHeads, HEAD_DIM, kImplBlockQ, BLOCK_KV, NUM_Q_STAGES,
      NUM_KV_STAGES, SPEC_THREADS, MATH_THREADS,
      MATH_THREADS / 128, SLOTS, REGS, EMIT, MIN_BLOCKS>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(reinterpret_cast<void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(static_cast<unsigned>(num_q_blocks), static_cast<unsigned>(kv_splits), 1u);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      static_cast<uint32_t>(seq_len), static_cast<uint32_t>(seq_len_kv),
      reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th_bucket.data_ptr<int32_t>(),
      // The ring always warm-starts from the seed histogram. This is sound
      // under the production exact-once prefix/suffix split.
      bcount.data_ptr<int32_t>(), static_cast<uint32_t>(num_buckets),
      static_cast<uint32_t>(topk), static_cast<uint32_t>(kv_splits),
      candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
      cand_cnt.data_ptr<int32_t>(), static_cast<uint32_t>(cand_cap), tm_q,
      tm_kv, tm_ks, tm_w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void scan(torch::Tensor q, torch::Tensor kv, torch::Tensor ks,
          torch::Tensor weights, torch::Tensor starts, torch::Tensor ends,
          torch::Tensor origin, torch::Tensor inv, torch::Tensor th,
          torch::Tensor cv, torch::Tensor ci, torch::Tensor cc, torch::Tensor hist,
          bool hot_valid) {
  const int splits = q.size(0) >= 4096 ? 1 : 2;
  if (hot_valid && q.size(0) == 8192 && kv.size(0) >= 131072 && kv.size(0) <= 135168) {
    mqa_logits_dsa_static_hot_litetopk_impl_t<8,4,2,6,96,64,128,128,4>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,splits);
  } else {
    mqa_logits_dsa_static_hot_litetopk_impl_t<8,4,2,6,96,32,128,128,4>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,splits);
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("scan",&scan);}
