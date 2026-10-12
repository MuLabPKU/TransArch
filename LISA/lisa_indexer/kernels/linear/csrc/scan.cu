#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <optional>
#include "scan.cuh"
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

constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_Q = 4;  // 128 q*h rows per UMMA tile / 32 heads
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 1;  // one q-block per CTA
constexpr int NUM_KV_STAGES = 4;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;       // B200
constexpr int ONLINE_FIXED_BUCKETS = 256;

#if defined(LISA_SCAN_RECOVERY)
template <int kImplHeads, int kImplBlockQ, int NUM_KV_STAGES, int SLOTS, int GATE, int REGS, int EMIT, int SLEEP, bool AGG, int FLUSH, int BLOCK_KV, int MATH_THREADS, int MIN_BLOCKS>
#else
template <int kImplHeads, int kImplBlockQ, int NUM_KV_STAGES, int SLOTS, int GATE, int REGS, int EMIT, int SLEEP, bool AGG, int FLUSH, int BLOCK_KV, int MATH_THREADS, int MIN_BLOCKS, int TUNE>
#endif
static int compute_smem_bytes_t(bool include_hist = true) {
  const int esz_fp8 = 1, esz_f32 = 4;
  const int smem_q = kImplBlockQ * kImplHeads * HEAD_DIM * esz_fp8;
  const int smem_w = align_up(kImplBlockQ * 4 * esz_f32, 512);
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
  const int smem_hist = include_hist
                            ? kImplBlockQ * 256 * (int)sizeof(int32_t)
                            : 0;  // per-CTA refresh histogram (NB<=256)
  return NUM_KV_STAGES * sizeof(uint64_t) + NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
         NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks + smem_barriers +
         smem_slots + smem_warpq + smem_hist;
}

#if defined(LISA_SCAN_RECOVERY)
template <int kImplHeads, int kImplBlockQ, int NUM_KV_STAGES, int SLOTS, int GATE, int REGS, int EMIT, int SLEEP, bool AGG, int FLUSH, int BLOCK_KV, int MATH_THREADS, int MIN_BLOCKS>
#else
template <int kImplHeads, int kImplBlockQ, int NUM_KV_STAGES, int SLOTS, int GATE, int REGS, int EMIT, int SLEEP, bool AGG, int FLUSH, int BLOCK_KV, int MATH_THREADS, int MIN_BLOCKS, int TUNE>
#endif
static void mqa_logits_dsa_static_hot_litetopk_impl_t(
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scales,
    torch::Tensor weights, torch::Tensor cu_start, torch::Tensor cu_end,
    torch::Tensor origin, torch::Tensor inv_delta, torch::Tensor th_bucket,
    torch::Tensor cand_val, torch::Tensor cand_idx, torch::Tensor cand_cnt,
    torch::Tensor bcount, int64_t num_buckets64, int64_t topk64, bool no_hist,
    bool exact_gate, std::optional<torch::Tensor> dump_slab = std::nullopt,
    std::optional<torch::Tensor> dump_counts = std::nullopt,
    int64_t dump_max_windows = 0) {
  TORCH_CHECK(!exact_gate || no_hist,
              "exact FP24 gate requires the histogram-free static scan");
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

  TORCH_CHECK(q.dim() == 3, "q must be [Q,2,128]");
  TORCH_CHECK(kv.dim() == 2, "kv must be [S,128]");
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  TORCH_CHECK(seq_len > 0 && seq_len_kv > 0, "Q and S must be nonzero");
  TORCH_CHECK(q.size(1) == kImplHeads && q.size(2) == HEAD_DIM &&
                  kv.size(1) == HEAD_DIM,
              "FP8x2 scan requires q [Q,2,128] and kv [S,128]");
  TORCH_CHECK(seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
              "packed candidates support at most 1M KV positions");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == seq_len &&
                  weights.size(1) == 4,
              "weights must be [Q,4]");
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
  const int num_buckets = static_cast<int>(num_buckets64);
  const int topk = static_cast<int>(topk64);
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
#if defined(LISA_SCAN_STATIC)
              seq_len_kv, HEAD_DIM, (BLOCK_KV > 256 ? 256 : BLOCK_KV), HEAD_DIM, HEAD_DIM);
#else
              seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
#endif
  auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
#if defined(LISA_SCAN_STATIC)
                       esz_f32, ks_aligned, 1, (BLOCK_KV > 256 ? 256 : BLOCK_KV), 1, 0, 0);
#else
                       esz_f32, ks_aligned, 1, BLOCK_KV, 1, 0, 0);
#endif
  auto tm_w =
      make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
              4, seq_len, 4, kImplBlockQ, 4, 0);

  const bool raw_dump = dump_slab.has_value();
  const int num_q_blocks = (seq_len + kImplBlockQ - 1) / kImplBlockQ;
  if (raw_dump) {
    TORCH_CHECK(no_hist && !exact_gate,
                "raw-dump emit requires the plain no-hist writer");
    TORCH_CHECK(dump_counts.has_value(),
                "raw-dump emit needs both the region slab and the counts slab");
    const auto& slab = *dump_slab;
    const auto& counts = *dump_counts;
    TORCH_CHECK(
        slab.is_cuda() && counts.is_cuda() && slab.device() == q.device() &&
            counts.device() == q.device() && slab.is_contiguous() &&
            counts.is_contiguous() && slab.scalar_type() == torch::kInt &&
            counts.scalar_type() == torch::kInt,
        "dump slabs must be contiguous int32 CUDA tensors");
    const int64_t region_words =
        static_cast<int64_t>(kImplBlockQ) * SLOTS * 32;
    const int64_t required_windows =
        (((static_cast<int64_t>(seq_len_kv) + BLOCK_KV - 1) / BLOCK_KV) +
         dsa_litetopk::kEmitChunkBlocks - 1) /
        dsa_litetopk::kEmitChunkBlocks;
    TORCH_CHECK(dump_max_windows >= required_windows,
                "dump_max_windows must cover the full KV range");
    const int64_t regions =
        static_cast<int64_t>(num_q_blocks) * 8 * dump_max_windows;
    TORCH_CHECK(slab.numel() >= regions * region_words &&
                    counts.numel() >= regions * 32,
                "dump slabs are smaller than the deterministic region layout");
  }
  // seq words + shadow words + pending/stale/last rows + per-lane
  // progress cursors, appended after the re-enabled histogram region.
  constexpr int kRingScratchBytes =
      2 * (MATH_THREADS / 32) * kImplBlockQ * static_cast<int>(sizeof(uint32_t)) +
      3 * kImplBlockQ * static_cast<int>(sizeof(int32_t)) +
      (MATH_THREADS / 32) * kImplBlockQ * 32;
  // The FP8 bucket-gate scan uses a warm-started ring daemon.
  // Exact-gate and histogram modes use separate non-ring specializations.
#if defined(LISA_SCAN_RECOVERY)
  const int smem = compute_smem_bytes_t<kImplHeads, kImplBlockQ, NUM_KV_STAGES, SLOTS, GATE, REGS, EMIT, SLEEP, AGG, FLUSH, BLOCK_KV, MATH_THREADS, MIN_BLOCKS>(!exact_gate) +
#else
  const int smem = compute_smem_bytes_t<kImplHeads, kImplBlockQ, NUM_KV_STAGES, SLOTS, GATE, REGS, EMIT, SLEEP, AGG, FLUSH, BLOCK_KV, MATH_THREADS, MIN_BLOCKS, TUNE>(!exact_gate) +
#endif
                   (no_hist && !exact_gate ? kRingScratchBytes : 0);
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
      kImplHeads, HEAD_DIM, kImplBlockQ, BLOCK_KV, NUM_Q_STAGES,
      NUM_KV_STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS,
#if defined(LISA_SCAN_RECOVERY)
      MATH_THREADS / 128, false, true, true, false, false, false, true, false, SLOTS, GATE, REGS, EMIT, SLEEP, AGG, FLUSH, MIN_BLOCKS>;
#elif defined(LISA_SCAN_STATIC)
      MATH_THREADS / 128, false, true, true, true, false, false, false, false, SLOTS, GATE, REGS, EMIT, SLEEP, AGG, FLUSH, MIN_BLOCKS, TUNE>;
#else
      MATH_THREADS / 128, false, true, true, false, false, false, true, false, SLOTS, GATE, REGS, EMIT, SLEEP, AGG, FLUSH, MIN_BLOCKS, TUNE>;
#endif
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(reinterpret_cast<void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      static_cast<uint32_t>(seq_len), static_cast<uint32_t>(seq_len_kv),
      reinterpret_cast<uint32_t*>(cu_start.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(cu_end.data_ptr<int>()),
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th_bucket.data_ptr<int32_t>(),
      // The ring always warm-starts from the seed histogram. This is sound
      // because prefix and suffix contribute once each; non-ring
      // specializations keep their existing bcount contract.
      bcount.data_ptr<int32_t>(), static_cast<uint32_t>(num_buckets),
      static_cast<uint32_t>(topk), 1u, 1u, 0u, 0ULL, 0u,
      candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
      cand_cnt.data_ptr<int32_t>(), static_cast<uint32_t>(cand_cap), tm_q,
      tm_kv, tm_ks, tm_w, tm_q, nullptr, nullptr, nullptr, nullptr, nullptr, 0u,
      nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u, nullptr, nullptr,
      raw_dump ? reinterpret_cast<uint32_t*>(dump_slab->data_ptr<int>())
               : nullptr,
      raw_dump ? reinterpret_cast<uint32_t*>(dump_counts->data_ptr<int>())
               : nullptr,
      static_cast<uint32_t>(raw_dump ? dump_max_windows : 0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void scan(torch::Tensor q, torch::Tensor kv, torch::Tensor ks,
          torch::Tensor weights, torch::Tensor starts, torch::Tensor ends,
          torch::Tensor origin, torch::Tensor inv, torch::Tensor th,
          torch::Tensor cv, torch::Tensor ci, torch::Tensor cc, torch::Tensor hist, int config) {
  switch(config) {
#if defined(LISA_SCAN_RECOVERY)
    case 29: mqa_logits_dsa_static_hot_litetopk_impl_t<2,8,2,7,8,128,64,2048,false,1,128,128,3>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,false); break;
    case 30: mqa_logits_dsa_static_hot_litetopk_impl_t<2,4,2,8,8,96,32,2048,false,1,128,128,4>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,false); break;
    case 31: mqa_logits_dsa_static_hot_litetopk_impl_t<2,4,2,8,8,96,64,2048,false,1,128,128,4>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,false); break;
#elif defined(LISA_SCAN_SYNC)
    case 3: mqa_logits_dsa_static_hot_litetopk_impl_t<2,4,2,8,8,96,32,2048,false,1,128,128,4,15>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,false); break;
#elif defined(LISA_SCAN_STATIC)
    case 8: mqa_logits_dsa_static_hot_litetopk_impl_t<2,8,2,4,8,112,32,1024,false,0,512,512,1,15>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,true); break;
    case 9: mqa_logits_dsa_static_hot_litetopk_impl_t<2,8,2,4,8,112,64,1024,false,0,512,512,1,15>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,true); break;
#elif defined(LISA_SCAN_FINAL)
    case 4: mqa_logits_dsa_static_hot_litetopk_impl_t<2,8,2,4,8,96,64,1024,false,1,256,256,2,31>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,false); break;
#else
    case 8: mqa_logits_dsa_static_hot_litetopk_impl_t<2,4,2,8,8,96,64,2048,false,1,128,128,4,7>(q,kv,ks,weights,starts,ends,origin,inv,th,cv,ci,cc,hist,256,2048,true,false); break;
#endif
    default: TORCH_CHECK(false, "unsupported LISA scan configuration");
  }
}

#if defined(LISA_SCAN_FINAL)
#include "finalize.cuh"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan", &scan);
#if defined(LISA_SCAN_FINAL)
  m.def("finalize_", [](torch::Tensor cv, torch::Tensor ci, torch::Tensor cc,
      torch::Tensor out, torch::Tensor status, torch::Tensor scratch,
      int64_t n, torch::Tensor threshold) {
    h2048_safe_topk_out_litetopk_(cv, ci, cc, out, status, scratch, n, false, threshold);
  });
#endif
}
