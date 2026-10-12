"""Hook at the DSA ragged-prefill boundary, before backend index conversion."""
from . import PrefillAdapter


class SGLangAdapter(PrefillAdapter):
    def select_prefill(self, q, gathered_k, gathered_scales, effective_weights,
                       starts, ends, out, *, context, topk=2048, transaction=None):
        import torch
        if gathered_k.dtype == torch.uint8:
            gathered_k = gathered_k.view(torch.float8_e4m3fn)
        if gathered_scales.dtype == torch.uint8:
            if not gathered_scales.is_contiguous() or gathered_scales.shape != (len(gathered_k), 4):
                return self.decline('scale_storage')
            gathered_scales = gathered_scales.view(torch.float32)
        if gathered_scales.shape == (len(gathered_k), 1):
            gathered_scales = gathered_scales.squeeze(-1)
        if effective_weights.shape == (*q.shape[:2], 1):
            effective_weights = effective_weights.squeeze(-1)
        # Caller must map logical request indices through its attention backend.
        return self.select_into(q, gathered_k, gathered_scales, effective_weights,
            starts, ends, out, context=context, topk=topk, transaction=transaction)


def transform_indices(metadata, indices):
    """Convert one request's logical indices to the active SGLang KV layout."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    if not envs.SGLANG_DSA_FUSE_TOPK.get() or metadata.force_unfused_topk:
        return  # The attention backend applies the conversion itself.
    if metadata.topk_transform_method == TopkTransformMethod.RAGGED:
        return  # A single request starts at offset zero in the gathered KV buffer.
    if metadata.topk_transform_method != TopkTransformMethod.PAGED:
        raise ValueError('unsupported SGLang top-k transform')
    from sglang.kernels.ops.attention.dsa.transform_index import transform_index_page_table_prefill

    indices.copy_(transform_index_page_table_prefill(
        page_table=metadata.get_page_table_1(), topk_indices=indices,
        extend_lens_cpu=[len(indices)], page_size=1,
        output_num_tokens=len(indices), cu_seqlens_q=metadata.attn_metadata.cu_seqlens_q,
    ))
