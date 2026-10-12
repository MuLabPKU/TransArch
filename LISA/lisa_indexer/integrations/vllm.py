"""vLLM adapter at the gathered FP8 prefill boundary."""
from . import PrefillAdapter


def prefill_logits_budget(builder, budget, num_prefills, num_decodes, query_lens, seq_lens):
    """Keep supported strict LISA prefills whole in the upstream indexer.

    The matched vLLM metadata builder uses this number only to split queries.
    LISA bounds its own short-context logits and has no long-context logits.
    Every case that can fall back to DSA retains the framework's memory budget.
    """
    import os

    variant = os.environ.get('LISA_INDEXER')
    if (variant not in ('linear', 'threshold', 'shared4')
            or os.environ.get('LISA_STRICT') != '1'
            or os.environ.get('LISA_SINGLE_REQUEST') != '1'
            or num_prefills != 1 or num_decodes != 0):
        return budget
    cfg = builder.vllm_config
    hf = cfg.model_config.hf_config
    heads = getattr(hf, 'index_n_heads', None)
    if (not cfg.model_config.enforce_eager or cfg.cache_config.enable_prefix_caching
            or builder.dcp_world_size != 1 or builder.pcp_world_size != 1
            or builder.compress_ratio != 1 or builder.use_fp4_indexer_cache
            or cfg.parallel_config.pipeline_parallel_size != 1
            or heads not in ((64,) if variant == 'shared4' else (32, 64))
            or getattr(hf, 'index_head_dim', None) != 128
            or getattr(hf, 'index_topk', None) != 2048):
        return budget
    query_length, sequence_length = int(query_lens[0]), int(seq_lens[0])
    if not 1 < query_length <= 8192 or not query_length <= sequence_length <= 1048576:
        return budget
    return max(budget, query_length * sequence_length * 4)


class VLLMAdapter(PrefillAdapter):
    def select_prefill(self, q, k, k_scale, effective_weights, starts, ends, out, *, context, topk=2048, transaction=None):
        # vLLM gather represents a FP32 scale as four raw uint8 bytes per token.
        import torch
        if k_scale.dtype == torch.uint8:
            if not k_scale.is_contiguous() or k_scale.shape != (len(k), 4):
                return self.decline('scale_storage')
            k_scale = k_scale.view(torch.float32).view(-1)
        elif k_scale.shape == (len(k), 1):
            k_scale = k_scale.squeeze(-1)
        return self.select_into(q, k, k_scale, effective_weights, starts, ends, out,
                                context=context, topk=topk, transaction=transaction)
