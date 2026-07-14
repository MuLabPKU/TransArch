"""OpenDSA distributed — context/expert parallel process groups + collectives."""
from .pg import (
    init_parallel, is_dist,
    cp_size, cp_rank, cp_group, ep_size, ep_rank, ep_group,
    local_shard, zigzag_local_gpos,
    all_gather_seq, AllGatherSeq, all_to_all,
    reduce_scatter_tensor, all_gather_into_tensor,
)

__all__ = [
    "init_parallel", "is_dist",
    "cp_size", "cp_rank", "cp_group", "ep_size", "ep_rank", "ep_group",
    "local_shard", "zigzag_local_gpos",
    "all_gather_seq", "AllGatherSeq", "all_to_all",
    "reduce_scatter_tensor", "all_gather_into_tensor",
]
