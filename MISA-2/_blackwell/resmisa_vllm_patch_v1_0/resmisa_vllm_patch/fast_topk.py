"""Fixed Torch token TopK and the sparse-attention token ID contract."""
import torch


def exact_topk(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Explicit Torch TopK used by the optimized package variants."""
    if scores.ndim != 2 or topk <= 0:
        raise ValueError("Expected scores [rows, columns] and positive topk")
    return torch.topk(scores, min(topk, scores.shape[1]), dim=-1,
                      sorted=True).indices.to(torch.int32)


def normalize_ids(ids: torch.Tensor) -> torch.Tensor:
    """Sort valid nonnegative IDs ascending, then pad with -1.

    Only order changes: duplicate IDs, if supplied by a broken selector, are not
    silently removed. Selectors must return unique valid IDs.
    """
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.where(ids >= 0, ids, sentinel).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered).to(torch.int32)


def select_window(scores, starts, ends, topk):
    """Return sorted window-relative IDs for half-open [starts, ends) rows.

    The scorer masks other columns to -infinity. Short rows include every valid
    token, independent of how the radix selector breaks masked-score ties.
    """
    absolute = exact_topk(scores, topk).long()
    k = absolute.shape[1]
    starts = starts.reshape(-1, 1)
    ends = ends.reshape(-1, 1)
    sequential = starts + torch.arange(k, device=scores.device)[None, :]
    absolute = torch.where(ends - starts <= k, sequential, absolute)
    relative = torch.where((absolute >= starts) & (absolute < ends), absolute - starts, -1)
    return normalize_ids(relative)
