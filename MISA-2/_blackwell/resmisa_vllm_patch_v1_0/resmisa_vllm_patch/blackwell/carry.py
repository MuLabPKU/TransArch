"""Per-layer HOT vote buffers and event-safe side-stream publication.

The selected subset is the September HOT12288 contract. Every completed
noninitial chunk publishes; no optional carry skipping, timing or DSA route.
"""

import torch

from .extensions import extension

HOT_PREFIX = 12288
CARRY_RECENT_ROWS = 1536
_HOT_CARRY = {}
_HOT_STREAM = {}
_CARRY_VOTE_BUFS = {}
_CARRY_TOPK_WORKSPACE = {}


def _carry_vote_hist(nv, dev, hot_key, waited_event=None):
    key = (str(dev), hot_key)
    entry = _CARRY_VOTE_BUFS.get(key)
    if entry is not None and entry["free_event"] is not None:
        event = entry["free_event"]
        if event is not waited_event:
            torch.cuda.current_stream(dev).wait_event(event)
    if entry is None or entry["buf"].numel() < nv:
        capacity = max(1024, 1 << (nv - 1).bit_length())
        hot = None if entry is None else entry.get("hot")
        ready = None if entry is None else entry.get("ready_event")
        entry = dict(
            buf=torch.zeros(capacity, dtype=torch.int32, device=dev),
            free_event=None,
            ready_event=torch.cuda.Event() if ready is None else ready,
            hot=torch.empty(HOT_PREFIX, dtype=torch.int64, device=dev) if hot is None else hot,
            needs_reset=False, dirty_extent=0,
        )
        _CARRY_VOTE_BUFS[key] = entry
    if entry["needs_reset"]:
        entry["buf"][:max(nv, entry["dirty_extent"])].zero_()
        entry["needs_reset"] = False
        entry["dirty_extent"] = 0
    entry["needs_reset"] = True
    entry["dirty_extent"] = max(entry["dirty_extent"], nv)
    return entry["buf"][:nv]


def _carry_topk_workspace(max_vote, dev):
    key = str(dev)
    bins = max_vote + 1
    entry = _CARRY_TOPK_WORKSPACE.get(key)
    if entry is None or entry["partial"].shape[1] < bins:
        entry = dict(
            partial=torch.empty((128, bins), dtype=torch.int16, device=dev),
            state=torch.zeros(136, dtype=torch.int32, device=dev),
        )
        _CARRY_TOPK_WORKSPACE[key] = entry
    return entry


def _publish_carry(hot_key, votes, nv, min_index, max_vote):
    if not (HOT_PREFIX <= nv - min_index and nv <= 135168 and 0 < max_vote <= 8192):
        raise ValueError("HOT carry publication outside its supported contract")
    dev = votes.device
    key = (str(dev), hot_key)
    entry = _CARRY_VOTE_BUFS[key]
    side = _HOT_STREAM.get(str(dev))
    if side is None:
        side = torch.cuda.Stream(device=dev)
        _HOT_STREAM[str(dev)] = side
    ext = extension("carry")
    # A missing custom operation is a build error, never an unreported Torch
    # selector substitution.
    if not hasattr(ext, "carry_votes_topk_reset_"):
        raise RuntimeError("Bundled LiteTopK build lacks the HOT carry operation")
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        votes.record_stream(side)
        hot = entry["hot"][:HOT_PREFIX]
        workspace = _carry_topk_workspace(max_vote, dev)
        ext.carry_votes_topk_reset_(votes, hot, workspace["partial"],
                                   workspace["state"], HOT_PREFIX, max_vote, min_index)
        entry["needs_reset"] = False
        entry["dirty_extent"] = 0
        ready = entry["ready_event"]
        ready.record(side)
    entry["free_event"] = ready
    _HOT_CARRY[key] = (hot, nv, ready, min_index)


def retire(device):
    """Release a device's carry after the caller has synchronized the device."""
    dev_key = str(device)
    for cache in (_HOT_CARRY, _CARRY_VOTE_BUFS):
        for key in list(cache):
            if key[0] == dev_key:
                del cache[key]
    _CARRY_TOPK_WORKSPACE.pop(dev_key, None)
