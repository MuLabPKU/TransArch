"""Single-request eager lifecycle for HOT carry and bounded native workspace.

The normal indexer supplies its stable layer key and CPU first-chunk metadata.
Starting the first layer's next request retires every prior request allocation
on that device. No evaluator/controller is required to establish this boundary.
"""
from contextlib import contextmanager
import threading

import torch

_REQUESTS = {}
_CALL_LOCK = threading.Lock()
MAX_LAYERS = 128


def _device(device=None):
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    result = torch.device(device)
    if result.type != "cuda":
        raise ValueError("Blackwell state belongs to a CUDA device")
    return result if result.index is not None else torch.device("cuda", torch.cuda.current_device())


def _retire(device):
    from . import carry, hot, native
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("HOT request retirement is prohibited during CUDA capture")
    # Includes the HOT publisher stream. Never discard workspaces while kernels
    # or carry votes can still be using their storage.
    torch.cuda.synchronize(device)
    carry.retire(device)
    for cache in (native._WS, hot._PLANS, hot._MAP):
        for key in list(cache):
            if str(key[0]) == str(device):
                del cache[key]


def reset_request(device=None):
    """Explicit safe boundary for embedding applications; not required per call.

    Invoke outside model.forward after completing the request. This resets only
    Blackwell selector resources, not the model's KV cache or parameters.
    """
    device = _device(device)
    if not _CALL_LOCK.acquire(blocking=False):
        raise RuntimeError("Cannot reset HOT state during an active selector call")
    try:
        _retire(device)
        _REQUESTS.pop(str(device), None)
    finally:
        _CALL_LOCK.release()


@contextmanager
def chunk(layer, m, n, device, first_chunk):
    """Validate one normal, ordered per-layer causal prefill invocation."""
    if not isinstance(layer, str) or not layer:
        raise ValueError("Blackwell prefill needs a stable indexer layer key")
    if bool(first_chunk) != (m == n):
        raise ValueError("first_chunk must match fresh single-request CPU metadata")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Blackwell HOT prefill requires eager execution")
    if not _CALL_LOCK.acquire(blocking=False):
        raise RuntimeError("Blackwell HOT supports one eager request per worker")
    try:
        device = _device(device)
        key = str(device)
        owner = (int(torch.cuda.current_stream(device).cuda_stream), threading.get_ident())
        request = _REQUESTS.get(key)
        new_request = request is None or (first_chunk and layer == request["first_layer"])
        if new_request:
            if not first_chunk:
                raise ValueError("HOT cannot resume a prefix without its request state; disable prefix caching")
            _retire(device)
            request = dict(first_layer=layer, owner=owner, progress={})
            _REQUESTS[key] = request
        elif request["owner"] != owner:
            raise RuntimeError("Concurrent or cross-stream HOT request is unsupported")
        if request.get("failed"):
            raise RuntimeError("Cannot continue a failed HOT request; reset at a safe boundary")
        previous = request["progress"].get(layer)
        if first_chunk:
            if previous is not None:
                raise RuntimeError("Duplicate/out-of-order first chunk for a HOT layer")
            if len(request["progress"]) >= MAX_LAYERS:
                raise RuntimeError("HOT request exceeded the supported layer-key limit")
        elif previous is None or n - m != previous:
            raise RuntimeError("HOT requires increasing contiguous per-layer causal chunks")
        try:
            yield
        except Exception:
            # The process/application must end the failed request. Do not let a
            # caller silently continue with a partially published carry state.
            request["failed"] = True
            raise
        else:
            if request.get("failed"):
                raise RuntimeError("Cannot continue a failed HOT request")
            request["progress"][layer] = n
    finally:
        _CALL_LOCK.release()


