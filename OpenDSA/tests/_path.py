"""Make tests runnable as scripts from the repository root.

This keeps commands like ``python tests/test_pg.py`` and
``torchrun --nproc_per_node=2 tests/test_cp_equiv.py`` working without requiring
the package to be installed first.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def shutdown_dist():
    """Best-effort torch.distributed shutdown for script-style tests."""
    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
