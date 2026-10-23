"""Build SM100 sources with user-installed DeepGEMM/CUTLASS headers.

Importing this module does not compile, download, or initialize CUDA. The first
selector call builds missing extensions in PyTorch's normal extension cache.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"
SOURCES = {
    "seed": ("seed", "dsa_litetopk.cu"),
    "scan": ("scan", "binding.cu"),
    "final": ("final", "binding.cu"),
    "pair_swap": ("pair_swap", "pair_swap.cu"),
    "carry": ("carry", "dsa_litetopk.cu"),
}


def external_include_paths():
    """Resolve the include roots documented in this variant's README."""
    roots = []
    for variable, header in (
        ("RESMISA_DEEPGEMM_INCLUDE", "deep_gemm/common/utils.cuh"),
        ("RESMISA_CUTLASS_INCLUDE", "cute/tensor.hpp"),
    ):
        value = os.environ.get(variable)
        root = Path(value).expanduser().resolve() if value else None
        if root is None or not (root / header).is_file():
            raise RuntimeError(
                f"Set {variable} to the downloaded include directory containing "
                f"{header}; follow the Blackwell README download commands.")
        roots.append(root)
    return roots


def source_digest(kind):
    folder, _ = SOURCES[kind]
    roots = [(folder, CSRC / folder)]
    if kind in ("seed", "scan", "carry"):
        roots += list(zip(("external/deep_gemm", "external/cutlass"),
                          external_include_paths()))
    digest = hashlib.sha256()
    if kind in ("seed", "carry"):
        shared_header = CSRC / "dense_topk_litetopk.cuh"
        digest.update(shared_header.name.encode())
        digest.update(shared_header.read_bytes())
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update((label + "/" + path.relative_to(root).as_posix()).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=len(SOURCES))
def extension(kind):
    if kind not in SOURCES:
        raise ValueError(f"Unknown Blackwell extension {kind!r}")
    from torch.utils.cpp_extension import load

    folder, source = SOURCES[kind]
    digest = source_digest(kind)
    # Names include source content; torch's builder also tracks compiler flags,
    # ABI and source changes, and serializes concurrent build processes.
    name = f"resmisa_bw_{kind}_{digest[:16]}"
    module = load(
        name=name,
        sources=[str(CSRC / folder / source)],
        extra_include_paths=[str(CSRC / folder),
                             *(str(root) for root in external_include_paths())],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda",
                           "-gencode=arch=compute_100a,code=sm_100a"],
        extra_ldflags=["-ldl"],
        verbose=False,
    )
    return module
