"""Lazy source builds. Importing lisa_indexer never compiles CUDA."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "linear_native": "kernels/linear/csrc/helpers.cu",
    "linear_recovery": "kernels/linear/csrc/scan.cu",
    "linear_main": "kernels/linear/csrc/scan.cu",
    "linear_sync": "kernels/linear/csrc/scan.cu",
    "linear_static": "kernels/linear/csrc/scan.cu",
    "linear_gatefinal": "kernels/linear/csrc/scan.cu",
    "linear_final": "kernels/linear/csrc/finalize.cu",
    "threshold_finish": "kernels/threshold/csrc/counted_finish.cu",
    "threshold_recovery": "kernels/threshold/csrc/bridge.cu",
}
SCAN_DEFINES = {
    "linear_recovery": "LISA_SCAN_RECOVERY",
    "linear_sync": "LISA_SCAN_SYNC",
    "linear_static": "LISA_SCAN_STATIC",
    "linear_gatefinal": "LISA_SCAN_FINAL",
}


def include_dirs():
    """Use external CUDA dependencies; see the pinned versions in README.md."""
    result = []
    for variable, header in [('CUTLASS_INCLUDE_DIR', 'cutlass/cutlass.h'),
                             ('DEEPGEMM_INCLUDE_DIR', 'deep_gemm/common/utils.cuh')]:
        value = os.environ.get(variable)
        if not value or not (Path(value).expanduser() / header).is_file():
            raise RuntimeError(f'Set {variable} to the include directory described in LISA/README.md')
        result.append(Path(value).expanduser().resolve())
    return result


@lru_cache(None)
def extension(name):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if name not in SOURCES:
        raise ValueError(f"unknown extension: {name}")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit/nvcc is required to build LISA")
    source = ROOT / SOURCES[name]
    includes = include_dirs() if name.startswith('linear_') and name != 'linear_final' else []
    # Include dependency contents and shared local headers in the cache key.
    digest = hashlib.sha256()
    for directory in [ROOT / 'kernels', *includes]:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.suffix in ('.cu', '.cuh', '.h', '.hpp'):
                digest.update(str(p.relative_to(directory)).encode())
                digest.update(p.read_bytes())
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo", "-UNDEBUG"]
    if name in SCAN_DEFINES:
        flags.append("-D" + SCAN_DEFINES[name])
    # Cross-compilation is supported without a GPU. Runtime input guards enforce
    # SM100 before any kernels execute; importing the package never calls here.
    return load(name=f"lisa_{name}_{digest.hexdigest()[:12]}", sources=[str(source)],
                extra_include_paths=[str(p) for p in includes],
                extra_cuda_cflags=flags, extra_ldflags=["-lcuda"],
                is_python_module=not name.startswith('threshold_'),
                verbose=os.environ.get('LISA_BUILD_VERBOSE', '0') == '1')


def build_all():
    for name in SOURCES:
        print(f"Building {name}", flush=True)
        extension(name)


if __name__ == '__main__':
    build_all()
