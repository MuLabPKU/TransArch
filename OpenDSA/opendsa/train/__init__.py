"""OpenDSA training — DSATrainer + warmup/sparse stages + stage conversion.

Keep imports lazy so lightweight submodules such as ``opendsa.train.zero2`` do not
pull in Trainer-only dependencies.
"""

__all__ = [
    "DSATrainer", "DSAArgs", "build_model", "build_tokenizer", "load_packed",
    "save_indexer", "load_indexer", "extract_indexer_state",
]


def __getattr__(name):
    if name == "DSATrainer":
        from .trainer import DSATrainer
        return DSATrainer
    if name in {"DSAArgs", "build_model", "build_tokenizer", "load_packed"}:
        from .build import DSAArgs, build_model, build_tokenizer, load_packed
        return {
            "DSAArgs": DSAArgs,
            "build_model": build_model,
            "build_tokenizer": build_tokenizer,
            "load_packed": load_packed,
        }[name]
    if name in {"save_indexer", "load_indexer", "extract_indexer_state"}:
        from .convert_stage import save_indexer, load_indexer, extract_indexer_state
        return {
            "save_indexer": save_indexer,
            "load_indexer": load_indexer,
            "extract_indexer_state": extract_indexer_state,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
