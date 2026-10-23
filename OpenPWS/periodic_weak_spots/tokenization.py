"""One tokenizer for every builder: HF loading and decode, GigaToken encode.

GigaToken runs the same ``tokenizer.json`` in a parallel Rust encoder. All
token-exact checks in this repository go through ``token_ids`` and
``token_ids_batch``, so generation and validation share one encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """The tokenizer interface the builders rely on; HF tokenizers also satisfy it."""

    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str: ...

    def __len__(self) -> int: ...


@runtime_checkable
class BatchTokenizer(Protocol):
    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = False
    ) -> Sequence[Sequence[int]]: ...


class FastTokenizer:
    """A pinned HF fast tokenizer whose encoder is GigaToken."""

    def __init__(self, hf_tokenizer: Any) -> None:
        from gigatoken import Tokenizer as GigaTokenizer

        self.hf = hf_tokenizer
        # Built from the pinned HF tokenizer itself, so both encoders share one vocabulary
        # and merge rules; decoding stays on HF.
        self._giga = GigaTokenizer(hf_tokenizer)
        self.eos_token_id = hf_tokenizer.eos_token_id
        self.all_special_ids = tuple(hf_tokenizer.all_special_ids)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("this repository never adds special tokens")
        return self._giga.encode(text).tolist()

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        if add_special_tokens:
            raise ValueError("this repository never adds special tokens")
        return self._giga.encode_batch_list(list(texts), parallel=True)

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str:
        return str(self.hf.decode(list(token_ids), **kwargs))

    def __len__(self) -> int:
        return len(self.hf)


def load_tokenizer(
    model_id: str,
    revision: str,
    *,
    local_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> FastTokenizer:
    """Load a pinned Hub tokenizer (or a local copy of it) behind GigaToken."""

    from transformers import AutoTokenizer

    if local_path is not None:
        hf = AutoTokenizer.from_pretrained(
            str(local_path), local_files_only=True, trust_remote_code=False, use_fast=True
        )
    else:
        hf = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            trust_remote_code=False,
            use_fast=True,
        )
    return FastTokenizer(hf)


def leading_special_ids(tokenizer: FastTokenizer) -> tuple[int, ...]:
    """Special tokens the HF tokenizer adds before a text by default, as vLLM does for raw text."""

    plain = tokenizer.hf.encode("a", add_special_tokens=False)
    full = tokenizer.hf.encode("a", add_special_tokens=True)
    # Appended specials would follow the query and change the scored next-token position.
    if full[len(full) - len(plain) :] != plain:
        raise ValueError("the tokenizer appends special tokens after the text")
    return tuple(int(token) for token in full[: len(full) - len(plain)])


def token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def token_ids_batch(tokenizer: Tokenizer, texts: Sequence[str]) -> list[list[int]]:
    if isinstance(tokenizer, BatchTokenizer):
        return [
            [int(token) for token in encoded]
            for encoded in tokenizer.encode_batch(texts, add_special_tokens=False)
        ]
    return [token_ids(tokenizer, text) for text in texts]


def decode(tokenizer: Tokenizer, ids: Sequence[int]) -> str:
    # No special-token stripping or space cleanup, so decoded text can re-encode exactly.
    return tokenizer.decode(
        list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def assert_additive(tokenizer: Tokenizer, *parts: str, label: str = "boundary") -> None:
    """Require that the parts encode to the same IDs separately and concatenated.

    Per-part token counts set positions, so no token may merge across a part boundary.
    """

    separate = [token for encoded in token_ids_batch(tokenizer, parts) for token in encoded]
    if token_ids(tokenizer, "".join(parts)) != separate:
        raise ValueError(f"non-additive token boundary: {label}")


def build_token_exact_padding(tokenizer: Tokenizer, token_count: int, filler_text: str) -> str:
    """Construct exactly ``token_count`` tokens of filler or fail without substitution."""

    if token_count < 0:
        raise ValueError("token_count cannot be negative")
    if token_count == 0:
        return ""
    if not filler_text:
        raise ValueError("filler text cannot be empty")
    repeated = filler_text
    encoded = token_ids(tokenizer, repeated)
    while len(encoded) < token_count:
        repeated += "\n" + filler_text
        encoded = token_ids(tokenizer, repeated)
    desired = encoded[:token_count]
    # A truncated token prefix can decode to text that re-encodes differently.
    candidate = decode(tokenizer, desired)
    if token_ids(tokenizer, candidate) != desired:
        raise ValueError("tokenizer cannot round-trip an exact padding prefix")
    return candidate
