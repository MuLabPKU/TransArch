"""Query-suffix knockouts in DeepSeek-V4 with Hugging Face Transformers (Appendix E.6).

DeepSeek-V4 has one shared KV head per layer, so a knockout replaces a layer's
whole attention readout: the output of ``self_attn.o_b_proj``, after the
attention output projection and before the hyper-connection. Each 128K prompt
splits into a 127,984-token prefix and a 16-token query suffix that starts at
the first occurrence of the query key. By default the prefix also starts with
the tokenizer's leading special tokens, so the model sees the same tokens as
the vLLM scorer, which passes raw text.

* Calibration prefills each calibration prompt's prefix in 1,024-token chunks,
  runs the suffix, and averages the readout of every layer at every suffix
  position, giving one position-aligned mean per ``(layer, suffix position)``.
* Evaluation prefills each prompt's prefix once and forks that cache for every
  suffix forward. A condition ``(layer, mask)`` replaces the readout of that
  layer at the masked suffix positions with their means; later suffix tokens
  and the target-token logits are recomputed. Because outputs vary with batch
  size, conditions run as one batch over the masks, each compared with the
  same slot of a clean batch of the same size.

The score is the designated gold target token (top-1, logit, NLL) without
autoregressive generation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from periodic_weak_spots.niah_128k.layout import QUERY_MIDDLE, QUERY_PREFIX, QUERY_SUFFIX
from periodic_weak_spots.tokenization import Tokenizer, token_ids

MASKS = ("first_query_key", "second_query_key", "both_query_keys", "full_suffix")
METRICS = ("top1", "gold_logit", "gold_nll")


@dataclass(frozen=True, slots=True)
class SuffixPrompt:
    """One tokenized 128K prompt, split at its query suffix, with the gold next token."""

    sample_id: str
    residue: int
    token_ids: tuple[int, ...]
    suffix_start: int
    second_key_offset: int
    gold_token_id: int

    @property
    def suffix_ids(self) -> tuple[int, ...]:
        return self.token_ids[self.suffix_start :]

    def mask_offsets(self, mask: str) -> tuple[int, ...]:
        """Suffix positions replaced by one of the four conditions."""

        # The suffix starts at the first key; the second key follows the query middle (13).
        return {
            "first_query_key": (0,),
            "second_query_key": (self.second_key_offset,),
            "both_query_keys": (0, self.second_key_offset),
            "full_suffix": tuple(range(len(self.suffix_ids))),
        }[mask]


def query_suffix(tokenizer: Tokenizer, prompt: str, prompt_ids: Sequence[int]) -> tuple[int, int]:
    """Token count of the query suffix (from the first query key) and the second key's offset."""

    query = prompt[prompt.rfind(QUERY_PREFIX) :]
    key = query[len(QUERY_PREFIX) :].partition(QUERY_MIDDLE)[0]
    if not key or query != QUERY_PREFIX + key + QUERY_MIDDLE + key + QUERY_SUFFIX:
        raise ValueError("prompt does not end with the canonical query")
    parts = [token_ids(tokenizer, part) for part in (key, QUERY_MIDDLE, key, QUERY_SUFFIX)]
    if len(parts[0]) != 1:
        raise ValueError("the query key must be one token")
    suffix = [token for part in parts for token in part]
    # Part-wise offsets are valid only if no token spans a part or the suffix boundary.
    if list(prompt_ids[-len(suffix) :]) != suffix:
        raise ValueError("query suffix tokens are not additive in the prompt")
    return len(suffix), 1 + len(parts[1])


def load_suffix_prompts(
    root: str | Path,
    tokenizer: Tokenizer,
    *,
    rows_per_residue: int | None = None,
    leading_ids: Sequence[int] = (),
) -> list[SuffixPrompt]:
    """Tokenized prompts of a built 128K dataset; optionally the first N sample IDs per residue.

    ``leading_ids`` (e.g. BOS) are prepended to every prompt; ``residue`` stays the
    dataset's label, which counts positions without them.
    """

    from periodic_weak_spots.niah_128k.score_vllm import load_rows

    # A fixed order makes `rows_per_residue` and `--shard` select the same prompts on every run.
    rows = sorted(load_rows(root), key=lambda row: (row["phase"], row["sample_id"]))
    taken: dict[int, int] = {}
    prompts = []
    for row in rows:
        residue = row["phase"]
        if rows_per_residue is not None and taken.get(residue, 0) >= rows_per_residue:
            continue
        taken[residue] = taken.get(residue, 0) + 1
        prompt = token_ids(tokenizer, row["prompt"])
        answer = token_ids(tokenizer, row["answer"][0])
        # The gold token must be the prompt's own next token, not a merge across the boundary.
        if len(answer) != 1 or token_ids(tokenizer, row["prompt"] + row["answer"][0]) != [
            *prompt,
            *answer,
        ]:
            raise ValueError(f"{row['sample_id']}: the answer is not one context-stable token")
        suffix_tokens, second_key_offset = query_suffix(tokenizer, row["prompt"], prompt)
        full = (*leading_ids, *prompt)
        prompts.append(
            SuffixPrompt(
                sample_id=row["sample_id"],
                residue=residue,
                token_ids=full,
                suffix_start=len(full) - suffix_tokens,
                second_key_offset=second_key_offset,
                gold_token_id=answer[0],
            )
        )
    return prompts


def load_model(path: str):
    """One expert-parallel DeepSeek-V4 replica across the ``torchrun`` world.

    The paper used one replica on 8 GPUs.
    """

    import torch.distributed as dist
    from transformers import DeepseekV4ForCausalLM
    from transformers.distributed import DistributedConfig

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    # The paper's runs used Transformers' portable Triton FP8 kernels.
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    model = DeepseekV4ForCausalLM.from_pretrained(
        path,
        dtype="auto",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        tp_plan="auto",
        attn_implementation="eager",
        experts_implementation="grouped_mm",
    )
    return model.eval()


@contextmanager
def readout_hooks(
    model,
    *,
    captures: list[torch.Tensor] | None = None,
    means: torch.Tensor | None = None,
    conditions: Sequence[tuple[int, Sequence[int]]] | None = None,
) -> Iterator[None]:
    """Capture or replace every layer's ``o_b_proj`` output during one suffix forward.

    With ``captures``, the forward appends a float32 ``[batch, layers, suffix,
    hidden]`` tensor. With ``means`` (``[layers, suffix, hidden]``) and one
    ``(layer, suffix offsets)`` condition per batch row, each row's readout is
    replaced at its offsets in its layer.
    """

    layers = model.model.layers
    if (captures is None) == (means is None or conditions is None):
        raise ValueError("pass captures, or both means and conditions")
    if conditions is not None:
        for layer, offsets in conditions:
            if not 0 <= layer < len(layers) or not offsets:
                raise ValueError(f"invalid condition ({layer}, {offsets})")
    pending: list[torch.Tensor] = []

    def hook(layer: int):
        def forward_hook(_module, _inputs, output: torch.Tensor) -> torch.Tensor | None:
            if captures is not None:
                pending.append(output.detach().float())
                if layer == len(layers) - 1:
                    captures.append(torch.stack(pending, dim=1))
                    pending.clear()
                return None
            if output.shape[0] != len(conditions):
                raise ValueError("pass one condition per batch row")
            rows = [row for row, (target, _) in enumerate(conditions) if target == layer]
            if not rows:
                return None
            # Batch row i carries condition i, so other rows keep this layer's clean readout.
            patched = output.clone()
            for row in rows:
                offsets = list(conditions[row][1])
                patched[row, offsets] = means[layer, offsets].to(output.device, output.dtype)
            return patched

        return forward_hook

    handles = [
        decoder.self_attn.o_b_proj.register_forward_hook(hook(layer))
        for layer, decoder in enumerate(layers)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def prefill_prefix(model, prompt: SuffixPrompt, *, chunk_size: int = 1024) -> DynamicCache:
    """Cache the prompt's prefix ``[0, suffix_start)``, prefilled in fixed-size chunks."""

    if chunk_size <= 0 or chunk_size % max(model.config.compress_rates.values()):
        raise ValueError("chunk_size must be a positive multiple of every compression rate")
    cache = DynamicCache(config=model.config)
    # Chunking bounds prefill memory at 128K; chunk starts stay aligned to compression windows.
    for start in range(0, prompt.suffix_start, chunk_size):
        chunk = prompt.token_ids[start : min(start + chunk_size, prompt.suffix_start)]
        model.model(
            input_ids=torch.tensor([chunk], device=model.device),
            past_key_values=cache,
            use_cache=True,
        )
    if cache.get_seq_length() != prompt.suffix_start:
        raise RuntimeError("the prefix cache does not cover the prefix")
    return cache


def fork_cache(prefix: DynamicCache, config, batch_size: int) -> DynamicCache:
    """Share the immutable prefix across ``batch_size`` rows in fresh cache objects.

    DeepSeek-V4 cache updates reassign tensors instead of writing in place, so
    expanded views keep the prefix intact while each fork records its own suffix.
    """

    def expand(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] != batch_size:
            if value.shape[0] != 1:
                raise ValueError(f"cannot fork a cache tensor of batch {value.shape[0]}")
            # A view, not a copy: forks never duplicate the 128K prefix.
            return value.expand(batch_size, *value.shape[1:])
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    fork = DynamicCache(config=config)
    for source, target in zip(prefix.layers, fork.layers, strict=True):
        for name, value in vars(source).items():
            setattr(target, name, expand(value))
    return fork


@torch.inference_mode()
def suffix_logits(
    model, prefix: DynamicCache, prompt: SuffixPrompt, batch_size: int
) -> torch.Tensor:
    """Target-token logits ``[batch, vocab]`` of the suffix on a fork of the prefix cache."""

    input_ids = torch.tensor([prompt.suffix_ids] * batch_size, device=model.device)
    outputs = model(
        input_ids=input_ids,
        past_key_values=fork_cache(prefix, model.config, batch_size),
        use_cache=True,
        logits_to_keep=1,
    )
    return outputs.logits[:, -1].float()


def _scores(logits: torch.Tensor, gold: int) -> list[dict[str, float]]:
    nll = F.cross_entropy(
        logits, torch.full((len(logits),), gold, device=logits.device), reduction="none"
    )
    return [
        {
            "top1": float(row.argmax() == gold),
            "gold_logit": float(row[gold]),
            "gold_nll": float(loss),
        }
        for row, loss in zip(logits, nll, strict=True)
    ]


def fit_suffix_means(
    model, prompts: Sequence[SuffixPrompt], *, chunk_size: int = 1024
) -> torch.Tensor:
    """Position-aligned mean readout ``[layers, suffix, hidden]`` over calibration prompts."""

    total: torch.Tensor | None = None
    for prompt in prompts:
        prefix = prefill_prefix(model, prompt, chunk_size=chunk_size)
        captures: list[torch.Tensor] = []
        with readout_hooks(model, captures=captures):
            suffix_logits(model, prefix, prompt, 1)
        value = captures[0][0].double()
        if total is not None and value.shape != total.shape:
            raise ValueError("calibration prompts must share one suffix length")
        total = value if total is None else total + value
    if total is None:
        raise ValueError("calibration prompts are empty")
    return (total / len(prompts)).float()


def evaluate_prompt(
    model,
    prompt: SuffixPrompt,
    means: torch.Tensor,
    *,
    layers: Sequence[int] | None = None,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Clean and knocked-out target-token scores for one prompt.

    ``matched_clean`` is a clean batch with one row per mask; condition
    ``(layer, mask)`` runs in the same batch slot as ``matched_clean[mask]``.
    """

    if means.shape[1] != len(prompt.suffix_ids):
        raise ValueError("means and prompt have different suffix lengths")
    prefix = prefill_prefix(model, prompt, chunk_size=chunk_size)
    gold = prompt.gold_token_id
    # Outputs depend on batch size, so the four masks of a layer share one batch and each
    # row is compared with the same row of a clean batch of four, not the batch-of-one run.
    matched = _scores(suffix_logits(model, prefix, prompt, len(MASKS)), gold)
    patched = []
    for layer in range(len(model.model.layers)) if layers is None else layers:
        conditions = [(layer, prompt.mask_offsets(mask)) for mask in MASKS]
        with readout_hooks(model, means=means, conditions=conditions):
            logits = suffix_logits(model, prefix, prompt, len(MASKS))
        patched += [
            {"layer": layer, "mask": mask, **score}
            for mask, score in zip(MASKS, _scores(logits, gold), strict=True)
        ]
    return {
        "sample_id": prompt.sample_id,
        "residue": prompt.residue,
        "clean": _scores(suffix_logits(model, prefix, prompt, 1), gold)[0],
        "matched_clean": dict(zip(MASKS, matched, strict=True)),
        "patched": patched,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average scores by residue; offsets are patched minus the matched clean batch."""

    residues = sorted({record["residue"] for record in records})

    def mean_by_residue(values: Sequence[tuple[int, float]]) -> dict[str, float]:
        sums: dict[int, list[float]] = {residue: [] for residue in residues}
        for residue, value in values:
            sums[residue].append(value)
        return {str(residue): sum(v) / len(v) for residue, v in sums.items()}

    clean = {
        metric: mean_by_residue([(r["residue"], r["clean"][metric]) for r in records])
        for metric in METRICS
    }
    masks = list(records[0]["matched_clean"])
    matched = {
        mask: {
            metric: mean_by_residue(
                [(r["residue"], r["matched_clean"][mask][metric]) for r in records]
            )
            for metric in METRICS
        }
        for mask in masks
    }
    # Conditions come from the first record; every record must cover the same `--layers`.
    conditions = []
    for layer, mask in dict.fromkeys((row["layer"], row["mask"]) for row in records[0]["patched"]):
        patched = {
            metric: mean_by_residue(
                [
                    (r["residue"], row[metric])
                    for r in records
                    for row in r["patched"]
                    if (row["layer"], row["mask"]) == (layer, mask)
                ]
            )
            for metric in METRICS
        }
        base = matched[mask]
        conditions.append(
            {
                "layer": layer,
                "mask": mask,
                "by_residue": patched,
                "offset_by_residue": {
                    metric: {r: patched[metric][r] - base[metric][r] for r in patched[metric]}
                    for metric in METRICS
                },
                "relative_top1_change_by_residue": {
                    r: None if base["top1"][r] == 0 else patched["top1"][r] / base["top1"][r] - 1
                    for r in patched["top1"]
                },
            }
        )
    return {
        "rows": len(records),
        "rows_by_residue": {str(r): sum(rec["residue"] == r for rec in records) for r in residues},
        "clean": clean,
        "matched_clean": matched,
        "conditions": conditions,
    }


__all__ = [
    "MASKS",
    "SuffixPrompt",
    "evaluate_prompt",
    "fit_suffix_means",
    "fork_cache",
    "load_model",
    "load_suffix_prompts",
    "prefill_prefix",
    "query_suffix",
    "readout_hooks",
    "suffix_logits",
    "summarize",
]
