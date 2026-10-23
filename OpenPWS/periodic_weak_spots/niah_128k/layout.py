"""Exact 128K prompt span accounting for the ``key:value。`` record form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from periodic_weak_spots.tokenization import (
    Tokenizer,
    assert_additive,
    build_token_exact_padding,
    token_ids,
)

if TYPE_CHECKING:
    from .bindings import Binding
    from .config import LayoutSpec
    from .schedule import PrefillAssignment

QUERY_PREFIX = "Now answer the question using the key-value pairs above. What is the value for key "
QUERY_MIDDLE = "? Answer with only the value. The value for key "
QUERY_SUFFIX = " is "

PAIR_FORM = "key_value"
# A record is key, ":", value, "。", one token each.
PAIR_TOKENS = 4
VALUE_OFFSET = 2


def render_query(key: str) -> str:
    return QUERY_PREFIX + key + QUERY_MIDDLE + key + QUERY_SUFFIX


@dataclass(frozen=True, slots=True)
class Scaffold:
    """Prompt pieces and token counts shared by every row."""

    intro_text: str
    intro_token_count: int
    body_prefix_text: str
    body_prefix_token_count: int
    tail_text: str
    tail_token_count: int
    query_token_count: int
    query_separator_token_count: int


def render_pair(key: str, value: str, spec: LayoutSpec) -> str:
    return f"{key}:{value}{spec.record_separator}"


def render_intro(spec: LayoutSpec) -> str:
    examples = (("猫", "树"), ("狗", "叶"), ("鸟", "根"))
    rendered = ", ".join(render_pair(key, value, spec) for key, value in examples)
    return (
        "Some special key-value pairs are hidden within the following text. "
        "Make sure to memorize them. I will quiz you about a key-value pair afterwards. "
        'Each key-value pair is written in the form key:value. The token before ":" '
        'is the key, and the token after ":" is the value.\n\n'
        f"For example, the text may contain {rendered}. "
        "These examples only explain the format.\n"
    )


def _single_token(tokenizer: Tokenizer, text: str, label: str) -> int:
    encoded = token_ids(tokenizer, text)
    if len(encoded) != 1:
        raise ValueError(f"{label} must be exactly one tokenizer token")
    return encoded[0]


def validate_pair_body(tokenizer: Tokenizer, binding: Binding, spec: LayoutSpec) -> list[int]:
    """Require every record, and the whole body, to tokenize as key, colon, value, separator."""

    colon = _single_token(tokenizer, ":", "colon")
    separator = _single_token(tokenizer, spec.record_separator, "separator")
    expected_body: list[int] = []
    texts: list[str] = []
    for pair in binding.pairs:
        expected = [pair.key_token_id, colon, pair.value_token_id, separator]
        text = render_pair(pair.key, pair.value, spec)
        if token_ids(tokenizer, text) != expected:
            raise ValueError("pair record is not token-exact")
        texts.append(text)
        expected_body.extend(expected)
    if token_ids(tokenizer, "".join(texts)) != expected_body:
        raise ValueError("adjacent records merge")
    return expected_body


def _aligned_intro(tokenizer: Tokenizer, filler_text: str, spec: LayoutSpec) -> tuple[str, int]:
    """Pad the intro with filler until its token count is a multiple of ``spec.alignment``."""

    intro = render_intro(spec)
    for padding_count in range(spec.alignment * 8):
        padding = build_token_exact_padding(tokenizer, padding_count, filler_text)
        candidate = intro + padding + "\n"
        count = len(token_ids(tokenizer, candidate))
        if count % spec.alignment == 0:
            return candidate, count
    raise ValueError("could not align the intro")


def _guarded_padding(
    tokenizer: Tokenizer, token_count: int, filler_text: str, guard: str, *, label: str
) -> str:
    """Exactly ``token_count`` tokens of filler between two single-token guards.

    The guards keep filler tokens from merging with neighboring components.
    """

    guard_id = _single_token(tokenizer, guard, f"{label} guard")
    inner_count = token_count - 2
    if inner_count < 0:
        raise ValueError(f"{label} is too short for both guards")
    inner = build_token_exact_padding(tokenizer, inner_count, filler_text)
    guarded = guard + inner + guard
    if token_ids(tokenizer, guarded) != [guard_id, *token_ids(tokenizer, inner), guard_id]:
        raise ValueError(f"{label} guard boundaries are non-additive")
    return guarded


def prepare_scaffold(
    tokenizer: Tokenizer,
    example_binding: Binding,
    *,
    alignment_filler: str,
    tail_filler: str,
    spec: LayoutSpec,
) -> Scaffold:
    body_ids = validate_pair_body(tokenizer, example_binding, spec)
    intro, intro_count = _aligned_intro(tokenizer, alignment_filler, spec)
    query_count = len(token_ids(tokenizer, render_query(example_binding.target.key)))
    separator_count = len(token_ids(tokenizer, spec.query_separator))
    # The target record then starts at target_pair_mean - natural_filler_tokens // 2 + prefill.
    # That offset (63,976) is a multiple of 8, so the prefill length alone sets the residue, and
    # the mean prefill of 24 puts the mean target start at target_pair_mean.
    body_prefix_count = (
        spec.target_pair_mean
        - intro_count
        - spec.target_index * PAIR_TOKENS
        - spec.natural_filler_tokens // 2
    )
    if body_prefix_count <= 0 or body_prefix_count % spec.alignment:
        raise ValueError("body prefix must be positive and preserve phase alignment")
    body_prefix = _guarded_padding(
        tokenizer, body_prefix_count, tail_filler, spec.body_prefix_guard_token, label="body prefix"
    )
    fixed_count = (
        intro_count
        + spec.natural_filler_tokens
        + body_prefix_count
        + len(body_ids)
        + separator_count
        + query_count
    )
    # Prefill + postfill always totals natural_filler_tokens, so one tail fits every row.
    tail_count = spec.prompt_tokens - fixed_count
    if tail_count < 0:
        raise ValueError("fixed prompt components exceed the prompt budget")
    tail = _guarded_padding(tokenizer, tail_count, tail_filler, spec.tail_guard_token, label="tail")
    if len(token_ids(tokenizer, tail)) != tail_count:
        raise ValueError("tail filler is not token exact")
    return Scaffold(
        intro_text=intro,
        intro_token_count=intro_count,
        body_prefix_text=body_prefix,
        body_prefix_token_count=body_prefix_count,
        tail_text=tail,
        tail_token_count=tail_count,
        query_token_count=query_count,
        query_separator_token_count=separator_count,
    )


def build_prompt_row(
    tokenizer: Tokenizer,
    binding: Binding,
    assignment: PrefillAssignment,
    scaffold: Scaffold,
    spec: LayoutSpec,
    *,
    sample_prefix: str,
) -> dict[str, Any]:
    """Assemble one prompt row with token-exact span metadata."""

    body_ids = validate_pair_body(tokenizer, binding, spec)
    body_text = "".join(render_pair(pair.key, pair.value, spec) for pair in binding.pairs)
    query_text = render_query(binding.target.key)
    query_ids = token_ids(tokenizer, query_text)
    if len(query_ids) != scaffold.query_token_count:
        raise ValueError("query token count varies across single-token keys")
    components = {
        "intro": scaffold.intro_text,
        "natural_prefill": assignment.prefill.text,
        "body_prefix": scaffold.body_prefix_text,
        "pair_body": body_text,
        "natural_postfill": assignment.postfill.text,
        "tail": scaffold.tail_text,
        "query_separator": spec.query_separator,
        "query": query_text,
    }
    # Additivity makes each component's own token count a valid span in the full prompt.
    assert_additive(
        tokenizer,
        *components.values(),
        label=f"{PAIR_FORM}/phase{assignment.phase}/row{assignment.row_index}",
    )
    prompt = "".join(components.values())
    prompt_ids = token_ids(tokenizer, prompt)
    if len(prompt_ids) != spec.prompt_tokens:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens, expected {spec.prompt_tokens}")
    counts = {
        "intro": scaffold.intro_token_count,
        "natural_prefill": assignment.prefill.token_count,
        "body_prefix": scaffold.body_prefix_token_count,
        "pair_body": len(body_ids),
        "natural_postfill": assignment.postfill.token_count,
        "tail": scaffold.tail_token_count,
        "query_separator": scaffold.query_separator_token_count,
        "query": len(query_ids),
    }
    starts: dict[str, int] = {}
    cursor = 0
    for label, count in counts.items():
        starts[label] = cursor
        cursor += count
    if cursor != spec.prompt_tokens:
        raise AssertionError("component spans do not sum to prompt length")
    target_pair_start = starts["pair_body"] + binding.target_index * PAIR_TOKENS
    target_value_start = target_pair_start + VALUE_OFFSET
    query_start = spec.prompt_tokens - counts["query"]
    if starts["query"] != query_start:
        raise AssertionError("query start must equal prompt length minus query length")
    if target_pair_start % spec.alignment != assignment.phase:
        raise ValueError("target phase equation is violated")
    coordinate = f"phase{assignment.phase}_{assignment.row_index:03d}"
    return {
        "sample_id": f"{sample_prefix}_{PAIR_FORM}_{coordinate}",
        "pair_form": PAIR_FORM,
        "phase": assignment.phase,
        "row_index": assignment.row_index,
        "prompt": prompt,
        "answer": [binding.target.value],
        "components": components,
        "metadata": {
            "prompt_token_count": spec.prompt_tokens,
            "model_prefill_token_count": spec.prompt_tokens,
            "phase_control_prefix_count": assignment.prefill.token_count,
            "fixed_natural_filler_count": spec.natural_filler_tokens,
            "pair_token_count": PAIR_TOKENS,
            "target_index": binding.target_index,
            "target_pair_token_start": target_pair_start,
            "target_value_token_start": target_value_start,
            "target_phase": target_pair_start % spec.alignment,
            "query_token_start": query_start,
            "query_token_count": counts["query"],
            "target_to_query_distance": query_start - target_value_start,
            "spans": {
                label: {"start": starts[label], "count": counts[label]} for label in components
            },
        },
        "binding_id": binding.binding_id,
        "sentence_pair_id": assignment.sentence_pair_id,
        "prefill_sentence_id": assignment.prefill.sentence_id,
        "postfill_sentence_id": assignment.postfill.sentence_id,
        "role_reversal_phase": assignment.role_reversal_phase,
        "role_reversal_row_index": assignment.role_reversal_row_index,
    }
