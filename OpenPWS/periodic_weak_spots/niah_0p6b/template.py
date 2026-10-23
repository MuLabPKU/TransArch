"""Token-exact rendering of the animal-name prompt templates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from string import Formatter
from typing import Any

from periodic_weak_spots.tokenization import token_ids


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(token_ids(tokenizer, text))


def _template_fields(template: str) -> tuple[str, ...]:
    fields: list[str] = []
    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name or format_spec or conversion:
            raise ValueError("prompt templates require named fields without modifiers")
        fields.append(field_name)
    return tuple(fields)


def pair_value_suffix(template: PromptTemplate) -> str:
    """Return the literal text following ``{value}`` in the pair template."""

    parts = list(Formatter().parse(template.pair_template))
    value_index = next(i for i, part in enumerate(parts) if part[1] == "value")
    return "".join(literal for literal, *_ in parts[value_index + 1 :])


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Literal prompt sections; ``{key}`` and ``{value}`` are single-token word slots."""

    layout_name: str
    task_descriptor: str
    filler_suffix: str
    mapping_start: str
    pair_template: str
    recall_query_prefix: str
    demonstration_template: str
    target_template: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.layout_name,
                self.task_descriptor,
                self.filler_suffix,
                self.mapping_start,
                self.pair_template,
                self.recall_query_prefix,
                self.demonstration_template,
                self.target_template,
            )
        ):
            raise ValueError("prompt template fields cannot be empty")
        expected = {
            "task_descriptor": (),
            "filler_suffix": (),
            "mapping_start": (),
            "pair_template": ("key", "value"),
            "recall_query_prefix": (),
            "demonstration_template": ("key", "value"),
            "target_template": ("key",),
        }
        for name, fields in expected.items():
            actual = _template_fields(getattr(self, name))
            if actual != fields:
                raise ValueError(
                    f"{name} fields must be exactly {fields} in that order, got {actual}"
                )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromptTemplate:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("prompt-template fields do not match schema")
        return cls(**{name: str(item) for name, item in value.items()})


@dataclass(frozen=True, slots=True)
class RenderedTemplate:
    """A rendered template and the token offsets of each field within it."""

    token_ids: tuple[int, ...]
    text: str
    field_positions: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class TemplateGeometry:
    """Token counts fixed by the templates, the same for every word binding."""

    task_descriptor_token_count: int
    mapping_start_token_count: int
    pair_token_count: int
    value_offset: int
    query_token_count: int


def render_template(
    template: str,
    tokenizer: Any,
    replacements: Mapping[str, tuple[str, int]],
) -> RenderedTemplate:
    """Encode literals independently and insert curated one-token fields."""

    ids: list[int] = []
    text_parts: list[str] = []
    field_positions: dict[str, list[int]] = defaultdict(list)
    for literal, field_name, _, _ in Formatter().parse(template):
        ids.extend(_encode(tokenizer, literal))
        text_parts.append(literal)
        if field_name is None:
            continue
        text, token_id = replacements[field_name]
        if _encode(tokenizer, text) != (token_id,):
            raise ValueError(f"replacement {field_name!r} is not the configured single token")
        field_positions[field_name].append(len(ids))
        ids.append(token_id)
        text_parts.append(text)
    rendered_text = "".join(text_parts)
    rendered_ids = tuple(ids)
    # Field offsets come from the piecewise encoding; they are exact only if encoding
    # the joined text yields the same IDs.
    if _encode(tokenizer, rendered_text) != rendered_ids:
        raise ValueError("rendered template is not additive under full-text encoding")
    return RenderedTemplate(
        token_ids=rendered_ids,
        text=rendered_text,
        field_positions={name: tuple(positions) for name, positions in field_positions.items()},
    )


def derive_template_geometry(
    template: PromptTemplate,
    tokenizer: Any,
    *,
    probe_key: tuple[str, int],
    probe_value: tuple[str, int],
    k_shot: int,
) -> TemplateGeometry:
    """Token counts and the in-pair value offset implied by the templates."""

    task_ids = _encode(tokenizer, template.task_descriptor)
    mapping_ids = _encode(tokenizer, template.mapping_start)
    prefix_ids = _encode(tokenizer, template.recall_query_prefix)
    pair = render_template(
        template.pair_template,
        tokenizer,
        {"key": probe_key, "value": probe_value},
    )
    if pair.field_positions["key"] != (0,):
        raise ValueError("pair template must begin with its key field")
    value_offset = pair.field_positions["value"][0]
    demonstration = render_template(
        template.demonstration_template,
        tokenizer,
        {"key": probe_key, "value": probe_value},
    )
    target = render_template(
        template.target_template,
        tokenizer,
        {"key": probe_key},
    )
    query_count = len(prefix_ids) + k_shot * len(demonstration.token_ids) + len(target.token_ids)
    return TemplateGeometry(
        task_descriptor_token_count=len(task_ids),
        mapping_start_token_count=len(mapping_ids),
        pair_token_count=len(pair.token_ids),
        value_offset=value_offset,
        query_token_count=query_count,
    )


__all__ = [
    "PromptTemplate",
    "RenderedTemplate",
    "TemplateGeometry",
    "derive_template_geometry",
    "pair_value_suffix",
    "render_template",
]
