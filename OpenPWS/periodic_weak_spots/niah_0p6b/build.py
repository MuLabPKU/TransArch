"""Curate filler sentences, assemble every phase replica, validate, and save the suite."""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import decode as _decode
from periodic_weak_spots.tokenization import token_ids as _token_ids
from periodic_weak_spots.tokenization import token_ids_batch

from .config import (
    Candidate,
    EpisodeRow,
    EpisodeSpec,
    Sentence,
    SentenceSourceSpec,
    SuiteManifest,
    TokenizerSpec,
    load_suite_config,
)
from .design import (
    build_exact_sentence_bins,
    design_a_conditional_pmf,
    select_complementary_fillers,
    select_design_a_length,
)
from .pool import load_pool
from .spacing import sample_bounded_composition
from .template import (
    PromptTemplate,
    TemplateGeometry,
    derive_template_geometry,
    pair_value_suffix,
    render_template,
)


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(_token_ids(tokenizer, text))


def solve_prefill_residue(
    *,
    target_residue: int,
    task_descriptor_length: int,
    mapping_start_length: int,
    pair_token_count: int,
    target_index: int,
    phase_base: int,
) -> int:
    """Prefill length residue that puts the target pair start on ``target_residue``."""

    # Each pair starts with its key, so the target pair start is the source key position t_K.
    return (
        target_residue
        - task_descriptor_length
        - mapping_start_length
        - pair_token_count * target_index
    ) % phase_base


def _bindings_for_sample(
    keys: Sequence[Candidate],
    values: Sequence[Candidate],
    *,
    pair_count: int,
    seed: int,
) -> tuple[tuple[Candidate, Candidate], ...]:
    """Pair ``pair_count`` sampled animals with ``pair_count`` sampled names, in body order."""

    if len(keys) < pair_count or len(values) < pair_count:
        raise ValueError("candidate pools do not cover pair_count")
    rng = random.Random(seed)
    selected_keys = rng.sample(list(keys), pair_count)
    selected_values = rng.sample(list(values), pair_count)
    return tuple(zip(selected_keys, selected_values, strict=True))


def _demonstrations_for_sample(
    *,
    pair_count: int,
    target_index: int,
    k_shot: int,
    seed: int,
) -> tuple[int, ...]:
    """Pick the non-target pairs restated as in-context demonstrations before the target key."""

    candidates = [index for index in range(pair_count) if index != target_index]
    return tuple(random.Random(seed).sample(candidates, k_shot))


def _render_query(
    spec: EpisodeSpec,
    tokenizer: Any,
    bindings: Sequence[tuple[Candidate, Candidate]],
    demonstration_pair_indices: Sequence[int],
) -> tuple[tuple[int, ...], str]:
    """Render the recall prefix, the demonstrations, and the target key that ends the prompt."""

    prefix_ids = _encode(tokenizer, spec.prompt_template.recall_query_prefix)
    query_ids = list(prefix_ids)
    query_text_parts = [spec.prompt_template.recall_query_prefix]
    for pair_index in demonstration_pair_indices:
        key, value = bindings[pair_index]
        rendered = render_template(
            spec.prompt_template.demonstration_template,
            tokenizer,
            {
                "key": (key.text, key.token_id),
                "value": (value.text, value.token_id),
            },
        )
        query_ids.extend(rendered.token_ids)
        query_text_parts.append(rendered.text)
    gold_key, _ = bindings[spec.target_index]
    target = render_template(
        spec.prompt_template.target_template,
        tokenizer,
        {"key": (gold_key.text, gold_key.token_id)},
    )
    query_ids.extend(target.token_ids)
    query_text_parts.append(target.text)
    query_text = "".join(query_text_parts)
    if _encode(tokenizer, query_text) != tuple(query_ids):
        raise ValueError("query components are not additive under full-text encoding")
    return tuple(query_ids), query_text


def _assembly_workers(requested: int | None, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested is not None:
        if requested <= 0:
            raise ValueError("workers must be positive")
        return min(requested, task_count)
    cpu = os.cpu_count() or 1
    return min(cpu, task_count, 32)


def assemble_episode(
    *,
    spec: EpisodeSpec,
    phase: int,
    sample_index: int,
    bindings: Sequence[tuple[Candidate, Candidate]],
    tokenizer: Any,
    demonstration_pair_indices: Sequence[int],
    sentence_bins: Mapping[int, Sequence[Sentence]],
    geometry: TemplateGeometry,
) -> EpisodeRow:
    """Assemble one semantic prompt; the gold answer remains next-token metadata."""

    base = spec.phase_base
    demonstration_pair_indices = tuple(demonstration_pair_indices)
    gold_key, gold_value = bindings[spec.target_index]
    prefill_residue = solve_prefill_residue(
        target_residue=phase,
        task_descriptor_length=geometry.task_descriptor_token_count,
        mapping_start_length=geometry.mapping_start_token_count,
        pair_token_count=geometry.pair_token_count,
        target_index=spec.target_index,
        phase_base=base,
    )
    filler_rng = random.Random(f"{spec.seed}:{sample_index}:{phase}")
    spacing = spec.body_spacing
    gap_extra_lengths: tuple[int, ...] = ()
    if spec.postfill_mode == "synthetic":
        # In-sequence Padding: fixed pad guards; the balanced-length draw that
        # Prefix Padding uses as its prefill is realized as pad gaps between pairs.
        if _encode(tokenizer, spacing.pad_text) != (spacing.pad_token_id,):
            raise ValueError("synthetic pad text does not match its pinned token ID")
        # Adjacent pad characters could merge into one token; check every run length used.
        for count in (*range(spacing.max_gap_extras + 1), spacing.outer_guard_tokens):
            if _encode(tokenizer, spacing.pad_text * count) != (spacing.pad_token_id,) * count:
                raise ValueError("synthetic pad repetitions are not token-additive")
        reference_prefill_length = select_design_a_length(
            base=base,
            residue=prefill_residue,
            rng=filler_rng,
        )
        # The pre-target gaps hold the draw minus the leading guard, so the key lands where a
        # prefill of the drawn length would put it; the post-target gaps take the rest of the
        # fixed gap budget, keeping the prompt length the same at every residue.
        x = reference_prefill_length - spacing.reference_filler_offset
        gap_rng = random.Random(f"{spec.seed}:{sample_index}:{phase}:bounded-body-gaps-v1")
        before = sample_bounded_composition(
            total=x,
            parts=spec.target_index,
            cap=spacing.max_gap_extras,
            rng=gap_rng,
        )
        after = sample_bounded_composition(
            total=spacing.total_gap_extras - x,
            parts=spec.pair_count - spec.target_index - 1,
            cap=spacing.max_gap_extras,
            rng=gap_rng,
        )
        gap_extra_lengths = (*before, *after)
        prefill_ids = (spacing.pad_token_id,) * spacing.outer_guard_tokens
        postfill_ids = prefill_ids
        prefill_text = spacing.pad_text * spacing.outer_guard_tokens
        postfill_text = prefill_text
        prefill_sentence_id = "synthetic-prefix-guard"
        postfill_sentence_id = "synthetic-suffix-guard"
    else:
        # Prefix Padding: the prefill sentence has the drawn length and the postfill sentence
        # completes it to ``fixed_filler_total``, so the query keeps one residue.
        prefill, postfill = select_complementary_fillers(
            sentence_bins,
            base=base,
            prefill_residue=prefill_residue,
            fixed_total=spec.fixed_filler_total,
            rng=filler_rng,
        )
        prefill_ids = prefill.token_ids
        postfill_ids = postfill.token_ids
        prefill_text = prefill.text
        postfill_text = postfill.text
        prefill_sentence_id = prefill.sentence_id
        postfill_sentence_id = postfill.sentence_id

    body_ids: list[int] = []
    body_text_parts: list[str] = []
    for pair_index, (key, value) in enumerate(bindings):
        rendered = render_template(
            spec.prompt_template.pair_template,
            tokenizer,
            {
                "key": (key.text, key.token_id),
                "value": (value.text, value.token_id),
            },
        )
        if len(rendered.token_ids) != geometry.pair_token_count:
            raise ValueError("pair template geometry changed across bindings")
        body_ids.extend(rendered.token_ids)
        body_text_parts.append(rendered.text)
        if pair_index < len(gap_extra_lengths):
            extra = gap_extra_lengths[pair_index]
            body_ids.extend((spacing.pad_token_id,) * extra)
            body_text_parts.append(spacing.pad_text * extra)
    body_text = "".join(body_text_parts)
    if _encode(tokenizer, body_text) != tuple(body_ids):
        raise ValueError("pair body is not additive under full-text encoding")

    query_ids, query_text = _render_query(
        spec,
        tokenizer,
        bindings,
        demonstration_pair_indices,
    )
    if len(query_ids) != geometry.query_token_count:
        raise ValueError("query geometry changed across bindings")
    task_ids = _encode(tokenizer, spec.prompt_template.task_descriptor)
    mapping_ids = _encode(tokenizer, spec.prompt_template.mapping_start)
    pair_region_start = len(task_ids) + len(prefill_ids) + len(mapping_ids)
    postfill_start = pair_region_start + len(body_ids)
    query_start = postfill_start + len(postfill_ids)
    token_ids = (
        *task_ids,
        *prefill_ids,
        *mapping_ids,
        *body_ids,
        *postfill_ids,
        *query_ids,
    )
    prompt_text = "".join(
        (
            spec.prompt_template.task_descriptor,
            prefill_text,
            spec.prompt_template.mapping_start,
            body_text,
            postfill_text,
            query_text,
        )
    )
    if _encode(tokenizer, prompt_text) != token_ids:
        raise ValueError("whole prompt is not additive under full-text encoding")
    # The gold name must also stay one token when appended, so the next-token target
    # matches what plain text continuation would produce.
    if _encode(tokenizer, prompt_text + gold_value.text) != (
        *token_ids,
        gold_value.token_id,
    ):
        raise ValueError("prompt plus gold answer fails exact full-text re-encoding")
    if len(token_ids) > spec.max_sequence_length:
        raise ValueError(
            f"generated sequence has {len(token_ids)} tokens, exceeding "
            f"max_sequence_length={spec.max_sequence_length}"
        )

    target_pair_start = (
        pair_region_start
        + spec.target_index * geometry.pair_token_count
        + sum(gap_extra_lengths[: spec.target_index])
    )
    target_value_position = target_pair_start + geometry.value_offset
    logical_sample_id = f"{spec.suite_id}:{sample_index:06d}"
    return EpisodeRow(
        sample_id=f"{logical_sample_id}:phase{phase:04d}",
        logical_sample_id=logical_sample_id,
        phase=phase,
        token_ids=tuple(token_ids),
        # Candidates are the bound names in body order, so the gold one sits at target_index.
        candidate_value_token_ids=tuple(value.token_id for _, value in bindings),
        gold_candidate_index=spec.target_index,
        gold_key_token_id=gold_key.token_id,
        gold_value_token_id=gold_value.token_id,
        demonstration_pair_indices=demonstration_pair_indices,
        pair_region_start=pair_region_start,
        target_pair_start=target_pair_start,
        target_value_position=target_value_position,
        postfill_start=postfill_start,
        query_start=query_start,
        scoring_position=len(token_ids) - 1,
        prefill_length=len(prefill_ids),
        postfill_length=len(postfill_ids),
        prefill_sentence_id=prefill_sentence_id,
        postfill_sentence_id=postfill_sentence_id,
        gap_extra_lengths=gap_extra_lengths,
    )


def generate_suite(
    spec: EpisodeSpec,
    *,
    keys: Sequence[Candidate],
    values: Sequence[Candidate],
    sentences: Sequence[Sentence],
    tokenizer: Any,
    tokenizer_spec: TokenizerSpec,
    workers: int | None = None,
) -> tuple[tuple[EpisodeRow, ...], SuiteManifest]:
    """Assemble every logical sample at every residue, then derive and check the manifest."""

    geometry = derive_template_geometry(
        spec.prompt_template,
        tokenizer,
        probe_key=(keys[0].text, keys[0].token_id),
        probe_value=(values[0].text, values[0].token_id),
        k_shot=spec.k_shot,
    )
    rows: list[EpisodeRow] = []
    sentence_bins = build_exact_sentence_bins(sentences)

    def rows_for_sample(sample_index: int) -> list[EpisodeRow]:
        # Bindings and demonstrations depend only on the sample index, so the residue
        # replicas of a logical sample differ only in padding.
        bindings = _bindings_for_sample(
            keys,
            values,
            pair_count=spec.pair_count,
            seed=spec.seed + sample_index,
        )
        demonstrations = _demonstrations_for_sample(
            pair_count=spec.pair_count,
            target_index=spec.target_index,
            k_shot=spec.k_shot,
            seed=spec.few_shot_seed + sample_index,
        )
        return [
            assemble_episode(
                spec=spec,
                phase=phase,
                sample_index=sample_index,
                bindings=bindings,
                tokenizer=tokenizer,
                demonstration_pair_indices=demonstrations,
                sentence_bins=sentence_bins,
                geometry=geometry,
            )
            for phase in range(spec.phase_base)
        ]

    sample_indices = range(
        spec.sample_index_start,
        spec.sample_index_start + spec.samples_per_residue,
    )
    worker_count = _assembly_workers(workers, spec.samples_per_residue)
    if worker_count == 1:
        for sample_index in sample_indices:
            rows.extend(rows_for_sample(sample_index))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for sample_rows in pool.map(rows_for_sample, sample_indices):
                rows.extend(sample_rows)
    rows.sort(key=lambda row: row.sample_id)
    sequence_lengths = [len(row.token_ids) for row in rows]
    query_residues: list[int] = []
    scoring_residues: list[int] = []
    for phase in range(spec.phase_base):
        phase_rows = [row for row in rows if row.phase == phase]
        phase_query_residues = {row.query_start % spec.phase_base for row in phase_rows}
        phase_scoring_residues = {row.scoring_position % spec.phase_base for row in phase_rows}
        if len(phase_query_residues) != 1 or len(phase_scoring_residues) != 1:
            raise RuntimeError("query/scoring residue varies within a target phase")
        query_residues.append(phase_query_residues.pop())
        scoring_residues.append(phase_scoring_residues.pop())
    manifest = SuiteManifest(
        suite_id=spec.suite_id,
        tokenizer_id=tokenizer_spec.tokenizer_id,
        tokenizer_revision=tokenizer_spec.revision,
        phase_base=spec.phase_base,
        strides=spec.strides,
        row_count=len(rows),
        samples_per_residue=spec.samples_per_residue,
        sample_index_start=spec.sample_index_start,
        candidate_count=spec.pair_count,
        row_ids=tuple(row.sample_id for row in rows),
        prompt_template=spec.prompt_template,
        target_index=spec.target_index,
        k_shot=spec.k_shot,
        pair_token_count=geometry.pair_token_count,
        value_offset=geometry.value_offset,
        postfill_mode=spec.postfill_mode,
        fixed_filler_total=spec.fixed_filler_total,
        task_descriptor_token_count=geometry.task_descriptor_token_count,
        mapping_start_token_count=geometry.mapping_start_token_count,
        query_token_count=geometry.query_token_count,
        query_residues=tuple(query_residues),
        scoring_residues=tuple(scoring_residues),
        sequence_length_min=min(sequence_lengths),
        sequence_length_max=max(sequence_lengths),
        maximum_sequence_length=spec.max_sequence_length,
        body_spacing=spec.body_spacing,
    )
    validate_generated_suite(rows, manifest)
    return tuple(rows), manifest


def validate_generated_suite(
    rows: Sequence[EpisodeRow],
    manifest: SuiteManifest,
) -> None:
    """Independently validate completeness, geometry, residues, and phase replicas."""

    if len(rows) != manifest.row_count:
        raise ValueError("stored row count differs from manifest")
    ids = [row.sample_id for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(manifest.row_ids):
        raise ValueError("stored rows are duplicated or incomplete")
    counts = [0] * manifest.phase_base
    conditional_prefill_lengths = {
        residue: {
            length
            for length, _ in design_a_conditional_pmf(
                base=manifest.phase_base,
                residue=residue,
            )
        }
        for residue in range(manifest.phase_base)
    }
    by_logical_sample: dict[str, list[EpisodeRow]] = defaultdict(list)
    for row in rows:
        if not 0 <= row.phase < manifest.phase_base:
            raise ValueError("row phase lies outside manifest")
        counts[row.phase] += 1
        by_logical_sample[row.logical_sample_id].append(row)
        if row.gold_candidate_index != manifest.target_index:
            raise ValueError("row does not use the manifest's fixed target index")
        if (
            len(row.candidate_value_token_ids) != manifest.candidate_count
            or len(set(row.candidate_value_token_ids)) != manifest.candidate_count
            or row.candidate_value_token_ids[manifest.target_index] != row.gold_value_token_id
        ):
            raise ValueError("row candidate values are incomplete, duplicated, or misaligned")
        if (
            len(row.demonstration_pair_indices) != manifest.k_shot
            or manifest.target_index in row.demonstration_pair_indices
            or len(set(row.demonstration_pair_indices)) != manifest.k_shot
            or any(
                not 0 <= index < manifest.candidate_count
                for index in row.demonstration_pair_indices
            )
        ):
            raise ValueError("row demonstrations are duplicated or leak the gold pair")
        expected_pair_region = (
            manifest.task_descriptor_token_count
            + row.prefill_length
            + manifest.mapping_start_token_count
        )
        if row.pair_region_start != expected_pair_region:
            raise ValueError("pair region does not follow descriptor/prefill/marker")
        spacing = manifest.body_spacing
        if spacing.mode == "randomized_gaps":
            if (
                len(row.gap_extra_lengths) != manifest.candidate_count - 1
                or any(not 0 <= extra <= spacing.max_gap_extras for extra in row.gap_extra_lengths)
                or sum(row.gap_extra_lengths) != spacing.total_gap_extras
            ):
                raise ValueError("row randomized gap allocation is invalid")
            x = sum(row.gap_extra_lengths[: manifest.target_index])
        else:
            if row.gap_extra_lengths:
                raise ValueError("packed row declares randomized gap lengths")
            x = 0
        if row.target_pair_start != (
            row.pair_region_start + manifest.target_index * manifest.pair_token_count + x
        ):
            raise ValueError("target pair start equation failed")
        if row.target_value_position != row.target_pair_start + manifest.value_offset:
            raise ValueError("target value offset equation failed")
        if row.postfill_start != (
            row.pair_region_start
            + manifest.candidate_count * manifest.pair_token_count
            + sum(row.gap_extra_lengths)
        ):
            raise ValueError("postfill does not follow the complete pair body")
        if row.query_start != row.postfill_start + row.postfill_length:
            raise ValueError("query does not follow the postfill sentence")
        if row.target_pair_start % manifest.phase_base != row.phase:
            raise ValueError("target residue equation failed")
        if row.token_ids[row.target_pair_start] != row.gold_key_token_id:
            raise ValueError("target key position is invalid")
        if row.token_ids[row.target_value_position] != row.gold_value_token_id:
            raise ValueError("target value position is invalid")
        if row.query_start % manifest.phase_base != manifest.query_residues[row.phase] or (
            row.scoring_position % manifest.phase_base != manifest.scoring_residues[row.phase]
        ):
            raise ValueError("query/scoring residue differs from its target-phase entry")
        if row.scoring_position != row.query_start + manifest.query_token_count - 1:
            raise ValueError("query/scoring equation failed")
        expected_prefill_residue = (
            row.phase
            - manifest.task_descriptor_token_count
            - manifest.mapping_start_token_count
            - manifest.pair_token_count * manifest.target_index
        ) % manifest.phase_base
        # In-sequence Padding has no prefill sentence; its balanced-length draw is recovered
        # from the pre-target gaps plus the leading guard.
        reference_prefill_length = (
            row.prefill_length if spacing.mode == "packed" else x + spacing.reference_filler_offset
        )
        if reference_prefill_length % manifest.phase_base != expected_prefill_residue:
            raise ValueError("row reference prefill violates its phase requirement")
        if reference_prefill_length not in conditional_prefill_lengths[expected_prefill_residue]:
            raise ValueError("row reference prefill lies outside the balanced-length support")
        if manifest.postfill_mode == "natural":
            if row.prefill_length + row.postfill_length != manifest.fixed_filler_total:
                raise ValueError("row violates exact filler control")
            if (
                row.prefill_length % manifest.phase_base
                != (-row.postfill_length) % manifest.phase_base
            ):
                raise ValueError("row violates complementary filler coupling")
        else:
            if (
                manifest.fixed_filler_total != 2 * spacing.outer_guard_tokens
                or row.prefill_length != spacing.outer_guard_tokens
                or row.postfill_length != spacing.outer_guard_tokens
                or row.prefill_sentence_id != "synthetic-prefix-guard"
                or row.postfill_sentence_id != "synthetic-suffix-guard"
            ):
                raise ValueError("synthetic row guard controls are invalid")
            prefix_guard_start = manifest.task_descriptor_token_count
            if (
                row.token_ids[prefix_guard_start : prefix_guard_start + spacing.outer_guard_tokens]
                != (spacing.pad_token_id,) * spacing.outer_guard_tokens
            ):
                raise ValueError("synthetic prefix guard tokens are invalid")
            if (
                row.token_ids[row.postfill_start : row.query_start]
                != (spacing.pad_token_id,) * spacing.outer_guard_tokens
            ):
                raise ValueError("synthetic suffix guard tokens are invalid")
            cursor = row.pair_region_start
            for pair_index in range(manifest.candidate_count):
                cursor += manifest.pair_token_count
                if pair_index < len(row.gap_extra_lengths):
                    extra = row.gap_extra_lengths[pair_index]
                    if row.token_ids[cursor : cursor + extra] != (spacing.pad_token_id,) * extra:
                        raise ValueError("synthetic inter-record gap tokens are invalid")
                    cursor += extra
            if cursor != row.postfill_start:
                raise ValueError("synthetic body cursor does not end at postfill")
        if len(row.token_ids) > manifest.maximum_sequence_length:
            raise ValueError("row exceeds the configured maximum sequence length")
    if any(count != manifest.samples_per_residue for count in counts):
        raise ValueError("phase cells do not match manifest-derived counts")
    expected_logical_ids = {
        f"{manifest.suite_id}:{sample_index:06d}"
        for sample_index in range(
            manifest.sample_index_start,
            manifest.sample_index_start + manifest.samples_per_residue,
        )
    }
    if set(by_logical_sample) != expected_logical_ids:
        raise ValueError("logical sample IDs do not match the manifest sample span")
    if (
        min(len(row.token_ids) for row in rows) != manifest.sequence_length_min
        or max(len(row.token_ids) for row in rows) != manifest.sequence_length_max
    ):
        raise ValueError("observed sequence bounds differ from the manifest")
    for replicas in by_logical_sample.values():
        if len(replicas) != manifest.phase_base or {row.phase for row in replicas} != set(
            range(manifest.phase_base)
        ):
            raise ValueError("logical sample does not contain every phase exactly once")
        reference = replicas[0]
        for row in replicas[1:]:
            if (
                row.demonstration_pair_indices != reference.demonstration_pair_indices
                or row.candidate_value_token_ids != reference.candidate_value_token_ids
                or row.gold_key_token_id != reference.gold_key_token_id
                or row.gold_value_token_id != reference.gold_value_token_id
            ):
                raise ValueError("phase replicas do not share bindings and demonstrations")
            if manifest.body_spacing.mode == "randomized_gaps" and (
                row.pair_region_start != reference.pair_region_start
                or row.postfill_start != reference.postfill_start
                or row.query_start != reference.query_start
                or len(row.token_ids) != len(reference.token_ids)
            ):
                raise ValueError("randomized-gap replicas moved fixed outer geometry")


_ENCODE_CHUNK_SIZE = 4096


def curate_capped_sentence_bins(
    tokenizer: Any,
    dataset: Any,
    source: SentenceSourceSpec,
    prompt_template: PromptTemplate,
    *,
    forbidden_token_ids: Collection[int] = (),
) -> tuple[Sentence, ...]:
    """Bin rendered filler sentences by exact token length, keeping only additive boundaries."""

    forbidden_ids = frozenset(forbidden_token_ids)
    shuffled = dataset.shuffle(seed=source.shuffle_seed)
    records = shuffled.select(range(min(len(shuffled), source.maximum_rows_scanned)))
    bins: dict[int, list[Sentence]] = defaultdict(list)
    seen_text: set[str] = set()
    rejections: Counter[str] = Counter()
    template = prompt_template
    task_ids = _encode(tokenizer, template.task_descriptor)
    mapping_start_ids = _encode(tokenizer, template.mapping_start)
    postfill_prefix = pair_value_suffix(template)
    postfill_prefix_ids = _encode(tokenizer, postfill_prefix)
    recall_prefix_ids = _encode(tokenizer, template.recall_query_prefix)
    required_lengths = range(source.minimum_token_length, source.maximum_token_length + 1)

    def bins_full() -> bool:
        return all(len(bins[length]) >= source.maximum_per_bin for length in required_lengths)

    def consume_chunk(scan_indices: Sequence[int], filler_texts: Sequence[str]) -> None:
        encodings = token_ids_batch(tokenizer, filler_texts)
        for scan_index, filler_text, encoded in zip(
            scan_indices, filler_texts, encodings, strict=True
        ):
            ids = tuple(encoded)
            if len(ids) not in required_lengths:
                continue
            # A pool token in the filler would add a stray occurrence of a key or name.
            if forbidden_ids.intersection(ids):
                rejections["forbidden_token"] += 1
                continue
            if len(bins[len(ids)]) >= source.maximum_per_bin:
                continue
            if _decode(tokenizer, ids) != filler_text:
                rejections["rendered_round_trip"] += 1
                continue
            # A sentence may fill either slot, so its tokens must not merge with the text
            # around the prefill or the postfill.
            if _encode(
                tokenizer, template.task_descriptor + filler_text + template.mapping_start
            ) != (*task_ids, *ids, *mapping_start_ids):
                rejections["prefill_boundary"] += 1
                continue
            if _encode(tokenizer, postfill_prefix + filler_text + template.recall_query_prefix) != (
                *postfill_prefix_ids,
                *ids,
                *recall_prefix_ids,
            ):
                rejections["postfill_boundary"] += 1
                continue
            bins[len(ids)].append(
                Sentence(
                    sentence_id=f"hq_shuffled_{scan_index:09d}",
                    text=filler_text,
                    token_ids=ids,
                    source_index=scan_index,
                )
            )

    pending_scan: list[int] = []
    pending_filler: list[str] = []
    for scan_index, record in enumerate(records):
        if len(pending_filler) >= _ENCODE_CHUNK_SIZE:
            consume_chunk(pending_scan, pending_filler)
            pending_scan, pending_filler = [], []
            if bins_full():
                break
        raw_text = record.get(source.text_field)
        if not isinstance(raw_text, str) or not raw_text or raw_text in seen_text:
            continue
        seen_text.add(raw_text)
        pending_scan.append(scan_index)
        pending_filler.append(raw_text + template.filler_suffix)
    if pending_filler:
        consume_chunk(pending_scan, pending_filler)
    missing = [length for length in required_lengths if not bins[length]]
    if missing:
        raise ValueError(
            f"empty exact-length filler bins {missing} after scanning {len(records)} rows; "
            f"rejections: {dict(rejections)}"
        )
    return tuple(sentence for length in required_lengths for sentence in bins[length])


def build_suite(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Any,
    cache_dir: str | None = None,
    workers: int | None = None,
) -> SuiteManifest:
    """Build one suite and save it as a Hugging Face dataset plus ``manifest.json``."""

    from datasets import Dataset, load_dataset

    config = load_suite_config(config_path)
    pool = load_pool(config.token_pool_manifest, tokenizer=tokenizer)
    if pool.tokenizer != config.tokenizer:
        raise ValueError("token pool and suite use different tokenizers")
    if pool.template != config.episode.prompt_template or pool.k_shot != config.episode.k_shot:
        raise ValueError("token pool and suite use different prompt geometry")
    source = config.sentence_source
    if source is None:
        sentences: tuple[Sentence, ...] = ()
    else:
        dataset = load_dataset(
            source.dataset_id, revision=source.revision, split=source.split, cache_dir=cache_dir
        )
        sentences = curate_capped_sentence_bins(
            tokenizer,
            dataset,
            source,
            config.episode.prompt_template,
            forbidden_token_ids=tuple(item.token_id for item in (*pool.keys, *pool.values)),
        )
    rows, manifest = generate_suite(
        config.episode,
        keys=pool.keys,
        values=pool.values,
        sentences=sentences,
        tokenizer=tokenizer,
        tokenizer_spec=config.tokenizer,
        workers=workers,
    )
    output = Path(output_dir)
    Dataset.from_list([row.to_dict() for row in rows]).save_to_disk(str(output))
    (output / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_suite(path: str | Path) -> tuple[tuple[EpisodeRow, ...], SuiteManifest]:
    """Load a saved suite and re-run the structural validator."""

    from datasets import load_from_disk

    directory = Path(path)
    manifest = SuiteManifest.from_dict(
        json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    )
    rows = tuple(EpisodeRow.from_dict(row) for row in load_from_disk(str(directory)))
    validate_generated_suite(rows, manifest)
    return rows, manifest
