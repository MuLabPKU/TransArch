"""Exact bounded-composition sampling for randomized record spacing."""

from __future__ import annotations

import random
from functools import cache


def sample_bounded_composition(
    *,
    total: int,
    parts: int,
    cap: int,
    rng: random.Random,
) -> tuple[int, ...]:
    """Sample uniformly from ordered vectors with a fixed sum and entry cap."""

    if total < 0 or parts <= 0 or cap < 0:
        raise ValueError("bounded-composition parameters are invalid")

    # Drawing each entry in proportion to the number of ways to complete the rest makes
    # the whole vector uniform, unlike capping independent draws.
    @cache
    def completion_count(remaining_parts: int, remaining_total: int) -> int:
        if remaining_parts == 0:
            return int(remaining_total == 0)
        if remaining_total < 0 or remaining_total > remaining_parts * cap:
            return 0
        return sum(
            completion_count(remaining_parts - 1, remaining_total - value)
            for value in range(cap + 1)
        )

    if completion_count(parts, total) == 0:
        raise ValueError("bounded composition has no feasible allocation")
    result: list[int] = []
    remaining = total
    for index in range(parts):
        remaining_parts = parts - index - 1
        weighted_values = tuple(
            (value, completion_count(remaining_parts, remaining - value))
            for value in range(cap + 1)
            if completion_count(remaining_parts, remaining - value)
        )
        draw = rng.randrange(sum(weight for _, weight in weighted_values))
        cumulative = 0
        for value, weight in weighted_values:
            cumulative += weight
            if draw < cumulative:
                result.append(value)
                remaining -= value
                break
    if remaining or len(result) != parts:
        raise RuntimeError("bounded-composition sampler failed")
    return tuple(result)


__all__ = ["sample_bounded_composition"]
