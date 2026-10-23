"""Exact integer arithmetic for the residue-balanced prefill-length design ("Design A")."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class DesignA:
    """Parameters for ``B*a + U1 + U2 + U3 + F``.

    ``B`` is ``modulus`` and ``a`` is ``start_multiple``. Each ``U`` is uniform
    on ``0..B-1`` and ``F`` is uniform on ``0..W``. The Design A width is
    selected so residue classes have equal mass, mean, and second raw moment.
    """

    modulus: int = 8
    start_multiple: int = 1
    n: int = 2

    @property
    def filler_width(self) -> int:
        # W = 2nB - 3(B - 1) makes the total span 2nB, so lengths cover [aB, (a + 2n)B].
        return 2 * self.n * self.modulus - 3 * (self.modulus - 1)

    @property
    def offset(self) -> int:
        return self.start_multiple * self.modulus

    @property
    def denominator(self) -> int:
        return self.modulus**3 * (self.filler_width + 1)

    def to_dict(self) -> dict[str, int]:
        return {**asdict(self), "filler_width": self.filler_width}


DEFAULT_DESIGN_A = DesignA()


def exact_counts(design: DesignA = DEFAULT_DESIGN_A) -> dict[int, int]:
    """Return integer multiplicities before normalization."""

    # Exact convolution of the four uniform draws; integers keep moment checks exact.
    counts: Counter[int] = Counter({0: 1})
    for width in (
        design.modulus - 1,
        design.modulus - 1,
        design.modulus - 1,
        design.filler_width,
    ):
        next_counts: Counter[int] = Counter()
        for subtotal, multiplicity in counts.items():
            for draw in range(width + 1):
                next_counts[subtotal + draw] += multiplicity
        counts = next_counts
    return {
        design.offset + length: counts[length] for length in range(min(counts), max(counts) + 1)
    }


def exact_pmf(design: DesignA = DEFAULT_DESIGN_A) -> dict[int, Fraction]:
    counts = exact_counts(design)
    denominator = sum(counts.values())
    return {length: Fraction(count, denominator) for length, count in counts.items()}


def balanced_length_pmf(modulus: int) -> dict[int, Fraction]:
    """Exact Design A PMF for one modulus."""

    return exact_pmf(DesignA(modulus=modulus))


def conditional_pmf(residue: int, design: DesignA = DEFAULT_DESIGN_A) -> dict[int, Fraction]:
    """Exact PMF of the lengths in one residue class mod ``design.modulus``."""

    counts = {
        length: count
        for length, count in exact_counts(design).items()
        if length % design.modulus == residue
    }
    denominator = sum(counts.values())
    return {length: Fraction(count, denominator) for length, count in counts.items()}


def raw_moment(distribution: dict[int, Fraction], order: int) -> Fraction:
    total = Fraction()
    for length, probability in distribution.items():
        total += (Fraction(length) ** order) * probability
    return total
