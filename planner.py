"""Candidate planner merging seed hypotheses with default spaces.

This module exposes a tiny planning helper used by the probing loop.  Given
an initial set of hypotheses (e.g. ``{"TEXT_EMB_d": 2048}``) it returns an
ordered candidate space where seeded values are prioritised but the default
space is preserved for back-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterator, Mapping, Sequence

DEFAULT_CANDIDATES: Dict[str, Sequence[Any]] = {
    "TEXT_EMB_d": [768, 1024, 1280, 1536, 2048, 4096],
    "HEAD": ["epsilon", "v", "flow"],
    "LATENT_C": [4, 8],
    "LATENT_SCALE": [8, 16],
    "VISION": ["ViT-L/14-grid", "ViT-H/14-grid", "SigLIP-H-map"],
    "AUDIO_SHAPE": [(1, 4, 32), (2, 4, 32)],
    # LLM-specific knobs
    "VOCAB": [32000, 64000],
    "ROPE": [128000, 256000],
    "KV_DTYPE": ["f16", "f32"],
}

ORDER = [
    "TEXT_EMB_d",
    "HEAD",
    "LATENT_C",
    "LATENT_SCALE",
    "VISION",
    "AUDIO_SHAPE",
    "VOCAB",
    "ROPE",
    "KV_DTYPE",
]


def plan_candidates(
    seed: Mapping[str, Any],
    defaults: Mapping[str, Sequence[Any]] | None = None,
) -> Dict[str, list[Any]]:
    """Merge ``seed`` with ``defaults`` to build candidate lists.

    Seeded values appear first in the list for their key.  Unknown keys in
    ``seed`` are carried through unchanged.
    """
    defaults = dict(defaults or DEFAULT_CANDIDATES)
    result: Dict[str, list[Any]] = {}
    for key, vals in defaults.items():
        if key in seed:
            val = seed[key]
            result[key] = [val, *(v for v in vals if v != val)]
        else:
            result[key] = list(vals)
    for key, val in seed.items():
        if key not in result:
            result[key] = [val]
    return result


@dataclass
class Planner:
    """Iterate through probe candidate combinations.

    The enumeration order favours collapsing large uncertainties first:
    ``TEXT_EMB_d → HEAD → LATENT_C → LATENT_SCALE → VISION → AUDIO_SHAPE → VOCAB → ROPE → KV_DTYPE``.
    """

    seed: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Sequence[Any]] = field(
        default_factory=lambda: DEFAULT_CANDIDATES
    )
    order: Sequence[str] = field(default_factory=lambda: ORDER)

    def __post_init__(self) -> None:
        self.candidates = plan_candidates(self.seed, self.defaults)
        keys = [k for k in self.order if k in self.candidates]
        self._keys = keys
        self._iter: Iterator[tuple[Any, ...]] = product(
            *(self.candidates[k] for k in keys)
        )

    def __iter__(self) -> "Planner":  # pragma: no cover - trivial
        return self

    def __next__(self) -> Dict[str, Any]:
        values = next(self._iter)
        return dict(zip(self._keys, values))

    def update(self, hyp: Mapping[str, Any]) -> None:
        """Prune candidate lists according to ``hyp`` and reset iteration.

        Any key present in ``hyp`` is fixed to its provided value.  If the
        value differs from the current candidate space the internal iterator is
        rebuilt so subsequent ``next`` calls reflect the new hypothesis state.
        """

        changed = False
        for key, val in hyp.items():
            current = self.candidates.get(key)
            if current == [val]:
                continue
            self.candidates[key] = [val]
            changed = True
        if changed:
            keys = [k for k in self.order if k in self.candidates]
            self._keys = keys
            self._iter = product(*(self.candidates[k] for k in keys))


__all__ = ["DEFAULT_CANDIDATES", "plan_candidates", "Planner"]
