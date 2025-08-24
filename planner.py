"""Candidate planner merging seed hypotheses with default spaces.

This module exposes a tiny planning helper used by the probing loop.  Given
an initial set of hypotheses (e.g. ``{"TEXT_EMB_d": 2048}``) it returns an
ordered candidate space where seeded values are prioritised but the default
space is preserved for back-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product, count
from heapq import heappush, heappop
import hashlib
import json
import math
from typing import Any, Dict, Iterator, Mapping, Sequence, Set, List, Tuple

from geometry import ConstraintSet

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


def probe_signature(combo: Mapping[str, Any]) -> str:
    """Stable hash for a probe ``combo``."""

    blob = json.dumps(combo, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


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
    failed_probes: Set[str] = field(default_factory=set)
    _queue: List[Tuple[float, int, Dict[str, Any]]] = field(init=False, default_factory=list)
    _counter: Iterator[int] = field(init=False, default_factory=count)

    def __post_init__(self) -> None:
        self.candidates = plan_candidates(self.seed, self.defaults)
        self._keys = [k for k in self.order if k in self.candidates]
        self._build_queue()

    def _build_queue(self) -> None:
        self._queue = []
        for values in product(*(self.candidates[k] for k in self._keys)):
            combo = dict(zip(self._keys, values))
            sig = probe_signature(combo)
            if sig in self.failed_probes:
                continue
            score = -self.score(combo)
            heappush(self._queue, (score, next(self._counter), combo))

    def score(self, combo: Mapping[str, Any]) -> float:  # pragma: no cover - default
        return 0.0

    def __iter__(self) -> "Planner":  # pragma: no cover - trivial
        return self

    def __next__(self) -> Dict[str, Any]:
        while self._queue:
            _score, _idx, combo = heappop(self._queue)
            sig = probe_signature(combo)
            if sig in self.failed_probes:
                continue
            return combo
        raise StopIteration

    def update(self, hyp: Mapping[str, Any]) -> None:
        """Prune candidate lists according to ``hyp`` and rebuild queue."""

        changed = False
        for key, val in hyp.items():
            current = self.candidates.get(key)
            if current == [val]:
                continue
            self.candidates[key] = [val]
            changed = True
        if changed:
            self._keys = [k for k in self.order if k in self.candidates]
            self._build_queue()


@dataclass
class GreedyPlanner(Planner):
    """Planner that ranks probes by estimated hypothesis collapse."""

    constraints: ConstraintSet = field(default_factory=ConstraintSet)

    def score(self, combo: Mapping[str, Any]) -> float:
        solved = self.constraints.solve()
        total = 1
        for k in self._keys:
            total *= 1 if k in solved else len(self.candidates[k])
        best = 0.0
        for k in self._keys:
            if k in solved:
                continue
            klen = len(self.candidates[k])
            if klen <= 1:
                continue
            after = total / klen
            diff = math.log(total) - math.log(after)
            if diff > best:
                best = diff
        return best


__all__ = [
    "DEFAULT_CANDIDATES",
    "plan_candidates",
    "probe_signature",
    "Planner",
    "GreedyPlanner",
]
