from __future__ import annotations

"""Simple dimensional constraint utilities.

This module provides tiny dataclasses for symbolic dimension variables and
constraints.  It is intentionally lightweight – just enough for the probing
loop to accumulate and solve equality/ divisibility rules extracted from
engine errors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, List, Union


@dataclass(frozen=True)
class DimVar:
    """Symbolic variable representing an unknown dimension."""

    name: str


class Constraint:
    """Base constraint type."""

    def propagate(self, assigns: Dict[str, Any]) -> bool:
        """Update ``assigns`` with any information implied by the constraint.

        Returns ``True`` if ``assigns`` was mutated.  Implementations may raise
        ``ValueError`` if a contradiction is detected.
        """

        raise NotImplementedError


@dataclass(frozen=True)
class Equality(Constraint):
    """Enforce that ``left`` equals ``right``."""

    left: DimVar
    right: Union[int, str, DimVar]

    def propagate(self, assigns: Dict[str, Any]) -> bool:
        lname = self.left.name
        r = self.right
        if isinstance(r, DimVar):
            rname = r.name
            if lname in assigns and rname in assigns:
                if assigns[lname] != assigns[rname]:
                    raise ValueError(f"{lname}={assigns[lname]} != {rname}={assigns[rname]}")
                return False
            if lname in assigns:
                assigns[rname] = assigns[lname]
                return True
            if rname in assigns:
                assigns[lname] = assigns[rname]
                return True
            return False
        else:
            if lname in assigns:
                if assigns[lname] != r:
                    raise ValueError(f"{lname}={assigns[lname]} != {r}")
                return False
            assigns[lname] = r
            return True


@dataclass(frozen=True)
class Divisible(Constraint):
    """Require that ``var`` is divisible by ``divisor``."""

    var: DimVar
    divisor: int

    def propagate(self, assigns: Dict[str, Any]) -> bool:
        name = self.var.name
        if name in assigns:
            val = assigns[name]
            if isinstance(val, int) and val % self.divisor != 0:
                raise ValueError(f"{val} not divisible by {self.divisor}")
        return False


@dataclass
class ConstraintSet:
    """Collection of constraints with a tiny solver."""

    constraints: List[Constraint] = field(default_factory=list)

    def add(self, *cons: Constraint) -> None:
        self.constraints.extend(cons)

    def extend(self, cons: Iterable[Constraint]) -> None:  # pragma: no cover - convenience
        self.constraints.extend(cons)

    def solve(self, seed: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Solve the constraint set returning concrete assignments.

        The solver performs a naive propagation pass over the constraints until
        no further progress can be made.
        """

        assigns: Dict[str, Any] = dict(seed or {})
        changed = True
        while changed:
            changed = False
            for c in self.constraints:
                if c.propagate(assigns):
                    changed = True
        return assigns


__all__ = ["DimVar", "Equality", "Divisible", "ConstraintSet"]
