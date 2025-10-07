from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from sympy import (
    symbols,
    sympify,
    Expr,
    N,
    Interval,
    ConditionSet,
    S,
    Intersection,
    Union,
    Complement,
    EmptySet,
)
from sympy.sets.sets import Set
from sympy.sets.fancysets import ProductSet  # type: ignore[attr-defined]

# Two real symbols for the plane
x, y = symbols("x y", real=True)


@dataclass(frozen=True)
class Circle:
    """
    Stores center and radius as SymPy Expr; we model the *disk* (filled circle),
    not the 1-D boundary curve.
    """

    cx: Expr
    cy: Expr
    r: Expr

    def __init__(
        self, cx: float | int | Expr, cy: float | int | Expr, r: float | int | Expr
    ):
        object.__setattr__(self, "cx", sympify(cx))
        object.__setattr__(self, "cy", sympify(cy))
        object.__setattr__(self, "r", sympify(r))

        # Validate radius (use SymPy attributes; silence Pylance where needed)
        if self.r.is_real is False:  # type: ignore[attr-defined]
            raise ValueError("Radius must be real.")
        if self.r.is_positive is True:  # type: ignore[attr-defined]
            return
        elif self.r.is_positive is False or self.r.is_zero is True:  # type: ignore[attr-defined]
            raise ValueError("Radius must be positive.")
        else:
            if float(self.r.evalf()) <= 0.0:  # type: ignore[arg-type]
                raise ValueError("Radius must be positive.")


def _disk_set(c: Circle) -> Set:
    """
    The closed disk {(x,y) in R^2 | (x-cx)^2 + (y-cy)^2 <= r^2} as a SymPy Set.
    """
    return ConditionSet(
        (x, y),
        (x - c.cx) ** 2 + (y - c.cy) ** 2 <= c.r**2,
        S.Reals**2,
    )


def _ambient_rectangle(circles: List[Circle], pad_ratio: float = 0.25) -> Set:
    """
    A rectangular ambient ProductSet that strictly contains all disks.
    Computed numerically (floats) for robust min/max, then converted to Intervals.
    """
    prec = 50
    xs_min = min(float(N(c.cx - c.r, prec)) for c in circles)  # type: ignore[arg-type]
    xs_max = max(float(N(c.cx + c.r, prec)) for c in circles)  # type: ignore[arg-type]
    ys_min = min(float(N(c.cy - c.r, prec)) for c in circles)  # type: ignore[arg-type]
    ys_max = max(float(N(c.cy + c.r, prec)) for c in circles)  # type: ignore[arg-type]

    span = max(xs_max - xs_min, ys_max - ys_min)
    pad = (span * pad_ratio) if span != 0.0 else 1.0

    x0, x1 = xs_min - pad, xs_max + pad
    y0, y1 = ys_min - pad, ys_max + pad

    X = Interval(x0, x1)
    Y = Interval(y0, y1)
    return ProductSet(X, Y)  # type: ignore[attr-defined]


def _bitmask_region(bits: str, disks: List[Set], ambient: Optional[Set]) -> Set:
    """
    Build the region for a bitmask using disk sets (not circle boundaries).
    - '1' → inside the corresponding disk
    - '0' → outside that disk
    For the all-zero mask, return ambient ∖ Union(disks). If ambient is None, return EmptySet.
    """
    inside: List[Set] = []
    outside: List[Set] = []
    for bit, d in zip(bits, disks):
        (inside if bit == "1" else outside).append(d)

    # Outside-all mask: ambient \ (union of all disks)
    if not inside and outside:
        if ambient is None:
            return EmptySet
        return Complement(ambient, Union(*outside))

    # Start with intersection of all required-inside disks (keep it symbolic; do NOT simplify away)
    region: Set = Intersection(*inside) if inside else EmptySet

    # Subtract all excluded disks
    if outside:
        region = Complement(region, Union(*outside))

    return region


def regions_for_arrangement(
    circles: List[Circle],
    *,
    include_outside: bool = False,
    ambient: Optional[Set] = None,
) -> Dict[str, Set]:
    """
    Compute geometric region(s) for each bitmask using *disk* regions.
    - If include_outside=True, the 000... mask is returned as (ambient ∖ Union(disks)).
      If ambient is not provided, a rectangular ProductSet is generated automatically.
    - If include_outside=False, the 000... mask is returned as EmptySet by design.

    Returns: dict mask -> SymPy Set (ConditionSet / Intersection / Union / Complement / EmptySet).
    """
    disks = [_disk_set(c) for c in circles]
    n = len(disks)

    if include_outside and ambient is None:
        ambient = _ambient_rectangle(circles)

    out: Dict[str, Set] = {}
    for i in range(2**n):
        mask = f"{i:0{n}b}"
        if not include_outside and all(ch == "0" for ch in mask):
            out[mask] = EmptySet
        else:
            out[mask] = _bitmask_region(mask, disks, ambient)
    return out
