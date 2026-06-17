"""Metric registry + shared statistical helpers.

Experiments register reducers with @metric("name"). A reducer takes results
already grouped by cell and returns a number or a small table (list of row dicts).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

from .trial import TrialResult

# name -> reducer(grouped: Dict[cell_tuple, List[TrialResult]], **params) -> Any
_REGISTRY: Dict[str, Callable] = {}


def metric(name: str):
    """Decorator registering a reducer under `name`."""

    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"metric {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def get_metric(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> List[str]:
    return sorted(_REGISTRY)


# --- grouping -----------------------------------------------------------------


def cell_key(cell: Dict[str, Any], axes: List[str] = None) -> Tuple:
    """Hashable key for a cell, optionally restricted to a subset of axes."""
    keys = axes if axes is not None else sorted(cell)
    return tuple((k, cell.get(k)) for k in keys)


def group_by_cell(
    results: List[TrialResult], axes: List[str] = None
) -> Dict[Tuple, List[TrialResult]]:
    grouped: Dict[Tuple, List[TrialResult]] = defaultdict(list)
    for r in results:
        if r.error:
            continue
        grouped[cell_key(r.cell, axes)].append(r)
    return dict(grouped)


# --- statistics (carried over from the prototype experiments) -----------------


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def two_proportion_p(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-sided two-proportion z-test p-value (pooled). Stdlib only.
    Returns 1.0 when undefined (empty group or zero variance)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def proportion(results: List[TrialResult], field: str, positive) -> Tuple[float, int]:
    """Fraction of results whose parsed[field] equals (or is in) `positive`."""
    vals = [r.parsed.get(field) for r in results if field in r.parsed]
    if not vals:
        return (0.0, 0)
    if callable(positive):
        hits = sum(1 for v in vals if positive(v))
    elif isinstance(positive, (set, list, tuple)):
        hits = sum(1 for v in vals if v in positive)
    else:
        hits = sum(1 for v in vals if v == positive)
    return (hits / len(vals), len(vals))


def mean(results: List[TrialResult], field: str) -> float:
    vals = [r.parsed[field] for r in results if isinstance(r.parsed.get(field), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0
