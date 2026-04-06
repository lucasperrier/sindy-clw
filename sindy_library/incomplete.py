"""sindy_library.incomplete

Incomplete CLW library for ablation: missing one or more true terms.

Default configuration drops `(P*Z/S)*sin(C)` — the hardest rational term
in the CLW system. This tests how gracefully recovery degrades when
truth ∉ span(library).
"""

from __future__ import annotations

import numpy as np
import pysindy as ps

from .physics_informed import _col, _safe_inv


def make_library(*, eps_inv: float = 1e-8, drop_terms: tuple[str, ...] = ("(P*Z/S)*sin(C)",)) -> ps.CustomLibrary:
    """Physics-informed CLW library with specified terms removed.

    Args:
        eps_inv: safety epsilon (only relevant if rational term is kept).
        drop_terms: tuple of feature names to exclude from the library.

    Returns:
        A CustomLibrary missing the specified terms.
    """

    def one(p, s, z, c):
        p = _col(p)
        return np.ones_like(p)

    def P(p, s, z, c):
        return _col(p)

    def S(p, s, z, c):
        return _col(s)

    def Z(p, s, z, c):
        return _col(z)

    def cosC(p, s, z, c):
        return np.cos(_col(c))

    def sinC(p, s, z, c):
        return np.sin(_col(c))

    def ZS_cosC(p, s, z, c):
        s = _col(s)
        z = _col(z)
        c = _col(c)
        return z * s * np.cos(c)

    def ZP_cosC(p, s, z, c):
        p = _col(p)
        z = _col(z)
        c = _col(c)
        return z * p * np.cos(c)

    def PS_cosC(p, s, z, c):
        p = _col(p)
        s = _col(s)
        c = _col(c)
        return p * s * np.cos(c)

    def PZ_over_S_sinC(p, s, z, c):
        p = _col(p)
        s = _col(s)
        z = _col(z)
        c = _col(c)
        return (p * z * _safe_inv(s, float(eps_inv))) * np.sin(c)

    all_items: list[tuple[callable, str]] = [
        (one, "1"),
        (P, "P"),
        (S, "S"),
        (Z, "Z"),
        (cosC, "cos(C)"),
        (sinC, "sin(C)"),
        (ZS_cosC, "Z*S*cos(C)"),
        (ZP_cosC, "Z*P*cos(C)"),
        (PS_cosC, "P*S*cos(C)"),
        (PZ_over_S_sinC, "(P*Z/S)*sin(C)"),
    ]

    drop_set = set(drop_terms)
    kept = [(f, n) for f, n in all_items if n not in drop_set]

    funcs = [f for f, _ in kept]
    names = [lambda *_args, _name=n: _name for _, n in kept]

    return ps.CustomLibrary(library_functions=funcs, function_names=names)
