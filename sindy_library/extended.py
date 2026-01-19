"""sindy_library.extended

Extended / less physics-informed CLW library.

This preserves representability of the true CLW dynamics by building products of
basis terms up to a chosen degree.

Used only in the extended-library comparison experiment.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pysindy as ps

from .physics_informed import _col, _safe_inv


def make_library(*, eps_inv: float = 1e-8, degree: int = 2) -> ps.CustomLibrary:
    deg = int(degree)
    if deg < 1:
        raise ValueError("degree must be >= 1")

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

    base_items: list[tuple[callable, str]] = [
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

    product_terms = [(f, n) for (f, n) in base_items if n != "1"]

    out_funcs: list[callable] = []
    out_names: list[callable] = []

    def _add_feature(func: callable, name: str) -> None:
        out_funcs.append(func)
        out_names.append(lambda *_args, _name=name: _name)

    # include base
    for f, name in base_items:
        _add_feature(f, str(name))

    for k in range(2, deg + 1):
        for combo in combinations_with_replacement(product_terms, k):
            funcs_k = [f for (f, _n) in combo]
            names_k = [str(_n) for (_f, _n) in combo]
            name = "*".join(names_k)

            def prod_feature(p, s, z, c, _funcs=tuple(funcs_k)):
                vals = None
                for ff in _funcs:
                    v = np.asarray(ff(p, s, z, c), dtype=float)
                    vals = v if vals is None else vals * v
                return np.asarray(vals, dtype=float)

            _add_feature(prod_feature, name)

    # deduplicate by name
    dedup_funcs: list[callable] = []
    dedup_names: list[callable] = []
    seen: set[str] = set()
    for f, nf in zip(out_funcs, out_names):
        n = str(nf(None))
        if n in seen:
            continue
        seen.add(n)
        dedup_funcs.append(f)
        dedup_names.append(nf)

    return ps.CustomLibrary(library_functions=dedup_funcs, function_names=dedup_names)
