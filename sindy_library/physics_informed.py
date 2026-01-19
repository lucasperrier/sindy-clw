"""sindy_library.physics_informed

CLW physics-informed candidate library used throughout the experiments.

This is the authoritative version of the old `make_clw_library`.
"""

from __future__ import annotations

import numpy as np
import pysindy as ps


def _safe_inv(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    tiny = float(eps)
    denom = np.where(np.abs(x) < tiny, np.sign(x) * tiny + (x == 0) * tiny, x)
    return 1.0 / denom


def _col(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError(f"Expected a single feature column, got shape {x.shape}")


def make_library(*, eps_inv: float = 1e-8) -> ps.CustomLibrary:
    """Physics-informed CLW library.

    State ordering: [P, S, Z, C].
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

    funcs = [one, P, S, Z, cosC, sinC, ZS_cosC, ZP_cosC, PS_cosC, PZ_over_S_sinC]
    names = [
        lambda *_: "1",
        lambda *_: "P",
        lambda *_: "S",
        lambda *_: "Z",
        lambda *_: "cos(C)",
        lambda *_: "sin(C)",
        lambda *_: "Z*S*cos(C)",
        lambda *_: "Z*P*cos(C)",
        lambda *_: "P*S*cos(C)",
        lambda *_: "(P*Z/S)*sin(C)",
    ]

    return ps.CustomLibrary(library_functions=funcs, function_names=names)
