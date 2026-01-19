"""sindy_utils.py

Shared utilities used by all experiments.

Design goals:
- extremely small surface area
- no abstraction layers
- PySINDy + STLSQ only
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps

STATE_NAMES = ["P", "S", "Z", "C"]


@dataclass(frozen=True)
class CLWParams:
    Gd: float = 2.0
    d: float = 2.0
    gz: float = 0.80

    def as_dict(self) -> dict[str, float]:
        return {"Gd": float(self.Gd), "d": float(self.d), "gz": float(self.gz)}


def integrate(rhs: callable, *, dt: float, T: float, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = float(dt)
    T = float(T)
    t_eval = np.arange(0.0, T + dt, dt, dtype=float)
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), np.asarray(x0, dtype=float), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.asarray(sol.t, dtype=float), np.asarray(sol.y.T, dtype=float)


def fit_sindy(
    X_list: list[np.ndarray],
    dX_list: list[np.ndarray],
    *,
    library: ps.FeatureLibrary,
    dt: float,
    threshold: float,
) -> ps.SINDy:
    opt = ps.STLSQ(threshold=float(threshold), alpha=0.0, normalize_columns=False)
    model = ps.SINDy(feature_library=library, optimizer=opt)

    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    dX = np.concatenate([np.asarray(dx, dtype=float) for dx in dX_list], axis=0)
    model.fit(X, t=float(dt), x_dot=dX)
    return model


def enforce_constant_only_in_Cdot(model: ps.SINDy) -> None:
    names = model.feature_library.get_feature_names(STATE_NAMES)
    if "1" not in names:
        return
    j = int(names.index("1"))
    coef = model.coefficients()
    coef[0, j] = 0.0
    coef[1, j] = 0.0
    coef[2, j] = 0.0


def count_nnz(model: ps.SINDy, *, nz_tol: float = 0.0) -> int:
    Xi = np.asarray(model.coefficients(), dtype=float)
    return int(np.sum(np.abs(Xi) > float(nz_tol)))


def select_model_by_score(results: list[dict], *, nnz_weight: float) -> dict:
    best: dict | None = None
    for r in results:
        mse = float(r["mse"])
        nnz = float(r["nnz"])
        score = float(np.log(mse + 1e-30) + float(nnz_weight) * nnz)
        cand = dict(r)
        cand["score"] = score
        if best is None or score < float(best["score"]):
            best = cand
    if best is None:
        raise ValueError("No models were fit.")
    return best


def identified_rhs_from_model(model: ps.SINDy) -> callable:
    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(1, 4)
        return np.asarray(model.predict(x), dtype=float).reshape(4,)

    return rhs
