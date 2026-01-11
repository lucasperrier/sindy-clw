"""main_experiment.py

Minimal SINDy demonstration for CLW, analogous to the Lorenz demo in
Brunton, Proctor, Kutz (2016).

Workflow:
1) Generate clean data from a known ODE (oracle derivatives).
2) Build a candidate library (includes constant term).
3) Perform sparse regression (SINDy / STLSQ) via threshold sweep.
4) Print + save identified coefficients.
5) Optional: simulate learned model for a short trajectory comparison.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps

from clw import clw_rhs
from data import simulate_short_bursts
from sindy_clw_lib import make_clw_library
from plot_validation import generate_validation_figures

STATE_NAMES = ["P", "S", "Z", "C"]


@dataclass
class Config:
    Gd: float = 2.0
    d: float = 2.0
    gz: float = 0.80

    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seed: int = 0

    thresholds: tuple[float, ...] = tuple(np.logspace(-3, -1, 21).astype(float).tolist())
    nnz_weight: float = 2e-3
    S_min: float = 0.2
    eps_inv: float = 1e-8

    outdir: str = "outputs"

    do_simulate: bool = False
    simulate_T: float = 2.0


def filter_small_S(X_list: list[np.ndarray], dX_list: list[np.ndarray], *, S_min: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X_out: list[np.ndarray] = []
    dX_out: list[np.ndarray] = []
    for X, dX in zip(X_list, dX_list):
        X = np.asarray(X, dtype=float)
        dX = np.asarray(dX, dtype=float)
        keep = X[:, 1] >= float(S_min)
        if np.any(keep):
            X_out.append(X[keep])
            dX_out.append(dX[keep])
    if not X_out:
        raise ValueError("All samples were filtered out by S_min.")
    return X_out, dX_out


def fit_sindy_clw(X_list: list[np.ndarray], dX_list: list[np.ndarray], *, dt: float, threshold: float, eps_inv: float) -> ps.SINDy:
    lib = make_clw_library(eps_inv=float(eps_inv))
    optimizer = ps.STLSQ(threshold=float(threshold), alpha=0.0, normalize_columns=False)
    model = ps.SINDy(feature_library=lib, optimizer=optimizer)
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    dX = np.concatenate([np.asarray(dx, dtype=float) for dx in dX_list], axis=0)
    model.fit(X, t=float(dt), x_dot=dX)

    # constant term only allowed in Cdot
    names = model.feature_library.get_feature_names(STATE_NAMES)
    if "1" in names:
        j = int(names.index("1"))
        coef = model.coefficients()
        coef[0, j] = 0.0
        coef[1, j] = 0.0
        coef[2, j] = 0.0
    return model


def count_nnz(model: ps.SINDy) -> int:
    return int(np.sum(np.abs(np.asarray(model.coefficients(), dtype=float)) > 0.0))


def derivative_mse(model: ps.SINDy, X_list: list[np.ndarray], dX_list: list[np.ndarray]) -> float:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    dX = np.concatenate([np.asarray(dx, dtype=float) for dx in dX_list], axis=0)
    dX_hat = np.asarray(model.predict(X), dtype=float)
    return float(np.mean((dX_hat - dX) ** 2))


def select_model_by_score(results: list[dict], *, nnz_weight: float) -> dict:
    best = None
    for r in results:
        score = float(np.log(float(r["mse"]) + 1e-30) + float(nnz_weight) * float(r["nnz"]))
        cand = dict(r)
        cand["score"] = score
        if best is None or score < float(best["score"]):
            best = cand
    if best is None:
        raise ValueError("No models were fit.")
    return best


def main() -> None:
    cfg = Config()
    params = {"Gd": float(cfg.Gd), "d": float(cfg.d), "gz": float(cfg.gz)}
    X_list, dX_list = simulate_short_bursts(params=params, n_traj=int(cfg.n_traj), T=float(cfg.burst_T), dt=float(cfg.dt), seed=int(cfg.seed))
    X_list, dX_list = filter_small_S(X_list, dX_list, S_min=float(cfg.S_min))

    results: list[dict] = []
    for thr in cfg.thresholds:
        m = fit_sindy_clw(X_list, dX_list, dt=float(cfg.dt), threshold=float(thr), eps_inv=float(cfg.eps_inv))
        results.append({"threshold": float(thr), "model": m, "mse": derivative_mse(m, X_list, dX_list), "nnz": count_nnz(m)})

    best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))
    model = best["model"]
    print("\n=== Identified SINDy Model (CLW) ===")
    print(f"selected_threshold={best['threshold']:.3e}, nnz={best['nnz']}, mse={best['mse']:.3e}, score={best['score']:.3f}")
    model.print()

    os.makedirs(cfg.outdir, exist_ok=True)
    save_path = os.path.join(cfg.outdir, "identified_model.npz")
    np.savez_compressed(
        save_path,
        threshold=np.asarray(float(best["threshold"]), dtype=float),
        feature_names=np.asarray(model.feature_library.get_feature_names(STATE_NAMES), dtype=object),
        coefficients=np.asarray(model.coefficients(), dtype=float),
    )
    print(f"\nSaved identified coefficients to: {save_path}")

    # Minimal validation figures (exactly two). These reuse the saved model
    # and do not change the identification procedure.
    generate_validation_figures(outdir=str(cfg.outdir))

    if bool(cfg.do_simulate):
        x0 = np.array([1.2, 1.0, 0.8, 0.5], dtype=float)
        t_eval = np.arange(0.0, float(cfg.simulate_T) + float(cfg.dt), float(cfg.dt))
        sol_true = solve_ivp(lambda t, x: clw_rhs(float(t), x, params), (float(t_eval[0]), float(t_eval[-1])), x0, t_eval=t_eval)
        if not sol_true.success:
            raise RuntimeError(f"True simulation failed: {sol_true.message}")
        X_true = np.asarray(sol_true.y.T, dtype=float)
        X_hat = np.asarray(model.simulate(x0, t_eval), dtype=float)
        err = np.linalg.norm(X_hat - X_true, axis=1)
        print(f"\nTrajectory compare (no plots): final L2 error = {float(err[-1]):.6g}")


if __name__ == "__main__":
    main()
