"""experiments.poster_baseline

Poster experiment (gold reference).

Requirements:
- CLW system
- oracle derivatives (no noise, no numerical differentiation)
- physics-informed library only

Outputs (to outputs/figures):
- fig_poster_timeseries_overlay.png      (short horizon overlay: true vs identified)
- fig_poster_phase_space_chaos.png       (long-horizon phase space with C-perturbation)

This script intentionally produces *no* noise sweeps and *no* coefficient tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: `python experiments/poster_baseline.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
from dataclasses import dataclass

import numpy as np

from clw import clw_rhs
from data import simulate_short_bursts

from sindy_library.physics_informed import make_library
from sindy_utils import CLWParams, STATE_NAMES, fit_sindy, integrate, enforce_constant_only_in_Cdot, identified_rhs_from_model
from plotting import plot_phase_space_psz, plot_timeseries_overlay_two


@dataclass(frozen=True)
class Config:
    params: CLWParams = CLWParams()

    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seed: int = 0

    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3
    eps_inv: float = 1e-8

    x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)
    T_short: float = 8.0
    T_long_phase: float = 100.0
    delta_C: float = 1e-6

    out_fig_dir: str = os.path.join("outputs", "figures")


def _select_best_model(*, X_list: list[np.ndarray], dX_list: list[np.ndarray], cfg: Config):
    import pysindy as ps

    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))

    results: list[dict] = []
    X = np.concatenate(X_list, axis=0)
    dX = np.concatenate(dX_list, axis=0)

    for thr in cfg.thresholds:
        model = fit_sindy(X_list, dX_list, library=lib, dt=cfg.dt, threshold=float(thr))
        enforce_constant_only_in_Cdot(model)
        mse = float(np.mean((model.predict(X) - dX) ** 2))
        nnz = int(np.sum(np.abs(model.coefficients()) > 0.0))
        score = float(np.log(mse + 1e-30) + float(cfg.nnz_weight) * float(nnz))
        results.append({"threshold": float(thr), "mse": mse, "nnz": nnz, "score": score, "model": model})

    best = min(results, key=lambda r: float(r["score"]))
    return best["model"]


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_fig_dir, exist_ok=True)

    params = cfg.params.as_dict()

    # training data (oracle derivatives)
    X_list, dX_list = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=cfg.seed)

    model = _select_best_model(X_list=X_list, dX_list=dX_list, cfg=cfg)
    rhs_hat = identified_rhs_from_model(model)
    rhs_true = lambda t, x: clw_rhs(t, x, params)

    # Figure 1: short-horizon overlay
    t, X_true = integrate(rhs_true, dt=cfg.dt, T=cfg.T_short, x0=np.asarray(cfg.x0, dtype=float))
    _, X_hat = integrate(rhs_hat, dt=cfg.dt, T=cfg.T_short, x0=np.asarray(cfg.x0, dtype=float))
    plot_timeseries_overlay_two(
        t=t,
        X_true=X_true,
        X_hat=X_hat,
        outpath=os.path.join(cfg.out_fig_dir, "fig_poster_timeseries_overlay.png"),
        title="CLW poster baseline: short-horizon overlay (true vs identified)",
    )

    # Figure 2: long-horizon chaos sensitivity (perturb C only)
    x0 = np.asarray(cfg.x0, dtype=float)
    x0_pert = x0.copy()
    x0_pert[3] += float(cfg.delta_C)

    _, X_a = integrate(rhs_true, dt=cfg.dt, T=cfg.T_long_phase, x0=x0)
    _, X_b = integrate(rhs_true, dt=cfg.dt, T=cfg.T_long_phase, x0=x0_pert)

    plot_phase_space_psz(
        X_a=X_a,
        X_b=X_b,
        outpath=os.path.join(cfg.out_fig_dir, "fig_poster_phase_space_chaos.png"),
        title=f"CLW poster baseline: chaos sensitivity (ΔC={cfg.delta_C:.0e})",
        label_a="CLW (x0)",
        label_b=f"CLW (x0 with C+ΔC, ΔC={cfg.delta_C:.0e})",
    )

    print(f"Wrote poster figures to: {cfg.out_fig_dir}")


if __name__ == "__main__":
    main()
