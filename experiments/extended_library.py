"""experiments.extended_library

Extended library experiment: physics-informed vs extended library.

Requirements:
- same data as poster baseline (oracle derivatives, no noise)
- compare libraries:
  - physics-informed
  - extended

Outputs:
Tables (outputs/tables):
- coef_recovery_extended_library.csv

Optional visual:
- fig_extended_library_overlay.png (short-horizon overlay comparison)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: `python experiments/extended_library.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
from dataclasses import dataclass

import numpy as np

from clw import clw_rhs
from data import simulate_short_bursts

from sindy_library.physics_informed import make_library as make_phys
from sindy_library.extended import make_library as make_ext

from sindy_utils import CLWParams, STATE_NAMES, count_nnz, enforce_constant_only_in_Cdot, fit_sindy, identified_rhs_from_model, integrate, select_model_by_score
from coeff_recovery import build_true_coefficients, coef_metrics
from plotting import plot_timeseries_overlay_two


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

    extended_degree: int = 2

    x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)
    T_short: float = 8.0

    out_tab_dir: str = os.path.join("outputs", "tables")
    out_fig_dir: str = os.path.join("outputs", "figures")


def _fit_best_model(*, X_list: list[np.ndarray], dX_list: list[np.ndarray], library, cfg: Config):
    library.fit(np.zeros((1, 4)))

    X = np.concatenate(X_list, axis=0)
    dX = np.concatenate(dX_list, axis=0)

    results: list[dict] = []
    for thr in cfg.thresholds:
        model = fit_sindy(X_list, dX_list, library=library, dt=cfg.dt, threshold=float(thr))
        enforce_constant_only_in_Cdot(model)
        mse = float(np.mean((model.predict(X) - dX) ** 2))
        results.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})

    best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))
    return best["model"]


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_tab_dir, exist_ok=True)
    os.makedirs(cfg.out_fig_dir, exist_ok=True)

    params = cfg.params.as_dict()

    X_list, dX_list = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=cfg.seed)

    lib_phys = make_phys(eps_inv=float(cfg.eps_inv))
    lib_ext = make_ext(eps_inv=float(cfg.eps_inv), degree=int(cfg.extended_degree))

    model_phys = _fit_best_model(X_list=X_list, dX_list=dX_list, library=lib_phys, cfg=cfg)
    model_ext = _fit_best_model(X_list=X_list, dX_list=dX_list, library=lib_ext, cfg=cfg)

    # build truth aligned to each library
    lib_phys.fit(np.zeros((1, 4)))
    lib_ext.fit(np.zeros((1, 4)))
    Xi_true_phys = build_true_coefficients(lib_phys.get_feature_names(STATE_NAMES), params)
    Xi_true_ext = build_true_coefficients(lib_ext.get_feature_names(STATE_NAMES), params)

    m_phys = coef_metrics(Xi_hat=np.asarray(model_phys.coefficients(), dtype=float), Xi_true=Xi_true_phys)
    m_ext = coef_metrics(Xi_hat=np.asarray(model_ext.coefficients(), dtype=float), Xi_true=Xi_true_ext)

    outpath = os.path.join(cfg.out_tab_dir, "coef_recovery_extended_library.csv")
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "nnz", "coef_rel_l2"])
        w.writeheader()
        w.writerow({"case": "physics_informed", "nnz": int(m_phys.nnz), "coef_rel_l2": float(m_phys.rel_l2)})
        w.writerow({"case": f"extended_degree_{cfg.extended_degree}", "nnz": int(m_ext.nnz), "coef_rel_l2": float(m_ext.rel_l2)})

    # optional short-horizon visual contrast (true vs extended model)
    rhs_true = lambda t, x: clw_rhs(t, x, params)
    t, X_true = integrate(rhs_true, dt=cfg.dt, T=cfg.T_short, x0=np.asarray(cfg.x0, dtype=float))
    _, X_hat = integrate(identified_rhs_from_model(model_ext), dt=cfg.dt, T=cfg.T_short, x0=np.asarray(cfg.x0, dtype=float))
    plot_timeseries_overlay_two(
        t=t,
        X_true=X_true,
        X_hat=X_hat,
        outpath=os.path.join(cfg.out_fig_dir, "fig_extended_library_overlay.png"),
        title=f"CLW: overlay using extended library (degree={cfg.extended_degree})",
        label_hat="Identified (extended)",
    )

    print(f"Wrote table to {outpath}")


if __name__ == "__main__":
    main()
