"""experiments.noise_state_oracle

Noise experiment (state noise + oracle derivatives). Physics-informed library only.

Noise protocol:
- add Gaussian noise to states X
- keep oracle derivatives dX clean

Outputs:
Figures (outputs/figures):
- fig_noise_state_error_vs_time.png
- fig_noise_state_timeseries_overlay.png        (shows η=0.001 and η=0.1)
- fig_noise_state_phase_space_eta0.001.png
- fig_noise_state_phase_space_eta0.1.png

Tables (outputs/tables):
- coef_recovery_state_oracle.csv                (over full η list)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: `python experiments/noise_state_oracle.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
from dataclasses import dataclass

import numpy as np

from clw import clw_rhs
from data import simulate_short_bursts

from sindy_library.physics_informed import make_library
from sindy_utils import CLWParams, STATE_NAMES, count_nnz, enforce_constant_only_in_Cdot, fit_sindy, identified_rhs_from_model, integrate, select_model_by_score
from plotting import plot_error_vs_time, plot_phase_space_psz, plot_timeseries_overlay_three
from coeff_recovery import build_true_coefficients, coef_metrics


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

    eta_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    focus_etas: tuple[float, float] = (1e-3, 1e-1)

    x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)
    error_T: float = 20.0
    overlay_T: float = 20.0

    out_fig_dir: str = os.path.join("outputs", "figures")
    out_tab_dir: str = os.path.join("outputs", "tables")


def _seed_for(*, base_seed: int, eta: float) -> int:
    key = f"{int(base_seed)}|{eta:.16g}|state_oracle".encode("utf-8")
    h = 2166136261
    for b in key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _compute_sigma(X_list: list[np.ndarray]) -> np.ndarray:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    return np.maximum(np.std(X, axis=0, ddof=0), 1e-12)


def _add_state_noise(X_list: list[np.ndarray], *, eta: float, sigma: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    sigma = np.asarray(sigma, dtype=float).reshape(1, 4)
    out: list[np.ndarray] = []
    for X in X_list:
        X = np.asarray(X, dtype=float)
        out.append(X + rng.normal(0.0, 1.0, size=X.shape) * (float(eta) * sigma))
    return out


def _fit_best_model(*, X_list: list[np.ndarray], dX_list: list[np.ndarray], cfg: Config):
    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))

    X = np.concatenate(X_list, axis=0)
    dX = np.concatenate(dX_list, axis=0)

    results: list[dict] = []
    for thr in cfg.thresholds:
        model = fit_sindy(X_list, dX_list, library=lib, dt=cfg.dt, threshold=float(thr))
        enforce_constant_only_in_Cdot(model)
        mse = float(np.mean((model.predict(X) - dX) ** 2))
        results.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})

    best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))
    return best["model"]


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_fig_dir, exist_ok=True)
    os.makedirs(cfg.out_tab_dir, exist_ok=True)

    params = cfg.params.as_dict()

    # clean bursts + oracle derivatives
    X_clean, dX_clean = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=cfg.seed)
    sigma_x = _compute_sigma(X_clean)

    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    models_by_eta: dict[float, object] = {}
    table_rows: list[dict] = []

    for eta in cfg.eta_list:
        rng = np.random.default_rng(_seed_for(base_seed=cfg.seed, eta=float(eta)))
        X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)
        model = _fit_best_model(X_list=X_noisy, dX_list=dX_clean, cfg=cfg)
        models_by_eta[float(eta)] = model

        m = coef_metrics(Xi_hat=np.asarray(model.coefficients(), dtype=float), Xi_true=Xi_true)
        table_rows.append({"eta": float(eta), "nnz": int(m.nnz), "coef_rel_l2": float(m.rel_l2)})

    # write coefficient table
    tab_path = os.path.join(cfg.out_tab_dir, "coef_recovery_state_oracle.csv")
    with open(tab_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eta", "nnz", "coef_rel_l2"])
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    # error vs time
    rhs_true = lambda t, x: clw_rhs(t, x, params)
    t_err, X_true = integrate(rhs_true, dt=cfg.dt, T=cfg.error_T, x0=np.asarray(cfg.x0, dtype=float))

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for eta in cfg.eta_list:
        rhs_hat = identified_rhs_from_model(models_by_eta[float(eta)])
        _, X_hat = integrate(rhs_hat, dt=cfg.dt, T=cfg.error_T, x0=np.asarray(cfg.x0, dtype=float))
        curves[float(eta)] = (t_err, np.maximum(np.linalg.norm(X_hat - X_true, axis=1), 1e-16))

    plot_error_vs_time(
        curves=curves,
        outpath=os.path.join(cfg.out_fig_dir, "fig_noise_state_error_vs_time.png"),
        title="CLW: trajectory error vs time (state noise; oracle derivatives)",
    )

    # overlay for eta in {0.001, 0.1}
    eta_low, eta_high = cfg.focus_etas
    t, X_ref = integrate(rhs_true, dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))
    _, X_low = integrate(identified_rhs_from_model(models_by_eta[float(eta_low)]), dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))
    _, X_high = integrate(identified_rhs_from_model(models_by_eta[float(eta_high)]), dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))

    plot_timeseries_overlay_three(
        t=t,
        X_true=X_ref,
        X_hat_low=X_low,
        X_hat_high=X_high,
        eta_low=float(eta_low),
        eta_high=float(eta_high),
        outpath=os.path.join(cfg.out_fig_dir, "fig_noise_state_timeseries_overlay.png"),
        title="CLW: time-series overlay (state noise; short horizon)",
    )

    # phase-space for the same etas
    for eta in cfg.focus_etas:
        tag = f"{float(eta):g}"
        _, X_hat = integrate(identified_rhs_from_model(models_by_eta[float(eta)]), dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))
        plot_phase_space_psz(
            X_a=X_ref,
            X_b=X_hat,
            outpath=os.path.join(cfg.out_fig_dir, f"fig_noise_state_phase_space_eta{tag}.png"),
            title=f"CLW: phase space (P,S,Z) — state noise, η={float(eta):g}",
            label_a="True",
            label_b="Identified",
        )

    print(f"Wrote figures to {cfg.out_fig_dir}")
    print(f"Wrote table to {tab_path}")


if __name__ == "__main__":
    main()
