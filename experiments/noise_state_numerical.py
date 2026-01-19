"""experiments.noise_state_numerical

Noise experiment (state noise + numerical derivatives). Physics-informed library only.

Scientific regime:
- add Gaussian noise to X
- estimate dX numerically from X_noisy

Outputs:
Tables (outputs/tables):
- coef_recovery_state_numerical.csv

No trajectory plots (by design).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: `python experiments/noise_state_numerical.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
from dataclasses import dataclass

import numpy as np

from data import simulate_short_bursts

from sindy_library.physics_informed import make_library
from sindy_utils import CLWParams, STATE_NAMES, count_nnz, enforce_constant_only_in_Cdot, fit_sindy, select_model_by_score
from coeff_recovery import build_true_coefficients, coef_metrics


@dataclass(frozen=True)
class Config:
    params: CLWParams = CLWParams()

    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seed: int = 0

    eta_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3
    eps_inv: float = 1e-8

    out_tab_dir: str = os.path.join("outputs", "tables")


def _seed_for(*, base_seed: int, eta: float) -> int:
    key = f"{int(base_seed)}|{eta:.16g}|state_numerical".encode("utf-8")
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


def _estimate_derivatives_fd(X_list: list[np.ndarray], *, dt: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for X in X_list:
        X = np.asarray(X, dtype=float)
        out.append(np.asarray(np.gradient(X, float(dt), axis=0, edge_order=2), dtype=float))
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
    os.makedirs(cfg.out_tab_dir, exist_ok=True)

    params = cfg.params.as_dict()

    # clean bursts (we'll contaminate measurements and then estimate derivatives)
    X_clean, _dX_oracle = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=cfg.seed)
    sigma_x = _compute_sigma(X_clean)

    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    rows: list[dict] = []
    for eta in cfg.eta_list:
        rng = np.random.default_rng(_seed_for(base_seed=cfg.seed, eta=float(eta)))
        X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)
        dX_est = _estimate_derivatives_fd(X_noisy, dt=cfg.dt)

        model = _fit_best_model(X_list=X_noisy, dX_list=dX_est, cfg=cfg)
        m = coef_metrics(Xi_hat=np.asarray(model.coefficients(), dtype=float), Xi_true=Xi_true)
        rows.append({"eta": float(eta), "nnz": int(m.nnz), "coef_rel_l2": float(m.rel_l2)})

    outpath = os.path.join(cfg.out_tab_dir, "coef_recovery_state_numerical.csv")
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eta", "nnz", "coef_rel_l2"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote table to {outpath}")


if __name__ == "__main__":
    main()
