"""experiments.sample_size_ablation

Sample-size ablation: sweep n_traj at fixed moderate noise (eta=0.01)
with oracle derivatives.

Measures how data volume affects coefficient recovery.

Outputs
-------
- outputs/tables/coef_recovery_sample_size_raw.csv
- outputs/tables/coef_recovery_sample_size.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from data import simulate_short_bursts
from sindy_library.physics_informed import make_library
from sindy_utils import (
    CLWParams,
    STATE_NAMES,
    count_nnz,
    enforce_constant_only_in_Cdot,
    fit_sindy,
    select_model_by_score,
    vector_field_error,
)
from coeff_recovery import build_true_coefficients, coef_metrics


@dataclass(frozen=True)
class Config:
    params: CLWParams = CLWParams()

    dt: float = 0.01
    burst_T: float = 5.0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    # Fixed noise level
    eta: float = 0.01

    # Sweep over number of trajectories
    n_traj_list: tuple[int, ...] = (25, 50, 100, 250, 500)

    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3
    eps_inv: float = 1e-8

    out_tab_dir: str = os.path.join("outputs", "tables")


def _seed_for(*, base_seed: int, n_traj: int) -> int:
    key = f"{int(base_seed)}|{n_traj}|sample_size".encode("utf-8")
    h = 2166136261
    for b in key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _compute_sigma(X_list: list[np.ndarray]) -> np.ndarray:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    return np.maximum(np.std(X, axis=0, ddof=0), 1e-12)


def _add_state_noise(
    X_list: list[np.ndarray], *, eta: float, sigma: np.ndarray, rng: np.random.Generator
) -> list[np.ndarray]:
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


def _write_raw_csv(path: str, rows: list[dict]) -> None:
    fields = ["seed", "n_traj", "nnz", "coef_rel_l2", "tpr", "fpr", "exact_support", "vf_error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_summary_csv(path: str, raw_rows: list[dict], n_traj_list: tuple[int, ...]) -> None:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in raw_rows:
        by_n[int(r["n_traj"])].append(r)

    fields = [
        "n_traj", "nnz_mean", "nnz_std", "rel_l2_mean", "rel_l2_std",
        "tpr_mean", "fpr_mean", "exact_support_frac", "vf_error_mean", "vf_error_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n in n_traj_list:
            g = by_n[int(n)]
            w.writerow({
                "n_traj": int(n),
                "nnz_mean": float(np.mean([r["nnz"] for r in g])),
                "nnz_std": float(np.std([r["nnz"] for r in g])),
                "rel_l2_mean": float(np.mean([r["coef_rel_l2"] for r in g])),
                "rel_l2_std": float(np.std([r["coef_rel_l2"] for r in g])),
                "tpr_mean": float(np.mean([r["tpr"] for r in g])),
                "fpr_mean": float(np.mean([r["fpr"] for r in g])),
                "exact_support_frac": float(np.mean([float(r["exact_support"]) for r in g])),
                "vf_error_mean": float(np.mean([r["vf_error"] for r in g])),
                "vf_error_std": float(np.std([r["vf_error"] for r in g])),
            })


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_tab_dir, exist_ok=True)

    params = cfg.params.as_dict()

    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    raw_rows: list[dict] = []

    for seed in cfg.seeds:
        # Generate maximum number of trajectories, then subsample
        max_n = max(cfg.n_traj_list)
        X_clean, dX_clean = simulate_short_bursts(
            params, n_traj=max_n, T=cfg.burst_T, dt=cfg.dt, seed=seed
        )
        sigma_x = _compute_sigma(X_clean)

        for n_traj in cfg.n_traj_list:
            rng = np.random.default_rng(_seed_for(base_seed=seed, n_traj=n_traj))

            # Subsample trajectories
            X_sub = X_clean[:n_traj]
            dX_sub = dX_clean[:n_traj]

            X_noisy = _add_state_noise(X_sub, eta=float(cfg.eta), sigma=sigma_x, rng=rng)

            model = _fit_best_model(X_list=X_noisy, dX_list=dX_sub, cfg=cfg)

            X_all = np.concatenate(X_sub, axis=0)
            dX_all = np.concatenate(dX_sub, axis=0)

            Xi_hat = np.asarray(model.coefficients(), dtype=float)
            m = coef_metrics(Xi_hat=Xi_hat, Xi_true=Xi_true)
            vf_err = vector_field_error(model, X_all, dX_all)
            raw_rows.append({
                "seed": int(seed), "n_traj": int(n_traj),
                "nnz": int(m.nnz), "coef_rel_l2": float(m.rel_l2),
                "tpr": float(m.tpr), "fpr": float(m.fpr),
                "exact_support": bool(m.exact_support), "vf_error": float(vf_err),
            })

    raw_path = os.path.join(cfg.out_tab_dir, "coef_recovery_sample_size_raw.csv")
    _write_raw_csv(raw_path, raw_rows)

    tab_path = os.path.join(cfg.out_tab_dir, "coef_recovery_sample_size.csv")
    _write_summary_csv(tab_path, raw_rows, cfg.n_traj_list)

    print(f"Wrote raw table to {raw_path}")
    print(f"Wrote summary table to {tab_path}")


if __name__ == "__main__":
    main()
