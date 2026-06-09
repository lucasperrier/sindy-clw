"""experiments.incomplete_library

Incomplete library ablation: library missing one or more true CLW terms.

Tests how gracefully recovery degrades when truth ∉ span(library).
Default: drop `(P*Z/S)*sin(C)` (the hardest rational term).

Runs the same η sweep as oracle/numerical experiments.

Outputs
-------
- outputs/tables/coef_recovery_incomplete_library_raw.csv
- outputs/tables/coef_recovery_incomplete_library.csv
- outputs/figures/fig_incomplete_library_overlay.png
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

from clw_model.clw import clw_rhs
from clw_model.data import simulate_short_bursts
from sindy_library.incomplete import make_library as make_incomplete
from clw_model.sindy_utils import (
    CLWParams,
    STATE_NAMES,
    count_nnz,
    enforce_constant_only_in_Cdot,
    fit_sindy,
    identified_rhs_from_model,
    integrate,
    select_model_by_score,
    vector_field_error,
)
from clw_model.coeff_recovery import build_true_coefficients_partial, coef_metrics
from clw_model.plotting import plot_error_vs_time, plot_timeseries_overlay_three


@dataclass(frozen=True)
class Config:
    params: CLWParams = CLWParams()

    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3
    eps_inv: float = 1e-8

    # Terms to drop from the physics-informed library
    drop_terms: tuple[str, ...] = ("(P*Z/S)*sin(C)",)

    eta_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    focus_etas: tuple[float, float] = (1e-3, 1e-1)

    x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)
    overlay_T: float = 20.0

    out_fig_dir: str = os.path.join("outputs", "figures")
    out_tab_dir: str = os.path.join("outputs", "tables")


def _seed_for(*, base_seed: int, eta: float) -> int:
    key = f"{int(base_seed)}|{eta:.16g}|incomplete_library".encode("utf-8")
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


def _fit_best_model(
    *, X_list: list[np.ndarray], dX_list: list[np.ndarray], library, cfg: Config
):
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


def _write_raw_csv(path: str, rows: list[dict]) -> None:
    fields = ["seed", "eta", "nnz", "coef_rel_l2", "tpr", "fpr", "exact_support", "vf_error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_summary_csv(path: str, raw_rows: list[dict], eta_list: tuple[float, ...]) -> None:
    by_eta: dict[float, list[dict]] = defaultdict(list)
    for r in raw_rows:
        by_eta[float(r["eta"])].append(r)

    fields = [
        "eta", "nnz_mean", "nnz_std", "rel_l2_mean", "rel_l2_std",
        "tpr_mean", "fpr_mean", "exact_support_frac", "vf_error_mean", "vf_error_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eta in eta_list:
            g = by_eta[float(eta)]
            w.writerow({
                "eta": float(eta),
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
    os.makedirs(cfg.out_fig_dir, exist_ok=True)
    os.makedirs(cfg.out_tab_dir, exist_ok=True)

    params = cfg.params.as_dict()

    lib = make_incomplete(eps_inv=float(cfg.eps_inv), drop_terms=cfg.drop_terms)
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients_partial(feature_names, params)

    raw_rows: list[dict] = []
    models_first_seed: dict[float, object] = {}

    for seed in cfg.seeds:
        X_clean, dX_clean = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=seed)
        sigma_x = _compute_sigma(X_clean)
        X_clean_all = np.concatenate(X_clean, axis=0)
        dX_clean_all = np.concatenate(dX_clean, axis=0)

        for eta in cfg.eta_list:
            rng = np.random.default_rng(_seed_for(base_seed=seed, eta=float(eta)))
            X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)

            model = _fit_best_model(
                X_list=X_noisy, dX_list=dX_clean, library=lib, cfg=cfg
            )

            Xi_hat = np.asarray(model.coefficients(), dtype=float)
            m = coef_metrics(Xi_hat=Xi_hat, Xi_true=Xi_true)
            vf_err = vector_field_error(model, X_clean_all, dX_clean_all)
            raw_rows.append({
                "seed": int(seed), "eta": float(eta),
                "nnz": int(m.nnz), "coef_rel_l2": float(m.rel_l2),
                "tpr": float(m.tpr), "fpr": float(m.fpr),
                "exact_support": bool(m.exact_support), "vf_error": float(vf_err),
            })

            if seed == cfg.seeds[0]:
                models_first_seed[float(eta)] = model

    raw_path = os.path.join(cfg.out_tab_dir, "coef_recovery_incomplete_library_raw.csv")
    _write_raw_csv(raw_path, raw_rows)

    tab_path = os.path.join(cfg.out_tab_dir, "coef_recovery_incomplete_library.csv")
    _write_summary_csv(tab_path, raw_rows, cfg.eta_list)

    # --- Figures (first seed only) ---
    rhs_true = lambda t, x: clw_rhs(t, x, params)
    t, X_true = integrate(rhs_true, dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))

    eta_low, eta_high = cfg.focus_etas
    _, X_hat_low = integrate(
        identified_rhs_from_model(models_first_seed[float(eta_low)]),
        dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float),
    )
    _, X_hat_high = integrate(
        identified_rhs_from_model(models_first_seed[float(eta_high)]),
        dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float),
    )

    dropped = ", ".join(cfg.drop_terms)
    plot_timeseries_overlay_three(
        t=t,
        X_true=X_true,
        X_hat_low=X_hat_low,
        X_hat_high=X_hat_high,
        eta_low=float(eta_low),
        eta_high=float(eta_high),
        outpath=os.path.join(cfg.out_fig_dir, "fig_incomplete_library_overlay.png"),
        title=f"CLW: incomplete library (dropped: {dropped})",
    )

    # Error vs time figure
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for eta in cfg.eta_list:
        rhs_hat = identified_rhs_from_model(models_first_seed[float(eta)])
        _, X_hat = integrate(rhs_hat, dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))
        curves[float(eta)] = (t, np.maximum(np.linalg.norm(X_hat - X_true, axis=1), 1e-16))

    plot_error_vs_time(
        curves=curves,
        outpath=os.path.join(cfg.out_fig_dir, "fig_incomplete_library_error_vs_time.png"),
        title=f"CLW: trajectory error (incomplete library, dropped: {dropped})",
    )

    print(f"Wrote raw table to {raw_path}")
    print(f"Wrote summary table to {tab_path}")
    print(f"Wrote figures to {cfg.out_fig_dir}")


if __name__ == "__main__":
    main()
