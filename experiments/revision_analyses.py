"""experiments.revision_analyses

Additional analyses requested during peer review:

1. False-positive term frequency table (extended library):
   Which non-true terms are most frequently selected across seeds and noise levels?

2. Threshold-path diagnostics:
   How do nnz, coefficient error, and score vary across STLSQ thresholds?

3. Validation-based model selection comparison:
   Does a held-out validation split change the selected model?

Outputs
-------
- outputs/tables/false_positive_terms.csv
- outputs/tables/threshold_path.csv
- outputs/tables/validation_comparison.csv
- outputs/figures/fig_threshold_path.pdf
- outputs/figures/fig_false_positive_heatmap.pdf
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clw import clw_rhs
from data import simulate_short_bursts
from sindy_library.extended import make_library as make_ext
from sindy_library.physics_informed import make_library as make_phys
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
    n_traj: int = 250
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3
    eps_inv: float = 1e-8
    eta_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    out_fig_dir: str = os.path.join("outputs", "figures")
    out_tab_dir: str = os.path.join("outputs", "tables")


def _seed_for(*, base_seed: int, eta: float) -> int:
    key = f"{int(base_seed)}|{eta:.16g}|extended_noise".encode("utf-8")
    h = 2166136261
    for b in key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _compute_sigma(X_list: list[np.ndarray]) -> np.ndarray:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    return np.maximum(np.std(X, axis=0, ddof=0), 1e-12)


def _add_state_noise(X_list, *, eta, sigma, rng):
    sigma = np.asarray(sigma, dtype=float).reshape(1, 4)
    return [X + rng.normal(0.0, 1.0, size=X.shape) * (float(eta) * sigma) for X in X_list]


# ── Analysis 1: False-positive term frequency ──────────────────────
def false_positive_analysis(cfg: Config) -> list[dict]:
    """For the extended library, identify WHICH non-true terms are selected."""
    params = cfg.params.as_dict()
    lib = make_ext(eps_inv=float(cfg.eps_inv), degree=2)
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    true_support = np.abs(Xi_true) > 0  # (4, 46)

    # Count how often each (equation, feature) pair is falsely selected
    fp_counts = np.zeros_like(Xi_true, dtype=int)
    total_runs = 0

    raw_rows = []

    for seed in cfg.seeds:
        X_clean, dX_clean = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=seed)
        sigma_x = _compute_sigma(X_clean)

        for eta in cfg.eta_list:
            rng = np.random.default_rng(_seed_for(base_seed=seed, eta=float(eta)))
            X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)

            # Fit best model (same as extended_library_noise.py)
            X = np.concatenate(X_noisy, axis=0)
            dX = np.concatenate(dX_clean, axis=0)
            results = []
            for thr in cfg.thresholds:
                model = fit_sindy(X_noisy, dX_clean, library=lib, dt=cfg.dt, threshold=float(thr))
                enforce_constant_only_in_Cdot(model)
                mse = float(np.mean((model.predict(X) - dX) ** 2))
                results.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})

            best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))
            model = best["model"]
            Xi_hat = np.asarray(model.coefficients(), dtype=float)

            hat_nz = np.abs(Xi_hat) > 0
            fp_mask = hat_nz & (~true_support)
            fp_counts += fp_mask.astype(int)
            total_runs += 1

            # Record individual false positive terms
            for eq_idx in range(4):
                for feat_idx in range(len(feature_names)):
                    if fp_mask[eq_idx, feat_idx]:
                        raw_rows.append({
                            "seed": seed, "eta": eta,
                            "equation": STATE_NAMES[eq_idx],
                            "term": feature_names[feat_idx],
                            "coefficient": float(Xi_hat[eq_idx, feat_idx]),
                        })

    # Build frequency table
    freq_rows = []
    for eq_idx in range(4):
        for feat_idx in range(len(feature_names)):
            if true_support[eq_idx, feat_idx]:
                continue  # Skip true terms
            count = int(fp_counts[eq_idx, feat_idx])
            if count > 0:
                freq_rows.append({
                    "equation": f"d{STATE_NAMES[eq_idx]}/dt",
                    "term": feature_names[feat_idx],
                    "frequency": count,
                    "fraction": round(count / total_runs, 3),
                })

    freq_rows.sort(key=lambda r: -r["frequency"])
    return freq_rows, raw_rows, fp_counts, total_runs, feature_names


# ── Analysis 2: Threshold-path diagnostics ─────────────────────────
def threshold_path_analysis(cfg: Config, eta: float = 0.01, seed: int = 0) -> list[dict]:
    """Show how nnz, coef_error, mse, and score change across thresholds."""
    params = cfg.params.as_dict()
    lib = make_phys(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    X_clean, dX_clean = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=seed)
    sigma_x = _compute_sigma(X_clean)

    rng_key = f"{seed}|{eta:.16g}|state_oracle".encode("utf-8")
    h = 2166136261
    for b in rng_key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    rng = np.random.default_rng(int(h))
    X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)

    X_all = np.concatenate(X_noisy, axis=0)
    dX_all = np.concatenate(dX_clean, axis=0)

    # Also prepare a held-out validation set
    n_train = int(len(X_noisy) * 0.8)
    X_train, dX_train = X_noisy[:n_train], dX_clean[:n_train]
    X_val, dX_val = X_noisy[n_train:], dX_clean[n_train:]
    X_val_all = np.concatenate(X_val, axis=0)
    dX_val_all = np.concatenate(dX_val, axis=0)

    rows = []
    for thr in cfg.thresholds:
        # Full-data fit
        model = fit_sindy(X_noisy, dX_clean, library=lib, dt=cfg.dt, threshold=float(thr))
        enforce_constant_only_in_Cdot(model)
        Xi_hat = np.asarray(model.coefficients(), dtype=float)
        m = coef_metrics(Xi_hat=Xi_hat, Xi_true=Xi_true)
        mse_train = float(np.mean((model.predict(X_all) - dX_all) ** 2))
        score = float(np.log(mse_train + 1e-30) + cfg.nnz_weight * m.nnz)

        # Validation-based fit
        model_cv = fit_sindy(X_train, dX_train, library=lib, dt=cfg.dt, threshold=float(thr))
        enforce_constant_only_in_Cdot(model_cv)
        mse_val = float(np.mean((model_cv.predict(X_val_all) - dX_val_all) ** 2))
        score_cv = float(np.log(mse_val + 1e-30) + cfg.nnz_weight * count_nnz(model_cv))

        rows.append({
            "threshold": float(thr),
            "nnz": m.nnz,
            "coef_rel_l2": m.rel_l2,
            "tpr": m.tpr,
            "fpr": m.fpr,
            "mse_train": mse_train,
            "score_train": score,
            "mse_val": mse_val,
            "score_val": score_cv,
        })

    return rows


# ── Analysis 3: Validation comparison ──────────────────────────────
def validation_comparison(cfg: Config) -> list[dict]:
    """Compare train-MSE vs held-out-val-MSE model selection across noise levels."""
    params = cfg.params.as_dict()
    lib = make_phys(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    rows = []
    for seed in cfg.seeds:
        X_clean, dX_clean = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=seed)
        sigma_x = _compute_sigma(X_clean)
        X_clean_all = np.concatenate(X_clean, axis=0)
        dX_clean_all = np.concatenate(dX_clean, axis=0)

        n_train = int(len(X_clean) * 0.8)

        for eta in cfg.eta_list:
            rng_key = f"{seed}|{eta:.16g}|state_oracle".encode("utf-8")
            h = 2166136261
            for b in rng_key:
                h ^= b
                h = (h * 16777619) & 0xFFFFFFFF
            rng = np.random.default_rng(int(h))
            X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)

            # --- Train-MSE selection (original) ---
            X_all = np.concatenate(X_noisy, axis=0)
            dX_all = np.concatenate(dX_clean, axis=0)
            results_train = []
            for thr in cfg.thresholds:
                model = fit_sindy(X_noisy, dX_clean, library=lib, dt=cfg.dt, threshold=float(thr))
                enforce_constant_only_in_Cdot(model)
                mse = float(np.mean((model.predict(X_all) - dX_all) ** 2))
                results_train.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})
            best_train = select_model_by_score(results_train, nnz_weight=float(cfg.nnz_weight))
            m_train = coef_metrics(Xi_hat=np.asarray(best_train["model"].coefficients(), dtype=float), Xi_true=Xi_true)

            # --- Validation-MSE selection ---
            X_tr, dX_tr = X_noisy[:n_train], dX_clean[:n_train]
            X_va, dX_va = X_noisy[n_train:], dX_clean[n_train:]
            X_va_all = np.concatenate(X_va, axis=0)
            dX_va_all = np.concatenate(dX_va, axis=0)

            results_val = []
            for thr in cfg.thresholds:
                model = fit_sindy(X_tr, dX_tr, library=lib, dt=cfg.dt, threshold=float(thr))
                enforce_constant_only_in_Cdot(model)
                mse_v = float(np.mean((model.predict(X_va_all) - dX_va_all) ** 2))
                results_val.append({"threshold": float(thr), "mse": mse_v, "nnz": count_nnz(model), "model": model})
            best_val = select_model_by_score(results_val, nnz_weight=float(cfg.nnz_weight))
            m_val = coef_metrics(Xi_hat=np.asarray(best_val["model"].coefficients(), dtype=float), Xi_true=Xi_true)

            rows.append({
                "seed": seed, "eta": eta,
                "nnz_train": m_train.nnz, "rel_l2_train": round(m_train.rel_l2, 6),
                "tpr_train": m_train.tpr, "fpr_train": m_train.fpr,
                "nnz_val": m_val.nnz, "rel_l2_val": round(m_val.rel_l2, 6),
                "tpr_val": m_val.tpr, "fpr_val": m_val.fpr,
                "threshold_train": best_train["threshold"],
                "threshold_val": best_val["threshold"],
            })

    return rows


# ── Plotting ───────────────────────────────────────────────────────
def plot_threshold_path(rows: list[dict], outpath: str) -> None:
    thresholds = [r["threshold"] for r in rows]
    nnz = [r["nnz"] for r in rows]
    coef_err = [r["coef_rel_l2"] for r in rows]
    score_train = [r["score_train"] for r in rows]
    score_val = [r["score_val"] for r in rows]

    best_train_idx = int(np.argmin(score_train))
    best_val_idx = int(np.argmin(score_val))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.semilogx(thresholds, nnz, "o-", markersize=4, color="tab:blue")
    ax.axhline(8, color="gray", linestyle=":", alpha=0.7, label="$k^\\star = 8$")
    ax.axvline(thresholds[best_train_idx], color="tab:orange", linestyle="--", alpha=0.7, label="Selected (train)")
    ax.axvline(thresholds[best_val_idx], color="tab:green", linestyle="--", alpha=0.7, label="Selected (val)")
    ax.set_xlabel("STLSQ threshold $\\lambda$")
    ax.set_ylabel("nnz")
    ax.set_title("(a) Sparsity vs threshold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.loglog(thresholds, coef_err, "o-", markersize=4, color="tab:blue")
    ax.axvline(thresholds[best_train_idx], color="tab:orange", linestyle="--", alpha=0.7, label="Selected (train)")
    ax.axvline(thresholds[best_val_idx], color="tab:green", linestyle="--", alpha=0.7, label="Selected (val)")
    ax.set_xlabel("STLSQ threshold $\\lambda$")
    ax.set_ylabel("$\\epsilon_{\\mathrm{coef}}$")
    ax.set_title("(b) Coefficient error vs threshold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2, which="both")

    ax = axes[2]
    ax.semilogx(thresholds, score_train, "o-", markersize=4, color="tab:orange", label="Train score")
    ax.semilogx(thresholds, score_val, "s-", markersize=4, color="tab:green", label="Val score")
    ax.axvline(thresholds[best_train_idx], color="tab:orange", linestyle="--", alpha=0.5)
    ax.axvline(thresholds[best_val_idx], color="tab:green", linestyle="--", alpha=0.5)
    ax.set_xlabel("STLSQ threshold $\\lambda$")
    ax.set_ylabel("Score")
    ax.set_title("(c) Model selection score")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_false_positive_heatmap(fp_counts, total_runs, feature_names, Xi_true, outpath: str) -> None:
    """Heatmap of false-positive frequency by equation and library term."""
    true_support = np.abs(Xi_true) > 0
    fp_frac = fp_counts.astype(float) / total_runs

    # Only show features that were falsely selected at least once
    any_fp = np.any(fp_counts > 0, axis=0) & ~np.any(true_support, axis=0)
    fp_indices = np.where(any_fp)[0]

    if len(fp_indices) == 0:
        print("  No false positives to plot")
        return

    # Also mark true positive features for context
    tp_indices = np.where(np.any(true_support, axis=0))[0]

    show_indices = np.concatenate([tp_indices, fp_indices])
    show_names = [feature_names[i] for i in show_indices]
    show_data = fp_frac[:, show_indices]

    fig, ax = plt.subplots(figsize=(max(8, len(show_indices) * 0.6), 3.5))
    im = ax.imshow(show_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    eq_labels = [f"$\\dot{{{n}}}$" for n in STATE_NAMES]
    ax.set_yticks(range(4))
    ax.set_yticklabels(eq_labels, fontsize=11)
    ax.set_xticks(range(len(show_names)))
    ax.set_xticklabels(show_names, rotation=55, ha="right", fontsize=8)

    # Annotate cells
    for i in range(4):
        for j in range(len(show_indices)):
            feat_idx = show_indices[j]
            if true_support[i, feat_idx]:
                ax.text(j, i, "TRUE", ha="center", va="center", fontsize=7, fontweight="bold", color="blue")
            elif fp_counts[i, feat_idx] > 0:
                ax.text(j, i, f"{fp_frac[i, feat_idx]:.2f}", ha="center", va="center", fontsize=7)

    # Separator between true and FP features
    if len(tp_indices) > 0 and len(fp_indices) > 0:
        ax.axvline(len(tp_indices) - 0.5, color="black", linewidth=1.5)

    fig.colorbar(im, ax=ax, shrink=0.8, label="False positive frequency")
    ax.set_title("False positive term frequency (extended library, oracle derivatives)", fontsize=11)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_fig_dir, exist_ok=True)
    os.makedirs(cfg.out_tab_dir, exist_ok=True)

    # ── 1. False-positive term analysis ─────────────────────────────
    print("Running false-positive term analysis (extended library)...")
    freq_rows, raw_fp_rows, fp_counts, total_runs, feature_names = false_positive_analysis(cfg)

    fp_path = os.path.join(cfg.out_tab_dir, "false_positive_terms.csv")
    with open(fp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["equation", "term", "frequency", "fraction"])
        w.writeheader()
        for r in freq_rows:
            w.writerow(r)
    print(f"  Wrote {fp_path} ({len(freq_rows)} terms)")

    # Top false positives
    print("\n  Top 10 false-positive terms:")
    for r in freq_rows[:10]:
        print(f"    {r['equation']:8s}  {r['term']:30s}  {r['frequency']:3d}/{total_runs} ({r['fraction']:.1%})")

    # Heatmap
    params = cfg.params.as_dict()
    lib = make_ext(eps_inv=float(cfg.eps_inv), degree=2)
    lib.fit(np.zeros((1, 4)))
    Xi_true = build_true_coefficients(lib.get_feature_names(STATE_NAMES), params)

    plot_false_positive_heatmap(
        fp_counts, total_runs, feature_names, Xi_true,
        os.path.join(cfg.out_fig_dir, "fig_false_positive_heatmap.pdf"),
    )
    print("  Wrote false-positive heatmap")

    # ── 2. Threshold-path diagnostics ───────────────────────────────
    print("\nRunning threshold-path analysis (η=0.01, seed=0)...")
    thr_rows = threshold_path_analysis(cfg, eta=0.01, seed=0)

    thr_path = os.path.join(cfg.out_tab_dir, "threshold_path.csv")
    with open(thr_path, "w", newline="", encoding="utf-8") as f:
        fields = list(thr_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in thr_rows:
            w.writerow(r)
    print(f"  Wrote {thr_path}")

    plot_threshold_path(thr_rows, os.path.join(cfg.out_fig_dir, "fig_threshold_path.pdf"))
    print("  Wrote threshold-path figure")

    # ── 3. Validation comparison ────────────────────────────────────
    print("\nRunning validation comparison (oracle, physics-informed)...")
    val_rows = validation_comparison(cfg)

    val_path = os.path.join(cfg.out_tab_dir, "validation_comparison.csv")
    with open(val_path, "w", newline="", encoding="utf-8") as f:
        fields = list(val_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in val_rows:
            w.writerow(r)
    print(f"  Wrote {val_path}")

    # Summarize
    by_eta = defaultdict(list)
    for r in val_rows:
        by_eta[float(r["eta"])].append(r)

    print("\n  Validation comparison summary (mean over seeds):")
    print(f"  {'η':>8s}  {'nnz_tr':>7s}  {'nnz_val':>7s}  {'ε_tr':>8s}  {'ε_val':>8s}  {'thr_tr':>8s}  {'thr_val':>8s}")
    for eta in sorted(by_eta.keys()):
        g = by_eta[eta]
        print(f"  {eta:8.0e}  "
              f"{np.mean([r['nnz_train'] for r in g]):7.1f}  "
              f"{np.mean([r['nnz_val'] for r in g]):7.1f}  "
              f"{np.mean([r['rel_l2_train'] for r in g]):8.4f}  "
              f"{np.mean([r['rel_l2_val'] for r in g]):8.4f}  "
              f"{np.mean([r['threshold_train'] for r in g]):8.4f}  "
              f"{np.mean([r['threshold_val'] for r in g]):8.4f}")

    print("\nAll revision analyses complete.")


if __name__ == "__main__":
    main()
