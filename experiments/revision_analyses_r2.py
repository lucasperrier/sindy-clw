"""experiments.revision_analyses_r2

Round-2 revision analyses:

1. BIC-based model selection for sample-size ablation
   (demonstrates that an n-dependent criterion resolves the non-monotonic anomaly)

2. Ensemble-SINDy comparison at 3 noise levels
   (provides a minimal comparator to quantify headroom beyond baseline STLSQ)

3. S→0 conditioning analysis
   (examines how often S is small in sampled trajectories, and library condition number)

4. Collinearity diagnostics for the extended library
   (singular-value spectrum and pairwise feature correlations on attractor data)

Outputs
-------
- outputs/tables/sample_size_bic.csv
- outputs/tables/ensemble_comparison.csv
- outputs/tables/s_conditioning.csv
- outputs/tables/collinearity_diagnostics.csv
- outputs/figures/fig_collinearity_svd.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Sparsity parameter is too big")

import pysindy as ps
from pysindy.optimizers import EnsembleOptimizer, STLSQ

from clw import clw_rhs
from data import simulate_short_bursts
from sindy_library.physics_informed import make_library as make_phys
from sindy_library.extended import make_library as make_ext
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


OUT_TAB = os.path.join("outputs", "tables")
OUT_FIG = os.path.join("outputs", "figures")
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)


# ── Shared helpers ─────────────────────────────────────────────────

def _compute_sigma(X_list: list[np.ndarray]) -> np.ndarray:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    return np.maximum(np.std(X, axis=0, ddof=0), 1e-12)


def _add_state_noise(X_list, *, eta, sigma, rng):
    sigma = np.asarray(sigma, dtype=float).reshape(1, 4)
    return [X + rng.normal(0.0, 1.0, size=X.shape) * (float(eta) * sigma) for X in X_list]


def _seed_for(*, base_seed: int, tag: str) -> int:
    key = f"{int(base_seed)}|{tag}".encode("utf-8")
    h = 2166136261
    for b in key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


# ── Analysis 1: BIC-based sample-size ablation ─────────────────────

def bic_sample_size_ablation():
    """Re-run sample-size ablation with BIC model selection.

    BIC = n * log(MSE) + k * log(n)
    where n = number of data points, k = nnz.
    """
    print("Running BIC sample-size ablation...")
    cfg_params = CLWParams()
    params = cfg_params.as_dict()
    dt = 0.01
    burst_T = 5.0
    eta = 0.01
    eps_inv = 1e-8
    seeds = (0, 1, 2, 3, 4)
    n_traj_list = (25, 50, 100, 250, 500)
    thresholds = tuple(np.logspace(-6, 0, 25).astype(float).tolist())

    lib = make_phys(eps_inv=eps_inv)
    lib.fit(np.zeros((1, 4)))
    feature_names = lib.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    raw_rows = []

    for seed in seeds:
        max_n = max(n_traj_list)
        X_clean, dX_clean = simulate_short_bursts(params, n_traj=max_n, T=burst_T, dt=dt, seed=seed)
        sigma_x = _compute_sigma(X_clean)

        for n_traj in n_traj_list:
            key = f"{int(seed)}|{n_traj}|sample_size".encode("utf-8")
            h = 2166136261
            for b in key:
                h ^= b
                h = (h * 16777619) & 0xFFFFFFFF
            rng = np.random.default_rng(int(h))

            X_sub = X_clean[:n_traj]
            dX_sub = dX_clean[:n_traj]
            X_noisy = _add_state_noise(X_sub, eta=eta, sigma=sigma_x, rng=rng)

            X_all = np.concatenate(X_noisy, axis=0)
            dX_all = np.concatenate(dX_sub, axis=0)
            n_data = X_all.shape[0]  # number of data points

            # Fit models across thresholds, select by BIC
            best_bic = None
            best_model = None
            for thr in thresholds:
                model = fit_sindy(X_noisy, dX_sub, library=lib, dt=dt, threshold=float(thr))
                enforce_constant_only_in_Cdot(model)
                mse = float(np.mean((model.predict(X_all) - dX_all) ** 2))
                nnz = count_nnz(model)
                # BIC: n * log(MSE) + k * log(n)
                bic = n_data * np.log(mse + 1e-30) + nnz * np.log(n_data)
                if best_bic is None or bic < best_bic:
                    best_bic = bic
                    best_model = model

            Xi_hat = np.asarray(best_model.coefficients(), dtype=float)
            m = coef_metrics(Xi_hat=Xi_hat, Xi_true=Xi_true)
            raw_rows.append({
                "seed": seed, "n_traj": n_traj,
                "nnz": m.nnz, "coef_rel_l2": round(m.rel_l2, 6),
                "tpr": m.tpr, "fpr": round(m.fpr, 6),
                "exact_support": m.exact_support,
            })

    # Summarize
    by_n = defaultdict(list)
    for r in raw_rows:
        by_n[r["n_traj"]].append(r)

    summary_rows = []
    for n_traj in n_traj_list:
        g = by_n[n_traj]
        summary_rows.append({
            "n_traj": n_traj,
            "nnz_mean": round(np.mean([r["nnz"] for r in g]), 1),
            "nnz_std": round(np.std([r["nnz"] for r in g]), 1),
            "rel_l2_mean": round(np.mean([r["coef_rel_l2"] for r in g]), 4),
            "rel_l2_std": round(np.std([r["coef_rel_l2"] for r in g]), 4),
            "tpr_mean": round(np.mean([r["tpr"] for r in g]), 3),
            "fpr_mean": round(np.mean([r["fpr"] for r in g]), 4),
            "exact_frac": round(np.mean([float(r["exact_support"]) for r in g]), 2),
        })

    path = os.path.join(OUT_TAB, "sample_size_bic.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Wrote {path}")

    print("\n  BIC sample-size ablation results:")
    print(f"  {'n_traj':>7} {'nnz':>6} {'ε_coef':>8} {'TPR':>6} {'FPR':>6} {'exact':>6}")
    for r in summary_rows:
        print(f"  {r['n_traj']:>7} {r['nnz_mean']:>6.1f} {r['rel_l2_mean']:>8.4f} "
              f"{r['tpr_mean']:>6.3f} {r['fpr_mean']:>6.4f} {r['exact_frac']:>6.2f}")

    return summary_rows


# ── Analysis 2: Ensemble-SINDy comparison ──────────────────────────

def ensemble_sindy_comparison():
    """Compare baseline STLSQ with ensemble (bagging) SINDy at 3 noise levels."""
    print("\nRunning ensemble-SINDy comparison...")
    params = CLWParams().as_dict()
    dt = 0.01
    burst_T = 5.0
    n_traj = 100
    eps_inv = 1e-8
    seeds = (0, 1, 2, 3, 4)
    eta_list = (1e-3, 1e-2, 1e-1)
    thresholds = tuple(np.logspace(-5, 0, 15).astype(float).tolist())
    nnz_weight = 2e-3

    lib_template = make_phys(eps_inv=eps_inv)
    lib_template.fit(np.zeros((1, 4)))
    feature_names = lib_template.get_feature_names(STATE_NAMES)
    Xi_true = build_true_coefficients(feature_names, params)

    rows = []

    for seed in seeds:
        X_clean, dX_clean = simulate_short_bursts(params, n_traj=n_traj, T=burst_T, dt=dt, seed=seed)
        sigma_x = _compute_sigma(X_clean)

        for eta in eta_list:
            print(f"  seed={seed}, eta={eta}")
            rng = np.random.default_rng(_seed_for(base_seed=seed, tag=f"{eta:.16g}|state_oracle"))
            X_noisy = _add_state_noise(X_clean, eta=eta, sigma=sigma_x, rng=rng)

            X_all = np.concatenate(X_noisy, axis=0)
            dX_all = np.concatenate(dX_clean, axis=0)

            # --- Baseline STLSQ ---
            results_stlsq = []
            for thr in thresholds:
                lib = make_phys(eps_inv=eps_inv)
                model = fit_sindy(X_noisy, dX_clean, library=lib, dt=dt, threshold=float(thr))
                enforce_constant_only_in_Cdot(model)
                mse = float(np.mean((model.predict(X_all) - dX_all) ** 2))
                results_stlsq.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})
            best_stlsq = select_model_by_score(results_stlsq, nnz_weight=nnz_weight)
            m_stlsq = coef_metrics(Xi_hat=np.asarray(best_stlsq["model"].coefficients(), dtype=float), Xi_true=Xi_true)

            # --- Ensemble SINDy (bagging, mean aggregation) ---
            results_ensemble = []
            for thr in thresholds:
                lib = make_phys(eps_inv=eps_inv)
                base_opt = STLSQ(threshold=float(thr), alpha=0.0, normalize_columns=False)
                ens_opt = EnsembleOptimizer(
                    opt=base_opt,
                    bagging=True,
                    n_models=10,
                    n_subset=int(X_all.shape[0] * 0.8),
                    replace=True,
                    ensemble_aggregator=lambda x: np.median(x, axis=0),
                )
                model = ps.SINDy(feature_library=lib, optimizer=ens_opt)
                model.fit(X_all, t=dt, x_dot=dX_all)
                # Guard against degenerate coef_ from high threshold
                coef_arr = np.asarray(model.optimizer.coef_, dtype=float)
                if coef_arr.ndim < 2:
                    n_feat = lib.n_output_features_ if hasattr(lib, 'n_output_features_') else 10
                    model.optimizer.coef_ = np.zeros((4, n_feat))
                enforce_constant_only_in_Cdot(model)
                try:
                    mse = float(np.mean((model.predict(X_all) - dX_all) ** 2))
                except Exception:
                    mse = float(np.mean(dX_all ** 2))  # null-model MSE
                results_ensemble.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})
            best_ens = select_model_by_score(results_ensemble, nnz_weight=nnz_weight)
            Xi_ens = np.atleast_2d(np.asarray(best_ens["model"].coefficients(), dtype=float))
            m_ens = coef_metrics(Xi_hat=Xi_ens, Xi_true=Xi_true)

            rows.append({
                "seed": seed, "eta": eta,
                "nnz_stlsq": m_stlsq.nnz, "rel_l2_stlsq": round(m_stlsq.rel_l2, 6),
                "tpr_stlsq": m_stlsq.tpr, "fpr_stlsq": round(m_stlsq.fpr, 6),
                "exact_stlsq": m_stlsq.exact_support,
                "nnz_ensemble": m_ens.nnz, "rel_l2_ensemble": round(m_ens.rel_l2, 6),
                "tpr_ensemble": m_ens.tpr, "fpr_ensemble": round(m_ens.fpr, 6),
                "exact_ensemble": m_ens.exact_support,
            })

    # Summarize
    by_eta = defaultdict(list)
    for r in rows:
        by_eta[r["eta"]].append(r)

    summary_rows = []
    for eta in eta_list:
        g = by_eta[eta]
        summary_rows.append({
            "eta": eta,
            "nnz_stlsq": round(np.mean([r["nnz_stlsq"] for r in g]), 1),
            "rel_l2_stlsq": round(np.mean([r["rel_l2_stlsq"] for r in g]), 4),
            "tpr_stlsq": round(np.mean([r["tpr_stlsq"] for r in g]), 3),
            "fpr_stlsq": round(np.mean([r["fpr_stlsq"] for r in g]), 4),
            "exact_stlsq": round(np.mean([float(r["exact_stlsq"]) for r in g]), 2),
            "nnz_ensemble": round(np.mean([r["nnz_ensemble"] for r in g]), 1),
            "rel_l2_ensemble": round(np.mean([r["rel_l2_ensemble"] for r in g]), 4),
            "tpr_ensemble": round(np.mean([r["tpr_ensemble"] for r in g]), 3),
            "fpr_ensemble": round(np.mean([r["fpr_ensemble"] for r in g]), 4),
            "exact_ensemble": round(np.mean([float(r["exact_ensemble"]) for r in g]), 2),
        })

    path = os.path.join(OUT_TAB, "ensemble_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Wrote {path}")

    # Also write raw results
    raw_path = os.path.join(OUT_TAB, "ensemble_comparison_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {raw_path}")

    print("\n  Ensemble comparison summary (mean over 5 seeds):")
    print(f"  {'η':>8}  {'nnz_ST':>6} {'ε_ST':>8} {'TPR_ST':>7}  {'nnz_EN':>6} {'ε_EN':>8} {'TPR_EN':>7}")
    for r in summary_rows:
        print(f"  {r['eta']:>8.0e}  {r['nnz_stlsq']:>6.1f} {r['rel_l2_stlsq']:>8.4f} {r['tpr_stlsq']:>7.3f}  "
              f"{r['nnz_ensemble']:>6.1f} {r['rel_l2_ensemble']:>8.4f} {r['tpr_ensemble']:>7.3f}")

    return summary_rows


# ── Analysis 3: S→0 conditioning analysis ──────────────────────────

def s_zero_conditioning():
    """Analyze how close S gets to 0 in sampled trajectories, and its
    effect on library conditioning."""
    print("\nRunning S→0 conditioning analysis...")
    params = CLWParams().as_dict()
    n_traj = 250
    dt = 0.01
    burst_T = 5.0

    X_clean, dX_clean = simulate_short_bursts(params, n_traj=n_traj, T=burst_T, dt=dt, seed=0)
    X_all = np.concatenate(X_clean, axis=0)

    S = X_all[:, 1]  # S is column index 1

    percentiles = [0, 1, 5, 10, 25, 50]
    pct_values = np.percentile(S, percentiles)

    # Fraction of time points where |S| < threshold
    thresholds_s = [0.01, 0.05, 0.1, 0.5]
    frac_below = {t: float(np.mean(np.abs(S) < t)) for t in thresholds_s}

    # Library condition number on clean data
    eps_inv = 1e-8
    lib_phys = make_phys(eps_inv=eps_inv)
    lib_phys.fit(np.zeros((1, 4)))
    Theta_phys = lib_phys.transform(X_all)

    lib_ext = make_ext(eps_inv=eps_inv, degree=2)
    lib_ext.fit(np.zeros((1, 4)))
    Theta_ext = lib_ext.transform(X_all)

    cond_phys = float(np.linalg.cond(Theta_phys))
    cond_ext = float(np.linalg.cond(Theta_ext))

    # Condition number on subsets stratified by min(S) per trajectory
    traj_min_S = [float(np.min(np.abs(X[:, 1]))) for X in X_clean]
    sorted_idx = np.argsort(traj_min_S)

    # Bottom 10% of trajectories (those closest to S=0)
    n_bottom = max(1, n_traj // 10)
    bottom_idx = sorted_idx[:n_bottom]
    X_bottom = np.concatenate([X_clean[i] for i in bottom_idx], axis=0)
    Theta_bottom_phys = lib_phys.transform(X_bottom)
    cond_bottom_phys = float(np.linalg.cond(Theta_bottom_phys))

    # Top 10% (those farthest from S=0)
    top_idx = sorted_idx[-n_bottom:]
    X_top = np.concatenate([X_clean[i] for i in top_idx], axis=0)
    Theta_top_phys = lib_phys.transform(X_top)
    cond_top_phys = float(np.linalg.cond(Theta_top_phys))

    # The rational term (PZ/S)sin(C) magnitude statistics
    feat_names = lib_phys.get_feature_names(STATE_NAMES)
    rat_idx = feat_names.index("(P*Z/S)*sin(C)")
    rational_col = Theta_phys[:, rat_idx]
    rat_pct = np.percentile(np.abs(rational_col), [50, 90, 95, 99, 100])

    results = {
        "S_min": float(np.min(S)),
        "S_pct1": float(pct_values[1]),
        "S_pct5": float(pct_values[2]),
        "S_pct10": float(pct_values[3]),
        "S_median": float(pct_values[5]),
        "frac_S_below_0.01": frac_below[0.01],
        "frac_S_below_0.05": frac_below[0.05],
        "frac_S_below_0.1": frac_below[0.1],
        "frac_S_below_0.5": frac_below[0.5],
        "cond_phys_all": cond_phys,
        "cond_ext_all": cond_ext,
        "cond_phys_bottom10pct": cond_bottom_phys,
        "cond_phys_top10pct": cond_top_phys,
        "rational_median": float(rat_pct[0]),
        "rational_p90": float(rat_pct[1]),
        "rational_p99": float(rat_pct[3]),
        "rational_max": float(rat_pct[4]),
    }

    path = os.path.join(OUT_TAB, "s_conditioning.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)
    print(f"  Wrote {path}")

    print("\n  S distribution:")
    print(f"    min(S)={results['S_min']:.4f}, 1st pct={results['S_pct1']:.4f}, "
          f"5th pct={results['S_pct5']:.4f}, median={results['S_median']:.4f}")
    print(f"    frac(|S|<0.01)={frac_below[0.01]:.4f}, frac(|S|<0.1)={frac_below[0.1]:.4f}, "
          f"frac(|S|<0.5)={frac_below[0.5]:.4f}")
    print(f"\n  Library condition numbers:")
    print(f"    Physics-informed (all traj): {cond_phys:.1f}")
    print(f"    Extended (all traj): {cond_ext:.1f}")
    print(f"    Physics-informed (bottom 10% by min|S|): {cond_bottom_phys:.1f}")
    print(f"    Physics-informed (top 10% by min|S|): {cond_top_phys:.1f}")
    print(f"\n  |(PZ/S)sin(C)| statistics:")
    print(f"    median={rat_pct[0]:.2f}, p90={rat_pct[1]:.2f}, "
          f"p99={rat_pct[3]:.2f}, max={rat_pct[4]:.2f}")

    return results


# ── Analysis 4: Collinearity diagnostics ───────────────────────────

def collinearity_diagnostics():
    """Compute pairwise feature correlations and SVD spectrum of the
    extended library matrix to explain false-positive patterns."""
    print("\nRunning collinearity diagnostics...")
    params = CLWParams().as_dict()
    n_traj = 250
    dt = 0.01
    burst_T = 5.0
    eps_inv = 1e-8

    X_clean, _ = simulate_short_bursts(params, n_traj=n_traj, T=burst_T, dt=dt, seed=0)
    X_all = np.concatenate(X_clean, axis=0)

    lib = make_ext(eps_inv=eps_inv, degree=2)
    lib.fit(np.zeros((1, 4)))
    Theta = lib.transform(X_all)
    feat_names = lib.get_feature_names(STATE_NAMES)

    # Normalize columns for correlation computation
    Theta_centered = Theta - Theta.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Theta_centered, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Theta_normed = Theta_centered / norms

    # Correlation matrix
    corr = Theta_normed.T @ Theta_normed / Theta_normed.shape[0]

    # Identify the true-term columns and the top false-positive columns
    true_terms = ["P", "S", "Z", "1", "cos(C)", "sin(C)",
                  "Z*S*cos(C)", "Z*P*cos(C)", "P*S*cos(C)", "(P*Z/S)*sin(C)"]
    # Top false positives from our analysis: S, P, sin(C) in C-dot equation
    # These correspond to the same library features
    fp_terms = ["S", "P", "sin(C)", "cos(C)", "Z"]

    # Compute correlations between the rational term and all other features
    rat_name = "(P*Z/S)*sin(C)"
    rat_idx = feat_names.index(rat_name)
    rat_corrs = []
    for i, name in enumerate(feat_names):
        if name == rat_name:
            continue
        rat_corrs.append({
            "feature": name,
            "corr_with_rational": round(float(corr[rat_idx, i]), 4),
            "abs_corr": round(abs(float(corr[rat_idx, i])), 4),
        })
    rat_corrs.sort(key=lambda r: -r["abs_corr"])

    # SVD of the full extended library
    U, s, Vt = np.linalg.svd(Theta, full_matrices=False)
    s_normalized = s / s[0]

    # Write correlation table (top 15 most-correlated with rational term)
    path_corr = os.path.join(OUT_TAB, "collinearity_diagnostics.csv")
    with open(path_corr, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "corr_with_rational", "abs_corr"])
        w.writeheader()
        for r in rat_corrs[:15]:
            w.writerow(r)
    print(f"  Wrote {path_corr}")

    # Plot SVD spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.semilogy(range(1, len(s_normalized) + 1), s_normalized, "o-", markersize=4, color="tab:blue")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value ($\\sigma_i / \\sigma_1$)")
    ax.set_title("(a) SVD spectrum of extended library ($p=46$)")
    ax.axhline(1e-10, color="gray", ls=":", alpha=0.7, label="$10^{-10}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation heatmap between (PZ/S)sin(C) and top features
    top_n = 10
    top_feats = [r["feature"] for r in rat_corrs[:top_n]]
    top_indices = [feat_names.index(f) for f in top_feats]
    all_idx = [rat_idx] + top_indices
    sub_corr = corr[np.ix_(all_idx, all_idx)]
    sub_names = [rat_name] + top_feats
    # Shorten names for display
    short_names = []
    for n in sub_names:
        if len(n) > 15:
            short_names.append(n[:14] + "…")
        else:
            short_names.append(n)

    ax2 = axes[1]
    im = ax2.imshow(np.abs(sub_corr), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(short_names)))
    ax2.set_yticks(range(len(short_names)))
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(short_names, fontsize=7)
    ax2.set_title("(b) |Correlation| with $(PZ/S)\\sin C$ and neighbors")
    fig.colorbar(im, ax=ax2, shrink=0.8)

    fig.tight_layout()
    fig_path = os.path.join(OUT_FIG, "fig_collinearity_svd.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_path}")

    print("\n  Top 10 features most correlated with (PZ/S)sin(C):")
    for r in rat_corrs[:10]:
        print(f"    {r['feature']:>25s}  |ρ| = {r['abs_corr']:.4f}")

    print(f"\n  SVD: σ_1={s[0]:.1f}, σ_46={s[-1]:.2e}, "
          f"condition number={s[0]/s[-1]:.1f}")
    # Effective rank (singular values > 1e-6 * σ_1)
    eff_rank = int(np.sum(s > 1e-6 * s[0]))
    print(f"  Effective rank (σ > 10⁻⁶·σ₁): {eff_rank}/{len(s)}")

    return rat_corrs, s_normalized


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    bic_csv = os.path.join(OUT_TAB, "sample_size_bic.csv")
    if os.path.exists(bic_csv):
        print(f"Skipping BIC ablation (already exists: {bic_csv})")
        bic_results = None
    else:
        bic_results = bic_sample_size_ablation()
    ensemble_results = ensemble_sindy_comparison()
    s_cond = s_zero_conditioning()
    collin_corrs, svd_spec = collinearity_diagnostics()
    print("\n✓ All round-2 revision analyses complete.")
