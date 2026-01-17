"""ablation_experiment.py

Run a clean ablation over:
- number of trajectories (n_traj)
- trajectory horizon length (T)

This script reuses the identification pipeline from `main_experiment.py`:
- simulate_short_bursts (oracle derivatives)
- filter_small_S
- threshold sweep (STLSQ)
- select model by score (log(mse) + nnz_weight * nnz)

Outputs:
- outputs/ablation_runs.csv      : one row per (n_traj, T, seed)
- outputs/ablation_summary.csv   : aggregated stats per (n_traj, T)
- outputs/fig_ablation_heatmap_support.png (optional)
- outputs/fig_ablation_fit_metrics.png    (optional)

The goal is to make *data regime* effects easy to study without changing
libraries/optimizers/selection across conditions.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from data import simulate_short_bursts

# Reuse core helpers from the main experiment to ensure consistency.
from main_experiment import (
    STATE_NAMES,
    count_nnz,
    derivative_mse,
    derivative_rmse,
    fit_sindy_clw,
    select_model_by_score,
    split_trajectories,
    short_rollout_rmse,
)

from coeff_recovery import build_true_coefficients, coefficient_recovery_rows, equation_summaries


EQ_NAMES = ["Pdot", "Sdot", "Zdot", "Cdot"]


@dataclass
class AblationConfig:
    # System parameters
    Gd: float = 2.0
    d: float = 2.0
    gz: float = 0.80

    # Data generation
    dt: float = 0.01
    S_min: float = 0.05
    eps_inv: float = 1e-8

    # Ablation grid
    n_traj_list: tuple[int, ...] = (25, 50, 100, 250, 500)
    T_list: tuple[float, ...] = (1.0, 2.5, 5.0, 7.5, 10.0)

    # Repetitions
    seeds: tuple[int, ...] = (0, 1, 2)

    # Identification sweep + selection (fixed across ablation)
    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3

    # Out-of-sample evaluation (generic fit metrics)
    eval_oos: bool = True
    train_frac: float = 0.8
    rollout_horizon_T: float = 0.5

    # Outputs
    outdir: str = "outputs"
    make_plots: bool = True


def _n_samples_total(X_list: list[np.ndarray]) -> int:
    return int(sum(int(np.asarray(X).shape[0]) for X in X_list))


def filter_drop_trajectories_below_S(
    X_list: list[np.ndarray],
    dX_list: list[np.ndarray],
    *,
    S_min: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Option A filtering: drop entire trajectories if any sample has S < S_min.

    This preserves contiguous, regularly-sampled time series, which is important
    if you later estimate derivatives from time-series data.

    Notes:
    - S is assumed to be the second state component (index 1), consistent with
      STATE_NAMES = ["P", "S", "Z", "C"].
    - If a trajectory violates S_min even once, it is discarded completely.
    """
    if len(X_list) != len(dX_list):
        raise ValueError("X_list and dX_list must have the same length")

    X_out: list[np.ndarray] = []
    dX_out: list[np.ndarray] = []
    thr = float(S_min)
    for X, dX in zip(X_list, dX_list):
        X = np.asarray(X, dtype=float)
        dX = np.asarray(dX, dtype=float)
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"Expected X shape (T, 4), got {X.shape}")
        if dX.shape != X.shape:
            raise ValueError(f"Expected dX shape {X.shape}, got {dX.shape}")

        if np.all(X[:, 1] >= thr):
            X_out.append(X)
            dX_out.append(dX)

    if not X_out:
        raise ValueError("All trajectories were discarded by S_min (Option A).")
    return X_out, dX_out


def _summaries_to_dict(summaries) -> dict[str, Any]:
    # Flatten per-equation recovery summaries into a dict.
    out: dict[str, Any] = {}
    for s in summaries:
        key = str(s.equation)
        out[f"{key}_tp"] = int(s.tp)
        out[f"{key}_fp"] = int(s.fp)
        out[f"{key}_fn"] = int(s.fn)
        out[f"{key}_precision"] = float(s.precision)
        out[f"{key}_recall"] = float(s.recall)
        out[f"{key}_l1_error"] = float(s.l1_error)
        out[f"{key}_l2_error"] = float(s.l2_error)
        out[f"{key}_l1_true"] = float(s.l1_true)
        out[f"{key}_l2_true"] = float(s.l2_true)
    return out


def run_single_condition(*, cfg: AblationConfig, n_traj: int, T: float, seed: int) -> dict[str, Any]:
    """Run identification once for a given (n_traj, T, seed) and return metrics."""

    params = {"Gd": float(cfg.Gd), "d": float(cfg.d), "gz": float(cfg.gz)}

    # --- Data generation (oracle derivatives)
    X_list_raw, dX_list_raw = simulate_short_bursts(
        params=params,
        n_traj=int(n_traj),
        T=float(T),
        dt=float(cfg.dt),
        seed=int(seed),
    )
    n_samples_raw = _n_samples_total(X_list_raw)

    n_traj_raw = int(len(X_list_raw))

    # Option A: drop whole trajectories if they ever go below S_min.
    try:
        X_list, dX_list = filter_drop_trajectories_below_S(X_list_raw, dX_list_raw, S_min=float(cfg.S_min))
        n_samples_used = _n_samples_total(X_list)
        n_traj_used = int(len(X_list))
    except ValueError:
        # If everything gets discarded by Option A, return a row of NaNs/zeros
        # so the full grid run continues.
        out: dict[str, Any] = {
            "n_traj": int(n_traj),
            "n_traj_raw": int(n_traj_raw),
            "n_traj_used": int(0),
            "did_oos_split": int(0),
            "T": float(T),
            "dt": float(cfg.dt),
            "seed": int(seed),
            "n_samples_raw": int(n_samples_raw),
            "n_samples_used": int(0),
            "selected_threshold": float("nan"),
            "nnz": int(0),
            "train_mse": float("nan"),
            "train_rmse": float("nan"),
            "test_derivative_rmse": float("nan"),
            "test_rollout_rmse": float("nan"),
            "fp_total": int(0),
            "fn_total": int(0),
            "l2_error_total": float("nan"),
        }
        # Also include per-equation keys so CSV columns stay consistent.
        for eq in EQ_NAMES:
            out[f"{eq}_tp"] = int(0)
            out[f"{eq}_fp"] = int(0)
            out[f"{eq}_fn"] = int(0)
            out[f"{eq}_precision"] = float("nan")
            out[f"{eq}_recall"] = float("nan")
            out[f"{eq}_l1_error"] = float("nan")
            out[f"{eq}_l2_error"] = float("nan")
            out[f"{eq}_l1_true"] = float("nan")
            out[f"{eq}_l2_true"] = float("nan")
        return out

    # --- Train/test split (by trajectory index)
    did_oos_split = False
    if bool(cfg.eval_oos) and n_traj_used >= 2:
        X_train, dX_train, X_test, dX_test = split_trajectories(
            X_list,
            dX_list,
            train_frac=float(cfg.train_frac),
            seed=int(seed),
        )
        did_oos_split = True
    else:
        # If Option A filtering leaves <2 trajectories, we cannot do a trajectory-level
        # train/test split. Continue with training-only metrics.
        X_train, dX_train = X_list, dX_list
        X_test, dX_test = [], []

    # --- Threshold sweep + model selection
    results: list[dict[str, Any]] = []
    for thr in cfg.thresholds:
        m = fit_sindy_clw(X_train, dX_train, dt=float(cfg.dt), threshold=float(thr), eps_inv=float(cfg.eps_inv))
        results.append(
            {
                "threshold": float(thr),
                "model": m,
                "mse_train": derivative_mse(m, X_train, dX_train),
                "nnz": count_nnz(m),
            }
        )

    best = select_model_by_score(
        [{"threshold": r["threshold"], "model": r["model"], "mse": r["mse_train"], "nnz": r["nnz"]} for r in results],
        nnz_weight=float(cfg.nnz_weight),
    )
    model = best["model"]

    # --- Generic fit metrics
    train_mse = float(best["mse"])
    train_rmse = float(np.sqrt(train_mse))
    nnz = int(best["nnz"])
    selected_threshold = float(best["threshold"])

    test_deriv_rmse = np.nan
    test_roll_rmse = np.nan
    if bool(did_oos_split):
        test_deriv_rmse = float(derivative_rmse(model, X_test, dX_test))
        # Rollout RMSE can fail if some test trajectories are too short after filtering.
        # Treat this as "metric not available" rather than aborting the whole ablation.
        try:
            test_roll_rmse = float(
                short_rollout_rmse(model, X_test, dt=float(cfg.dt), horizon_T=float(cfg.rollout_horizon_T))
            )
        except ValueError:
            test_roll_rmse = float("nan")

    # --- Coefficient recovery metrics
    feature_names = model.feature_library.get_feature_names(STATE_NAMES)
    Xi_hat = np.asarray(model.coefficients(), dtype=float)
    Xi_true = build_true_coefficients(feature_names, params)

    rows = coefficient_recovery_rows(
        feature_names,
        Xi_true,
        Xi_hat,
        equation_names=EQ_NAMES,
        nz_tol=0.0,
        rel_floor=1e-12,
        include_tn=False,
    )
    summaries = equation_summaries(rows, equation_names=EQ_NAMES, Xi_true=Xi_true, Xi_hat=Xi_hat, nz_tol=0.0)
    rec = _summaries_to_dict(summaries)

    # Useful summary scalars (across equations)
    fp_total = int(sum(rec[f"{eq}_fp"] for eq in EQ_NAMES))
    fn_total = int(sum(rec[f"{eq}_fn"] for eq in EQ_NAMES))
    l2_err_total = float(sum(rec[f"{eq}_l2_error"] for eq in EQ_NAMES))

    out: dict[str, Any] = {
        "n_traj": int(n_traj),
        "n_traj_raw": int(n_traj_raw),
        "n_traj_used": int(n_traj_used),
        "did_oos_split": int(1 if did_oos_split else 0),
        "T": float(T),
        "dt": float(cfg.dt),
        "seed": int(seed),
        "n_samples_raw": int(n_samples_raw),
        "n_samples_used": int(n_samples_used),
        "selected_threshold": float(selected_threshold),
        "nnz": int(nnz),
        "train_mse": float(train_mse),
        "train_rmse": float(train_rmse),
        "test_derivative_rmse": float(test_deriv_rmse),
        "test_rollout_rmse": float(test_roll_rmse),
        "fp_total": int(fp_total),
        "fn_total": int(fn_total),
        "l2_error_total": float(l2_err_total),
    }
    out.update(rec)
    return out


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Stable column ordering: core fields first, then everything else sorted.
    core = [
        "n_traj",
        "n_traj_raw",
        "n_traj_used",
        "did_oos_split",
        "T",
        "dt",
        "seed",
        "n_samples_raw",
        "n_samples_used",
        "selected_threshold",
        "nnz",
        "train_mse",
        "train_rmse",
        "test_derivative_rmse",
        "test_rollout_rmse",
        "fp_total",
        "fn_total",
        "l2_error_total",
    ]
    extra = sorted([k for k in rows[0].keys() if k not in core])
    fieldnames = core + extra

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _group_key(r: dict[str, Any]) -> tuple[int, float]:
    return (int(r["n_traj"]), float(r["T"]))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metrics across seeds for each (n_traj, T)."""
    if not rows:
        return []

    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)

    def stats(vals: list[float]) -> tuple[float, float]:
        """Return (mean, std) ignoring NaNs.

        If all values are NaN, returns (nan, nan).
        """
        a = np.asarray(vals, dtype=float)
        if a.size == 0:
            return float("nan"), float("nan")
        if np.all(np.isnan(a)):
            return float("nan"), float("nan")
        return float(np.nanmean(a)), float(np.nanstd(a, ddof=0))

    # Choose a small set of summary columns to keep the table readable.
    summary_cols = [
        "n_traj_used",
        "n_samples_used",
        "selected_threshold",
        "nnz",
        "train_mse",
        "test_derivative_rmse",
        "test_rollout_rmse",
        "fp_total",
        "fn_total",
        "l2_error_total",
        # per-equation support metrics
        *[f"{eq}_fp" for eq in EQ_NAMES],
        *[f"{eq}_fn" for eq in EQ_NAMES],
        *[f"{eq}_precision" for eq in EQ_NAMES],
        *[f"{eq}_recall" for eq in EQ_NAMES],
    ]

    out: list[dict[str, Any]] = []
    for (n_traj, T), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        row: dict[str, Any] = {
            "n_traj": int(n_traj),
            "T": float(T),
            "n_reps": int(len(rs)),
        }

        for col in summary_cols:
            vals = [float(r[col]) for r in rs]
            m, s = stats(vals)
            row[f"{col}_mean"] = float(m)
            row[f"{col}_std"] = float(s)

        # Int-friendly aggregated support error counts
        row["fp_total_sum"] = int(sum(int(r["fp_total"]) for r in rs))
        row["fn_total_sum"] = int(sum(int(r["fn_total"]) for r in rs))

        out.append(row)
    return out


def plot_optional(*, cfg: AblationConfig, summary_rows: list[dict[str, Any]]) -> None:
    """Optional plotting (kept lightweight, uses matplotlib already in requirements)."""
    if not summary_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(cfg.outdir, exist_ok=True)

    n_traj_vals = sorted({int(r["n_traj"]) for r in summary_rows})
    T_vals = sorted({float(r["T"]) for r in summary_rows})

    # Heatmap: mean fp_total + fn_total across seeds
    M = np.full((len(T_vals), len(n_traj_vals)), np.nan, dtype=float)
    for r in summary_rows:
        i = T_vals.index(float(r["T"]))
        j = n_traj_vals.index(int(r["n_traj"]))
        M[i, j] = float(r["fp_total_mean"]) + float(r["fn_total_mean"])

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    im = ax.imshow(M, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(n_traj_vals)))
    ax.set_xticklabels([str(x) for x in n_traj_vals])
    ax.set_yticks(np.arange(len(T_vals)))
    ax.set_yticklabels([str(t) for t in T_vals])
    ax.set_xlabel("n_traj")
    ax.set_ylabel("T")
    ax.set_title("Ablation: mean(FP+FN) across seeds (lower is better)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean FP+FN")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "fig_ablation_heatmap_support.png"), dpi=200)
    plt.close(fig)

    # Heatmap: coefficient error (L2) across seeds
    l2_err = np.full((len(T_vals), len(n_traj_vals)), np.nan, dtype=float)
    for i, T in enumerate(T_vals):
        for j, n in enumerate(n_traj_vals):
            matches = [r for r in summary_rows if int(r["n_traj"]) == int(n) and float(r["T"]) == float(T)]
            if matches:
                l2_err[i, j] = float(matches[0].get("l2_error_total_mean", float("nan")))

    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(n_traj_vals)), 1.2 * max(4, len(T_vals))))
    im = ax.imshow(l2_err, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(n_traj_vals)))
    ax.set_xticklabels([str(n) for n in n_traj_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(T_vals)))
    ax.set_yticklabels([str(T) for T in T_vals])
    ax.set_xlabel("n_traj")
    ax.set_ylabel("T")
    ax.set_title("Ablation: mean coefficient L2 error across seeds")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean L2 error")
    fig.savefig(os.path.join(cfg.outdir, "fig_ablation_heatmap_l2_error.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Line plot: test derivative RMSE vs n_traj (one line per T)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    for T in T_vals:
        xs = []
        ys = []
        yerr = []
        for n in n_traj_vals:
            rr = next((r for r in summary_rows if int(r["n_traj"]) == n and float(r["T"]) == T), None)
            if rr is None:
                continue
            xs.append(n)
            ys.append(float(rr["test_derivative_rmse_mean"]))
            yerr.append(float(rr["test_derivative_rmse_std"]))
        if xs:
            ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=f"T={T:g}")

    ax.set_xscale("log")
    ax.set_xlabel("n_traj (log scale)")
    ax.set_ylabel("test_derivative_rmse (mean ± std)")
    ax.set_title("Ablation: out-of-sample derivative RMSE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "fig_ablation_fit_metrics.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = AblationConfig()
    os.makedirs(cfg.outdir, exist_ok=True)

    run_rows: list[dict[str, Any]] = []

    # STLSQ can emit warnings for very large thresholds during the sweep; they are expected.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*eliminated all coefficients.*", category=UserWarning)

        for T in cfg.T_list:
            for n_traj in cfg.n_traj_list:
                for seed in cfg.seeds:
                    print(f"Running: n_traj={n_traj}, T={T:g}, seed={seed}")
                    r = run_single_condition(cfg=cfg, n_traj=int(n_traj), T=float(T), seed=int(seed))
                    run_rows.append(r)

    runs_csv = os.path.join(cfg.outdir, "ablation_runs.csv")
    write_csv(runs_csv, run_rows)
    print(f"Wrote per-run ablation table: {runs_csv}")

    summary_rows = aggregate(run_rows)
    summary_csv = os.path.join(cfg.outdir, "ablation_summary.csv")
    write_csv(summary_csv, summary_rows)
    print(f"Wrote aggregated ablation table: {summary_csv}")

    if bool(cfg.make_plots):
        plot_optional(cfg=cfg, summary_rows=summary_rows)
        print(f"Wrote optional plots to: {cfg.outdir}/fig_ablation_heatmap_support.png and fig_ablation_fit_metrics.png")


if __name__ == "__main__":
    main()
