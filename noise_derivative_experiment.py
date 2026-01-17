"""noise_derivative_experiment.py

Noise sweep experiment: additive Gaussian measurement noise on state observations
followed by **numerical derivative estimation**.

This is the more realistic setting:
- observe X_noisy(t) = X(t) + noise
- estimate dX_hat from X_noisy
- fit SINDy on (X_noisy, dX_hat)

We keep the same Option-2 selection scheme as noise_state_experiment:
threshold sweep + minimize log(mse) + nnz_weight*nnz.

Outputs (written to outputs/)
-----------------------------
- noise_derivative_summary.csv
- fig_noise_derivative_error_vs_time.png
- fig_noise_derivative_timeseries_overlay.png
- fig_noise_derivative_support_metrics.png
- fig_noise_derivative_coeff_error.png

Notes
-----
Derivative estimation is a major failure mode under noise. This script includes
2 estimators you can switch between:
- 'finite_difference' (simple, fast; very noise-sensitive)
- 'savgol' (Savitzky–Golay smoothing + derivative; generally better)

"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

import pysindy as ps

from clw import clw_rhs
from data import simulate_short_bursts
from sindy_clw_lib import make_clw_library

from coeff_recovery import build_true_coefficients, coefficient_recovery_rows, equation_summaries


STATE_NAMES = ["P", "S", "Z", "C"]
EQ_NAMES = ["Pdot", "Sdot", "Zdot", "Cdot"]


@dataclass
class Config:
    # System params
    Gd: float = 2.0
    d: float = 2.0
    gz: float = 0.80

    # Data generation
    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seed: int = 0
    S_min: float = 0.05
    eps_inv: float = 1e-8

    # Threshold sweep / selection
    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3

    # Noise sweep
    eta_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    n_trials: int = 8

    # Derivative estimation configuration
    deriv_method: Literal["finite_difference", "savgol"] = "savgol"

    # Savitzky–Golay settings (must be odd window length)
    sg_window: int = 31
    sg_poly: int = 3

    # Plots: same as state-noise experiment
    error_T: float = 20.0
    error_x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)

    overlay_eta_low: float = 1e-2
    overlay_eta_high: float = 10.0
    overlay_T: float = 20.0

    outdir: str = "outputs"


def filter_drop_trajectories_below_S(
    X_list: list[np.ndarray],
    dX_list: list[np.ndarray],
    *,
    S_min: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(X_list) != len(dX_list):
        raise ValueError("X_list and dX_list must have the same length")

    thr = float(S_min)
    X_out: list[np.ndarray] = []
    dX_out: list[np.ndarray] = []

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
        raise ValueError("All trajectories were discarded by S_min.")
    return X_out, dX_out


def fit_sindy_clw(*, X_list: list[np.ndarray], dX_list: list[np.ndarray], dt: float, threshold: float, eps_inv: float) -> ps.SINDy:
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


def _seed_for(*, base_seed: int, eta: float, trial: int) -> int:
    key = f"{int(base_seed)}|{eta:.16g}|{int(trial)}".encode("utf-8")
    h = 2166136261
    for b in key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def compute_sigma_from_data(X_list: list[np.ndarray]) -> np.ndarray:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma = np.maximum(sigma, 1e-12)
    return sigma


def add_state_noise_eta(X_list: list[np.ndarray], *, eta: float, rng: np.random.Generator, sigma: np.ndarray) -> list[np.ndarray]:
    eta = float(eta)
    sigma = np.asarray(sigma, dtype=float).reshape(1, 4)
    out: list[np.ndarray] = []
    for X in X_list:
        X = np.asarray(X, dtype=float)
        noise = rng.normal(loc=0.0, scale=eta, size=X.shape) * sigma
        out.append(X + noise)
    return out


def estimate_derivatives(
    X_list: list[np.ndarray],
    *,
    dt: float,
    method: Literal["finite_difference", "savgol"],
    sg_window: int,
    sg_poly: int,
) -> list[np.ndarray]:
    """Estimate dX from noisy state time series.

    Each trajectory is assumed regularly sampled with time step dt.
    """
    dt = float(dt)

    out: list[np.ndarray] = []
    if method == "finite_difference":
        for X in X_list:
            X = np.asarray(X, dtype=float)
            dX = np.gradient(X, dt, axis=0, edge_order=2)
            out.append(np.asarray(dX, dtype=float))
        return out

    if method == "savgol":
        win = int(sg_window)
        if win % 2 == 0:
            win += 1
        poly = int(sg_poly)
        if poly >= win:
            raise ValueError("sg_poly must be < sg_window")

        for X in X_list:
            X = np.asarray(X, dtype=float)
            T = int(X.shape[0])
            # Ensure window fits; if too short, fall back to finite differences.
            if T < win + 2:
                dX = np.gradient(X, dt, axis=0, edge_order=2)
                out.append(np.asarray(dX, dtype=float))
                continue

            # Filter along time axis; derivative=1 to compute d/dt
            dX = savgol_filter(X, window_length=win, polyorder=poly, deriv=1, delta=dt, axis=0, mode="interp")
            out.append(np.asarray(dX, dtype=float))
        return out

    raise ValueError(f"Unknown derivative method: {method}")


def rel_fro_error(Xi_hat: np.ndarray, Xi_true: np.ndarray) -> float:
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    Xi_true = np.asarray(Xi_true, dtype=float)
    num = float(np.linalg.norm(Xi_hat - Xi_true))
    den = float(np.linalg.norm(Xi_true))
    return num / den if den > 0 else num


def f1(precision: float, recall: float) -> float:
    p = float(precision)
    r = float(recall)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 1.0


def integrate(rhs: callable, *, t_span: tuple[float, float], dt: float, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(float(t_span[0]), float(t_span[1]) + float(dt), float(dt))
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), np.asarray(x0, dtype=float), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.asarray(sol.t, dtype=float), np.asarray(sol.y.T, dtype=float)


def identified_rhs_from_model(model: ps.SINDy, *, eps_inv: float) -> callable:
    coeff = np.asarray(model.coefficients(), dtype=float)
    lib = make_clw_library(eps_inv=float(eps_inv))
    lib.fit(np.zeros((1, 4), dtype=float))

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(4,)
        theta = np.asarray(lib.transform(x[None, :]), dtype=float).reshape(-1)
        return np.asarray(coeff @ theta, dtype=float)

    return rhs


def plot_error_vs_time(curves: dict[float, tuple[np.ndarray, np.ndarray]], *, outpath: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.2))
    for eta, (t, err) in sorted(curves.items(), key=lambda kv: float(kv[0])):
        ax.plot(t, err, linewidth=1.6, label=f"η={eta:g}")

    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Trajectory error ||x̂(t) - x(t)||₂")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_timeseries_overlay(
    *,
    t: np.ndarray,
    X_true: np.ndarray,
    X_low: np.ndarray,
    X_high: np.ndarray,
    eta_low: float,
    eta_high: float,
    outpath: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(9.0, 8.0), sharex=True)
    for i, name in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(t, X_true[:, i], color="black", linewidth=1.6, label="True" if i == 0 else None)
        ax.plot(t, X_low[:, i], color="tab:red", linestyle="--", linewidth=1.2, label=(f"Identified (η={eta_low:g})" if i == 0 else None))
        ax.plot(t, X_high[:, i], color="tab:blue", linestyle=":", linewidth=1.2, label=(f"Identified (η={eta_high:g})" if i == 0 else None))
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title(title)
        if i == len(STATE_NAMES) - 1:
            ax.set_xlabel("t")
        if i == 0:
            ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_support_metrics(summary_rows: list[dict], *, outpath: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etas = np.asarray([float(r["eta"]) for r in summary_rows], dtype=float)
    prec = np.asarray([float(r["precision_mean"]) for r in summary_rows], dtype=float)
    rec = np.asarray([float(r["recall_mean"]) for r in summary_rows], dtype=float)
    f1s = np.asarray([float(r["f1_mean"]) for r in summary_rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0))
    ax.plot(etas, prec, marker="o", label="precision")
    ax.plot(etas, rec, marker="o", label="recall")
    ax.plot(etas, f1s, marker="o", label="F1")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Noise amplitude η (log scale)")
    ax.set_ylabel("Support metric")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_coeff_error(summary_rows: list[dict], *, outpath: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etas = np.asarray([float(r["eta"]) for r in summary_rows], dtype=float)
    err = np.asarray([float(r["coef_rel_fro_mean"]) for r in summary_rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0))
    ax.plot(etas, err, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise amplitude η (log scale)")
    ax.set_ylabel("Relative coefficient error (Frobenius)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def write_summary_csv(rows: Iterable[dict], *, path: str) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    cfg = Config()
    params = {"Gd": float(cfg.Gd), "d": float(cfg.d), "gz": float(cfg.gz)}
    os.makedirs(cfg.outdir, exist_ok=True)

    # Generate clean trajectories (we only use clean dX_list to define "truth" for metrics)
    X_list, dX_list_true = simulate_short_bursts(params=params, n_traj=int(cfg.n_traj), T=float(cfg.burst_T), dt=float(cfg.dt), seed=int(cfg.seed))
    X_list, dX_list_true = filter_drop_trajectories_below_S(X_list, dX_list_true, S_min=float(cfg.S_min))

    sigma = compute_sigma_from_data(X_list)

    summary: list[dict] = []
    best_models_by_eta: dict[float, ps.SINDy] = {}

    for eta in cfg.eta_list:
        precision_trials = []
        recall_trials = []
        f1_trials = []
        coef_err_trials = []
        nnz_trials = []
        thr_trials = []

        for trial in range(int(cfg.n_trials)):
            rng = np.random.default_rng(_seed_for(base_seed=int(cfg.seed), eta=float(eta), trial=int(trial)))
            X_noisy = add_state_noise_eta(X_list, eta=float(eta), rng=rng, sigma=sigma)
            dX_hat = estimate_derivatives(
                X_noisy,
                dt=float(cfg.dt),
                method=str(cfg.deriv_method),
                sg_window=int(cfg.sg_window),
                sg_poly=int(cfg.sg_poly),
            )

            results: list[dict] = []
            for thr in cfg.thresholds:
                m = fit_sindy_clw(X_list=X_noisy, dX_list=dX_hat, dt=float(cfg.dt), threshold=float(thr), eps_inv=float(cfg.eps_inv))
                # Score uses derivative MSE vs the (estimated) regression target dX_hat
                results.append({"threshold": float(thr), "model": m, "mse": derivative_mse(m, X_noisy, dX_hat), "nnz": count_nnz(m)})

            best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))
            model = best["model"]

            feature_names = model.feature_library.get_feature_names(STATE_NAMES)
            Xi_hat = np.asarray(model.coefficients(), dtype=float)
            Xi_true = build_true_coefficients(feature_names, params)

            # Evaluate recovery vs ground truth coefficients (still well-defined)
            rows = coefficient_recovery_rows(
                feature_names,
                Xi_true,
                Xi_hat,
                equation_names=EQ_NAMES,
                nz_tol=0.0,
                rel_floor=1e-12,
                include_tn=False,
            )
            sums = equation_summaries(rows, equation_names=EQ_NAMES, Xi_true=Xi_true, Xi_hat=Xi_hat, nz_tol=0.0)

            prec = float(np.mean([s.precision for s in sums]))
            rec = float(np.mean([s.recall for s in sums]))

            precision_trials.append(prec)
            recall_trials.append(rec)
            f1_trials.append(f1(prec, rec))
            coef_err_trials.append(rel_fro_error(Xi_hat, Xi_true))
            nnz_trials.append(int(best["nnz"]))
            thr_trials.append(float(best["threshold"]))

        # Representative model for plotting: trial=0
        rng0 = np.random.default_rng(_seed_for(base_seed=int(cfg.seed), eta=float(eta), trial=0))
        X_noisy0 = add_state_noise_eta(X_list, eta=float(eta), rng=rng0, sigma=sigma)
        dX_hat0 = estimate_derivatives(
            X_noisy0,
            dt=float(cfg.dt),
            method=str(cfg.deriv_method),
            sg_window=int(cfg.sg_window),
            sg_poly=int(cfg.sg_poly),
        )

        results0: list[dict] = []
        for thr in cfg.thresholds:
            m = fit_sindy_clw(X_list=X_noisy0, dX_list=dX_hat0, dt=float(cfg.dt), threshold=float(thr), eps_inv=float(cfg.eps_inv))
            results0.append({"threshold": float(thr), "model": m, "mse": derivative_mse(m, X_noisy0, dX_hat0), "nnz": count_nnz(m)})
        best0 = select_model_by_score(results0, nnz_weight=float(cfg.nnz_weight))
        best_models_by_eta[float(eta)] = best0["model"]

        summary.append(
            {
                "eta": float(eta),
                "precision_mean": float(np.mean(precision_trials)),
                "precision_std": float(np.std(precision_trials, ddof=0)),
                "recall_mean": float(np.mean(recall_trials)),
                "recall_std": float(np.std(recall_trials, ddof=0)),
                "f1_mean": float(np.mean(f1_trials)),
                "f1_std": float(np.std(f1_trials, ddof=0)),
                "coef_rel_fro_mean": float(np.mean(coef_err_trials)),
                "coef_rel_fro_std": float(np.std(coef_err_trials, ddof=0)),
                "nnz_mean": float(np.mean(nnz_trials)),
                "nnz_std": float(np.std(nnz_trials, ddof=0)),
                "threshold_mean": float(np.mean(thr_trials)),
                "threshold_std": float(np.std(thr_trials, ddof=0)),
                "deriv_method": str(cfg.deriv_method),
                "sg_window": int(cfg.sg_window),
                "sg_poly": int(cfg.sg_poly),
            }
        )

    summary_csv = os.path.join(cfg.outdir, "noise_derivative_summary.csv")
    write_summary_csv(summary, path=summary_csv)
    print(f"Saved derivative-noise sweep summary to: {summary_csv}")

    title_suffix = f"(derivative={cfg.deriv_method})"
    plot_support_metrics(summary, outpath=os.path.join(cfg.outdir, "fig_noise_derivative_support_metrics.png"), title=f"CLW: term selection vs derivative estimation noise {title_suffix}")
    plot_coeff_error(summary, outpath=os.path.join(cfg.outdir, "fig_noise_derivative_coeff_error.png"), title=f"CLW: coefficient recovery error vs derivative estimation noise {title_suffix}")

    # Simulated trajectory error vs time (compare identified model to true CLW trajectory)
    t_err, X_true = integrate(lambda t, x: clw_rhs(t, x, params), t_span=(0.0, float(cfg.error_T)), dt=float(cfg.dt), x0=np.asarray(cfg.error_x0, dtype=float))

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for eta in cfg.eta_list:
        model = best_models_by_eta[float(eta)]
        rhs_hat = identified_rhs_from_model(model, eps_inv=float(cfg.eps_inv))
        _, X_hat = integrate(rhs_hat, t_span=(0.0, float(cfg.error_T)), dt=float(cfg.dt), x0=np.asarray(cfg.error_x0, dtype=float))
        err = np.linalg.norm(X_hat - X_true, axis=1)
        curves[float(eta)] = (t_err, np.maximum(err, 1e-16))

    plot_error_vs_time(
        curves,
        outpath=os.path.join(cfg.outdir, "fig_noise_derivative_error_vs_time.png"),
        title=f"CLW: trajectory error vs time for increasing derivative estimation noise {title_suffix}",
    )

    # Timeseries overlay
    eta_low = float(cfg.overlay_eta_low)
    eta_high = float(cfg.overlay_eta_high)
    m_low = best_models_by_eta[eta_low]
    m_high = best_models_by_eta[eta_high]
    rhs_low = identified_rhs_from_model(m_low, eps_inv=float(cfg.eps_inv))
    rhs_high = identified_rhs_from_model(m_high, eps_inv=float(cfg.eps_inv))

    t_ov, X_true_ov = integrate(lambda t, x: clw_rhs(t, x, params), t_span=(0.0, float(cfg.overlay_T)), dt=float(cfg.dt), x0=np.asarray(cfg.error_x0, dtype=float))
    _, X_low_ov = integrate(rhs_low, t_span=(0.0, float(cfg.overlay_T)), dt=float(cfg.dt), x0=np.asarray(cfg.error_x0, dtype=float))
    _, X_high_ov = integrate(rhs_high, t_span=(0.0, float(cfg.overlay_T)), dt=float(cfg.dt), x0=np.asarray(cfg.error_x0, dtype=float))

    plot_timeseries_overlay(
        t=t_ov,
        X_true=X_true_ov,
        X_low=X_low_ov,
        X_high=X_high_ov,
        eta_low=eta_low,
        eta_high=eta_high,
        outpath=os.path.join(cfg.outdir, "fig_noise_derivative_timeseries_overlay.png"),
        title=f"CLW: time series overlay (true vs identified under derivative estimation noise) {title_suffix}",
    )

    print("Saved derivative-noise figures to outputs/")


if __name__ == "__main__":
    main()
