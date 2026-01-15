"""main_experiment.py

Minimal SINDy demonstration for CLW, analogous to the Lorenz demo in
Brunton, Proctor, Kutz (2016).

Workflow:
1) Generate clean data from a known ODE (oracle derivatives).
2) Build a candidate library (includes constant term).
3) Perform sparse regression (SINDy / STLSQ) via threshold sweep.
4) Print + save identified coefficients.
5) Optional: simulate learned model for a short trajectory comparison.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps

from clw import clw_rhs
from data import simulate_short_bursts
from sindy_clw_lib import make_clw_library
from plot_validation import generate_validation_figures, plot_threshold_pareto

STATE_NAMES = ["P", "S", "Z", "C"]


@dataclass
class Config:
    Gd: float = 2.0
    d: float = 2.0
    gz: float = 0.80

    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seed: int = 0

    # Wider sweep so we actually see sparsity/accuracy trade-offs even in nearly noise-free settings.
    # (Log-spaced thresholds for STLSQ.)
    thresholds: tuple[float, ...] = tuple(np.logspace(-6, 0, 25).astype(float).tolist())
    nnz_weight: float = 2e-3
    S_min: float = 0.2
    eps_inv: float = 1e-8

    outdir: str = "outputs"

    do_simulate: bool = False
    simulate_T: float = 2.0

    # --- Out-of-sample evaluation (trajectory-level split) ---
    # Fit on a subset of trajectories (initial conditions) and evaluate on held-out trajectories.
    train_frac: float = 0.8
    eval_oos: bool = True

    # Short-horizon rollout evaluation on test trajectories
    rollout_horizon_T: float = 0.5
    # Rollout always starts at t0=0 for each test trajectory.

    # Print a compact sweep table (threshold -> nnz, mse, score)
    print_sweep_table: bool = True


def filter_small_S(X_list: list[np.ndarray], dX_list: list[np.ndarray], *, S_min: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X_out: list[np.ndarray] = []
    dX_out: list[np.ndarray] = []
    for X, dX in zip(X_list, dX_list):
        X = np.asarray(X, dtype=float)
        dX = np.asarray(dX, dtype=float)
        keep = X[:, 1] >= float(S_min)
        if np.any(keep):
            X_out.append(X[keep])
            dX_out.append(dX[keep])
    if not X_out:
        raise ValueError("All samples were filtered out by S_min.")
    return X_out, dX_out


def fit_sindy_clw(X_list: list[np.ndarray], dX_list: list[np.ndarray], *, dt: float, threshold: float, eps_inv: float) -> ps.SINDy:
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


def derivative_rmse(model: ps.SINDy, X_list: list[np.ndarray], dX_list: list[np.ndarray]) -> float:
    r"""Vector-field RMSE comparing \hat f(x) to ground-truth f(x) on provided samples."""
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    dX = np.concatenate([np.asarray(dx, dtype=float) for dx in dX_list], axis=0)
    dX_hat = np.asarray(model.predict(X), dtype=float)
    mse = float(np.mean((dX_hat - dX) ** 2))
    return float(np.sqrt(mse))


def split_trajectories(
    X_list: list[np.ndarray],
    dX_list: list[np.ndarray],
    *,
    train_frac: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split trajectory lists into train/test by trajectory index (no time-sample leakage)."""
    if len(X_list) != len(dX_list):
        raise ValueError("X_list and dX_list must have the same length")
    n = len(X_list)
    if n < 2:
        raise ValueError("Need at least 2 trajectories to do a train/test split")
    frac = float(train_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(np.floor(frac * n))
    n_train = max(1, min(n - 1, n_train))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train = [X_list[int(i)] for i in train_idx]
    dX_train = [dX_list[int(i)] for i in train_idx]
    X_test = [X_list[int(i)] for i in test_idx]
    dX_test = [dX_list[int(i)] for i in test_idx]
    return X_train, dX_train, X_test, dX_test


def short_rollout_rmse(
    model: ps.SINDy,
    X_test_list: list[np.ndarray],
    *,
    dt: float,
    horizon_T: float,
) -> float:
    """Short-horizon rollout RMSE on held-out trajectories.

    For each test trajectory, we start at t0=0, simulate the learned model
    forward for `horizon_T`, and compare to the true trajectory over that horizon.
    """
    if not X_test_list:
        raise ValueError("Need at least 1 test trajectory")
    dt = float(dt)
    H = int(np.round(float(horizon_T) / dt))
    H = max(1, H)

    # Require that every trajectory supports the full horizon from t0=0.
    min_Tn = min(int(np.asarray(X, dtype=float).shape[0]) for X in X_test_list)
    if min_Tn < H + 1:
        raise ValueError("Not enough samples to compute rollout RMSE (horizon too long?)")

    sq_sum = 0.0
    count = 0
    for X_true in X_test_list:
        X_true = np.asarray(X_true, dtype=float)
        if X_true.ndim != 2:
            raise ValueError(f"Expected trajectory X shape (T, d), got {X_true.shape}")
        x0 = X_true[0]
        t_eval = np.arange(0.0, (H + 1) * dt, dt, dtype=float)
        X_hat = np.asarray(model.simulate(np.asarray(x0, dtype=float), t_eval), dtype=float)
        X_ref = X_true[: H + 1]
        if X_hat.shape != X_ref.shape:
            raise ValueError(f"Model simulation shape {X_hat.shape} does not match reference horizon {X_ref.shape}")
        err = X_hat - X_ref
        sq_sum += float(np.sum(err**2))
        count += int(err.size)

    if count == 0:
        raise ValueError("Not enough samples to compute rollout RMSE (horizon too long?)")
    return float(np.sqrt(sq_sum / float(count)))


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


def main() -> None:
    cfg = Config()
    params = {"Gd": float(cfg.Gd), "d": float(cfg.d), "gz": float(cfg.gz)}
    X_list, dX_list = simulate_short_bursts(params=params, n_traj=int(cfg.n_traj), T=float(cfg.burst_T), dt=float(cfg.dt), seed=int(cfg.seed))
    X_list, dX_list = filter_small_S(X_list, dX_list, S_min=float(cfg.S_min))

    if bool(cfg.eval_oos):
        X_train, dX_train, X_test, dX_test = split_trajectories(
            X_list,
            dX_list,
            train_frac=float(cfg.train_frac),
            seed=int(cfg.seed),
        )
        print(f"\nTrajectory split: n_train={len(X_train)}, n_test={len(X_test)} (train_frac={float(cfg.train_frac):.2f})")
    else:
        X_train, dX_train, X_test, dX_test = X_list, dX_list, [], []

    results: list[dict] = []
    for thr in cfg.thresholds:
        m = fit_sindy_clw(X_train, dX_train, dt=float(cfg.dt), threshold=float(thr), eps_inv=float(cfg.eps_inv))
        results.append({"threshold": float(thr), "model": m, "mse": derivative_mse(m, X_train, dX_train), "nnz": count_nnz(m)})

    if bool(cfg.print_sweep_table):
        # Compact table, sorted by threshold ascending
        rows = sorted(
            (
                (
                    float(r["threshold"]),
                    int(r["nnz"]),
                    float(r["mse"]),
                    float(np.log(float(r["mse"]) + 1e-30) + float(cfg.nnz_weight) * float(r["nnz"])),
                )
                for r in results
            ),
            key=lambda x: x[0],
        )
        print("\n=== Threshold sweep summary ===")
        print("threshold\tnnz\tmse\tscore")
        for thr, nnz, mse, score in rows:
            print(f"{thr:.2e}\t{nnz:d}\t{mse:.3e}\t{score:.6f}")

    best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))
    model = best["model"]
    print("\n=== Identified SINDy Model (CLW) ===")
    print(f"selected_threshold={best['threshold']:.3e}, nnz={best['nnz']}, mse={best['mse']:.3e}, score={best['score']:.3f}")
    model.print()

    os.makedirs(cfg.outdir, exist_ok=True)
    save_path = os.path.join(cfg.outdir, "identified_model.npz")
    np.savez_compressed(
        save_path,
        threshold=np.asarray(float(best["threshold"]), dtype=float),
        feature_names=np.asarray(model.feature_library.get_feature_names(STATE_NAMES), dtype=object),
        coefficients=np.asarray(model.coefficients(), dtype=float),
    )
    print(f"\nSaved identified coefficients to: {save_path}")

    # --- Out-of-sample metrics on held-out trajectories (test only) ---
    if bool(cfg.eval_oos):
        test_deriv_rmse = derivative_rmse(model, X_test, dX_test)
        test_roll_rmse = short_rollout_rmse(
            model,
            X_test,
            dt=float(cfg.dt),
            horizon_T=float(cfg.rollout_horizon_T),
        )
        print("\n=== Out-of-sample evaluation (test only) ===")
        print(f"test_derivative_rmse = {test_deriv_rmse:.6e}")
        print(
            "test_rollout_rmse     = "
            f"{test_roll_rmse:.6e} (horizon_T={float(cfg.rollout_horizon_T):.3g}, start_index=0)"
        )
        metrics_path = os.path.join(cfg.outdir, "oos_metrics.npz")
        np.savez_compressed(
            metrics_path,
            test_derivative_rmse=np.asarray(float(test_deriv_rmse), dtype=float),
            test_rollout_rmse=np.asarray(float(test_roll_rmse), dtype=float),
            train_frac=np.asarray(float(cfg.train_frac), dtype=float),
            n_train=np.asarray(int(len(X_train)), dtype=int),
            n_test=np.asarray(int(len(X_test)), dtype=int),
            rollout_horizon_T=np.asarray(float(cfg.rollout_horizon_T), dtype=float),
        )
        print(f"Saved out-of-sample metrics to: {metrics_path}")

    pareto_path = plot_threshold_pareto(
        results,
        outdir=str(cfg.outdir),
        filename="fig_pareto_thresholds.png",
        best_threshold=float(best["threshold"]),
        log_mse=True,
    )
    print(f"Saved Pareto front figure to: {pareto_path}")

    # Minimal validation figures (exactly two). These reuse the saved model
    # and do not change the identification procedure.
    generate_validation_figures(outdir=str(cfg.outdir))

    if bool(cfg.do_simulate):
        x0 = np.array([1.2, 1.0, 0.8, 0.5], dtype=float)
        t_eval = np.arange(0.0, float(cfg.simulate_T) + float(cfg.dt), float(cfg.dt))
        sol_true = solve_ivp(lambda t, x: clw_rhs(float(t), x, params), (float(t_eval[0]), float(t_eval[-1])), x0, t_eval=t_eval)
        if not sol_true.success:
            raise RuntimeError(f"True simulation failed: {sol_true.message}")
        X_true = np.asarray(sol_true.y.T, dtype=float)
        X_hat = np.asarray(model.simulate(x0, t_eval), dtype=float)
        err = np.linalg.norm(X_hat - X_true, axis=1)
        print(f"\nTrajectory compare (no plots): final L2 error = {float(err[-1]):.6g}")


if __name__ == "__main__":
    main()
