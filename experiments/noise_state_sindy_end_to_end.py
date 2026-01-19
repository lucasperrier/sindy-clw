"""experiments.noise_state_sindy_end_to_end

SINDy end-to-end on noisy measurements (all η).

Scientific regime implemented here:
- simulate clean CLW trajectories X(t)
- add Gaussian measurement noise to states:

     X_noisy = X + eta * sigma_X * N(0, 1)

- let PySINDy compute derivatives internally (SmoothedFiniteDifference)
- fit with the physics-informed CLW library
- fit per-trajectory (no concatenation before differentiation)
- NO state standardization (keep physical units)
- NO masking / dropping samples based on S (keep uniform grid)

Outputs
-------
A) outputs/tables/coef_recovery_state_sindy_internal.csv
    columns: eta, nnz, coef_rel_l2

B) outputs/figures/fig_noise_state_sindy_internal_overlay.png
    short-horizon overlay for eta=0.001 and eta=0.1 (4x2: P,S,Z,C)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: `python3 experiments/noise_state_sindy_end_to_end.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
from dataclasses import dataclass

import numpy as np
import pysindy as ps

from clw import clw_rhs
from data import simulate_short_bursts
from coeff_recovery import build_true_coefficients, coef_metrics
from sindy_library.physics_informed import make_library
from sindy_utils import CLWParams, STATE_NAMES, count_nnz, enforce_constant_only_in_Cdot, integrate, select_model_by_score


@dataclass(frozen=True)
class Config:
    params: CLWParams = CLWParams()

    # data generation
    dt: float = 0.01
    burst_T: float = 5.0
    n_traj: int = 250
    seed: int = 0

    # noise sweep
    eta_list: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    focus_etas: tuple[float, float] = (1e-3, 1e-1)

    # numerical safety for rational-like term (P*Z/S)*sin(C)
    # NOTE: safety for 1/S is handled inside the feature library via a safe inverse.
    eps_inv: float = 1e-8

    # Numerical safety WITHOUT masking (MANDATORY): clip values only for feature
    # evaluation so we keep a uniform time grid.
    S_clip_min: float = 1e-3
    X_clip_abs: float = 1e3

    # INTERNAL DIFFERENTIATION (MANDATORY): SmoothedFiniteDifference
    # (Savitzky–Golay smoother configured via smoother_kws)
    sfd_order: int = 2
    sfd_window_length: int = 21
    sfd_polyorder: int = 3

    # regularized sparse regression (MANDATORY)
    stlsq_alpha: float = 1e-3

    # thresholds: gentler than oracle-derivative experiments
    thresholds: tuple[float, ...] = tuple(np.logspace(-6, -1, 18).astype(float).tolist())
    nnz_weight: float = 5e-4

    # short-horizon overlay
    x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)
    overlay_T: float = 20.0

    out_fig_dir: str = os.path.join("outputs", "figures")
    out_tab_dir: str = os.path.join("outputs", "tables")


def _seed_for(*, base_seed: int, eta: float) -> int:
    key = f"{int(base_seed)}|{eta:.16g}|state_sindy_end_to_end".encode("utf-8")
    h = 2166136261
    for b in key:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _compute_sigma(X_list: list[np.ndarray]) -> np.ndarray:
    X = np.concatenate([np.asarray(x, dtype=float) for x in X_list], axis=0)
    return np.maximum(np.std(X, axis=0, ddof=0), 1e-12)


def _clip_for_features(X: np.ndarray, *, S_clip_min: float, X_clip_abs: float) -> np.ndarray:
    """Clip states for safe feature evaluation.

    This is NOT masking: it preserves the sample count and uniform dt.
    """

    X = np.asarray(X, dtype=float)
    Xc = np.clip(X, -float(X_clip_abs), float(X_clip_abs)).copy()
    Xc[:, 1] = np.maximum(Xc[:, 1], float(S_clip_min))
    return Xc


def _rollout_forward_euler(*, model: ps.SINDy, x0: np.ndarray, dt: float, T: float, eps_S: float) -> tuple[np.ndarray, np.ndarray]:
    """Short-horizon roll-out with a fixed time step.

    We keep the time grid uniform (required for the experiment regime) and avoid
    adaptive ODE solvers that can fail when the learned model is imperfect.

    Numerical safety: we clip S from below in the *simulation* state used to
    evaluate features (this does not drop samples).
    """

    dt = float(dt)
    T = float(T)
    t = np.arange(0.0, T + dt, dt, dtype=float)
    X = np.zeros((t.size, 4), dtype=float)
    X[0] = np.asarray(x0, dtype=float).reshape(4,)

    for k in range(t.size - 1):
        x = np.asarray(X[k], dtype=float)
        x_feat = x.copy()
        x_feat[1] = max(float(x_feat[1]), float(eps_S))
        x_feat = np.clip(x_feat, -1e3, 1e3)
        dx = np.asarray(model.predict(x_feat.reshape(1, 4)), dtype=float).reshape(4,)
        X[k + 1] = x + dt * dx

    return t, X


def _add_state_noise(
    X_list: list[np.ndarray], *, eta: float, sigma: np.ndarray, rng: np.random.Generator
) -> list[np.ndarray]:
    """Noise is added to measured states (MANDATORY)."""
    sigma = np.asarray(sigma, dtype=float).reshape(1, 4)
    out: list[np.ndarray] = []
    for X in X_list:
        X = np.asarray(X, dtype=float)
        out.append(X + rng.normal(0.0, 1.0, size=X.shape) * (float(eta) * sigma))
    return out


def _fit_and_select_model_end_to_end(
    *,
    X_noisy_list: list[np.ndarray],
    cfg: Config,
) -> tuple[ps.SINDy, np.ndarray]:
    """Fit SINDy end-to-end on noisy measurements with internal differentiation.

    Returns the selected model and the internally differentiated dX used for
    self-consistent model selection (concatenated across trajectories).
    """

    # (MANDATORY) Keep uniform time grid: do NOT drop/mask samples.
    # For numerical safety we only clip values used for feature evaluation.
    X_used_list = [
        _clip_for_features(np.asarray(x, dtype=float), S_clip_min=float(cfg.S_clip_min), X_clip_abs=float(cfg.X_clip_abs))
        for x in X_noisy_list
    ]

    # Physics-informed library only.
    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))

    # (MANDATORY) Internal differentiator (public API).
    diff = ps.SmoothedFiniteDifference(
        order=int(cfg.sfd_order),
        smoother_kws={"window_length": int(cfg.sfd_window_length), "polyorder": int(cfg.sfd_polyorder)},
    )

    # Create a base model instance; we'll rebuild with varying thresholds.
    results: list[dict] = []
    for thr in cfg.thresholds:
        optimizer = ps.STLSQ(
            threshold=float(thr),
            alpha=float(cfg.stlsq_alpha),
            normalize_columns=True,  # (MANDATORY)
        )
        model = ps.SINDy(feature_library=lib, optimizer=optimizer, differentiation_method=diff)

        # (MANDATORY) True internal differentiation: do NOT pass x_dot.
        # (MANDATORY) Per-trajectory differentiation: pass a list of trajectories.
        model.fit(X_used_list, t=float(cfg.dt))
        enforce_constant_only_in_Cdot(model)

        # (MANDATORY) Self-consistent selection score:
        # compare model.predict(X) against derivatives computed by the same
        # internal differentiator on the same fitting data.
        dX_int_list = [np.asarray(diff(x, t=float(cfg.dt)), dtype=float) for x in X_used_list]
        X_all = np.concatenate(X_used_list, axis=0)
        dX_all = np.concatenate(dX_int_list, axis=0)
        mse = float(np.mean((model.predict(X_all) - dX_all) ** 2))
        results.append({"threshold": float(thr), "mse": mse, "nnz": count_nnz(model), "model": model})

    best = select_model_by_score(results, nnz_weight=float(cfg.nnz_weight))

    # Return dX_int for the chosen threshold (for coefficient recovery).
    dX_best_list = [np.asarray(diff(x, t=float(cfg.dt)), dtype=float) for x in X_used_list]
    dX_best_all = np.concatenate(dX_best_list, axis=0)
    return best["model"], dX_best_all


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_fig_dir, exist_ok=True)
    os.makedirs(cfg.out_tab_dir, exist_ok=True)

    params = cfg.params.as_dict()

    X_clean, _dX_oracle = simulate_short_bursts(params, n_traj=cfg.n_traj, T=cfg.burst_T, dt=cfg.dt, seed=cfg.seed)
    sigma_x = _compute_sigma(X_clean)

    lib = make_library(eps_inv=float(cfg.eps_inv))
    lib.fit(np.zeros((1, 4)))
    feature_names = list(lib.get_feature_names(STATE_NAMES))
    Xi_true = build_true_coefficients(feature_names, params)

    models_by_eta: dict[float, ps.SINDy] = {}
    table_rows: list[dict] = []

    for eta in cfg.eta_list:
        rng = np.random.default_rng(_seed_for(base_seed=cfg.seed, eta=float(eta)))
        X_noisy = _add_state_noise(X_clean, eta=float(eta), sigma=sigma_x, rng=rng)

        model, dX_int_all = _fit_and_select_model_end_to_end(X_noisy_list=X_noisy, cfg=cfg)

        # Coefficient recovery: coefficients are already in physical coordinates
        # since we did not standardize the state.
        Xi_hat = np.asarray(model.coefficients(), dtype=float)
        m = coef_metrics(Xi_hat=Xi_hat, Xi_true=Xi_true)
        table_rows.append({"eta": float(eta), "nnz": int(m.nnz), "coef_rel_l2": float(m.rel_l2)})

        models_by_eta[float(eta)] = model

    tab_path = os.path.join(cfg.out_tab_dir, "coef_recovery_state_sindy_internal.csv")
    with open(tab_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eta", "nnz", "coef_rel_l2"])
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    # Overlay figure: 2-column layout (eta=0.001 and eta=0.1), 4 stacked rows.
    eta_low, eta_high = cfg.focus_etas
    model_low = models_by_eta[float(eta_low)]
    model_high = models_by_eta[float(eta_high)]

    rhs_true = lambda t, x: clw_rhs(t, x, params)
    t, X_true = integrate(rhs_true, dt=cfg.dt, T=cfg.overlay_T, x0=np.asarray(cfg.x0, dtype=float))

    # Identified model overlay: explicit fixed-step roll-out (short horizon only).
    # We clip S in feature evaluation to avoid extreme inverse values.
    t_hat, X_hat_low = _rollout_forward_euler(
        model=model_low,
        x0=np.asarray(cfg.x0, dtype=float),
        dt=cfg.dt,
        T=cfg.overlay_T,
        eps_S=float(cfg.eps_inv),
    )
    _, X_hat_high = _rollout_forward_euler(
        model=model_high,
        x0=np.asarray(cfg.x0, dtype=float),
        dt=cfg.dt,
        T=cfg.overlay_T,
        eps_S=float(cfg.eps_inv),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 2, figsize=(12.0, 8.0), sharex="col")
    for i, name in enumerate(STATE_NAMES):
        ax = axs[i, 0]
        ax.plot(t, X_true[:, i], color="black", linewidth=1.5, label="True CLW")
        ax.plot(t, X_hat_low[:, i], color="tab:red", linestyle="--", linewidth=1.3, label="Identified")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title(f"η={float(eta_low):g}")
            ax.legend(loc="upper right", frameon=False)
        if i == 3:
            ax.set_xlabel("t")

        ax = axs[i, 1]
        ax.plot(t, X_true[:, i], color="black", linewidth=1.5, label="True CLW")
        ax.plot(t, X_hat_high[:, i], color="tab:red", linestyle="--", linewidth=1.3, label="Identified")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title(f"η={float(eta_high):g}")
            ax.legend(loc="upper right", frameon=False)
        if i == 3:
            ax.set_xlabel("t")

    fig.suptitle("SINDy end-to-end on noisy measurements (internal differentiation, short horizon)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig_path = os.path.join(cfg.out_fig_dir, "fig_noise_state_sindy_internal_overlay.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Wrote table to {tab_path}")
    print(f"Wrote overlay figure to {fig_path}")

    # Error vs time figure: one curve per eta (short horizon only).
    # We roll out each identified model from the SAME initial condition.
    errs_by_eta: dict[float, np.ndarray] = {}
    for eta in cfg.eta_list:
        model = models_by_eta[float(eta)]
        _, X_hat = _rollout_forward_euler(
            model=model,
            x0=np.asarray(cfg.x0, dtype=float),
            dt=cfg.dt,
            T=cfg.overlay_T,
            eps_S=float(cfg.S_clip_min),
        )
        err = np.linalg.norm(X_true - X_hat, axis=1)
        errs_by_eta[float(eta)] = np.asarray(err, dtype=float)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8.0, 4.5))
    for eta in cfg.eta_list:
        ax2.plot(t, errs_by_eta[float(eta)], linewidth=1.4, label=f"η={float(eta):g}")
    ax2.set_yscale("log")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$\|x(t) - \hat{x}(t)\|_2$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(ncol=2, frameon=False)
    ax2.set_title("SINDy end-to-end on noisy measurements (internal differentiation)")
    fig2.tight_layout()

    fig2_path = os.path.join(cfg.out_fig_dir, "fig_noise_state_sindy_internal_error_vs_time.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    print(f"Wrote error-vs-time figure to {fig2_path}")


if __name__ == "__main__":
    main()
