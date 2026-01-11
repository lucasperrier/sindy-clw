"""plot_validation.py

Generate minimal validation figures for the CLW SINDy demo.

Scope constraints:
- Reuses the saved identified model at outputs/identified_model.npz
- Uses solve_ivp for all integrations
- Produces exactly two figures:
  1) outputs/fig_timeseries_comparison.png
  2) outputs/fig_phase_space_PSZ.png

This file does not modify the identification procedure.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.integrate import solve_ivp

from clw import clw_rhs
from sindy_clw_lib import make_clw_library

STATE_NAMES = ["P", "S", "Z", "C"]


def _load_identified_model(npz_path: str) -> tuple[list[str], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    coefficients = np.asarray(data["coefficients"], dtype=float)
    if coefficients.shape[0] != 4:
        raise ValueError(f"Expected coefficients with 4 rows (one per state), got shape {coefficients.shape}")
    if coefficients.shape[1] != len(feature_names):
        raise ValueError(
            f"Mismatch: coefficients has {coefficients.shape[1]} columns but feature_names has {len(feature_names)} entries"
        )
    return feature_names, coefficients


def _identified_rhs_from_library(*, coefficients: np.ndarray, eps_inv: float) -> callable:
    """Construct RHS(t, x) = Theta(x) @ Xi^T using the existing feature library."""

    lib = make_clw_library(eps_inv=float(eps_inv))

    # Ensure internal state (if any) is initialized for transform.
    # CustomLibrary doesn't require training data, but PySINDy follows the
    # sklearn-style fit/transform contract.
    lib.fit(np.zeros((1, 4), dtype=float))

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (4,):
            x = x.reshape(4,)

        # Build Theta(x) with the library's own transform method so the
        # feature ordering matches `feature_names` stored in the npz.
        theta_vals = np.asarray(lib.transform(x[None, :]), dtype=float).reshape(-1)
        dx = coefficients @ theta_vals
        return np.asarray(dx, dtype=float)

    return rhs


def _integrate(rhs: callable, *, t_span: tuple[float, float], dt: float, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(float(t_span[0]), float(t_span[1]) + float(dt), float(dt))
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), np.asarray(x0, dtype=float), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    X = np.asarray(sol.y.T, dtype=float)
    t = np.asarray(sol.t, dtype=float)
    return t, X


def generate_validation_figures(
    *,
    outdir: str = "outputs",
    model_npz: str | None = None,
    params: dict | None = None,
    x0: np.ndarray | None = None,
    dt: float = 0.01,
    T_short: float = 8.0,
    T_phase: float = 100.0,
    delta_norm: float = 1e-6,
    eps_inv: float = 1e-8,
) -> tuple[str, str]:
    """Generate the two required validation figures.

    Returns:
        (timeseries_path, phase_path)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (required for 3D)

    if params is None:
        params = {"Gd": 2.0, "d": 2.0, "gz": 0.80}
    if x0 is None:
        x0 = np.array([1.2, 1.0, 0.8, 0.5], dtype=float)

    os.makedirs(outdir, exist_ok=True)
    if model_npz is None:
        model_npz = os.path.join(outdir, "identified_model.npz")

    _, coefficients = _load_identified_model(str(model_npz))
    rhs_ident = _identified_rhs_from_library(coefficients=coefficients, eps_inv=float(eps_inv))

    # --- Figure 1: time-series comparison (short horizon)
    t1, X_true_1 = _integrate(lambda t, x: clw_rhs(t, x, params), t_span=(0.0, float(T_short)), dt=float(dt), x0=x0)
    _, X_hat_1 = _integrate(rhs_ident, t_span=(0.0, float(T_short)), dt=float(dt), x0=x0)

    fig1, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for i, name in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(t1, X_true_1[:, i], color="black", linewidth=1.5, label="True")
        ax.plot(t1, X_hat_1[:, i], color="tab:red", linestyle="--", linewidth=1.3, label="Identified")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title("CLW: time-series comparison (true vs SINDy-identified)")
        if i == len(STATE_NAMES) - 1:
            ax.set_xlabel("t")
        if i == 0:
            ax.legend(loc="upper right", frameon=False)

    fig1.tight_layout()
    fig_timeseries_path = os.path.join(outdir, "fig_timeseries_comparison.png")
    fig1.savefig(fig_timeseries_path, dpi=200)
    plt.close(fig1)

    # --- Figure 2: phase-space (P,S,Z) with infinitesimal perturbation
    # True: x0, Identified: x0 + delta
    rng = np.random.default_rng(0)
    delta = rng.normal(size=4)
    delta = delta / np.linalg.norm(delta) * float(delta_norm)
    x0_pert = np.asarray(x0, dtype=float).copy()
    x0_pert[3] += float(delta_norm)   # perturb only C

    _, X_true_2 = _integrate(lambda t, x: clw_rhs(t, x, params), t_span=(0.0, float(T_phase)), dt=float(dt), x0=x0)
    _, X_hat_2 = _integrate(rhs_ident, t_span=(0.0, float(T_phase)), dt=float(dt), x0=x0_pert)

    fig2 = plt.figure(figsize=(8, 6))
    ax3d = fig2.add_subplot(111, projection="3d")
    ax3d.plot(
        X_true_2[:, 0],
        X_true_2[:, 1],
        X_true_2[:, 2],
        color="black",
        linewidth=1.6,
        alpha=0.85,
        label="CLW (x0)",
    )
    ax3d.plot(
        X_hat_2[:, 0],
        X_hat_2[:, 1],
        X_hat_2[:, 2],
        color="tab:red",
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        label=f"CLW (x0 + δ), ||δ||={float(delta_norm):.0e}",
    )
    ax3d.set_title(
        f"CLW: phase-space trajectory (P,S,Z)\nchaotic divergence from initial-condition perturbation (||δ||={float(delta_norm):.0e})"
    )
    ax3d.set_xlabel("P")
    ax3d.set_ylabel("S")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left", frameon=False)

    fig2.tight_layout()
    fig_phase_path = os.path.join(outdir, "fig_phase_space_PSZ.png")
    fig2.savefig(fig_phase_path, dpi=200)
    plt.close(fig2)

    return fig_timeseries_path, fig_phase_path


def main() -> None:
    generate_validation_figures()


if __name__ == "__main__":
    main()
