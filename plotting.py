"""plotting.py

Minimal plotting helpers.

Only what the experiments need:
- short-horizon time-series overlays
- error-vs-time
- PSZ phase-space plot with optional perturbation comparison

All plots are saved using the non-interactive Agg backend.
"""

from __future__ import annotations

import numpy as np

from sindy_utils import STATE_NAMES


def plot_error_vs_time(*, curves: dict[float, tuple[np.ndarray, np.ndarray]], outpath: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0))
    for eta, (t, err) in curves.items():
        ax.plot(t, err, linewidth=1.4, label=f"η={eta:g}")

    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$||\hat x(t) - x(t)||_2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_timeseries_overlay_two(
    *,
    t: np.ndarray,
    X_true: np.ndarray,
    X_hat: np.ndarray,
    outpath: str,
    title: str,
    label_true: str = "True",
    label_hat: str = "Identified",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(9.0, 8.0), sharex=True)
    for i, name in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(t, X_true[:, i], color="black", linewidth=1.5, label=label_true)
        ax.plot(t, X_hat[:, i], color="tab:red", linestyle="--", linewidth=1.3, label=label_hat)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", frameon=False)
        if i == 3:
            ax.set_xlabel("t")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_timeseries_overlay_three(
    *,
    t: np.ndarray,
    X_true: np.ndarray,
    X_hat_low: np.ndarray,
    X_hat_high: np.ndarray,
    eta_low: float,
    eta_high: float,
    outpath: str,
    title: str,
) -> None:
    """Legacy overlay: True + (η_low) + (η_high) in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(9.0, 8.0), sharex=True)
    for i, name in enumerate(STATE_NAMES):
        ax = axs[i]
        ax.plot(t, X_true[:, i], color="black", linewidth=1.5, label="True")
        ax.plot(t, X_hat_low[:, i], color="tab:blue", linestyle="--", linewidth=1.2, label=f"Identified (η={eta_low:g})")
        ax.plot(t, X_hat_high[:, i], color="tab:red", linestyle="--", linewidth=1.2, label=f"Identified (η={eta_high:g})")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", frameon=False)
        if i == 3:
            ax.set_xlabel("t")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_phase_space_psz(
    *,
    X_a: np.ndarray,
    X_b: np.ndarray,
    outpath: str,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(X_a[:, 0], X_a[:, 1], X_a[:, 2], color="black", linewidth=1.6, alpha=0.85, label=label_a)
    ax.plot(X_b[:, 0], X_b[:, 1], X_b[:, 2], color="tab:red", linestyle="--", linewidth=1.6, alpha=0.85, label=label_b)

    ax.set_xlabel("P")
    ax.set_ylabel("S")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
