"""experiments.paper_figures

Generate all paper-quality figures from existing experiment outputs.

Reads CSVs and re-simulates only what is needed for trajectory overlays.

Figures produced
----------------
Fig 1: CLW system overview (sample trajectory + phase portrait)
Fig 2: Coefficient rel_l2 vs η, one curve per derivative regime
Fig 3: Support TPR vs η, same layout
Fig 4: Short-horizon overlay at η=0.01, all 3 regimes (3×4 grid)
Fig 5: Error-vs-time in Lyapunov-time units, all regimes overlaid
Fig 6: Library ablation — rel_l2 for physics-informed vs extended vs incomplete
Fig S1: SFD sensitivity heatmap
Fig S2: Long-horizon chaos demonstration
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from clw_model.clw import clw_rhs
from clw_model.sindy_utils import CLWParams, STATE_NAMES, integrate
from clw_model.lyapunov import max_lyapunov_exponent


@dataclass(frozen=True)
class FigConfig:
    params: CLWParams = CLWParams()
    dt: float = 0.01
    x0: tuple[float, float, float, float] = (1.2, 1.0, 0.8, 0.5)

    T_short: float = 8.0
    T_long: float = 100.0
    T_error: float = 20.0
    delta_C: float = 1e-6

    out_dir: str = os.path.join(str(REPO_ROOT), "outputs", "figures")
    tab_dir: str = os.path.join(str(REPO_ROOT), "outputs", "tables")


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_eta_metric(rows: list[dict], metric: str) -> tuple[list[float], list[float], list[float]]:
    """Return (etas, means, stds) from summary CSV rows."""
    etas, means, stds = [], [], []
    for r in rows:
        etas.append(float(r["eta"]))
        means.append(float(r[f"{metric}_mean"]))
        std_key = f"{metric}_std"
        stds.append(float(r[std_key]) if std_key in r else 0.0)
    return etas, means, stds


# ── Fig 1: CLW overview ─────────────────────────────────────────────
def fig1_clw_overview(cfg: FigConfig) -> None:
    params = cfg.params.as_dict()
    rhs = lambda t, x: clw_rhs(t, x, params)
    t, X = integrate(rhs, dt=cfg.dt, T=cfg.T_short, x0=np.asarray(cfg.x0, dtype=float))

    fig = plt.figure(figsize=(14, 5))

    # Left: time series
    ax_ts = fig.add_axes([0.06, 0.12, 0.42, 0.80])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, name in enumerate(STATE_NAMES):
        ax_ts.plot(t, X[:, i], linewidth=1.5, color=colors[i], label=name)
    ax_ts.set_xlabel("t", fontsize=12)
    ax_ts.set_ylabel("State", fontsize=12)
    ax_ts.set_title("(a) CLW time series", fontsize=13)
    ax_ts.legend(loc="upper right", frameon=False, fontsize=11)
    ax_ts.grid(True, alpha=0.2)

    # Right: 3D phase portrait (P, S, Z)
    _, X_long = integrate(rhs, dt=cfg.dt, T=cfg.T_long, x0=np.asarray(cfg.x0, dtype=float))
    ax_3d = fig.add_axes([0.55, 0.10, 0.42, 0.85], projection="3d")
    ax_3d.plot(X_long[:, 0], X_long[:, 1], X_long[:, 2], linewidth=0.4, color="black", alpha=0.7)
    ax_3d.set_xlabel("P", fontsize=10)
    ax_3d.set_ylabel("S", fontsize=10)
    ax_3d.set_zlabel("Z", fontsize=10)
    ax_3d.set_title("(b) Phase portrait (P, S, Z)", fontsize=13)

    fig.savefig(os.path.join(cfg.out_dir, "fig1_clw_overview.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig1_clw_overview.png"), dpi=200)
    plt.close(fig)
    print("  Fig 1 done")


# ── Fig 2: rel_l2 vs η ──────────────────────────────────────────────
def fig2_rel_l2_vs_eta(cfg: FigConfig) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    regime_files = [
        ("Oracle", "coef_recovery_state_oracle.csv", "tab:blue", "o"),
        ("Numerical FD", "coef_recovery_state_numerical.csv", "tab:orange", "s"),
        ("SINDy SFD", "coef_recovery_state_sindy_internal.csv", "tab:green", "^"),
    ]
    for label, fname, color, marker in regime_files:
        path = os.path.join(cfg.tab_dir, fname)
        if not os.path.isfile(path):
            continue
        rows = _read_csv(path)
        etas, means, stds = _extract_eta_metric(rows, "rel_l2")
        ax.errorbar(etas, means, yerr=stds, label=label, color=color, marker=marker,
                     capsize=3, linewidth=1.5, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Noise level $\eta$", fontsize=12)
    ax.set_ylabel(r"Relative coefficient error $\|\hat{\Xi} - \Xi\|_F / \|\Xi\|_F$", fontsize=12)
    ax.set_title("Coefficient recovery vs noise level", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.2, which="both")
    fig.tight_layout()

    fig.savefig(os.path.join(cfg.out_dir, "fig2_rel_l2_vs_eta.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig2_rel_l2_vs_eta.png"), dpi=200)
    plt.close(fig)
    print("  Fig 2 done")


# ── Fig 3: TPR vs η ─────────────────────────────────────────────────
def fig3_tpr_vs_eta(cfg: FigConfig) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    regime_files = [
        ("Oracle", "coef_recovery_state_oracle.csv", "tab:blue", "o"),
        ("Numerical FD", "coef_recovery_state_numerical.csv", "tab:orange", "s"),
        ("SINDy SFD", "coef_recovery_state_sindy_internal.csv", "tab:green", "^"),
    ]
    for label, fname, color, marker in regime_files:
        path = os.path.join(cfg.tab_dir, fname)
        if not os.path.isfile(path):
            continue
        rows = _read_csv(path)
        etas = [float(r["eta"]) for r in rows]
        tpr = [float(r["tpr_mean"]) for r in rows]
        ax.plot(etas, tpr, label=label, color=color, marker=marker, linewidth=1.5, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Noise level $\eta$", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Support recovery vs noise level", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.2, which="both")
    fig.tight_layout()

    fig.savefig(os.path.join(cfg.out_dir, "fig3_tpr_vs_eta.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig3_tpr_vs_eta.png"), dpi=200)
    plt.close(fig)
    print("  Fig 3 done")


# ── Fig 4: Short-horizon overlay (3 regimes × 4 states) at η=0.01 ──
# This would require refitting models. Instead we create it from existing figures
# or skip it if models aren't cached. For the paper we note this requires
# running the individual experiments which produce their own overlays.
# We create a placeholder that combines error-vs-time from three regimes.
def fig4_short_horizon_error(cfg: FigConfig) -> None:
    """Error vs time for all three derivative regimes, overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute Lyapunov time for the x-axis normalization
    params = cfg.params.as_dict()
    try:
        lam_max = max_lyapunov_exponent(params, T=200.0, dt_renorm=0.5)
        T_lyap = 1.0 / lam_max if lam_max > 0 else None
    except Exception:
        T_lyap = None

    rhs_true = lambda t, x: clw_rhs(t, x, params)
    t, X_true = integrate(rhs_true, dt=cfg.dt, T=cfg.T_error, x0=np.asarray(cfg.x0, dtype=float))

    x_axis = t / T_lyap if T_lyap else t
    x_label = r"$t / T_\lambda$" if T_lyap else "t"

    # We can't refit models here, but we produce the structure.
    # The individual experiment scripts produce the actual error-vs-time figures.
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(r"$\|x(t) - \hat{x}(t)\|_2$", fontsize=12)
    ax.set_title("Trajectory error in Lyapunov-time units", fontsize=13)
    if T_lyap:
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label=r"$T_\lambda$")
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "fig4_error_vs_lyapunov_time.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig4_error_vs_lyapunov_time.png"), dpi=200)
    plt.close(fig)
    print("  Fig 4 done (skeleton)")


# ── Fig 5: Library ablation bars ─────────────────────────────────────
def fig5_library_ablation(cfg: FigConfig) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Extended library (clean, oracle)
    ext_path = os.path.join(cfg.tab_dir, "coef_recovery_extended_library.csv")
    ext_noise_path = os.path.join(cfg.tab_dir, "coef_recovery_extended_noise.csv")
    inc_path = os.path.join(cfg.tab_dir, "coef_recovery_incomplete_library.csv")

    data = {}
    if os.path.isfile(ext_path):
        rows = _read_csv(ext_path)
        for r in rows:
            case = str(r["case"])
            data[case] = {"rel_l2": float(r["rel_l2_mean"]), "rel_l2_std": float(r["rel_l2_std"])}

    if os.path.isfile(ext_noise_path):
        rows = _read_csv(ext_noise_path)
        # Pick eta=0.01 as representative
        for r in rows:
            if abs(float(r["eta"]) - 0.01) < 1e-6:
                data["extended_noise_eta0.01"] = {"rel_l2": float(r["rel_l2_mean"]), "rel_l2_std": float(r["rel_l2_std"])}

    if os.path.isfile(inc_path):
        rows = _read_csv(inc_path)
        for r in rows:
            if abs(float(r["eta"]) - 0.01) < 1e-6:
                data["incomplete_eta0.01"] = {"rel_l2": float(r["rel_l2_mean"]), "rel_l2_std": float(r["rel_l2_std"])}

    if data:
        labels = list(data.keys())
        vals = [data[k]["rel_l2"] for k in labels]
        errs = [data[k]["rel_l2_std"] for k in labels]

        # Nicer labels
        label_map = {
            "physics_informed": "Physics-\ninformed",
            "extended_degree_2": "Extended\n(clean)",
            "extended_noise_eta0.01": "Extended\n(η=0.01)",
            "incomplete_eta0.01": "Incomplete\n(η=0.01)",
        }
        nice_labels = [label_map.get(l, l) for l in labels]

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(labels)]
        bars = ax.bar(range(len(labels)), vals, yerr=errs, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(nice_labels, fontsize=10)
        ax.set_ylabel(r"Relative coefficient error", fontsize=12)
        ax.set_title("Library ablation: coefficient recovery", fontsize=13)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2, axis="y", which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "fig5_library_ablation.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig5_library_ablation.png"), dpi=200)
    plt.close(fig)
    print("  Fig 5 done")


# ── Fig 6: Sample-size ablation ─────────────────────────────────────
def fig6_sample_size(cfg: FigConfig) -> None:
    path = os.path.join(cfg.tab_dir, "coef_recovery_sample_size.csv")
    if not os.path.isfile(path):
        print("  Fig 6 skipped (no sample-size data)")
        return

    rows = _read_csv(path)
    n_trajs = [int(r["n_traj"]) for r in rows]
    rel_l2 = [float(r["rel_l2_mean"]) for r in rows]
    rel_l2_std = [float(r["rel_l2_std"]) for r in rows]
    tpr = [float(r["tpr_mean"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.errorbar(n_trajs, rel_l2, yerr=rel_l2_std, marker="o", capsize=3, linewidth=1.5, color="tab:blue")
    ax1.set_xlabel("Number of trajectories", fontsize=12)
    ax1.set_ylabel(r"Relative coefficient error", fontsize=12)
    ax1.set_title(r"(a) Coefficient error vs data volume ($\eta$=0.01)", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.2, which="both")

    ax2.plot(n_trajs, tpr, marker="s", linewidth=1.5, color="tab:orange")
    ax2.set_xlabel("Number of trajectories", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title(r"(b) Support recovery vs data volume ($\eta$=0.01)", fontsize=12)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "fig6_sample_size_ablation.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig6_sample_size_ablation.png"), dpi=200)
    plt.close(fig)
    print("  Fig 6 done")


# ── Fig S1: SFD sensitivity heatmap ─────────────────────────────────
def figS1_sfd_sensitivity(cfg: FigConfig) -> None:
    path = os.path.join(cfg.tab_dir, "sfd_sensitivity.csv")
    if not os.path.isfile(path):
        print("  Fig S1 skipped (no SFD sensitivity data)")
        return

    rows = _read_csv(path)

    etas = sorted(set(float(r["eta"]) for r in rows))

    fig, axes = plt.subplots(1, len(etas), figsize=(5 * len(etas), 4), sharey=True)
    if len(etas) == 1:
        axes = [axes]

    for ax, eta in zip(axes, etas):
        sub = [r for r in rows if abs(float(r["eta"]) - eta) < 1e-10]
        wls = sorted(set(int(r["window_length"]) for r in sub))
        pos = sorted(set(int(r["polyorder"]) for r in sub))

        grid = np.full((len(pos), len(wls)), np.nan)
        for r in sub:
            i = pos.index(int(r["polyorder"]))
            j = wls.index(int(r["window_length"]))
            grid[i, j] = float(r["rel_l2_mean"])

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(wls)))
        ax.set_xticklabels([str(w) for w in wls])
        ax.set_yticks(range(len(pos)))
        ax.set_yticklabels([str(p) for p in pos])
        ax.set_xlabel("Window length")
        if ax == axes[0]:
            ax.set_ylabel("Polyorder")
        ax.set_title(f"η={eta:g}")

        # Annotate cells
        for i in range(len(pos)):
            for j in range(len(wls)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if grid[i, j] > np.nanmedian(grid) else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="rel_l2")

    fig.suptitle("SFD sensitivity: rel_l2 by window length × polyorder", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "figS1_sfd_sensitivity.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(cfg.out_dir, "figS1_sfd_sensitivity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig S1 done")


# ── Fig S2: Long-horizon chaos demo ─────────────────────────────────
def figS2_chaos_demo(cfg: FigConfig) -> None:
    params = cfg.params.as_dict()
    rhs = lambda t, x: clw_rhs(t, x, params)

    x0 = np.asarray(cfg.x0, dtype=float)
    x0_pert = x0.copy()
    x0_pert[3] += float(cfg.delta_C)

    t, X_a = integrate(rhs, dt=cfg.dt, T=cfg.T_long, x0=x0)
    _, X_b = integrate(rhs, dt=cfg.dt, T=cfg.T_long, x0=x0_pert)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i, name in enumerate(STATE_NAMES):
        axs[i].plot(t, X_a[:, i], color="black", linewidth=1.0, label="x₀")
        axs[i].plot(t, X_b[:, i], color="tab:red", linewidth=1.0, alpha=0.7, label=f"x₀ + ΔC={cfg.delta_C:.0e}")
        axs[i].set_ylabel(name, fontsize=11)
        axs[i].grid(True, alpha=0.2)
        if i == 0:
            axs[i].set_title("Long-horizon chaos demonstration", fontsize=13)
            axs[i].legend(loc="upper right", frameon=False, fontsize=10)
    axs[-1].set_xlabel("t", fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "figS2_chaos_demo.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "figS2_chaos_demo.png"), dpi=200)
    plt.close(fig)
    print("  Fig S2 done")


# ── Fig 7: FPR vs η ─────────────────────────────────────────────────
def fig7_fpr_vs_eta(cfg: FigConfig) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    regime_files = [
        ("Oracle", "coef_recovery_state_oracle.csv", "tab:blue", "o"),
        ("Numerical FD", "coef_recovery_state_numerical.csv", "tab:orange", "s"),
        ("SINDy SFD", "coef_recovery_state_sindy_internal.csv", "tab:green", "^"),
    ]
    for label, fname, color, marker in regime_files:
        path = os.path.join(cfg.tab_dir, fname)
        if not os.path.isfile(path):
            continue
        rows = _read_csv(path)
        etas = [float(r["eta"]) for r in rows]
        fpr = [float(r["fpr_mean"]) for r in rows]
        ax.plot(etas, fpr, label=label, color=color, marker=marker, linewidth=1.5, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Noise level $\eta$", fontsize=12)
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("False positives vs noise level", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.2, which="both")
    fig.tight_layout()

    fig.savefig(os.path.join(cfg.out_dir, "fig7_fpr_vs_eta.pdf"), dpi=300)
    fig.savefig(os.path.join(cfg.out_dir, "fig7_fpr_vs_eta.png"), dpi=200)
    plt.close(fig)
    print("  Fig 7 done")


def main() -> None:
    cfg = FigConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("Generating paper figures...")

    fig1_clw_overview(cfg)
    fig2_rel_l2_vs_eta(cfg)
    fig3_tpr_vs_eta(cfg)
    fig4_short_horizon_error(cfg)
    fig5_library_ablation(cfg)
    fig6_sample_size(cfg)
    fig7_fpr_vs_eta(cfg)
    figS1_sfd_sensitivity(cfg)
    figS2_chaos_demo(cfg)

    print(f"\nAll figures written to {cfg.out_dir}")


if __name__ == "__main__":
    main()
