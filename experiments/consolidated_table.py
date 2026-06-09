"""experiments.consolidated_table

Build a single consolidated comparison CSV from the per-experiment summary CSVs.

Reads existing summary tables (produced by each experiment) and joins them into
one master table for the paper.

Outputs
-------
- outputs/tables/consolidated_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import os


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    tab_dir = os.path.join(str(REPO_ROOT), "outputs", "tables")

    out_fields = [
        "regime", "eta",
        "nnz_mean", "nnz_std",
        "rel_l2_mean", "rel_l2_std",
        "tpr_mean", "fpr_mean", "exact_support_frac",
        "vf_error_mean", "vf_error_std",
    ]

    rows: list[dict] = []

    # --- Oracle derivatives ---
    oracle_path = os.path.join(tab_dir, "coef_recovery_state_oracle.csv")
    if os.path.isfile(oracle_path):
        for r in _read_csv(oracle_path):
            rows.append({
                "regime": "oracle",
                "eta": r["eta"],
                "nnz_mean": r["nnz_mean"], "nnz_std": r["nnz_std"],
                "rel_l2_mean": r["rel_l2_mean"], "rel_l2_std": r["rel_l2_std"],
                "tpr_mean": r["tpr_mean"], "fpr_mean": r["fpr_mean"],
                "exact_support_frac": r["exact_support_frac"],
                "vf_error_mean": r["vf_error_mean"], "vf_error_std": r["vf_error_std"],
            })

    # --- Numerical FD ---
    num_path = os.path.join(tab_dir, "coef_recovery_state_numerical.csv")
    if os.path.isfile(num_path):
        for r in _read_csv(num_path):
            rows.append({
                "regime": "numerical_fd",
                "eta": r["eta"],
                "nnz_mean": r["nnz_mean"], "nnz_std": r["nnz_std"],
                "rel_l2_mean": r["rel_l2_mean"], "rel_l2_std": r["rel_l2_std"],
                "tpr_mean": r["tpr_mean"], "fpr_mean": r["fpr_mean"],
                "exact_support_frac": r["exact_support_frac"],
                "vf_error_mean": r["vf_error_mean"], "vf_error_std": r["vf_error_std"],
            })

    # --- SINDy end-to-end (SFD) ---
    sfd_path = os.path.join(tab_dir, "coef_recovery_state_sindy_internal.csv")
    if os.path.isfile(sfd_path):
        for r in _read_csv(sfd_path):
            rows.append({
                "regime": "sindy_sfd",
                "eta": r["eta"],
                "nnz_mean": r["nnz_mean"], "nnz_std": r["nnz_std"],
                "rel_l2_mean": r["rel_l2_mean"], "rel_l2_std": r["rel_l2_std"],
                "tpr_mean": r["tpr_mean"], "fpr_mean": r["fpr_mean"],
                "exact_support_frac": r["exact_support_frac"],
                "vf_error_mean": r["vf_error_mean"], "vf_error_std": r["vf_error_std"],
            })

    # --- Extended library under noise ---
    ext_noise_path = os.path.join(tab_dir, "coef_recovery_extended_noise.csv")
    if os.path.isfile(ext_noise_path):
        for r in _read_csv(ext_noise_path):
            rows.append({
                "regime": "extended_lib_oracle",
                "eta": r["eta"],
                "nnz_mean": r["nnz_mean"], "nnz_std": r["nnz_std"],
                "rel_l2_mean": r["rel_l2_mean"], "rel_l2_std": r["rel_l2_std"],
                "tpr_mean": r["tpr_mean"], "fpr_mean": r["fpr_mean"],
                "exact_support_frac": r["exact_support_frac"],
                "vf_error_mean": r["vf_error_mean"], "vf_error_std": r["vf_error_std"],
            })

    # --- Incomplete library ---
    inc_path = os.path.join(tab_dir, "coef_recovery_incomplete_library.csv")
    if os.path.isfile(inc_path):
        for r in _read_csv(inc_path):
            rows.append({
                "regime": "incomplete_lib_oracle",
                "eta": r["eta"],
                "nnz_mean": r["nnz_mean"], "nnz_std": r["nnz_std"],
                "rel_l2_mean": r["rel_l2_mean"], "rel_l2_std": r["rel_l2_std"],
                "tpr_mean": r["tpr_mean"], "fpr_mean": r["fpr_mean"],
                "exact_support_frac": r["exact_support_frac"],
                "vf_error_mean": r["vf_error_mean"], "vf_error_std": r["vf_error_std"],
            })

    out_path = os.path.join(tab_dir, "consolidated_comparison.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote consolidated table ({len(rows)} rows) to {out_path}")


if __name__ == "__main__":
    main()
