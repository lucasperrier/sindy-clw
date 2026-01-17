"""Export ablation summary CSV to a paper-friendly Markdown table.

Reads `outputs/ablation_summary.csv` produced by `ablation_experiment.py` and writes
`outputs/ablation_summary.md`.

Design goals:
- No extra dependencies (uses only stdlib).
- Column selection is configurable at the top of the file.
- Formats numeric columns in scientific notation by default.

Usage:
    python3 export_ablation_table.py

Optional:
    python3 export_ablation_table.py --in outputs/ablation_summary.csv --out outputs/ablation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Any


DEFAULT_COLUMNS = [
    "n_traj",
    "T",
    "n_reps",
    "n_traj_used_mean",
    "n_samples_used_mean",
    "nnz_mean",
    "fp_total_mean",
    "fn_total_mean",
    "l2_error_total_mean",
    "test_derivative_rmse_mean",
    "test_rollout_rmse_mean",
]


def _is_nan(x: float) -> bool:
    return bool(math.isnan(x))


def _parse_float(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def format_cell(col: str, v: str) -> str:
    """Format a CSV cell as Markdown.

    - Keep integers (n_traj, n_reps, nnz...) without scientific notation when possible.
    - For floats use a compact scientific format.
    """

    fv = _parse_float(v)

    # pass through non-numerics
    if fv is None:
        return "" if v is None else str(v)

    if _is_nan(fv):
        return ""

    # heuristics for integer-looking columns
    if col in {"n_traj", "n_reps"}:
        return str(int(round(fv)))

    if col.endswith("_mean") and any(
        col.startswith(prefix)
        for prefix in ("n_", "fp_", "fn_", "nnz")
    ):
        # counts/averages of ints: show with 1 decimal max
        if abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return f"{fv:.2f}"

    if col in {"T"}:
        # nice small floats
        if abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return f"{fv:g}"

    # Use scientific for tiny/huge; otherwise a compact fixed.
    afv = abs(fv)
    if afv != 0.0 and (afv < 1e-3 or afv >= 1e4):
        return f"{fv:.2e}"
    return f"{fv:.4g}"


def write_markdown_table(rows: list[dict[str, str]], columns: list[str], out_path: str) -> None:
    # header
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, sep]
    for r in rows:
        line = "| " + " | ".join(format_cell(c, r.get(c, "")) for c in columns) + " |"
        lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="outputs/ablation_summary.csv")
    p.add_argument("--out", dest="out_path", default="outputs/ablation_summary.md")
    p.add_argument(
        "--cols",
        dest="cols",
        default=",")
    args = p.parse_args(argv)

    columns = DEFAULT_COLUMNS
    if args.cols and args.cols != ",":
        columns = [c.strip() for c in args.cols.split(",") if c.strip()]

    with open(args.in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = [dict(r) for r in reader]

    # stable sort: (n_traj, T)
    def key(r: dict[str, str]) -> tuple[int, float]:
        n = int(float(r.get("n_traj", "0") or 0))
        T = float(r.get("T", "nan") or float("nan"))
        return (n, T)

    rows.sort(key=key)

    write_markdown_table(rows, columns, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
