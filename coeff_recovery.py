"""Coefficient recovery utilities.

Goal: compare a known ground-truth coefficient matrix (Xi_true) against an
identified SINDy coefficient matrix (Xi_hat) in a robust way.

Robustness features:
- Reports the union of nonzero terms from truth and identified model.
- Handles missing/extra terms (true term not identified, spurious identified term).
- Provides absolute and relative errors per coefficient.
- Provides per-equation summary metrics (TP/FP/FN, precision/recall, L1/L2 errors).

This repo uses a fixed CustomLibrary (`make_clw_library`) so by default we can
construct Xi_true from (feature_names, params).
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RecoveryRow:
    equation: str
    term: str
    true: float
    identified: float
    abs_error: float
    rel_error: float
    status: str  # "TP", "FN", "FP", "TN" (TN usually omitted)


@dataclass(frozen=True)
class EquationSummary:
    equation: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    l1_error: float
    l2_error: float
    l1_true: float
    l2_true: float


def _as_1d_str_list(x: Sequence) -> list[str]:
    return [str(s) for s in list(x)]


def build_true_coefficients(feature_names: Sequence[str], params: dict[str, float]) -> np.ndarray:
    """Build CLW ground-truth coefficients aligned to `feature_names`.

    Returns:
        Xi_true: shape (4, n_features)

    Raises:
        KeyError if a required feature name is absent.
    """
    names = _as_1d_str_list(feature_names)
    idx = {name: j for j, name in enumerate(names)}

    required = ["1", "P", "S", "Z", "Z*S*cos(C)", "Z*P*cos(C)", "P*S*cos(C)", "(P*Z/S)*sin(C)"]
    missing = [r for r in required if r not in idx]
    if missing:
        raise KeyError(f"Feature library is missing required terms for CLW truth: {missing}")

    Gd = float(params["Gd"])
    gz = float(params["gz"])
    d = float(params["d"])

    Xi = np.zeros((4, len(names)), dtype=float)

    # P' = P - 2 Z S cos(C)
    Xi[0, idx["P"]] = 1.0
    Xi[0, idx["Z*S*cos(C)"]] = -2.0

    # S' = -Gd S + Z P cos(C)
    Xi[1, idx["S"]] = -Gd
    Xi[1, idx["Z*P*cos(C)"]] = 1.0

    # Z' = -gz Z + 2 P S cos(C)
    Xi[2, idx["Z"]] = -gz
    Xi[2, idx["P*S*cos(C)"]] = 2.0

    # C' = d - (P Z / S) sin(C)
    Xi[3, idx["1"]] = d
    Xi[3, idx["(P*Z/S)*sin(C)"]] = -1.0

    return Xi


def coefficient_recovery_rows(
    feature_names: Sequence[str],
    Xi_true: np.ndarray,
    Xi_hat: np.ndarray,
    *,
    equation_names: Sequence[str],
    nz_tol: float = 0.0,
    rel_floor: float = 1e-12,
    include_tn: bool = False,
) -> list[RecoveryRow]:
    """Create row-wise comparison.

    Union logic:
    - By default we include a row if either |true|>tol OR |identified|>tol.
    - This makes the table informative even when supports differ.

    Status definitions (per coefficient):
    - TP: both nonzero
    - FN: true nonzero, identified zero (missed true term)
    - FP: true zero, identified nonzero (spurious term)
    - TN: both zero (usually omitted)
    """
    feature_names = _as_1d_str_list(feature_names)
    Xi_true = np.asarray(Xi_true, dtype=float)
    Xi_hat = np.asarray(Xi_hat, dtype=float)

    if Xi_true.shape != Xi_hat.shape:
        raise ValueError(f"Shape mismatch: Xi_true {Xi_true.shape} vs Xi_hat {Xi_hat.shape}")
    if Xi_true.shape[0] != len(equation_names):
        raise ValueError("equation_names length must match number of rows in Xi")
    if Xi_true.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match number of columns in Xi")

    tol = float(nz_tol)
    rows: list[RecoveryRow] = []

    for i, eq in enumerate(equation_names):
        for j, term in enumerate(feature_names):
            t = float(Xi_true[i, j])
            h = float(Xi_hat[i, j])
            t_nz = abs(t) > tol
            h_nz = abs(h) > tol

            if not include_tn and (not t_nz) and (not h_nz):
                continue

            status = "TN"
            if t_nz and h_nz:
                status = "TP"
            elif t_nz and (not h_nz):
                status = "FN"
            elif (not t_nz) and h_nz:
                status = "FP"

            abs_err = abs(h - t)
            rel_err = abs_err / max(abs(t), float(rel_floor))

            rows.append(
                RecoveryRow(
                    equation=str(eq),
                    term=str(term),
                    true=t,
                    identified=h,
                    abs_error=float(abs_err),
                    rel_error=float(rel_err),
                    status=status,
                )
            )

    # Helpful ordering: group by equation, then show FN/FP first, then TP.
    status_rank = {"FN": 0, "FP": 1, "TP": 2, "TN": 3}
    rows.sort(key=lambda r: (r.equation, status_rank.get(r.status, 99), -abs(r.true), -abs(r.identified), r.term))
    return rows


def equation_summaries(
    rows: Iterable[RecoveryRow],
    *,
    equation_names: Sequence[str],
    Xi_true: np.ndarray,
    Xi_hat: np.ndarray,
    nz_tol: float = 0.0,
) -> list[EquationSummary]:
    """Compute per-equation support + coefficient error summaries."""
    tol = float(nz_tol)
    Xi_true = np.asarray(Xi_true, dtype=float)
    Xi_hat = np.asarray(Xi_hat, dtype=float)

    out: list[EquationSummary] = []
    for i, eq in enumerate(equation_names):
        t = Xi_true[i]
        h = Xi_hat[i]

        t_nz = np.abs(t) > tol
        h_nz = np.abs(h) > tol

        tp = int(np.sum(t_nz & h_nz))
        fp = int(np.sum((~t_nz) & h_nz))
        fn = int(np.sum(t_nz & (~h_nz)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0

        diff = h - t
        l1_error = float(np.sum(np.abs(diff)))
        l2_error = float(np.sqrt(np.sum(diff**2)))
        l1_true = float(np.sum(np.abs(t)))
        l2_true = float(np.sqrt(np.sum(t**2)))

        out.append(
            EquationSummary(
                equation=str(eq),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                l1_error=l1_error,
                l2_error=l2_error,
                l1_true=l1_true,
                l2_true=l2_true,
            )
        )

    return out


def print_recovery_table(rows: Sequence[RecoveryRow], *, max_rows_per_eq: int | None = None) -> None:
    """Pretty-print a compact table to stdout."""
    by_eq: dict[str, list[RecoveryRow]] = {}
    for r in rows:
        by_eq.setdefault(r.equation, []).append(r)

    def fmt(x: float) -> str:
        # Clean up machine-precision noise so reports are readable.
        # Anything below ~1e-12 is effectively zero for this repo's scales.
        if abs(x) < 1e-12:
            return "0"
        ax = abs(x)
        # Use a consistent, standard number of significant figures.
        # Scientific for very small/large, fixed-ish otherwise.
        if ax < 1e-3 or ax >= 1e3:
            return f"{x:.3g}"
        return f"{x:.6g}"

    for eq, eq_rows in by_eq.items():
        print(f"\n--- Coefficient recovery: {eq} ---")

        shown = eq_rows
        if max_rows_per_eq is not None:
            shown = shown[: int(max_rows_per_eq)]

        # Pre-format strings so we can compute widths for clean alignment.
        header = {
            "status": "status",
            "term": "term",
            "true": "true",
            "identified": "identified",
            "abs_error": "|err|",
            "rel_error": "rel_err",
        }

        formatted = []
        for r in shown:
            formatted.append(
                {
                    "status": str(r.status),
                    "term": str(r.term),
                    "true": fmt(r.true),
                    "identified": fmt(r.identified),
                    "abs_error": fmt(r.abs_error),
                    "rel_error": fmt(r.rel_error),
                }
            )

        # Column widths (include header).
        def col_width(key: str) -> int:
            return max(
                [len(header[key])] + [len(row[key]) for row in formatted],
                default=len(header[key]),
            )

        w_status = col_width("status")
        w_term = col_width("term")
        w_true = col_width("true")
        w_id = col_width("identified")
        w_abs = col_width("abs_error")
        w_rel = col_width("rel_error")

        # Header
        print(
            f"{header['status']:<{w_status}}  {header['term']:<{w_term}}  "
            f"{header['true']:>{w_true}}  {header['identified']:>{w_id}}  "
            f"{header['abs_error']:>{w_abs}}  {header['rel_error']:>{w_rel}}"
        )

        # Rows
        for row in formatted:
            print(
                f"{row['status']:<{w_status}}  {row['term']:<{w_term}}  "
                f"{row['true']:>{w_true}}  {row['identified']:>{w_id}}  "
                f"{row['abs_error']:>{w_abs}}  {row['rel_error']:>{w_rel}}"
            )

        if max_rows_per_eq is not None and len(eq_rows) > int(max_rows_per_eq):
            print(f"... ({len(eq_rows) - int(max_rows_per_eq)} more rows omitted)")


def print_equation_summaries(summaries: Sequence[EquationSummary]) -> None:
    print("\n=== Coefficient recovery summary (per equation) ===")
    # Use fixed-width columns (not tabs) for consistent alignment across terminals.
    header = ["eq", "TP", "FP", "FN", "precision", "recall", "L1_err", "L2_err"]

    def fmt_err(x: float) -> str:
        # Snap near-machine-precision values to 0 for readability.
        if abs(x) < 1e-12:
            return "0"
        return f"{x:.3g}"
    rows = []
    for s in summaries:
        rows.append(
            [
                str(s.equation),
                str(int(s.tp)),
                str(int(s.fp)),
                str(int(s.fn)),
                f"{s.precision:.3f}",
                f"{s.recall:.3f}",
                fmt_err(s.l1_error),
                fmt_err(s.l2_error),
            ]
        )

    widths = [
        max(len(header[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(header))
    ]

    def fmt_row(vals: list[str]) -> str:
        out = []
        for i, v in enumerate(vals):
            # left-align eq label, right-align numeric columns
            if i == 0:
                out.append(f"{v:<{widths[i]}}")
            else:
                out.append(f"{v:>{widths[i]}}")
        return "  ".join(out)

    print(fmt_row(header))
    for r in rows:
        print(fmt_row(r))


def save_recovery_csv(rows: Sequence[RecoveryRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["equation", "term", "status", "true", "identified", "abs_error", "rel_error"])
        for r in rows:
            w.writerow([r.equation, r.term, r.status, r.true, r.identified, r.abs_error, r.rel_error])


def save_recovery_markdown(rows: Sequence[RecoveryRow], path: str) -> None:
    # Simple Markdown table, grouped by equation.
    by_eq: dict[str, list[RecoveryRow]] = {}
    for r in rows:
        by_eq.setdefault(r.equation, []).append(r)

    def fmt(x: float) -> str:
        if abs(x) < 1e-12:
            return "0"
        ax = abs(x)
        if ax < 1e-3 or ax >= 1e3:
            return f"{x:.3g}"
        return f"{x:.6g}"

    lines: list[str] = []
    lines.append("# Coefficient recovery\n")
    for eq, eq_rows in by_eq.items():
        lines.append(f"## {eq}\n")
        lines.append("| status | term | true | identified | abs_error | rel_error |\n")
        lines.append("|---:|:---|---:|---:|---:|---:|\n")
        for r in eq_rows:
            lines.append(
                f"| {r.status} | `{r.term}` | {fmt(r.true)} | {fmt(r.identified)} | {fmt(r.abs_error)} | {fmt(r.rel_error)} |\n"
            )
        lines.append("\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
