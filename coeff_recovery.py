"""coeff_recovery.py

Minimal coefficient recovery helper.

We only need the ground-truth coefficient matrix aligned to the CLW library's
feature names, so the noise experiment can report:
- number of nonzero coefficients (nnz)
- relative coefficient error (L2 / Frobenius)

Everything else from the previous, more feature-rich analysis was removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def build_true_coefficients(feature_names: Sequence[str], params: dict[str, float]) -> np.ndarray:
    """Return the CLW ground-truth coefficient matrix Xi_true.

    Args:
        feature_names: names from `make_clw_library().get_feature_names(...)`.
        params: dict with keys {'Gd', 'gz', 'd'}.

    Returns:
        Xi_true with shape (4, n_features).

    Raises:
        KeyError if required features are missing.
    """

    names = [str(s) for s in list(feature_names)]
    idx = {name: j for j, name in enumerate(names)}

    required = [
        "1",
        "P",
        "S",
        "Z",
        "Z*S*cos(C)",
        "Z*P*cos(C)",
        "P*S*cos(C)",
        "(P*Z/S)*sin(C)",
    ]
    missing = [r for r in required if r not in idx]
    if missing:
        raise KeyError(f"Feature library is missing required CLW terms: {missing}")

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


@dataclass(frozen=True)
class CoefMetrics:
    nnz: int
    rel_l2: float


def coef_metrics(*, Xi_hat: np.ndarray, Xi_true: np.ndarray, nz_tol: float = 0.0) -> CoefMetrics:
    """Return simple metrics used in the paper-style tables."""
    Xi_hat = np.asarray(Xi_hat, dtype=float)
    Xi_true = np.asarray(Xi_true, dtype=float)
    nnz = int(np.sum(np.abs(Xi_hat) > float(nz_tol)))
    num = float(np.linalg.norm(Xi_hat - Xi_true))
    den = float(np.linalg.norm(Xi_true))
    rel_l2 = num / den if den > 0 else num
    return CoefMetrics(nnz=nnz, rel_l2=rel_l2)
