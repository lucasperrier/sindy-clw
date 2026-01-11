import numpy as np
import pysindy as ps

def _safe_inv(x, eps=1e-8):
    # elementwise safe 1/x
    x = np.asarray(x, dtype=float)
    tiny = eps
    # keep sign when near zero; avoid division by 0
    denom = np.where(np.abs(x) < tiny, np.sign(x) * tiny + (x == 0) * tiny, x)
    return 1.0 / denom


def _col(x) -> np.ndarray:
    """Convert PySINDy feature inputs into a 1D array of samples."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    # If we get a full state matrix here, callers should be using _as_state.
    raise ValueError(f"Expected a single feature column, got shape {x.shape}")


def _as_state(X: np.ndarray) -> np.ndarray:
    """Ensure X is (n_samples, 4) for manual calls like model.predict(X)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        if X.shape[0] != 4:
            raise ValueError(f"Expected state dimension 4, got shape {X.shape}")
        return X[None, :]
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X shape (n_samples, 4), got shape {X.shape}")
    return X


def make_clw_library(eps_inv: float = 1e-8) -> ps.CustomLibrary:
    """
    Custom feature library for CLW (Eqs. 12–15):
      P' = P - 2 Z S cos(C)
      S' = -Gd S + Z P cos(C)
      Z' = -gz Z + 2 P S cos(C)
      C' = d - (P Z / S) sin(C)

    State ordering must be: x = [P, S, Z, C]

    Notes:
    - Functions are written to accept both X shape (n_samples, 4) and (4,).
    - Each function returns a 1D array of length n_samples (or length 1 for 1D input).
    """
    # IMPORTANT: PySINDy CustomLibrary expects each function to accept
    # the state variables as separate positional arguments (columns), not a full X.
    # So signatures must be f(P, S, Z, C) and operate elementwise.
    def one(p, s, z, c):
        # Explicit constant feature column.
        # (Avoid relying on an implicit optimizer bias so it can be masked per-equation.)
        p = _col(p)
        return np.ones_like(p)

    def P(p, s, z, c):
        return _col(p)

    def S(p, s, z, c):
        return _col(s)

    def Z(p, s, z, c):
        return _col(z)

    def cosC(p, s, z, c):
        return np.cos(_col(c))

    def sinC(p, s, z, c):
        return np.sin(_col(c))

    def ZS_cosC(p, s, z, c):
        s = _col(s)
        z = _col(z)
        c = _col(c)
        return z * s * np.cos(c)

    def ZP_cosC(p, s, z, c):
        p = _col(p)
        z = _col(z)
        c = _col(c)
        return z * p * np.cos(c)

    def PS_cosC(p, s, z, c):
        p = _col(p)
        s = _col(s)
        c = _col(c)
        return p * s * np.cos(c)

    def PZ_over_S_sinC(p, s, z, c):
        p = _col(p)
        s = _col(s)
        z = _col(z)
        c = _col(c)
        return (p * z * _safe_inv(s, eps_inv)) * np.sin(c)

    funcs = [
        one,
        P, S, Z,
        cosC, sinC,
        ZS_cosC, ZP_cosC, PS_cosC,
        PZ_over_S_sinC,
    ]

    names = [
        lambda *_: "1",
        lambda *_: "P",
        lambda *_: "S",
        lambda *_: "Z",
        lambda *_: "cos(C)",
        lambda *_: "sin(C)",
        lambda *_: "Z*S*cos(C)",
        lambda *_: "Z*P*cos(C)",
        lambda *_: "P*S*cos(C)",
        lambda *_: "(P*Z/S)*sin(C)",
    ]

    return ps.CustomLibrary(library_functions=funcs, function_names=names)





