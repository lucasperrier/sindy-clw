"""lyapunov.py

Maximal Lyapunov exponent estimation for the CLW system via QR decomposition
of the tangent-space evolution along a reference trajectory.

Algorithm (Benettin et al., 1980):
1. Integrate the CLW system + variational equation (state + tangent vectors).
2. Periodically QR-decompose the tangent matrix to prevent collapse.
3. Accumulate log of the diagonal of R → Lyapunov exponents.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def clw_jacobian(x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Jacobian of the CLW RHS at state x = [P, S, Z, C]."""
    P, S, Z, C = x
    Gd = float(params["Gd"])
    gz = float(params["gz"])

    S_safe = S if abs(S) > 1e-8 else 1e-8

    cosC = np.cos(C)
    sinC = np.sin(C)

    # dP/d(P,S,Z,C)
    J = np.zeros((4, 4), dtype=float)

    # P' = P - 2 Z S cos(C)
    J[0, 0] = 1.0                       # dP'/dP
    J[0, 1] = -2.0 * Z * cosC           # dP'/dS
    J[0, 2] = -2.0 * S * cosC           # dP'/dZ
    J[0, 3] = 2.0 * Z * S * sinC        # dP'/dC

    # S' = -Gd S + Z P cos(C)
    J[1, 0] = Z * cosC                   # dS'/dP
    J[1, 1] = -Gd                        # dS'/dS
    J[1, 2] = P * cosC                   # dS'/dZ
    J[1, 3] = -Z * P * sinC              # dS'/dC

    # Z' = -gz Z + 2 P S cos(C)
    J[2, 0] = 2.0 * S * cosC             # dZ'/dP
    J[2, 1] = 2.0 * P * cosC             # dZ'/dS
    J[2, 2] = -gz                        # dZ'/dZ
    J[2, 3] = -2.0 * P * S * sinC        # dZ'/dC

    # C' = d - (P Z / S) sin(C)
    J[3, 0] = -(Z / S_safe) * sinC       # dC'/dP
    J[3, 1] = (P * Z / S_safe**2) * sinC  # dC'/dS
    J[3, 2] = -(P / S_safe) * sinC       # dC'/dZ
    J[3, 3] = -(P * Z / S_safe) * cosC   # dC'/dC

    return J


def _augmented_rhs(t, y, params):
    """RHS for the augmented system: 4 state + 4x4 tangent = 20 variables."""
    x = y[:4]
    Q = y[4:].reshape(4, 4)

    from clw import clw_rhs
    dx = clw_rhs(t, x, params)
    J = clw_jacobian(x, params)
    dQ = J @ Q

    return np.concatenate([dx, dQ.ravel()])


def compute_lyapunov_exponents(
    params: dict[str, float],
    *,
    x0: np.ndarray | None = None,
    T: float = 500.0,
    dt_renorm: float = 1.0,
    transient: float = 50.0,
) -> np.ndarray:
    """Estimate all 4 Lyapunov exponents of the CLW system.

    Args:
        params: CLW parameter dict with keys {'Gd', 'gz', 'd'}.
        x0: initial condition (4,). Default: [1.2, 1.0, 0.8, 0.5].
        T: total integration time after transient.
        dt_renorm: QR renormalization interval.
        transient: discard this initial time to reach the attractor.

    Returns:
        Array of 4 Lyapunov exponents (largest first).
    """
    if x0 is None:
        x0 = np.array([1.2, 1.0, 0.8, 0.5], dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()

    # --- discard transient ---
    if transient > 0:
        from clw import clw_rhs
        sol = solve_ivp(
            lambda t, y: clw_rhs(t, y, params),
            (0.0, transient),
            x0,
            rtol=1e-9, atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x0 = sol.y[:, -1].copy()

    # --- main integration with QR renormalization ---
    Q = np.eye(4, dtype=float)
    n_steps = int(T / dt_renorm)
    log_diag = np.zeros((n_steps, 4), dtype=float)

    x = x0.copy()
    for k in range(n_steps):
        y0 = np.concatenate([x, Q.ravel()])
        sol = solve_ivp(
            lambda t, y: _augmented_rhs(t, y, params),
            (0.0, dt_renorm),
            y0,
            rtol=1e-9, atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        y_end = sol.y[:, -1]
        x = y_end[:4].copy()
        M = y_end[4:].reshape(4, 4)

        Q, R = np.linalg.qr(M)
        # Ensure positive diagonal of R for consistent sign
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs
        R = np.diag(signs) @ R

        log_diag[k] = np.log(np.abs(np.diag(R)))

    lyap = np.cumsum(log_diag, axis=0) / (np.arange(1, n_steps + 1)[:, None] * dt_renorm)
    # Return converged values (last step), sorted largest first
    exponents = lyap[-1]
    return np.sort(exponents)[::-1]


def max_lyapunov_exponent(
    params: dict[str, float],
    **kwargs,
) -> float:
    """Return the maximal Lyapunov exponent (scalar)."""
    return float(compute_lyapunov_exponents(params, **kwargs)[0])


def lyapunov_time(params: dict[str, float], **kwargs) -> float:
    """Return the Lyapunov time T_λ = 1 / λ_max."""
    lam = max_lyapunov_exponent(params, **kwargs)
    if lam <= 0:
        raise ValueError(f"Non-positive max Lyapunov exponent: {lam}")
    return 1.0 / lam
