import numpy as np

def clw_rhs(t, x, params):
    """
    Chen–Lin–White system (Eqs. 12–15)
    x = [P, S, Z, C]
    """
    P, S, Z, C = x
    Gd = params["Gd"]
    gz = params["gz"]
    d  = params["d"]

    # Avoid division by zero in dC/dt
    if abs(S) < 1e-8:
        S = 1e-8

    dP = P - 2.0 * Z * S * np.cos(C)
    dS = -Gd * S + Z * P * np.cos(C)
    dZ = -gz * Z + 2.0 * P * S * np.cos(C)
    dC = d - (P * Z / S) * np.sin(C)

    return np.array([dP, dS, dZ, dC])
