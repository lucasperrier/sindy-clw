import numpy as np
from scipy.integrate import solve_ivp

from clw import clw_rhs

def simulate_short_bursts(params,
                          n_traj=200,
                          T=5.0,
                          dt=0.01,
                          seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0, T + dt, dt)

    X_list, dX_list = [], []

    for _ in range(n_traj):
        x0 = rng.uniform(low=0.5, high=2.0, size=4)
        sol = solve_ivp(
            lambda t, x: clw_rhs(t, x, params),
            (0, T), x0, t_eval=t_eval,
            rtol=1e-9, atol=1e-12
        )

        X = sol.y.T
        dX = np.array([clw_rhs(t, x, params) for t, x in zip(sol.t, X)])

        X_list.append(X)
        dX_list.append(dX)

    return X_list, dX_list
