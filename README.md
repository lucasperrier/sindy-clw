# sindy-clw (minimal SINDy identification)

This repo is a hard-pruned, minimal example demonstrating the core SINDy claim:

> With noise-free simulated data and **oracle derivatives** from the CLW system, sparse regression (SINDy/STLSQ) can recover a parsimonious model in the provided candidate library.

## What’s included

- `clw.py`: the ground-truth CLW ODE right-hand side.
- `data.py`: simulation of many short trajectories using `solve_ivp`, plus oracle derivatives by re-evaluating `clw_rhs` along each trajectory.
- `sindy_clw_lib.py`: the candidate feature library used by SINDy.
- `main_experiment.py`: end-to-end script: simulate → build library → fit SINDy (STLSQ) → print equations → save coefficients.

## Quickstart

### 1) Environment

You need Python plus:

- `numpy`
- `scipy`
- `pysindy`

Install however you manage environments (venv/conda). This repo purposely avoids extra tooling.

### 2) Run the identification

```bash
python main_experiment.py
```

Expected terminal output looks like:

```
=== Identified SINDy Model (CLW) ===
selected_threshold=..., nnz=..., mse=..., score=...
(x0)' = ...
(x1)' = ...
(x2)' = ...
(x3)' = ...

Saved identified coefficients to: outputs/identified_model.npz
```

### 3) Saved result artifact

The script writes:

- `outputs/identified_model.npz`

That NPZ contains:

- `coefficients`: array of shape `(4, n_features)`
- `feature_names`: list/array of feature name strings (length `n_features`)
- `threshold`: the selected STLSQ sparsity threshold

You can inspect it with:

```bash
python - <<'PY'
import numpy as np
z = np.load('outputs/identified_model.npz', allow_pickle=True)
print('keys:', z.files)
print('coefficients shape:', z['coefficients'].shape)
print('threshold:', z['threshold'])
print('feature_names:', list(z['feature_names']))
PY
```

## Notes on the “oracle derivative” setup

- Trajectories are integrated with `solve_ivp`.
- Derivatives are *not* estimated by finite differences.
- Instead, the derivative at each sampled time is computed by re-evaluating the known RHS `clw_rhs(t, x, params)`.

That is the cleanest setting for verifying sparse regression recovers the correct functional form.
# sindy-clw
