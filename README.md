# sindy-clw (curated CLW SINDy mini-research repo)

This repo is intentionally narrow: it contains only the code needed to run a small set of CLW / SINDy experiments, organized into distinct regimes.

All experiment entrypoints live in `experiments/`.

## Install

Install the dependencies in `requirements.txt`.

## Run the experiments

All outputs go to:

- `outputs/figures/`
- `outputs/tables/`

You can run everything in one go via:

```bash
python experiments/run_all.py
```

### Poster baseline (no noise, oracle derivatives, physics-informed library)

- Script: `experiments/poster_baseline.py`
- Outputs:
	- `outputs/figures/fig_poster_timeseries_overlay.png`
	- `outputs/figures/fig_poster_phase_space_chaos.png`

Notes:

- The long-horizon figure is a *chaos sensitivity* demo: it compares the true CLW system from $x_0$ vs the true CLW system from $x_0$ with a small perturbation only in $C$.

### State noise + oracle derivatives (physics-informed library)

- Script: `experiments/noise_state_oracle.py`
- Outputs:
	- `outputs/figures/fig_noise_state_oracle_error_vs_time.png`
	- `outputs/figures/fig_noise_state_oracle_timeseries_overlay.png` (shows $\eta\in\{0.001, 0.1\}$)
	- `outputs/figures/fig_noise_state_oracle_phase_space_eta0.001.png`
	- `outputs/figures/fig_noise_state_oracle_phase_space_eta0.1.png`
	- `outputs/tables/coef_recovery_state_oracle.csv`

Noise protocol: Gaussian noise is added to the observed states $X$; derivatives are **oracle** (computed from the true CLW RHS).

### State noise + numerical derivatives (distinct regime)

- Script: `experiments/noise_state_numerical.py`
- Output (table only):
	- `outputs/tables/coef_recovery_state_numerical.csv`

Noise protocol: Gaussian noise is added to $X$ and derivatives are estimated numerically from the noisy states (finite differences).

### Extended library comparison (no noise, oracle derivatives)

- Script: `experiments/extended_library.py`
- Outputs:
	- `outputs/tables/coef_recovery_extended_library.csv`
	- (optional) `outputs/figures/fig_extended_library_overlay.png`

This compares fits using the physics-informed library vs an extended library (products of basis terms).

## Code map

- `experiments/`: experiment entrypoints.
- `sindy_library/physics_informed.py`: authoritative physics-informed CLW library.
- `sindy_library/extended.py`: extended library construction.
- `sindy_utils.py`: shared fit/integration utilities.
- `plotting.py`: minimal plotting helpers used by experiments.
- `coeff_recovery.py`: ground-truth coefficients + coefficient recovery metrics.
