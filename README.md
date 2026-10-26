# Reproducibility package: Absorbing margin-liquidation cascades

This repository contains the code, fixed random seeds, numerical outputs, validation checks, and figure-generation scripts for the study of finite-size effects, heavy-tailed exposures, precursor avalanches, and transient impact in absorbing margin-liquidation cascades.

The package is self-contained and contains computational materials only.

## Quick start

Create the environment with either:

```bash
conda env create -f environment.yml
conda activate absorbing-margin
```

or install the Python dependencies listed in `requirements.txt`.

Run a fast integrity check:

```bash
python run_all.py --quick
```

This performs analytic smoke tests, regenerates Figures 1–9 from the bundled numerical results, checks the principal quantitative targets, validates the targeted controls, and compares regenerated figures with the bundled expected figures.

To rerun all stochastic experiments from their fixed seeds:

```bash
python run_all.py --full
```

The full workflow regenerates every numerical output in `results/`, rebuilds the figures in `figures/`, and runs all validation checks.

## Repository structure

- `model.py` — analytic threshold model, folds, cusp calculations, and event-driven absorbing protocol.
- `aval_theory.py` — Borel-mixture avalanche calculation.
- `pathsim.py` — path-resolved partial-deleveraging simulator.
- `exp_finiteN.py` — finite-cohort experiment for lognormal exposures.
- `exp_pareto.py` — Pareto exposure experiments.
- `exp_avalanche.py` — avalanche-size experiments for equal, lognormal, and Pareto exposures.
- `exp_path.py` — cohort-size, ramp-rate, and impact-memory experiments.
- `bm_parabola.py` — Brownian-motion-with-parabolic-drift constants.
- `avalanche_size_sweep.py` — finite-size convergence check for the avalanche distribution.
- `stalled_ignition_diagnostic.py` — matched-cohort diagnostic of stalled ignition under transient impact.
- `dominant_holder_meanfield_controls.py` — event-definition-independent dominant-holder test and binary mean-field benchmark for the path-resolved population.
- `make_figs.py` — builds Figures 1–9 from numerical outputs.
- `results/` — numerical outputs used by the figure-generation and validation scripts.
- `figures/` — generated figure files.
- `expected_results/` — bundled expected numerical outputs for reproducibility checks.
- `expected_figures/` — bundled expected figure files.
- `tests/smoke_test.py` — fast analytic checks.
- `validate_results.py` — validation of principal quantitative results.
- `validate_diagnostics.py` — validation of avalanche-size and stalled-ignition diagnostics.
- `validate_controls.py` — validation of dominant-holder and path-model mean-field controls.
- `validate_figures.py` — SHA-256 comparison of generated and expected figures.
- `RESULTS_MAP.md` — map from scientific results to code and outputs.
- `RESULTS_INDEX.md` — description of numerical output files.
- `CONTROLS.md` — targeted scientific controls and their interpretation.
- `metadata/RANDOM_SEEDS.md` — random-seed inventory.
- `metadata/software_versions.txt` — tested software environment.
- `metadata/PACKAGE_INFO.md` — package scope and provenance.
- `metadata/MANIFEST.sha256` — SHA-256 manifest of package files.

## Main experiments

| Analysis | Command | Main outputs |
|---|---|---|
| Brownian-parabola constants | `python bm_parabola.py` | `results/bm_parabola_constants.csv` |
| Finite-cohort scaling | `python exp_finiteN.py` | `results/finiteN_lognormal.csv`, `results/finiteN_theory_constants.csv` |
| Heavy-tailed exposures | `python exp_pareto.py <mu>` | `results/pareto_raw_mu*.npz`, `results/finiteN_pareto.csv` |
| Avalanche distributions | `python exp_avalanche.py <kind>` | `results/avalanches_*.csv` |
| Path-resolved cohort-size study | `python exp_path.py size` | `results/path_size.csv`, `results/path_curves_size.npz` |
| Ramp-rate study | `python exp_path.py ramp` | `results/path_ramp.csv`, `results/path_curves_ramp.npz` |
| Impact-memory study | `python exp_path.py memory` | `results/path_memory.csv`, `results/path_curves_memory.npz` |
| Avalanche finite-size check | `python avalanche_size_sweep.py` | `results/avalanche_finiteN_sweep.csv` |
| Stalled-ignition diagnostic | `python stalled_ignition_diagnostic.py` | `results/stalled_ignition_diagnostic.csv` |
| Dominant-holder and mean-field controls | `python dominant_holder_meanfield_controls.py` | `results/dominant_holder_event_independent.csv`, `results/path_model_meanfield_reference.csv` |

## Reproducibility conventions

All stochastic experiments use fixed seeds listed in `metadata/RANDOM_SEEDS.md`. Deterministic quantities are checked with tight tolerances. Monte Carlo summaries use small numerical tolerances to allow harmless differences across platforms and numerical libraries.

The `expected_results/` and `expected_figures/` directories are immutable comparison targets. A full rerun writes only to `results/` and `figures/`.

## Scope

The simulations are theoretical and dimensionless. The package does not contain empirical market data and does not provide calibration or forecasts for a specific security.
