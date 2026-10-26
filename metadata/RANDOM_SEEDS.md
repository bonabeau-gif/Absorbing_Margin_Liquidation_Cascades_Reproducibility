# Random-seed inventory

All RNGs use NumPy `default_rng` unless noted.

- `exp_finiteN.py`: seed `20260917`.
- `bm_parabola.py`: seed `7`.
- `exp_pareto.py`: seed `int(1000*mu)+17` for each tail index; therefore 1267, 1517, 1767, 2517, and 4017 for mu = 1.25, 1.5, 1.75, 2.5, and 4.
- `exp_avalanche.py`: seed `99` for the lognormal run; seed `101` for equal and Pareto-1.5 runs.
- `exp_path.py size`: seed `5000+N` separately for each cohort size N; the same seed rule is used for transient and permanent impact at a given N.
- `exp_path.py ramp`: seed `777` reset at every ramp duration so the same sequence of cohorts is used for every T.
- `exp_path.py memory`: seed `31337` reset at every delay/feedback/half-life combination so the same sequence of cohorts is used along each sweep.

Fixed seeds remove Monte Carlo sampling drift within the tested software stack. Small floating-point differences may remain across platforms, BLAS/libm implementations, or package versions; `validate_results.py` therefore uses exact/tight checks for deterministic quantities and modest tolerances for stochastic summaries.
- `avalanche_size_sweep.py`: seed `20260922+N`, giving 20385922, 20760922, and 22260922 for N = 125000, 500000, and 2000000.
- `stalled_ignition_diagnostic.py`: seed `97531`.
- `dominant_holder_meanfield_controls.py`: seed `20260922` for the finite-cohort dominant-holder calculation; the mean-field calculation is deterministic.
