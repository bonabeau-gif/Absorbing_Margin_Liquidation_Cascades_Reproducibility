# Scientific results to computational outputs

| Scientific result | Code | Output / validation |
|---|---|---|
| Reversible equality map, absorbing upper-fold transition, and fold coordinates | `model.py`, `make_figs.py` | `figures/fig1_skeleton.png`, `results/finiteN_theory_constants.csv`, `validate_results.py` |
| Cusp dependence on initiating drawdown and maintenance ratio | `model.py`, `make_figs.py` | `figures/fig2_cusp.png` |
| Exposure-leverage tilt shifts the cusp | `model.py`, `make_figs.py` | `results/cusp_vs_tilt.csv`, `figures/fig3_tilt.png` |
| Fold onset under five impact laws | `model.py`, `make_figs.py` | `results/impact_law_onsets.csv`, `figures/fig4_impact_laws.png` |
| Finite-cohort dispersion, mean shift, and takeoff-state scaling | `exp_finiteN.py`, `bm_parabola.py` | `results/finiteN_lognormal.csv`, `results/finiteN_theory_constants.csv`, `figures/fig5_finiteN.png` |
| Heavy-tail core scaling and late-collapse tail | `exp_pareto.py`, `make_figs.py` | `results/finiteN_pareto.csv`, `results/pareto_exponents.csv`, `results/pareto_late_collapse.csv`, `figures/fig6_heavy_tails.png` |
| Event-independent dominant-holder dilution | `dominant_holder_meanfield_controls.py` | `results/dominant_holder_event_independent.csv` |
| Equal-exposure avalanche distribution | `exp_avalanche.py`, `aval_theory.py` | `results/avalanches_equal.csv`, `figures/fig7_avalanches.png` |
| Avalanche finite-size discrepancy decreases with cohort size | `avalanche_size_sweep.py` | `results/avalanche_finiteN_sweep.csv`, `validate_diagnostics.py` |
| Finite intrinsic cascade duration and ramp-rate sharpening | `exp_path.py ramp`, `pathsim.py` | `results/path_ramp.csv`, `figures/fig8_path.png` |
| Cohort-size behavior under permanent and transient impact | `exp_path.py size`, `pathsim.py` | `results/path_size.csv`, `figures/fig8_path.png` |
| Direct stalled-ignition diagnostic | `stalled_ignition_diagnostic.py` | `results/stalled_ignition_diagnostic.csv` |
| Impact-memory crossover and fragmentation | `exp_path.py memory`, `pathsim.py` | `results/path_memory.csv`, `figures/fig9_memory.png` |
| Binary mean-field benchmark for path-resolved population | `dominant_holder_meanfield_controls.py` | `results/path_model_meanfield_reference.csv` |
