# Targeted scientific controls

## Dominant-holder dilution

`dominant_holder_meanfield_controls.py` tests the dilution effect without defining collapse as the field of the largest avalanche. For a conservative holder that remains intact through the background instability, the script compares:

1. the upper fold of the mean-field background map after the holder absorbs a fixed share of total exposure, and
2. the simulated external field at which the remaining cohort first loses 25% of its own exposure.

The comparison is stored in `results/dominant_holder_event_independent.csv`.

## Path-model mean-field benchmark

The same script computes a binary, complete-liquidation, permanent-impact mean-field benchmark for the population used in the path-resolved simulations. It uses the same linear-to-square-root impact function and the same truncated maintenance-ratio distribution. The output is `results/path_model_meanfield_reference.csv`.

This benchmark separates the binary mean-field instability from the additional effects introduced by partial deleveraging, heterogeneous execution delays, sale-rate limits, finite price relaxation, and transient impact.

## Avalanche finite-size convergence

`avalanche_size_sweep.py` measures the discrepancy between the equal-exposure avalanche simulation and the Borel-mixture prediction over increasing cohort sizes. The output is `results/avalanche_finiteN_sweep.csv`.

## Stalled ignition

`stalled_ignition_diagnostic.py` follows matched transient-impact cohorts and measures the size of the first recruitment wave, decay of the unabsorbed impact stock before later ignition, and the quiet interval between waves. The output is `results/stalled_ignition_diagnostic.csv`.
