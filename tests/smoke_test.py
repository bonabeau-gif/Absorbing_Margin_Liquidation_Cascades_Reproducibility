#!/usr/bin/env python3
"""Fast deterministic checks of the analytic implementation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from model import folds, landing, cusp_power, onset_fixed_l, event_driven

# Benchmark fold for the baseline parameter set.
(Bp, dp), (Bm, dm) = folds(2.0, 0.16, 0.25, 0.30, 0.30)
assert abs(Bp - 0.02931704) < 2e-6
assert abs(dp - 0.10892667) < 2e-6
assert abs(landing(2.0, 0.16, 0.25, 0.30, 0.30) - 0.71129135) < 2e-5

# Cusp at d0=0.10, m=.25, sigma=.16, chi=0.
Bc, Lc, lc = cusp_power(0.16, 0.0, 0.25, 0.10)
assert abs(Lc - 0.23989205) < 2e-6
assert abs(lc - 2.17328776) < 2e-6

# All five impact laws have finite fold onset.
for law in ("square-root", "linear-to-square-root", "linear", "logarithmic", "saturating"):
    x, B = onset_fixed_l(2.0, 0.16, 0.25, 0.30, law)
    assert np.isfinite(x) and x > 0 and 0 < B < 1

# Event-driven record logic on a tiny cohort.
t = np.array([0.10, 0.12, 0.11, 0.20])
q = np.array([1.0, 2.0, 1.0, 1.0])
need, rec, Lb, w = event_driven(t, q, 0.3)
assert need.shape == rec.dtype.type(need.size) * (1,) if False else need.shape == (4,)
assert rec.size >= 1 and np.all(np.diff(rec) > 0)
assert abs(w.sum() - 1.0) < 1e-12
print("smoke tests passed")
