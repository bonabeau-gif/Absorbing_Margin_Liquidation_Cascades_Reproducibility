#!/usr/bin/env python3
"""Validate avalanche finite-size and stalled-ignition diagnostics."""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent / 'results'
A = pd.read_csv(R/'avalanche_finiteN_sweep.csv')
S = pd.read_csv(R/'stalled_ignition_diagnostic.csv')
assert list(A.N) == [125000, 500000, 2000000]
expected = [1.2469717299954548, 1.1984874929735894, 1.0807452210420057]
for x,y in zip(A.ratio_sim_theory_200_1000, expected): assert abs(x-y) < 1e-12
late=S[S.late==True]; other=S[S.late==False]
assert len(late)==24 and len(S)==128
assert abs(late.first_wave_sold.mean()-0.09605477354327756)<1e-12
assert abs(other.first_wave_sold.mean()-0.3686489473451916)<1e-12
assert abs(late.U_decay_fraction.median()-0.6912654713126946)<1e-12
print('diagnostic checks: PASS')
