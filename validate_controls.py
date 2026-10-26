#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parent/'results'
D=pd.read_csv(R/'dominant_holder_event_independent.csv')
X=pd.read_csv(R/'path_model_meanfield_reference.csv')
assert list(D.holder_share)==[0.0,0.1,0.2]
exp_theory=[0.12745927555917533,0.13082913322780065,0.13508522704074982]
exp_sim=[0.12783787337300484,0.13135775419835152,0.13561310829275486]
for a,b in zip(D.meanfield_fold_field,exp_theory): assert abs(a-b)<1e-12
for a,b in zip(D.simulation_field_bg25,exp_sim): assert abs(a-b)<1e-12
assert (D.sim_minus_theory.abs()<5.4e-4).all()
row=X[X.case=='binary_mean_field_permanent'].iloc[0]
assert abs(row.transition_field-0.10468681982772746)<1e-12
assert abs(row.total_drawdown_at_fold-0.13683338)<1e-8
assert abs(row.B_at_fold-0.01891222)<1e-8
print('targeted controls: PASS')
