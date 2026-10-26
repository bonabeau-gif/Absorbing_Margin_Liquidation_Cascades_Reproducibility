#!/usr/bin/env python3
"""Validate generated outputs against the bundled expected outputs and quantitative targets.

The deterministic analytic outputs are checked tightly. Fixed-seed Monte Carlo summaries are checked
with small tolerances to allow harmless numerical/library differences across platforms.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, REF = ROOT / "results", ROOT / "expected_results"
report = []

def ok(name, cond, detail=""):
    report.append((name, bool(cond), detail))

def close(name, x, y, atol=1e-6, rtol=1e-6):
    cond = np.isclose(x, y, atol=atol, rtol=rtol)
    ok(name, cond, f"got {x:.8g}; target {y:.8g}; atol={atol:g}, rtol={rtol:g}")

def csv_compare(name, rtol, atol):
    a = pd.read_csv(RES / name); b = pd.read_csv(REF / name)
    ok(name + " rows", list(a.columns) == list(b.columns) and len(a) == len(b), f"rows={len(a)}, expected={len(b)}")
    if list(a.columns) != list(b.columns) or len(a) != len(b): return
    nums = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    aa, bb = a[nums].to_numpy(float), b[nums].to_numpy(float)
    good = np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True)
    maxerr = np.nanmax(np.abs(aa-bb)) if aa.size else 0.0
    ok(name + " numeric", good, f"max abs difference={maxerr:.3g}; tolerance rtol={rtol:g}, atol={atol:g}")

# Required outputs.
required = [
    "finiteN_lognormal.csv", "finiteN_theory_constants.csv", "finiteN_pareto.csv",
    "impact_law_onsets.csv", "cusp_vs_tilt.csv", "pareto_exponents.csv", "pareto_late_collapse.csv",
    "avalanches_equal.csv", "avalanches_lognormal.csv", "avalanches_1.5.csv",
    "path_size.csv", "path_ramp.csv", "path_memory.csv",
    "path_curves_size.npz", "path_curves_ramp.npz", "path_curves_memory.npz",
]
for fn in required: ok("exists: " + fn, (RES/fn).exists())

# Compare summary tables to bundled expected outputs.
for fn in ("finiteN_theory_constants.csv", "impact_law_onsets.csv", "cusp_vs_tilt.csv"):
    csv_compare(fn, 2e-6, 2e-8)
for fn in ("finiteN_lognormal.csv", "finiteN_pareto.csv", "pareto_exponents.csv", "pareto_late_collapse.csv",
           "path_size.csv", "path_ramp.csv", "path_memory.csv"):
    csv_compare(fn, 0.03, 3e-4)

# Key quantitative values.
th = pd.read_csv(RES / "finiteN_theory_constants.csv").iloc[0]
close("fold B+", th.Bp, 0.0293170398, 2e-7, 0)
close("fold d0+", th.d0p, 0.1089266716, 2e-7, 0)
close("finite-N dispersion coefficient", th.sd_coef, 0.26819267, 2e-6, 0)
close("finite-N mean-shift coefficient", th.shift_coef, 0.90997893, 2e-6, 0)
close("finite-N takeoff coefficient", th.loc_coef, 0.29945134, 2e-6, 0)

imp = pd.read_csv(RES / "impact_law_onsets.csv").set_index("law")
for law, target in {"square-root":.240509, "linear-to-square-root":.221326, "linear":.232824,
                    "logarithmic":.184128, "saturating":.137967}.items():
    close("impact onset " + law, imp.loc[law,"Lambda_onset"], target, 3e-5, 0)

til = pd.read_csv(RES / "cusp_vs_tilt.csv")
for x, L, ell in [(-.8,.276218,2.377743),(0,.239892,2.173288),(.8,.211687,1.999794)]:
    r = til.iloc[(til.chi-x).abs().argmin()]
    close(f"tilt Lambda chi={x}", r.Lambda_c, L, 3e-5, 0)
    close(f"tilt leverage chi={x}", r.l_star, ell, 3e-5, 0)

f = pd.read_csv(RES / "finiteN_lognormal.csv")
sl_sd = np.polyfit(np.log(f.N), np.log(f.sd_field), 1)[0]
sl_shift = np.polyfit(np.log(f.N), np.log(f.mean_shift), 1)[0]
ok("finite-N exponent field", -0.53 < sl_sd < -0.43, f"fit={sl_sd:.3f}; expected near -0.48")
ok("finite-N exponent shift", -0.72 < sl_shift < -0.62, f"fit={sl_shift:.3f}; expected near -0.67")
last = f.iloc[-1]
close("large-N mean jump", last.mean_jump, 0.68377, 0.003, 0)

pe = pd.read_csv(RES / "pareto_exponents.csv").set_index("mu")
for mu, target in {1.25:.23,1.5:.31,1.75:.37,2.5:.47,4.0:.50}.items():
    close(f"Pareto IQR exponent mu={mu}", pe.loc[mu,"iqr_exponent"], target, 0.03, 0)

pr = pd.read_csv(RES / "path_ramp.csv").set_index("T")
close("ramp width steps T=4000", pr.loc[4000,"width_steps_mean"], 32.2, 1.2, 0)
close("ramp field width T=250", pr.loc[250,"width_field_mean"], .0296, .002, 0)
close("ramp field width T=64000", pr.loc[64000,"width_field_mean"], .00014, .00003, 0)

ps = pd.read_csv(RES / "path_size.csv")
for N, target in [(2000,.0064),(8000,.0070),(64000,.0051)]:
    v = ps[(ps.half_life==80)&(ps.N==N)].iloc[0].mid_field_sd
    close(f"transient transition-field sd N={N}", v, target, .0012, 0)
for N, target in [(1000,.0043),(64000,.0007)]:
    v = ps[(ps.half_life==-1)&(ps.N==N)].iloc[0].mid_field_sd
    close(f"permanent transition-field sd N={N}", v, target, .0008, 0)

pm = pd.read_csv(RES / "path_memory.csv")
g = pm[(pm.delay==5)&(pm.Lam==.5)].set_index("half_life")
for h, target in [(-1,.857),(80,.805),(20,.613),(3,.231)]:
    close(f"largest 200-step event h={h}", g.loc[h,"largest_event_mean"], target, .025 if h!=20 else .04, 0)
base = pm[(pm.delay==5)&(pm.Lam==0)].iloc[0]
close("no-feedback largest 200-step event", base.largest_event_mean, .046, .006, 0)

# Reference figure and current figure presence.
for i in range(1,10):
    ok(f"figure {i}", (ROOT/f"figures/fig{i}_" ).parent.exists() and any((ROOT/"figures").glob(f"fig{i}_*.png")))

npass=sum(x[1] for x in report); n=len(report)
lines=["# Validation report", "", f"Passed **{npass}/{n}** checks.", ""]
for name, passed, detail in report:
    lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}" + (f": {detail}" if detail else ""))
(ROOT/"VALIDATION_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
for row in report: print(("PASS" if row[1] else "FAIL"), row[0], row[2])
if npass != n:
    print(f"\n{n-npass} validation checks failed. See VALIDATION_REPORT.md")
    raise SystemExit(1)
print(f"\nAll {n} validation checks passed.")
