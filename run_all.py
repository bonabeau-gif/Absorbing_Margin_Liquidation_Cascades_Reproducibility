#!/usr/bin/env python3
"""Reproduce the numerical results and figures in this package.

Usage
-----
python run_all.py --quick   # smoke tests, rebuild figures from bundled results, validate
python run_all.py --full    # rerun all stochastic experiments from fixed seeds, rebuild figures, validate

Reruns write to results/ and figures/. Bundled expected outputs in expected_results/
and expected_figures/ are read-only comparison targets.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"

GENERATED = [
    "finiteN_lognormal.csv", "finiteN_theory_constants.csv", "bm_parabola_constants.csv",
    "finiteN_pareto.csv", "pareto_raw_mu1.25.npz", "pareto_raw_mu1.5.npz",
    "pareto_raw_mu1.75.npz", "pareto_raw_mu2.5.npz", "pareto_raw_mu4.npz",
    "avalanches_equal.csv", "avalanches_lognormal.csv", "avalanches_1.5.csv",
    "path_size.csv", "path_curves_size.npz", "path_ramp.csv", "path_curves_ramp.npz",
    "path_memory.csv", "path_curves_memory.npz", "impact_law_onsets.csv", "cusp_vs_tilt.csv",
    "pareto_exponents.csv", "pareto_late_collapse.csv", "avalanche_finiteN_sweep.csv",
    "stalled_ignition_diagnostic.csv", "dominant_holder_event_independent.csv",
    "path_model_meanfield_reference.csv",
]

def run(label: str, cmd: list[str]) -> None:
    LOGS.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"\n[{label}] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (LOGS / f"{label}.log").write_text(p.stdout, encoding="utf-8")
    if p.returncode:
        print(p.stdout[-4000:])
        raise SystemExit(f"stage {label} failed with exit code {p.returncode}")
    print(f"[{label}] done in {time.time()-t0:.1f} s", flush=True)

def clean_generated() -> None:
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    for name in GENERATED:
        p = RESULTS / name
        if p.exists(): p.unlink()
    for p in FIGS.glob("fig*.png"): p.unlink()

def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="smoke tests + figures from bundled results + validation")
    g.add_argument("--full", action="store_true", help="rerun every numerical experiment from fixed seeds")
    args = ap.parse_args()

    if args.quick:
        run("smoke", [sys.executable, "tests/smoke_test.py"])
        run("figures", [sys.executable, "make_figs.py"])
    else:
        clean_generated()
        run("bm_parabola", [sys.executable, "bm_parabola.py"])
        run("finiteN", [sys.executable, "exp_finiteN.py"])
        for mu in ("1.25", "1.5", "1.75", "2.5", "4"):
            run(f"pareto_{mu.replace('.', '_')}", [sys.executable, "exp_pareto.py", mu])
        for kind in ("equal", "lognormal", "1.5"):
            run(f"avalanche_{kind.replace('.', '_')}", [sys.executable, "exp_avalanche.py", kind])
        for which in ("size", "ramp", "memory"):
            run(f"path_{which}", [sys.executable, "exp_path.py", which])
        run("avalanche_size_sweep", [sys.executable, "avalanche_size_sweep.py"])
        run("stalled_ignition", [sys.executable, "stalled_ignition_diagnostic.py"])
        run("dominant_holder_meanfield", [sys.executable, "dominant_holder_meanfield_controls.py"])
        run("figures", [sys.executable, "make_figs.py"])

    run("validate_results", [sys.executable, "validate_results.py"])
    run("validate_diagnostics", [sys.executable, "validate_diagnostics.py"])
    run("validate_controls", [sys.executable, "validate_controls.py"])
    run("validate_figures", [sys.executable, "validate_figures.py"])
    print("\nReproduction workflow completed successfully.")

if __name__ == "__main__":
    main()
