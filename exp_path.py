"""Path-resolved experiments: cohort-size convergence, ramp-rate sharpening, impact-memory sweep (Sec. X)."""
import numpy as np, csv, sys
from pathsim import *
LAM, GAM, FLOOR, HOLD, NC = 0.5, 0.22, 0.88, 500, 32
def stats(a): a = np.asarray(a, float); return a.mean(), a.std(ddof=1)
def write(fn, rows):
    with open(fn, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
which = sys.argv[1]; curves = {}
if which == "size":
    rows = []; NCS = 48
    for hI in [80.0, -1.0]:
        for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            rng = np.random.default_rng(5000 + N); res = []
            for c in range(NCS):
                p = run_path(*make_cohort(rng, N), LAM, GAM, hI, 0.02, 0.30, 3000, HOLD, FLOOR); res.append(diagnostics(p, 3000, .02, .30))
                if c == 0 and hI > 0 and N in (2000, 8000, 32000): curves["N%d" % N] = p[::5].astype(np.float32)
            res = np.array(res); curves["mid_h%d_N%d" % (hI, N)] = res[:, 3]
            med = np.median(res[:, 3])
            rows.append(dict(half_life=hI, N=N, cohorts=NCS, final_sold_mean=res[:,0].mean(), final_sold_sd=res[:,0].std(ddof=1), mid_field_mean=res[:,3].mean(),
                             mid_field_median=med, mid_field_sd=res[:,3].std(ddof=1), mid_field_iqr=np.subtract(*np.percentile(res[:,3], [75, 25])),
                             frac_late=np.mean(res[:,3] > med + 0.008), width_steps_mean=res[:,2].mean(), width_steps_sd=res[:,2].std(ddof=1)))
            r = rows[-1]; print(hI, N, "mid %.4f sd %.4f iqr %.4f late %.2f  Sf %.3f  width %.1f" % (r["mid_field_mean"], r["mid_field_sd"], r["mid_field_iqr"], r["frac_late"], r["final_sold_mean"], r["width_steps_mean"])); sys.stdout.flush()
    write("results/path_size.csv", rows)
if which == "ramp":
    rows = []
    for T in [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]:
        rng = np.random.default_rng(777); res = []          # same cohorts at every ramp duration
        for c in range(NC):
            p = run_path(*make_cohort(rng, 8000), LAM, GAM, 80.0, 0.02, 0.30, T, HOLD, FLOOR); res.append(diagnostics(p, T, .02, .30))
            if c == 0 and T in (1000, 4000, 16000): curves["T%d" % T] = p[:: max(1, T//1000)].astype(np.float32)
        res = np.array(res); wf = res[:,5] - res[:,4]
        rows.append(dict(T=T, cohorts=NC, width_steps_mean=res[:,2].mean(), width_steps_sd=res[:,2].std(ddof=1), width_field_mean=wf.mean(),
                         width_field_sd=wf.std(ddof=1), mid_field_mean=res[:,3].mean(), mid_field_sd=res[:,3].std(ddof=1), final_sold_mean=res[:,0].mean()))
        print(rows[-1]); sys.stdout.flush()
    write("results/path_ramp.csv", rows)
if which == "memory":
    rows = []; T = 4600; hi = 0.45; W = 200
    hgrid = [3, 4, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 48, 57, 68, 80, 113, 160, 320, 640, -1]
    for delay in [1.0, 5.0, 15.0]:
        for Lam in [LAM, 0.0]:
            for hI in (hgrid if Lam > 0 else [80]):
                rng = np.random.default_rng(31337); res = []
                for c in range(NC):
                    p = run_path(*make_cohort(rng, 8000, delay=delay), Lam, GAM, float(hI), 0.02, hi, T, HOLD, FLOOR)
                    inc = p[W:] - p[:-W]; k = inc.argmax(); up = inc > 0.10
                    nev = int(up[0]) + int(np.sum(up[1:] & ~up[:-1]))
                    t30 = int(T*(0.30-0.02)/(hi-0.02))
                    res.append((p[-1], inc[k], 0.02 + (hi-0.02)*min(k + W//2, T)/T, p[t30], nev))
                    if c == 0 and delay == 5.0 and Lam > 0 and hI in (4, 12, 20, 80, -1): curves["h%d" % hI] = p[::5].astype(np.float32)
                    if c == 0 and delay == 5.0 and Lam == 0: curves["baseline"] = p[::5].astype(np.float32)
                res = np.array(res)
                rows.append(dict(delay=delay, Lam=Lam, half_life=hI, cohorts=NC, final_sold_mean=res[:,0].mean(), final_sold_sd=res[:,0].std(ddof=1),
                                 largest_event_mean=res[:,1].mean(), largest_event_sd=res[:,1].std(ddof=1), event_field_mean=res[:,2].mean(),
                                 event_field_sd=res[:,2].std(ddof=1), sold_at_030_mean=res[:,3].mean(), sold_at_030_sd=res[:,3].std(ddof=1),
                                 n_events_mean=res[:,4].mean()))
                r = rows[-1]; print(delay, Lam, hI, "J %.3f+-%.3f  field %.3f+-%.3f  S(0.30) %.3f+-%.3f  Sf %.3f  nev %.2f" % (r["largest_event_mean"], r["largest_event_sd"], r["event_field_mean"], r["event_field_sd"], r["sold_at_030_mean"], r["sold_at_030_sd"], r["final_sold_mean"], r["n_events_mean"])); sys.stdout.flush()
    write("results/path_memory.csv", rows)
np.savez_compressed("results/path_curves_%s.npz" % which, **curves)
