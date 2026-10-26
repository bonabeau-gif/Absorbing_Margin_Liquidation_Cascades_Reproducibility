"""Collapse-field statistics with Pareto-distributed exposures independent of leverage (Sec. VIII)."""
import numpy as np, csv, sys, os
from model import *
l, s, m, Lam, chi = 2.0, 0.16, 0.25, 0.30, 0.0
(Bp, dp), (Bm, dm) = folds(l, s, m, chi, Lam)
mu = float(sys.argv[1])
rng = np.random.default_rng(int(1000*mu) + 17)
rows = []
sched = [(1000,20000),(4000,10000),(16000,5000),(64000,2000),(256000,1000),(1024000,400)] if mu < 2 else [(1000,4000),(4000,4000),(16000,2000),(64000,1000),(256000,600),(1024000,250)]
raw = {}
for N, R in sched:
    out = np.empty((R, 4))
    for r in range(R):
        Z = rng.standard_normal(N); lev = np.maximum(1, l*(1+s*Z))
        Q = (1.0 - rng.random(N))**(-1.0/mu)
        need, rec, Lb, w = event_driven((1/lev-m)/(1-m), Q, Lam)
        Lc = np.append(Lb, 1.0); sizes = Lc[np.append(rec[1:], N)] - Lc[rec]; j = sizes.argmax()
        out[r] = need[rec[j]], sizes[j], w[rec[j]]*N, w.max()
    raw[str(N)] = out[:, :2].astype(np.float32)
    d = out[:, 0]; q = np.percentile(d, [5, 25, 50, 75, 95])
    rows.append(dict(mu=mu, N=N, R=R, d0p=dp, mean=d.mean(), sd=d.std(ddof=1), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4],
                     iqr=q[3]-q[1], mean_jump=out[:,1].mean(), sd_jump=out[:,1].std(ddof=1),
                     median_trigger_relweight=np.median(out[:,2]),
                     median_maxweight=np.median(out[:,3])))
    print(N, "sd %.5f iqr %.5f med-shift %.5f trig %.2f" % (rows[-1]["sd"], rows[-1]["iqr"], q[2]-dp, rows[-1]["median_trigger_relweight"])); sys.stdout.flush()
np.savez_compressed("results/pareto_raw_mu%s.npz" % sys.argv[1], **raw)
fn = "results/finiteN_pareto.csv"; new = not os.path.exists(fn)
with open(fn, "a", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys())
    if new: wr.writeheader()
    wr.writerows(rows)
