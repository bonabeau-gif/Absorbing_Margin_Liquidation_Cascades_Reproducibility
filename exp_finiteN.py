"""Finite-cohort collapse statistics with lognormal exposures (Sec. VII) and parameter-free Daniels-type predictions."""
import numpy as np, csv, sys
from scipy.stats import norm
from model import *
l, s, m, Lam, sq, rho = 2.0, 0.16, 0.25, 0.30, 0.75, 0.4
chi = rho*sq
(Bp, dp), (Bm, dm) = folds(l, s, m, chi, Lam)
h = 1e-4
kappa = -(d0_of_B(Bp+h, l, s, m, chi, Lam) - 2*dp + d0_of_B(Bp-h, l, s, m, chi, Lam))/h**2
Tp = dt_dB(Bp, l, s, m, chi); zp = norm.ppf(Bp)
V = np.exp(sq**2)*((1-Bp)**2*norm.cdf(zp+chi) + Bp**2*(1-norm.cdf(zp+chi)))   # N*Var of weighted empirical CDF at B+
q2 = np.exp(sq**2 - chi*zp - chi**2/2)                                           # local variance rate
eps0 = Tp*np.sqrt(q2)
EM, SDM, SDARG = 0.9961930, 0.5775, np.sqrt(0.26355964)*2**(2/3)
pred = dict(Bp=Bp, d0p=dp, kappa=kappa, Tp=Tp, sd_coef=Tp*np.sqrt(V), shift_coef=EM*(eps0**4/kappa)**(1/3),
            loc_coef=SDARG*(eps0/kappa)**(2/3), jump_inf=landing(l, s, m, chi, Lam)-Bp)
print(pred)
rng = np.random.default_rng(20260917)
rows = []
for N, R in [(250,4000),(1000,4000),(4000,4000),(16000,3000),(64000,2000),(256000,1000)]:
    out = np.empty((R, 3))
    for r in range(R):
        Z = rng.standard_normal(N); U = rng.standard_normal(N)
        lev = np.maximum(1, l*(1+s*Z)); Q = np.exp(sq*(rho*Z + np.sqrt(1-rho**2)*U))
        need, rec, Lb, w = event_driven((1/lev-m)/(1-m), Q, Lam)
        Lc = np.append(Lb, 1.0); sizes = Lc[np.append(rec[1:], N)] - Lc[rec]; j = sizes.argmax()
        out[r] = need[rec[j]], sizes[j], Lb[rec[j]]
    d, J, L0 = out.T
    se = lambda x: x.std(ddof=1)/np.sqrt(R)
    rows.append(dict(N=N, R=R, mean_shift=d.mean()-dp, se_shift=se(d), sd_field=d.std(ddof=1),
                     sd_takeoff=L0.std(ddof=1), mean_takeoff=L0.mean(), mean_jump=J.mean(), sd_jump=J.std(ddof=1),
                     pred_shift=pred["shift_coef"]*N**(-2/3), pred_sd=np.sqrt(pred["sd_coef"]**2/N + (SDM/EM*pred["shift_coef"])**2*N**(-4/3)),
                     pred_sd_lead=pred["sd_coef"]/np.sqrt(N), pred_loc=pred["loc_coef"]*N**(-1/3)))
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()}); sys.stdout.flush()
with open("results/finiteN_lognormal.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
with open("results/finiteN_theory_constants.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=pred.keys()); wr.writeheader(); wr.writerow(pred)
