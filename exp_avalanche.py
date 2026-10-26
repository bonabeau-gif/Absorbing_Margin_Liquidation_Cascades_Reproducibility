"""Precursor avalanche-size statistics (account counts) before the macroscopic collapse (Sec. IX)."""
import numpy as np, sys, csv
from model import *
l, s, m, Lam = 2.0, 0.16, 0.25, 0.30
kind = sys.argv[1]; N = 2_000_000; R = 40
rng = np.random.default_rng(99 if kind == "lognormal" else 101)
bins = np.unique(np.round(np.logspace(0, 4.5, 28)).astype(int)); bins = np.append(bins, bins[-1]*2)
H_all = np.zeros(len(bins)-1); H_near = np.zeros(len(bins)-1); n_all = n_near = 0
for r in range(R):
    Z = rng.standard_normal(N); lev = np.maximum(1, l*(1+s*Z))
    if kind == "equal":
        Q = np.ones(N)
    elif kind == "lognormal":
        sq, rho = 0.75, 0.4; Q = np.exp(sq*(rho*Z + np.sqrt(1-rho**2)*rng.standard_normal(N)))
    else:
        mu = float(kind); Q = (1.0 - rng.random(N))**(-1.0/mu)
    need, rec, Lb, w = event_driven((1/lev-m)/(1-m), Q, Lam)
    cnt = np.diff(np.append(rec, N)); Lc = np.append(Lb, 1.0); ex = Lc[np.append(rec[1:], N)] - Lc[rec]
    j = ex.argmax(); dc = need[rec[j]]
    S = cnt[:j]; d = need[rec[:j]]
    near = d > dc - 0.002
    H_all += np.histogram(S, bins)[0]; H_near += np.histogram(S[near], bins)[0]; n_all += S.size; n_near += near.sum()
wd = np.diff(bins); ctr = np.sqrt(bins[:-1]*(bins[1:]-1).clip(1))
with open("results/avalanches_%s.csv" % kind, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["S_center", "bin_lo", "bin_hi", "pdf_all", "count_all", "pdf_near", "count_near"])
    for k in range(len(wd)):
        wr.writerow([ctr[k], bins[k], bins[k+1], H_all[k]/wd[k]/n_all, int(H_all[k]), H_near[k]/wd[k]/max(n_near, 1), int(H_near[k])])
def slope(H, lo, hi):
    k = (ctr >= lo) & (ctr <= hi) & (H > 20); return np.polyfit(np.log(ctr[k]), np.log(H[k]/wd[k]), 1)[0]
print(kind, "integrated slope [5,500]: %.3f   near-fold slope [3,300]: %.3f   n_all %d n_near %d" % (slope(H_all, 5, 500), slope(H_near, 3, 300), n_all, n_near))
