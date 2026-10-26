import numpy as np, pandas as pd, sys
from model import *
from aval_theory import integrated_borel
l,s,m,Lam=2.0,0.16,0.25,0.30
bins=np.unique(np.round(np.logspace(0,3.6,24)).astype(int)); bins=np.append(bins,bins[-1]*2)
rows=[]
for N,R in [(125_000,60),(500_000,30),(2_000_000,12)]:
    rng=np.random.default_rng(20260922+N)
    H=np.zeros(len(bins)-1); nav=0
    for r in range(R):
        Z=rng.standard_normal(N); lev=np.maximum(1,l*(1+s*Z)); Q=np.ones(N)
        need,rec,Lb,w=event_driven((1/lev-m)/(1-m),Q,Lam)
        cnt=np.diff(np.append(rec,N)); Lc=np.append(Lb,1.0); ex=Lc[np.append(rec[1:],N)]-Lc[rec]
        j=ex.argmax(); S=cnt[:j]
        H += np.histogram(S,bins)[0]; nav += len(S)
    wd=np.diff(bins); ctr=np.sqrt(bins[:-1]*(bins[1:]-1).clip(1)); pdf=H/wd/nav
    th=np.array([integrated_borel(np.arange(lo,hi)).mean() for lo,hi in zip(bins[:-1],bins[1:])])
    mask=(ctr>=200)&(ctr<=1000)&(H>=10)
    # count-weighted mean log ratio and mean absolute relative discrepancy
    weights=H[mask]/H[mask].sum()
    ratio=np.exp(np.sum(weights*np.log(pdf[mask]/th[mask])))
    mare=np.sum(weights*np.abs(pdf[mask]/th[mask]-1))
    rows.append(dict(N=N,R=R,n_avalanches=nav,ratio_sim_theory_200_1000=ratio,mean_abs_rel_error_200_1000=mare,total_counts_200_1000=int(H[mask].sum())))
    print(rows[-1]); sys.stdout.flush()
pd.DataFrame(rows).to_csv('results/avalanche_finiteN_sweep.csv',index=False)
