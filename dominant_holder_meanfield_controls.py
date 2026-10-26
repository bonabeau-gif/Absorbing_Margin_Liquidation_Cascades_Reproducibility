import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq
from model import event_driven, t_of_B

# A. Event-definition-independent dominant-holder check
l,s,m,Lam=2.0,0.16,0.25,0.30
rng=np.random.default_rng(20260922)
N=100_000
Z=rng.standard_normal(N)
lev=np.maximum(1.0,l*(1+s*Z))
Q=np.ones(N)
t=(1/lev-m)/(1-m)

def meanfield_fold(omega):
    # conservative holder stays intact through the background fold
    def d0(B):
        return t_of_B(B,l,s,m,0.0)-Lam*np.sqrt((1.0-omega)*B)
    r=minimize_scalar(lambda B:-d0(B),bounds=(1e-6,0.2),method='bounded',options={'xatol':1e-13})
    return float(r.x), float(d0(r.x))

def simulated_fixed_background_loss(omega, frac=.25):
    if omega==0:
        tt=t.copy(); QQ=Q.copy(); bg=np.ones(N,dtype=bool)
    else:
        qstar=omega*N/(1-omega)
        lstar=max(1.0,l*(1+s*(-2.0))) # conservative holder, z=-2
        tstar=(1/lstar-m)/(1-m)
        tt=np.append(t,tstar); QQ=np.append(Q,qstar); bg=np.r_[np.ones(N,dtype=bool),False]
    o=np.argsort(tt,kind='stable'); ts=tt[o]; q=QQ[o]; bg=bg[o]
    w=q/q.sum(); cw=np.cumsum(w); Lb=np.empty_like(cw); Lb[0]=0; Lb[1:]=cw[:-1]
    need=ts-Lam*np.sqrt(Lb)
    rec=np.flatnonzero(need>=np.maximum.accumulate(need)); ends=np.r_[rec[1:],len(tt)]
    bg_total=w[bg].sum(); sold_bg=0.0
    largest=(-1.0,np.nan)
    for start,end in zip(rec,ends):
        event=w[start:end].sum()
        if event>largest[0]: largest=(event,float(need[start]))
        sold_bg += w[start:end][bg[start:end]].sum()
        if sold_bg/bg_total >= frac:
            return float(need[start]), float(sold_bg/bg_total), float(largest[1])
    raise RuntimeError('threshold not reached')

rows=[]
for omega in [0.0,0.10,0.20]:
    Bf,df=meanfield_fold(omega)
    sf,bgfrac,op=simulated_fixed_background_loss(omega,.25)
    rows.append(dict(holder_share=omega, meanfield_fold_B=Bf, meanfield_fold_field=df,
                     simulation_field_bg25=sf, bg_fraction_after_event=bgfrac,
                     sim_minus_theory=sf-df, largest_avalanche_field_seen_to_bg25=op))
pd.DataFrame(rows).to_csv('results/dominant_holder_event_independent.csv',index=False)

# B. Mean-field reference for Table II population in Sec. X
# Binary complete liquidation + permanent impact, same linear-to-square-root f and truncated m distribution.
l=2.0; s=0.16; sq=0.75; rho=0.40; chi=rho*sq
mmean=0.25; msd=0.025; Ztr=norm.cdf(4)-norm.cdf(-4)
Lam=0.50; xc=0.02

def fcross(x):
    return (np.sqrt(1+x/xc)-1)/(np.sqrt(1+1/xc)-1)

def Fd(d):
    def integ(mm):
        A=mm+(1-mm)*d
        zthr=(1/(l*A)-1)/s
        p=1-norm.cdf(zthr-chi)
        dens=norm.pdf((mm-mmean)/msd)/(msd*Ztr)
        return p*dens
    return quad(integ,mmean-4*msd,mmean+4*msd,epsabs=1e-12,epsrel=1e-11,limit=200)[0]

def d0d(d): return d-Lam*fcross(Fd(d))
r=minimize_scalar(lambda d:-d0d(d),bounds=(0.05,0.22),method='bounded',options={'xatol':1e-13})
dtot=float(r.x); d0=float(d0d(dtot)); B=float(Fd(dtot))
# Existing large-N path-resolved medians from archived path_size.csv
ps=pd.read_csv('results/path_size.csv')
def largeN(h):
    z=ps[(ps.half_life==h)&(ps.N>=16000)]
    return float(z.mid_field_median.mean()), float(z.mid_field_median.min()), float(z.mid_field_median.max())
perm=largeN(-1.0); trans=largeN(80.0)
out=pd.DataFrame([
    dict(case='binary_mean_field_permanent',transition_field=d0,low=np.nan,high=np.nan,total_drawdown_at_fold=dtot,B_at_fold=B),
    dict(case='path_resolved_permanent_largeN_median',transition_field=perm[0],low=perm[1],high=perm[2],total_drawdown_at_fold=np.nan,B_at_fold=np.nan),
    dict(case='path_resolved_half_life80_largeN_median',transition_field=trans[0],low=trans[1],high=trans[2],total_drawdown_at_fold=np.nan,B_at_fold=np.nan),
])
out.to_csv('results/path_model_meanfield_reference.csv',index=False)
print('dominant holder')
print(pd.DataFrame(rows).to_string(index=False))
print('\nsection X')
print(out.to_string(index=False))
