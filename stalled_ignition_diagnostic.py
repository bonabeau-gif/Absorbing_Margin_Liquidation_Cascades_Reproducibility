import numpy as np, pandas as pd
from pathsim import make_cohort, run_path
LAM,GAM,FLOOR,HOLD=0.5,0.22,0.88,500
N,T=4000,3000; dlo,dhi=.02,.30
rng=np.random.default_rng(97531)
rows=[]

def fld(t): return dlo+(dhi-dlo)*min(t,T)/T

def recU(path,h=80.0):
    u=np.diff(np.r_[0.,path]); delta=2**(-1/h); U=np.empty_like(path); x=0.
    for i,ui in enumerate(u): x=delta*x+ui; U[i]=x
    return u,U

for c in range(128):
    cohort=make_cohort(rng,N)
    p=run_path(*cohort,LAM,GAM,80.0,dlo,dhi,T,HOLD,FLOOR)
    t50=int(np.searchsorted(p,.5*p[-1])); field=fld(t50)
    u,U=recU(p)
    # detect first recruitment episode: first 100-step increment > 0.01
    W=100; inc=p[W:]-p[:-W]; ids=np.where(inc>.01)[0]
    if len(ids):
        ts=int(ids[0])
        # first episode ends at first point after ts+W where 100-step increment falls below 0.005
        after=np.where(inc[min(ts+W,len(inc)-1):] < .005)[0]
        te=(min(ts+W,len(inc)-1)+int(after[0])) if len(after) else min(t50,ts+400)
        te=min(te,t50)
        peak=float(U[ts:te+1].max()) if te>=ts else np.nan
        j=ts+int(np.argmax(U[ts:te+1])) if te>=ts else ts
        if t50>j+10:
            mn=float(U[j:t50].min()); decay=(peak-mn)/peak if peak>0 else np.nan
        else: mn=peak; decay=0.
        sold_first=float(p[te]-p[max(0,ts-1)])
        gap=t50-te
    else: ts=te=-1; peak=mn=decay=sold_first=gap=np.nan
    rows.append(dict(cohort=c,field=field,final=p[-1],t50=t50,first_start=ts,first_end=te,first_wave_sold=sold_first,U_peak=peak,U_min_before_ignition=mn,U_decay_fraction=decay,gap_steps=gap))

df=pd.DataFrame(rows)
med=df.field.median(); df['late']=df.field>med+.008
print('median field',med,'late fraction',df.late.mean())
print(df.groupby('late')[['field','first_wave_sold','U_decay_fraction','gap_steps']].agg(['count','mean','median']).to_string())
print('late decay > .25',np.mean(df.loc[df.late,'U_decay_fraction']>.25) if df.late.any() else np.nan)
print('nonlate decay > .25',np.mean(df.loc[~df.late,'U_decay_fraction']>.25))
df.to_csv('results/stalled_ignition_control2.csv',index=False)
