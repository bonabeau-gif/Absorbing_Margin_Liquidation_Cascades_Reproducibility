"""Path-resolved partial-deleveraging simulation (Sec. X)."""
import numpy as np
from numba import njit

XC = 0.02
@njit(cache=True)
def f_cross(x):
    return (np.sqrt(1.0 + x/XC) - 1.0)/(np.sqrt(1.0 + 1.0/XC) - 1.0)

@njit(cache=True)
def run_path(s0, D0, mi, tau, ri, Lam, gam, hI, d_lo, d_hi, T, hold, dfloor):
    """Returns sold-fraction path (length T+hold). hI<=0 means permanent impact."""
    N = s0.size
    sh = s0.copy(); D = D0.copy(); cnt = np.zeros(N, np.int64)
    S0 = s0.sum(); sold = 0.0; U = 0.0; d = d_lo
    delta = 1.0 if hI <= 0 else 2.0**(-1.0/hI)
    path = np.empty(T + hold)
    for t in range(T + hold):
        d0 = d_lo + (d_hi - d_lo)*min(t, T)/T
        P = 1.0 - d; u = 0.0
        for i in range(N):
            if sh[i] <= 0.0: continue
            A = P*sh[i]; E = A - D[i]
            if E < mi[i]*A:
                cnt[i] += 1
                if cnt[i] > tau[i]:
                    x = A - E/mi[i] if E > 0.0 else A
                    if x > ri[i]*A: x = ri[i]*A
                    if x > A: x = A
                    q = x/P; sh[i] -= q; D[i] -= x; u += q
            else:
                cnt[i] = 0
        u /= S0; sold += u
        U = delta*U + u
        d = d + gam*(d0 + Lam*f_cross(U) - d)
        if d > dfloor: d = dfloor
        path[t] = sold
    return path

def make_cohort(rng, N, l=2.0, s=0.16, sq=0.75, rho=0.4, m_mean=0.25, m_sd=0.025, delay=5.0, r_mean=0.25, r_sd=0.10):
    Z = rng.standard_normal(N); U = rng.standard_normal(N)
    lev = np.maximum(1.0, l*(1 + s*Z)); Q = np.exp(sq*(rho*Z + np.sqrt(1 - rho**2)*U))
    mi = rng.normal(m_mean, m_sd, N)
    bad = (mi < m_mean - 4*m_sd) | (mi > m_mean + 4*m_sd)
    while bad.any():
        mi[bad] = rng.normal(m_mean, m_sd, bad.sum()); bad = (mi < m_mean - 4*m_sd) | (mi > m_mean + 4*m_sd)
    tau = rng.poisson(delay, N).astype(np.int64)
    k = r_mean*(1 - r_mean)/r_sd**2 - 1
    ri = rng.beta(r_mean*k, (1 - r_mean)*k, N)
    return Q, Q*(1 - 1/lev), mi, tau, ri

def diagnostics(path, T, d_lo, d_hi, W=200):
    """final sold fraction; 10-90% width (steps) of the largest W-window jump region; midpoint field; max W-step increase."""
    Sf = path[-1]
    inc = path[W:] - path[:-W]; JW = inc.max()
    t10 = np.searchsorted(path, 0.1*Sf); t50 = np.searchsorted(path, 0.5*Sf); t90 = np.searchsorted(path, 0.9*Sf)
    fld = lambda t: d_lo + (d_hi - d_lo)*min(t, T)/T
    return Sf, JW, t90 - t10, fld(t50), fld(t10), fld(t90)
