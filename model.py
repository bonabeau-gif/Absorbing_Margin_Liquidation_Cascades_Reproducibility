"""Common definitions for the absorbing margin-liquidation cascade model."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar

XC = 0.02      # crossover scale of the linear-to-square-root law
KLOG = 10.0    # logarithmic law
KSAT = 4.0     # saturating law

IMPACT = {
    "square-root":           (lambda x: np.sqrt(x),                       lambda x: 0.5/np.sqrt(x)),
    "linear-to-square-root": (lambda x: (np.sqrt(1+x/XC)-1)/(np.sqrt(1+1/XC)-1),
                              lambda x: 0.5/XC/np.sqrt(1+x/XC)/(np.sqrt(1+1/XC)-1)),
    "linear":                (lambda x: x,                                lambda x: np.ones_like(x)),
    "logarithmic":           (lambda x: np.log1p(KLOG*x)/np.log1p(KLOG),  lambda x: KLOG/(1+KLOG*x)/np.log1p(KLOG)),
    "saturating":            (lambda x: (1-np.exp(-KSAT*x))/(1-np.exp(-KSAT)),
                              lambda x: KSAT*np.exp(-KSAT*x)/(1-np.exp(-KSAT))),
}

def t_of_B(B, l, s, m, chi):
    """Exposure-weighted threshold-field quantile t(B): field at which fraction B is breached."""
    z = norm.ppf(B); A = 1 - s*(z - chi)
    return (1/(l*A) - m)/(1 - m)

def dt_dB(B, l, s, m, chi):
    z = norm.ppf(B); A = 1 - s*(z - chi)
    return s/((1 - m)*l*A**2*norm.pdf(z))

def d0_of_B(B, l, s, m, chi, Lam, law="square-root"):
    return t_of_B(B, l, s, m, chi) - Lam*IMPACT[law][0](B)

def folds(l, s, m, chi, Lam, law="square-root", n=400001):
    B = np.linspace(1e-6, 0.95, n); y = d0_of_B(B, l, s, m, chi, Lam, law)
    dy = np.diff(y); i = np.where(np.sign(dy[1:]) != np.sign(dy[:-1]))[0] + 1
    out = []
    for k in i:
        g = lambda b: (d0_of_B(b+1e-7, l, s, m, chi, Lam, law) - d0_of_B(b-1e-7, l, s, m, chi, Lam, law))
        b = brentq(g, B[k-2], B[k+2]); out.append((b, float(d0_of_B(b, l, s, m, chi, Lam, law))))
    return out

def landing(l, s, m, chi, Lam, law="square-root"):
    (Bp, dp), (Bm, dm) = folds(l, s, m, chi, Lam, law)[:2]
    return brentq(lambda b: d0_of_B(b, l, s, m, chi, Lam, law) - dp, Bm, 0.999)

def Lsp_power(B, s, chi, m, d0, a=0.5):
    z = norm.ppf(B); A = 1 - s*(z - chi); c = 1 - m; b0 = m + c*d0
    den = a*A*norm.pdf(z) - s*B
    return np.where(den > 0, s*b0*B**(1-a)/(c*np.where(den > 0, den, 1)), np.inf)

def cusp_power(s, chi, m, d0, a=0.5):
    r = minimize_scalar(lambda B: float(Lsp_power(B, s, chi, m, d0, a)), bounds=(1e-4, 0.6), method="bounded",
                        options=dict(xatol=1e-10))
    B = r.x; z = norm.ppf(B); A = 1 - s*(z - chi); c = 1 - m; b0 = m + c*d0
    return B, r.fun, 1/(A*(b0 + c*r.fun*B**a))

def onset_fixed_l(l, s, m, chi, law):
    B = np.linspace(1e-6, 0.95, 400001)
    L = dt_dB(B, l, s, m, chi)/IMPACT[law][1](B)
    return float(L.min()), float(B[L.argmin()])

def event_driven(t, Q, Lam, f=np.sqrt):
    """Quasistatic absorbing protocol (Appendix C). t: threshold fields, Q: exposures.
    Returns need (field required to reach each sorted account), record indices, cumulative exposure grid."""
    o = np.argsort(t, kind="stable"); t = t[o]; w = Q[o]/Q.sum()
    cw = np.cumsum(w); Lb = np.empty_like(cw); Lb[0] = 0.0; Lb[1:] = cw[:-1]
    need = t - Lam*f(Lb)
    rec = np.flatnonzero(need >= np.maximum.accumulate(need))
    return need, rec, Lb, w
