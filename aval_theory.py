"""Borel-mixture prediction for the integrated precursor avalanche distribution (equal exposures)."""
import numpy as np
from scipy.special import gammaln
from model import *
def integrated_borel(S, l=2.0, s=0.16, m=0.25, chi=0.0, Lam=0.30, n=20000):
    Bp = folds(l, s, m, chi, Lam)[0][0]
    B = np.linspace(1e-9, Bp, n); R = Lam*IMPACT["square-root"][1](B)/dt_dB(B, l, s, m, chi)
    R = np.clip(R, 1e-12, 1.0)
    S = np.asarray(S, float)[:, None]
    logp = -R*S + (S-1)*np.log(R*S) - gammaln(S+1)
    D = np.trapezoid((1-R)*np.exp(logp), B, axis=1)
    norm_ = np.trapezoid(1-R, B)
    return D/norm_
if __name__ == "__main__":
    import pandas as pd
    d = pd.read_csv("results/avalanches_equal.csv"); d = d[d.count_all > 30]
    for lo, hi, pdf in zip(d.bin_lo, d.bin_hi, d.pdf_all):
        Ss = np.arange(lo, hi); th = integrated_borel(Ss).mean()
        print(int(lo), int(hi), "sim %.3e  theory %.3e  ratio %.3f" % (pdf, th, pdf/th))
