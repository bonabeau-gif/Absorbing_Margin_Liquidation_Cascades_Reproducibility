"""Monte Carlo constants for M = max_v [W(v) - v^2/2], two-sided standard Brownian motion."""
import numpy as np
rng = np.random.default_rng(7)
dt, V, R = 5e-4, 5.0, 20000  # sd[M] from this run is used only for a subleading correction
n = int(V/dt); v = dt*np.arange(1, n+1); par = 0.5*v**2
M = np.empty(R); arg = np.empty(R)
for r in range(R):
    a = np.cumsum(rng.standard_normal(n))*np.sqrt(dt) - par
    b = np.cumsum(rng.standard_normal(n))*np.sqrt(dt) - par
    ia, ib = a.argmax(), b.argmax()
    if max(a[ia], b[ib]) < 0: M[r], arg[r] = 0.0, 0.0
    elif a[ia] >= b[ib]: M[r], arg[r] = a[ia], v[ia]
    else: M[r], arg[r] = b[ib], -v[ib]
corr = 0.5826*np.sqrt(dt)   # standard discretization correction for the maximum of BM
print("E[M] raw %.4f  corrected %.4f +- %.4f" % (M.mean(), M.mean()+corr, M.std()/np.sqrt(R)))
print("sd[M] %.4f   sd[argmax] %.4f" % (M.std(), arg.std()))
np.savetxt("results/bm_parabola_constants.csv", [[M.mean()+corr, M.std(), arg.std()]], header="E_M,sd_M,sd_argmax", delimiter=",", comments="")
