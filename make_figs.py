"""Generate all study figures from model.py and results/*.csv."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model import *
from aval_theory import integrated_borel
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7, "figure.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.4})
lab = lambda ax, s: ax.set_title(s, loc="left", fontweight="bold")

# ---- Fig 1: skeleton vs absorbing
l, s, m, chi, Lam = 2.0, 0.16, 0.25, 0.30, 0.30
(Bp, dp), (Bm, dm) = folds(l, s, m, chi, Lam); Bl = landing(l, s, m, chi, Lam)
B = np.linspace(1e-6, 0.9, 4000); d = d0_of_B(B, l, s, m, chi, Lam)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
ax[0].plot(d, B); ax[0].plot([dp, dm], [Bp, Bm], "ko", ms=3); ax[0].set_xlim(0.05, 0.16); ax[0].set_xlabel("external drawdown $d_0$"); ax[0].set_ylabel("exposure fraction $B$"); ax[0].set_title("equality map"); lab(ax[0], "(a)")
lo = B < Bp; hi = B > Bl
ax[1].plot(d[lo], B[lo], "C0"); ax[1].plot(d[hi], B[hi], "C0"); ax[1].plot([dp, dp], [Bp, Bl], "C0:"); ax[1].axvline(dp, color="0.5", ls="--", lw=0.8)
ax[1].set_xlim(0.05, 0.16); ax[1].set_ylim(ax[0].get_ylim()); ax[1].set_xlabel("external drawdown $d_0$"); ax[1].set_ylabel("liquidated exposure $L$"); ax[1].set_title("absorbing update from $L_0=0$"); lab(ax[1], "(b)")
fig.tight_layout(); fig.savefig("figures/fig1_skeleton.png"); plt.close(fig)

# ---- Fig 2: cusp vs d0, m
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8)); dd = np.linspace(0.02, 0.40, 60)
for mm in [0.20, 0.25, 0.30, 0.35]:
    c = np.array([cusp_power(0.16, 0.0, mm, x)[1:] for x in dd]); ax[0].plot(dd, c[:, 0], label="$m=%.2f$" % mm); ax[1].plot(dd, c[:, 1])
ax[0].set_xlabel("initiating drawdown $d_0$"); ax[0].set_ylabel(r"$\Lambda_c$"); ax[0].legend(); lab(ax[0], "(a)")
ax[1].set_xlabel("initiating drawdown $d_0$"); ax[1].set_ylabel(r"cusp leverage $\bar\ell^*$"); lab(ax[1], "(b)")
fig.tight_layout(); fig.savefig("figures/fig2_cusp.png"); plt.close(fig)

# ---- Fig 3: exposure tilt
cc = np.linspace(-0.8, 0.8, 41); c = np.array([cusp_power(0.16, x, 0.25, 0.10)[1:] for x in cc])
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for k, yl in enumerate([r"$\Lambda_c$", r"cusp leverage $\bar\ell^*$"]):
    ax[k].plot(cc, c[:, k]); ax[k].axvline(0, color="0.5", ls="--", lw=0.8); ax[k].set_xlabel(r"exposure tilt $\chi=\rho_{qz}s_q$"); ax[k].set_ylabel(yl); lab(ax[k], "(%s)" % "ab"[k])
fig.tight_layout(); fig.savefig("figures/fig3_tilt.png"); plt.close(fig)
pd.DataFrame(dict(chi=cc, Lambda_c=c[:, 0], l_star=c[:, 1])).to_csv("results/cusp_vs_tilt.csv", index=False)

# ---- Fig 4: impact laws
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8)); x = np.linspace(1e-6, 1, 800); Bb = np.linspace(1e-6, 0.8, 3000); rows = []
for name, (f, fp) in IMPACT.items():
    on, Bon = onset_fixed_l(2.0, 0.16, 0.25, 0.30, name); rows.append(dict(law=name, Lambda_onset=on, B_onset=Bon))
    ax[0].plot(x, f(x), label=name); ax[1].plot(Bb, d0_of_B(Bb, 2.0, 0.16, 0.25, 0.30, 1.25*on, name))
pd.DataFrame(rows).to_csv("results/impact_law_onsets.csv", index=False)
ax[0].set_xlabel("sold exposure fraction"); ax[0].set_ylabel("normalized impact $f$"); ax[0].legend(); lab(ax[0], "(a)")
ax[1].set_ylim(0.0, 0.2); ax[1].set_xlabel("exposure fraction $B$"); ax[1].set_ylabel("external field $d_0$"); lab(ax[1], "(b)")
fig.tight_layout(); fig.savefig("figures/fig4_impact_laws.png"); plt.close(fig)

# ---- Fig 5: finite-N lognormal
f = pd.read_csv("results/finiteN_lognormal.csv"); th = pd.read_csv("results/finiteN_theory_constants.csv").iloc[0]; Ng = np.geomspace(200, 4e5, 50)
fig, ax = plt.subplots(1, 2, figsize=(7, 3.0))
ax[0].errorbar(f.N, f.mean_shift, yerr=2*f.se_shift, fmt="o", ms=4, label="simulation"); ax[0].plot(Ng, th.shift_coef*Ng**(-2/3), "k-", lw=1, label=r"Eq. (28), $\propto N^{-2/3}$")
ax[0].plot(Ng, th.sd_coef*Ng**-0.5*0.5, "k:", lw=0.8, label=r"$\propto N^{-1/2}$ (guide)")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("cohort size $N$"); ax[0].set_ylabel(r"mean shift $\langle d_{0,+}^{(N)}\rangle-d_{0,+}$"); ax[0].legend(loc="lower left"); lab(ax[0], "(a)")
ax[1].plot(f.N, f.sd_field, "o", ms=4, label="s.d. collapse field"); ax[1].plot(Ng, th.sd_coef*Ng**-0.5, "C0-", lw=1)
ax[1].plot(f.N, f.sd_takeoff, "s", ms=4, color="C1", label="s.d. takeoff state"); ax[1].plot(Ng, th.loc_coef*Ng**(-1/3), "C1-", lw=1)
ax[1].plot(f.N, f.sd_jump, "^", ms=4, color="C2", label="s.d. jump size")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("cohort size $N$"); ax[1].set_ylabel("standard deviation"); ax[1].legend(loc="lower left"); lab(ax[1], "(b)")
fig.tight_layout(); fig.savefig("figures/fig5_finiteN.png"); plt.close(fig)

# ---- Fig 6: heavy tails
p = pd.read_csv("results/finiteN_pareto.csv"); fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7)); ex = []
for k, (mu, g) in enumerate(p.groupby("mu")):
    ax[0].plot(g.N, g.iqr, "o-", ms=3, lw=0.8, color="C%d" % k, label=r"$\mu=%g$" % mu)
    ex.append((mu, -np.polyfit(np.log(g.N), np.log(g.iqr), 1)[0], -np.polyfit(np.log(g.N), np.log(g.sd), 1)[0]))
ex = np.array(ex); pd.DataFrame(ex, columns=["mu", "iqr_exponent", "sd_exponent"]).to_csv("results/pareto_exponents.csv", index=False)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("$N$"); ax[0].set_ylabel("IQR of collapse field"); ax[0].legend(); lab(ax[0], "(a)")
mg = np.linspace(1.05, 4.2, 200)
ax[1].plot(mg, np.where(mg < 2, 1-1/mg, 0.5), "k-", lw=1, label=r"$\min(1-1/\mu,\,1/2)$"); ax[1].plot(mg, np.where(mg < 2, (mg-1)/2, 0.5), "k--", lw=1, label=r"$\min((\mu-1)/2,\,1/2)$")
ax[1].plot(ex[:, 0], ex[:, 1], "o", ms=4, label="IQR exponent"); ax[1].plot(ex[:, 0], ex[:, 2], "s", ms=4, mfc="none", label="s.d. exponent")
ax[1].set_xlabel(r"tail index $\mu$"); ax[1].set_ylabel("decay exponent"); ax[1].set_ylim(0, 0.62); ax[1].legend(loc="lower right"); lab(ax[1], "(b)")
d0p = p.d0p.iloc[0]; late = []
for k, mu in enumerate(["1.25", "1.5", "1.75"]):
    raw = np.load("results/pareto_raw_mu%s.npz" % mu); Ns = np.array([int(n) for n in raw.files]); pr = np.array([np.mean(raw[n][:, 0] > d0p + 0.04) for n in raw.files]); R = np.array([len(raw[n]) for n in raw.files])
    ok = pr*R >= 3; ax[2].plot(Ns[ok], pr[ok], "o", ms=4, color="C%d" % k, label=r"$\mu=%s$" % mu); ax[2].plot(Ns, pr[0]*(Ns/Ns[0])**(1-float(mu)), "-", lw=0.8, color="C%d" % k)
    late += [dict(mu=mu, N=n, prob_late=a, cohorts=r) for n, a, r in zip(Ns, pr, R)]
pd.DataFrame(late).to_csv("results/pareto_late_collapse.csv", index=False)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("$N$"); ax[2].set_ylabel(r"$P(d_{0,+}^{(N)}>d_{0,+}+0.04)$"); ax[2].legend(); lab(ax[2], "(c)")
fig.tight_layout(); fig.savefig("figures/fig6_heavy_tails.png"); plt.close(fig)

# ---- Fig 7: avalanches
fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))
for k, (nm, lb) in enumerate([("equal", "equal exposures"), ("lognormal", "lognormal, $s_q=0.75$"), ("1.5", r"Pareto, $\mu=1.5$")]):
    a = pd.read_csv("results/avalanches_%s.csv" % nm); a = a[a.count_all >= 10]; ax.plot(a.S_center, a.pdf_all, "os^"[k], ms=3.5, mfc="none", color="C%d" % k, label=lb)
Sg = np.unique(np.round(np.geomspace(1, 3000, 60))).astype(int); ax.plot(Sg, integrated_borel(Sg), "C0-", lw=1, label="Borel mixture, Eq. (32)")
ax.plot(Sg[Sg > 30], 0.9*Sg[Sg > 30]**-2.5, "k:", lw=0.8, label=r"$S^{-5/2}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("avalanche size $S$ (accounts)"); ax.set_ylabel("$D(S)$"); ax.legend()
fig.tight_layout(); fig.savefig("figures/fig7_avalanches.png"); plt.close(fig)

# ---- Fig 8: path-resolved, size and ramp
cs = np.load("results/path_curves_size.npz"); cr = np.load("results/path_curves_ramp.npz"); ps = pd.read_csv("results/path_size.csv"); pr_ = pd.read_csv("results/path_ramp.csv")
fig, ax = plt.subplots(2, 2, figsize=(7, 5.2)); ax = ax.ravel()
for N in (2000, 8000, 32000):
    y = cs["N%d" % N]; t = 5*np.arange(y.size); k = t <= 3000; ax[0].plot(0.02 + 0.28*t[k]/3000, y[k], label="$N=%d$" % N)
ax[0].set_xlabel("$d_0$"); ax[0].set_ylabel("sold exposure"); ax[0].legend(); lab(ax[0], "(a)")
for hI, lb, mk in [(80.0, "half-life 80", "o"), (-1.0, "permanent impact", "s")]:
    g = ps[ps.half_life == hI]; ax[1].plot(g.N, g.mid_field_sd, mk + "-", ms=4, lw=0.8, label=lb)
Ng = np.array([1e3, 6.4e4]); ax[1].plot(Ng, 0.0043*(Ng/1e3)**-0.5, "k:", lw=0.8, label=r"$N^{-1/2}$")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("$N$"); ax[1].set_ylabel("s.d. of transition field"); ax[1].legend(); lab(ax[1], "(b)")
for T in (1000, 4000, 16000):
    y = cr["T%d" % T]; st = max(1, T//1000); t = st*np.arange(y.size); k = t <= T; ax[2].plot(0.02 + 0.28*t[k]/T, y[k], label="$T=%d$" % T)
ax[2].set_xlim(0.19, 0.26); ax[2].set_xlabel("$d_0$"); ax[2].set_ylabel("sold exposure"); ax[2].legend(); lab(ax[2], "(c)")
ax[3].errorbar(pr_["T"], pr_.width_field_mean, yerr=pr_.width_field_sd, fmt="o", ms=4, label="width in $d_0$"); ax[3].plot(pr_["T"], 32.4*0.28/pr_["T"], "k-", lw=0.8, label=r"$\tau_c\,\Delta d_0/T$, $\tau_c=32.4$")
ax[3].set_xscale("log"); ax[3].set_yscale("log"); ax[3].set_xlabel("ramp duration $T$ (steps)"); ax[3].set_ylabel("10–90% width in $d_0$"); ax[3].legend(loc="lower left"); lab(ax[3], "(d)")
a2 = ax[3].inset_axes([0.55, 0.55, 0.42, 0.4]); a2.errorbar(pr_["T"], pr_.width_steps_mean, yerr=pr_.width_steps_sd, fmt="s", ms=3, color="C1"); a2.set_xscale("log"); a2.set_ylim(20, 45); a2.text(0.5, 0.9, "width (steps)", transform=a2.transAxes, ha="center", va="top", fontsize=7); a2.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig("figures/fig8_path.png"); plt.close(fig)

# ---- Fig 9: impact memory
cm = np.load("results/path_curves_memory.npz"); pm = pd.read_csv("results/path_memory.csv"); fb = pm[pm.Lam > 0].copy(); fb["h"] = fb.half_life.replace(-1, 2000)
fig, ax = plt.subplots(2, 2, figsize=(7, 5.2)); ax = ax.ravel()
for key, lb in [("h-1", "permanent"), ("h80", "half-life 80"), ("h20", "20"), ("h12", "12"), ("h4", "4"), ("baseline", r"$\Lambda=0$")]:
    y = cm[key]; t = 5*np.arange(y.size); k = t <= 4600; ax[0].plot(0.02 + 0.43*t[k]/4600, y[k], "k--" if key == "baseline" else "-", lw=1, label=lb)
ax[0].set_xlabel("$d_0$"); ax[0].set_ylabel("sold exposure"); ax[0].legend(); lab(ax[0], "(a)")
for k, (dl, g) in enumerate(fb.groupby("delay")):
    ax[1].errorbar(g.h, g.largest_event_mean, yerr=g.largest_event_sd, fmt="os^"[k] + "-", ms=3, lw=0.7, label="mean delay %g" % dl)
    ax[2].errorbar(g.h, g.event_field_mean, yerr=g.event_field_sd, fmt="os^"[k] + "-", ms=3, lw=0.7)
    ax[3].plot(g.h, g.n_events_mean, "os^"[k] + "-", ms=3, lw=0.7)
ax[1].axhline(pm[pm.Lam == 0].largest_event_mean.mean(), color="k", ls="--", lw=0.8, label=r"$\Lambda=0$ baseline")
for a, yl, tg in zip(ax[1:], ["largest 200-step event $J$", "field of largest event", "events with $J>0.1$ per cohort"], "bcd"):
    a.set_xscale("log"); a.set_xlabel("impact half-life $h_I$ (steps)"); a.set_ylabel(yl); lab(a, "(%s)" % tg)
ax[1].legend(loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig9_memory.png"); plt.close(fig)
print("done")
