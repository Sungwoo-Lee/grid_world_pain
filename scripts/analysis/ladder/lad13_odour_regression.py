"""FIGURE 13 - The false alarm, with everything else held fixed.

QUESTION. Figure 9 showed bush dwell rising with the strength of a rabbit's smell. That is already
causal, because the smell is drawn at random. But it is a raw contrast: a world with strong-smelling
rabbits might differ in other ways too. This figure repeats the test inside a regression that
adjusts for the number of bushes, the number of food patches, how far the agent spawned from cover,
the predator's detection range, attack delay, attack range and stamina, the predator's own smell,
and the wound and hunger the agent woke up with.

WHAT THE TWO BARS MEAN. `pred_olf_intensity` is how strongly this episode's predator smelled;
`rab_olf_intensity` is how strongly its rabbit smelled. A positive predator bar is the agent doing
its job. A positive rabbit bar is the agent hiding from an animal that has never hurt it and cannot.
The rabbit bar is the false-alarm rate expressed in percentage points of lost foraging time.

HOW IT IS COMPUTED. Quasi-binomial regression on the episode-level bush-dwell rate, restricted to
episodes with exactly one predator and one rabbit so that "the predator's smell" and "the rabbit's
smell" are each a single well-defined number rather than an average over several animals. Standard
errors are scaled by the Pearson overdispersion. Bars are the effect of a one-standard-deviation
change in odour intensity, in percentage points.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv, numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

MODEL = "M3 + rabbit smell, 1 predator + 1 rabbit"
GLM_ROOT = "results/analysis/lad"
arms = [a for a in L.ARM_ORDER if os.path.exists(f"{GLM_ROOT}/{a}/multivariate.csv")]

def get(a):
    rows = [r for r in csv.DictReader(open(f"{GLM_ROOT}/{a}/multivariate.csv"))
            if r["model"] == MODEL]
    d = {r["term"]: (float(r["dpp_per_sd"]), float(r["p"])) for r in rows}
    return d.get("pred_olf_intensity", (np.nan, 1)), d.get("rab_olf_intensity", (np.nan, 1))

P = np.array([get(a)[0][0] for a in arms]); Pp = np.array([get(a)[0][1] for a in arms])
R = np.array([get(a)[1][0] for a in arms]); Rp = np.array([get(a)[1][1] for a in arms])

fig, ax = plt.subplots(figsize=(11.2, 0.40 * len(arms) + 2.9))
y = np.arange(len(arms))[::-1]; h = 0.36
ax.barh(y + h/2, P, height=h, color=PL.THREAT, edgecolor="none",
        label="stronger PREDATOR smell  (a real cue - hiding is correct)")
ax.barh(y - h/2, R, height=h, color=PL.HARMLESS, edgecolor="none",
        label="stronger RABBIT smell  (harmless - hiding is a false alarm)")
ax.axvline(0, color=PL.INK, lw=1)
ax.set_yticks(y); ax.set_yticklabels(PL.arm_ylabels(arms), fontsize=8)
ax.set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
ax.set_xlabel("effect on bush dwell of a one-standard-deviation stronger smell  "
              "(percentage points)\nadjusted for bushes, food, cover distance, predator traits, "
              "and the agent's starting wound and hunger")
ax.grid(axis="y", visible=False)
for i in range(len(arms)):
    for val, off, pv in ((P[i], +h/2, Pp[i]), (R[i], -h/2, Rp[i])):
        if np.isfinite(val):
            ax.text(val + np.sign(val) * 0.06, y[i] + off,
                    f"{val:+.2f}" + ("" if pv < 0.001 else " (n.s.)"),
                    va="center", ha="left" if val >= 0 else "right", fontsize=7, color=PL.INK)
hi = np.nanmax(np.concatenate([P, R])); lo = min(np.nanmin(np.concatenate([P, R])), 0.0)
ax.set_xlim(lo - 0.35 - abs(lo) * 0.5, hi * 1.42 + 0.35)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=8.5)
fig.text(0.5, -0.08, "'(n.s.)' marks an effect that is not distinguishable from zero at p < 0.001 "
         "after scaling for overdispersion", ha="center", fontsize=7.6, color=PL.MUTED)
PL.finish(fig, f"{L.FIG_ROOT}/lad13_odour_regression.png")
print(f"{'arm':22}{'predator smell':>16}{'rabbit smell':>15}")
for i, a in enumerate(arms):
    print(f"{a:22}{P[i]:>+16.3f}{R[i]:>+15.3f}")
