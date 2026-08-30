"""FIGURE 2 - Isolating one sensor change at a time.

QUESTION. Figure 1 ranks the arms, but a ranking confounds everything: the sharp-sighted agent
differs from the blind one in several settings at once. The ladder was built so that most arms are
ONE setting away from a named reference arm. This figure shows only those single-variable steps,
so each bar is the effect of exactly one knob.

HOW TO READ IT. Each row names a change and its reference. A bar to the right means the change
made the agent live longer (top panel) or hide more (bottom panel). `_ladder.ARM_REFERENCE` holds
the pairing and is derived from the configs, not assumed.

HOW IT IS COMPUTED. The difference of the two arms' pooled values from Figure 1. Because every
arm replayed the SAME 300,000 worlds, the two members of a pair met identical predators, identical
food and identical starting wounds - the difference is the sensor change and nothing else.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all()
pairs = [(a, r) for a, r in L.ARM_REFERENCE.items()]
pairs.sort(key=lambda p: L.ARM_ORDER.index(p[0]))
lab = [f"{L.ARM_LABEL[a][1]}\n(vs {r})" for a, r in pairs]
d_surv = [D[a]["mean_survival"] - D[r]["mean_survival"] for a, r in pairs]
d_dwell = [D[a]["bush_dwell_pct"] - D[r]["bush_dwell_pct"] for a, r in pairs]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
y = np.arange(len(pairs))[::-1]
for k, (vals, xl, ttl) in enumerate([
        (d_surv, "change in mean survival  (steps, vs the reference arm)\n"
                  "left of zero = the change SHORTENED life",
         "Effect of the change on how long it lives"),
        (d_dwell, "change in bush dwell  (percentage points, vs the reference arm)\n"
                  "left of zero = the change REDUCED hiding",
         "Effect of the change on how much it hides")]):
    c = [PL.THREAT if v < 0 else PL.HARMLESS for v in vals]
    ax[k].barh(y, vals, color=c, height=0.7, edgecolor="none")
    ax[k].axvline(0, color=PL.INK, lw=1)
    ax[k].set_xlabel(xl); ax[k].set_title(ttl, fontsize=9, color=PL.MUTED, loc="left", pad=8)
    ax[k].grid(axis="y", visible=False)
    m = max(abs(np.array(vals))) * 1.35
    ax[k].set_xlim(-m, m)
    for i, v in enumerate(vals):
        off = m * 0.03
        ax[k].text(v + (off if v >= 0 else -off), y[i],
                   f"{v:+.1f}" + ("" if k == 0 else " pp"),
                   va="center", ha="left" if v >= 0 else "right", fontsize=8, color=PL.INK)
ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=8)
ax[0].set_ylabel("single-variable sensor change")
PL.finish(fig, f"{L.FIG_ROOT}/lad02_single_variable_steps.png")
print(f"{'change':34}{'vs':16}{'d survival':>12}{'d dwell':>11}")
for (a, r), s, d in zip(pairs, d_surv, d_dwell):
    print(f"{L.ARM_LABEL[a][1][:33]:34}{r:16}{s:>+12.1f}{d:>+10.1f}pp")
