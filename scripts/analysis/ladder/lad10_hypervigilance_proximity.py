"""FIGURE 10 - Hypervigilance: does a wound make the agent treat HARMLESS cues as threats?

QUESTION. This is the measure the ladder was built for, and it is a sharper question than "does a
wounded agent hide more". Figure 8 already shows that it does. Hiding more is ordinary caution. The
claim that would earn the word hypervigilance is a shift in what the agent treats as evidence of
danger - that when wounded, an AMBIGUOUS and harmless cue starts driving the same defence a real
threat does. So: when the agent wakes up badly wounded, does a nearby RABBIT push it into cover
more than when it wakes up nearly unhurt, and does it do so MORE than a nearby predator does?

WHY THE PREDATOR PANEL IS THE CONTROL AND NOT A SECOND RESULT. Hiding harder from a predator when
wounded is adaptive, and almost any account predicts it. If the wound raised the rabbit response and
the predator response by the same amount, the agent has simply become more defensive across the
board - a gain change, not a change in what counts as a threat. Only a rabbit shift that OUTRUNS the
predator shift is a criterion shift. The third panel puts the two shifts on one axis so that
comparison is read directly rather than inferred by eye across two different scales.

HOW IT IS COMPUTED. Within each starting-wound quarter separately,
    P(in bush | nearest animal 1-2 cells away) - P(in bush | nearest animal 6+ cells away).
The shift is the heaviest quarter (start wound 75-100) minus the lightest (0-25). The starting
wound is drawn uniformly at random by the environment before the agent acts, and the agent has no
sensor for it other than the interoceptive nociceptor - `injury_observable` is false - so this
contrast is causal. Predictors are read off the previous row.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER
LOW, HIGH = (0,), (3,)
eff = lambda a, k, b: L.proximity_effect(D[a]["grids"][f"{k}_bush"], D[a]["grids"][f"{k}_tot"], b)
rab_lo = np.array([eff(a, "rd", LOW) for a in arms]); rab_hi = np.array([eff(a, "rd", HIGH) for a in arms])
pre_lo = np.array([eff(a, "pd", LOW) for a in arms]); pre_hi = np.array([eff(a, "pd", HIGH) for a in arms])
d_rab, d_pre = rab_hi - rab_lo, pre_hi - pre_lo

fig, ax = plt.subplots(1, 3, figsize=(14.6, 6.2), sharey=True,
                       gridspec_kw={"width_ratios": [1, 1, 1.05]})
y = np.arange(len(arms)); h = 0.36
for a_ in ax:
    a_.set_yticks(np.arange(len(arms)) + 0.5, minor=True)
for j, (lo, hi, ttl, unit) in enumerate([
        (rab_lo, rab_hi, "A.  RABBIT nearby - harmless, so any response is wasted\n(same scale as panel B)", "rabbit"),
        (pre_lo, pre_hi, "B.  PREDATOR nearby - a real threat (control)\n(same scale as panel A)", "predator")]):
    ax[j].barh(y + h/2, lo, height=h, color=PL.WOUND_LO, edgecolor="none",
               label="woke up nearly unhurt  (start wound 0-25)")
    ax[j].barh(y - h/2, hi, height=h, color=PL.WOUND_HI, edgecolor="none",
               label="woke up badly wounded  (start wound 75-100)")
    ax[j].axvline(0, color=PL.INK, lw=1)
    ax[j].set_title(ttl, fontsize=9.5, loc="left", pad=8)
    ax[j].set_xlabel(f"hiding triggered by a nearby {unit}\n"
                     "a DIFFERENCE, in percentage points\n"
                     "dwell at 1-2 cells MINUS at 6+ cells")
    ax[j].grid(axis="y", visible=False)
    # A and B SHARE a scale. Drawing the rabbit panel on its own tighter axis made a response
    # of a few points look like the predator panel's forty, which is the opposite of the finding.
    allv = np.r_[rab_lo, rab_hi, pre_lo, pre_hi]
    ax[j].set_xlim(min(0, allv.min()) - 2.5, allv.max() * 1.06)

ax[2].barh(y + h/2, d_rab, height=h, color=PL.HARMLESS, edgecolor="none",
           label="shift in the RABBIT response")
ax[2].barh(y - h/2, d_pre, height=h, color=PL.THREAT, edgecolor="none",
           label="shift in the PREDATOR response")
ax[2].axvline(0, color=PL.INK, lw=1)
ax[2].set_title("C.  THE TEST: how much the wound moved each response\n"
                "(both on one scale - a longer blue bar than red would be hypervigilance)",
                fontsize=9.5, loc="left", pad=8)
ax[2].set_xlabel("shift caused by waking up badly wounded\n"
                 "a DIFFERENCE of two differences, in percentage points\n"
                 "the response above at wound 75-100 MINUS at 0-25")
ax[2].grid(axis="y", visible=True, color=PL.GRID, lw=0.5)
m2 = max(np.max(np.abs(np.r_[d_rab, d_pre])), 1e-6)
ax[2].set_xlim(min(0, np.min(np.r_[d_rab, d_pre])) - m2 * 0.45, m2 * 1.45)
for i in range(len(arms)):
    PL.outward_label(ax[2], d_rab[i], y[i] + h/2, m2 * 0.035, fontsize=6.8, color=PL.HARMLESS)
    PL.outward_label(ax[2], d_pre[i], y[i] - h/2, m2 * 0.035, fontsize=6.8, color=PL.THREAT)
ax[0].set_yticks(y); ax[0].set_yticklabels(PL.arm_ylabels(arms), fontsize=8)
ax[0].set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
ax[0].legend(loc="lower center", bbox_to_anchor=(1.03, 1.14), ncol=2, fontsize=8.5)
ax[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=1, fontsize=8.5)
POP = L.population()
def _q(key, bins, q):
    return sum(int(np.asarray(D[a]["grids"][f"{key}_tot"], float)[list(bins)][:, list(q)].sum())
               for a in arms)
L.record_samples("lad10_hypervigilance_proximity", [
    dict(what="step rows, rabbit panel, lightest wound quarter",
         used=_q("rd", L.NEAR_BINS, LOW) + _q("rd", L.FAR_BINS, LOW), total=POP["steps"],
         note="near and far bins together, for episodes that began with a wound of 0-25"),
    dict(what="step rows, rabbit panel, heaviest wound quarter",
         used=_q("rd", L.NEAR_BINS, HIGH) + _q("rd", L.FAR_BINS, HIGH), total=POP["steps"], note=""),
    dict(what="step rows, predator panel, lightest wound quarter",
         used=_q("pd", L.NEAR_BINS, LOW) + _q("pd", L.FAR_BINS, LOW), total=POP["steps"],
         note="smaller than the rabbit rows because a third of episodes contain no predator"),
    dict(what="step rows, predator panel, heaviest wound quarter",
         used=_q("pd", L.NEAR_BINS, HIGH) + _q("pd", L.FAR_BINS, HIGH), total=POP["steps"], note="")])

PL.assert_labels_fit(fig, ax)
PL.finish(fig, f"{L.FIG_ROOT}/lad10_hypervigilance_proximity.png")
print(f"{'arm':22}{'rab lo':>9}{'rab hi':>9}{'shift':>8}  |{'pred lo':>9}{'pred hi':>9}{'shift':>8}")
for i, a in enumerate(arms):
    print(f"{a:22}{rab_lo[i]:>9.1f}{rab_hi[i]:>9.1f}{d_rab[i]:>+8.1f}  |"
          f"{pre_lo[i]:>9.1f}{pre_hi[i]:>9.1f}{d_pre[i]:>+8.1f}")
