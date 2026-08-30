"""FIGURE 11 - Where hypervigilance DOES show up: the smell of a harmless animal.

QUESTION. Figure 10 looked for hypervigilance on the proximity channel and did not find it: a wound
raised the agent's response to a nearby predator more than to a nearby rabbit, which is heightened
caution, not a change in what counts as evidence of danger. But proximity is not the only cue the
agent has, and it is not the ambiguous one. Smell is. Each animal's odour is drawn fresh and at
random every episode, and a rabbit's odour carries no danger whatsoever - so any response to it is
a false alarm by construction. This figure asks whether waking up wounded makes that false alarm
LOUDER.

WHAT THE ANSWER IS. It does, and it does so selectively. In the arms whose sight can resolve what
it is looking at, the response to a rabbit's smell grows when the agent starts the episode wounded.
In the arms that never had reliable sight to begin with, it does not - those agents were already
responding to rabbit odour at full strength, wounded or not, so there is no headroom. The effect is
largest in the three occlusion arms, where sight is present but intermittently blocked by scenery.

The reading this supports: a wound does not make the agent generically more afraid. It makes the
agent lean harder on an ambiguous channel - and that shows up only in an agent that has a reliable
channel to lean AWAY from.

HOW IT IS COMPUTED. Within each arm, episodes are split by the odour intensity drawn for their
rabbits (four quartiles; intensity is the sum of the two olfactory channels that separate predators
from rabbits, with the channels derived from the run's own config). The slope is bush dwell in the
strongest-smelling quarter minus the weakest, over the episode's first 25 steps. That slope is
computed twice: once over episodes that began nearly unhurt (start wound 0-25) and once over those
that began badly wounded (75-100). The bar is the difference. Episodes containing no rabbit are
excluded - they have no rabbit odour, and letting them fall into a bin would make the reference
group "episodes with no rabbit in them".
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER

def slope(a, kind, ib):
    o = D[a]["odour"]
    b = np.asarray(o[f"{kind}_bush"], float)[:, ib]
    t = np.asarray(o[f"{kind}_tot"], float)[:, ib]
    c = np.where(t >= 1000, 100 * b / np.maximum(t, 1), np.nan)
    return c[3] - c[0]

r_lo = np.array([slope(a, "rab", 0) for a in arms])
r_hi = np.array([slope(a, "rab", 3) for a in arms])
p_lo = np.array([slope(a, "pred", 0) for a in arms])
p_hi = np.array([slope(a, "pred", 3) for a in arms])
d_r, d_p = r_hi - r_lo, p_hi - p_lo

resolves_identity = lambda a: L.resolves_identity(D[a]["sensory"])
grp = np.array([resolves_identity(a) for a in arms])
CY, CN = PL.GROUP_YES, PL.GROUP_NO

fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.9), sharey=True,
                       gridspec_kw={"width_ratios": [1.2, 1]})
y = np.arange(len(arms)); h = 0.36
ax[0].barh(y + h/2, r_lo, height=h, color=PL.WOUND_LO, edgecolor="none",
           label="episodes that began nearly unhurt  (start wound 0-25)")
ax[0].barh(y - h/2, r_hi, height=h, color=PL.WOUND_HI, edgecolor="none",
           label="episodes that began badly wounded  (start wound 75-100)")
ax[0].axvline(0, color=PL.INK, lw=1)
ax[0].set_yticks(y); ax[0].set_yticklabels(PL.arm_ylabels(arms), fontsize=8)
ax[0].set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
ax[0].set_xlabel("response to a strong RABBIT smell  (percentage points)\n"
                 "bush dwell in the strongest-smelling quarter minus the weakest")
ax[0].grid(axis="y", visible=False)
ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=8.5)
ax[0].set_xlim(min(0, r_lo.min(), r_hi.min()) - 0.6, max(r_lo.max(), r_hi.max()) * 1.16)

ax[1].barh(y + h/2, d_r, height=h, color=[CY if g else CN for g in grp], edgecolor="none")
ax[1].barh(y - h/2, d_p, height=h, color=PL.GRID, edgecolor="none")
ax[1].axvline(0, color=PL.INK, lw=1)
ax[1].set_xlabel("how much the wound AMPLIFIED that response  (percentage points)\n"
                 "wounded slope minus unhurt slope")
ax[1].grid(axis="y", visible=False)
m = max(np.max(np.abs(np.r_[d_r, d_p])), 1e-6)
ax[1].set_xlim(min(0, np.min(np.r_[d_r, d_p])) - m * 0.45, m * 1.5)
for i in range(len(arms)):
    PL.outward_label(ax[1], d_r[i], y[i] + h/2, m * 0.04, "{:+.2f}", fontsize=6.9)
    PL.outward_label(ax[1], d_p[i], y[i] - h/2, m * 0.04, "{:+.2f}", color=PL.MUTED, fontsize=6.9)
hh = [plt.Rectangle((0, 0), 1, 1, color=CY, label="RABBIT smell - arm whose sight resolves identity"),
      plt.Rectangle((0, 0), 1, 1, color=CN, label="RABBIT smell - arm whose sight does not"),
      plt.Rectangle((0, 0), 1, 1, color=PL.GRID, label="PREDATOR smell (control, all arms)")]
ax[1].legend(handles=hh, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=8)
PL.finish(fig, f"{L.FIG_ROOT}/lad11_hypervigilance_odour.png")
print(f"{'arm':22}{'rab unhurt':>12}{'rab wounded':>13}{'amplified':>11}   "
      f"{'pred amplified':>15}   resolves identity")
for i, a in enumerate(arms):
    print(f"{a:22}{r_lo[i]:>+12.2f}{r_hi[i]:>+13.2f}{d_r[i]:>+11.2f}   {d_p[i]:>+15.2f}   {grp[i]}")
