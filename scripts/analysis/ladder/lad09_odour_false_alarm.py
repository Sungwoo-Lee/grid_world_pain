"""FIGURE 9 - The smell of a rabbit, and whether a wound makes it scarier.

QUESTION. Each animal's odour is drawn fresh for every episode, independently of everything else.
That makes odour strength a second randomised handle - a clean one, because the agent cannot
choose what a rabbit smells like. If bush dwell rises with the RABBIT odour draw, the agent is
responding to a cue that carries no danger at all. This figure asks whether it does, whether that
depends on what the arm can see, and whether waking up wounded makes it worse.

WHY THIS IS THE SHARPEST TEST IN THE REPORT. The predator panel is what a competent agent should
do: hide more when the world happens to contain strong-smelling predators. The rabbit panel is
pure false alarm - and because the draw is random, no confound can produce it.

HOW IT IS COMPUTED. Within each arm, episodes are split into quartiles of the mean odour intensity
drawn for its rabbits (and separately, its predators); intensity is the sum of the two olfactory
channels that separate the two classes, with the channels derived from the run's own config rather
than assumed. Episodes containing no predator (or no rabbit) are excluded from the corresponding
panel - they have no odour draw at all, and letting them fall into a bin would quietly make
"episodes with no predator in them" the comparison group.

WHY THE PREDATOR PANEL DIPS AT THE LOUDEST QUARTILE. That dip is real behaviour, not an artefact.
Intensity is the SUM of the two odour channels while what marks an animal as a predator is their
DIFFERENCE, so an episode whose predators smell very loudly has both channels near their ceiling
and the difference is squeezed toward zero. Measured on this run: mean predator-ness is 0.12 in the
loudest quartile against 0.18-0.22 in the other three. The loudest predators are the least
distinguishable ones, and the agent responds to them less. Bush dwell is pooled over each episode's first 25 steps. Solid lines are episodes
that began nearly unhurt (start wound 0-25), dashed lines those that began badly wounded (75-100);
the gap between a pair is how much the wound amplified the response to a harmless smell.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER; col = PL.arm_colors()
x = np.arange(4)
QN = ["weakest\nquarter", "2nd", "3rd", "strongest\nquarter"]

def curve(a, kind, ib):
    o = D[a]["odour"]
    b = np.asarray(o[f"{kind}_bush"], float)[:, ib]
    t = np.asarray(o[f"{kind}_tot"], float)[:, ib]
    return np.where(t >= 1000, 100 * b / np.maximum(t, 1), np.nan)

fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)
for j, (kind, ttl, xl) in enumerate([
        ("rab", "RABBIT odour - carries no danger, so any slope is a false alarm",
         "how strongly this episode's RABBITS happened to smell\n(quartile of the randomised odour draw)"),
        ("pred", "PREDATOR odour - a real cue (control panel)",
         "how strongly this episode's PREDATORS happened to smell\n(quartile of the randomised odour draw)")]):
    for a in arms:
        ax[j].plot(x, curve(a, kind, 0), lw=1.6, marker="o", ms=3.4, color=col[a],
                   label=f"{a} - {L.ARM_LABEL[a][0]}")
        ax[j].plot(x, curve(a, kind, 3), lw=1.3, ls="--", marker="s", ms=3.0, color=col[a],
                   alpha=0.85)
    ax[j].set_xticks(x); ax[j].set_xticklabels(QN, fontsize=8)
    ax[j].set_xlabel(xl)
    ax[j].set_title(ttl, fontsize=9.5, loc="left", pad=8)
ax[0].set_ylabel("bush dwell over the episode's first 25 steps\n(% of those steps spent in a bush)")
h = [plt.Line2D([], [], color=PL.MUTED, lw=1.7, marker="o", ms=4,
                label="episodes that began nearly unhurt  (start wound 0-25)"),
     plt.Line2D([], [], color=PL.MUTED, lw=1.4, ls="--", marker="s", ms=3.6,
                label="episodes that began badly wounded  (start wound 75-100)")]
ax[0].legend(handles=h, loc="lower center", bbox_to_anchor=(1.05, 1.13), ncol=2, fontsize=8.5)
ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.4,
             title="sensor-ladder arm", title_fontsize=8)
PL.finish(fig, f"{L.FIG_ROOT}/lad09_odour_false_alarm.png")
print(f"{'arm':22}{'rabbit slope':>14}{'  (unhurt)':>12}{'rabbit slope':>14}{'  (wounded)':>13}"
      f"{'predator slope':>16}")
for a in arms:
    r0, r3 = curve(a, "rab", 0), curve(a, "rab", 3)
    p0 = curve(a, "pred", 0)
    print(f"{a:22}{r0[3]-r0[0]:>+13.2f}{'':>12}{r3[3]-r3[0]:>+13.2f}{'':>13}{p0[3]-p0[0]:>+15.2f}")
