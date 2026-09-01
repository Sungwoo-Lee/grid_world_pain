"""FIGURE 8 - Does waking up wounded make the agent hide? (the causal test)

QUESTION. Everywhere else in this report, a wounded agent is a suspicious comparison: it got hurt
by doing something, so its later behaviour is contaminated by whatever it was doing. This
environment removes that problem. `body.random_start_injury` is true, so at the start of every
episode the agent is handed a wound drawn uniformly from 0 to 100 that it did nothing to earn.
Behaviour that tracks THAT number is caused by the wound.

WHY IT MATTERS FOR THE SENSOR LADDER. The agent cannot see its own wound - `injury_observable` is
false. The only route from an injury level to behaviour is the interoceptive nociceptor, which
convolves a twelve-slot buffer of injury LEVELS with an alpha kernel. So this figure asks whether
that internal channel changes behaviour, and whether the answer depends on what the agent can
sense of the outside world.

HOW IT IS COMPUTED. Episodes are split into four equal quarters of starting wound (0-25, 25-50,
50-75, 75-100). Bush dwell is pooled over the first 25 steps of each episode only - the wound
recovers over time, so a whole-episode average would dilute the assigned dose with whatever the
agent's own behaviour produced later. The t=0 row is excluded.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER; col = PL.arm_colors()
x = np.arange(4)
curves = {a: L.rate(D[a]["grids"]["dw_early"], D[a]["grids"]["dwt_early"]) for a in arms}
slope = {a: curves[a][3] - curves[a][0] for a in arms}
GRP = {a: L.resolves_identity(D[a]["sensory"]) for a in arms}
# A_baseline is the one arm whose line goes DOWN, so it is the one the prose names.
SPOT = ("A_baseline", "V4_blur05")

fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.5),
                       gridspec_kw={"width_ratios": [1.1, 1]})
h = PL.group_lines(ax[0], x, curves, GRP, spotlight=SPOT, label_end=True)
ax[0].set_xlim(-0.25, 3.7)
ax[0].set_xticks(x); ax[0].set_xticklabels(L.INJ_NAMES)
ax[0].set_xlabel("wound the agent was handed at the start of the episode\n"
                 "(injury level, 0-100, assigned at random by the environment)")
ax[0].set_ylabel("bush dwell over the episode's first 25 steps\n(% of those steps spent in a bush)")
ax[0].set_title("Response to a wound the agent did not earn", fontsize=9.5, loc="left", pad=8)
ax[0].legend(handles=h, loc="lower right", fontsize=7.8)

y = np.arange(len(arms))
v = np.array([slope[a] for a in arms])
ax[1].barh(y, v, color=[col[a] for a in arms], height=0.72, edgecolor="none")
ax[1].axvline(0, color=PL.INK, lw=1)
ax[1].set_yticks(y); ax[1].set_yticklabels(PL.arm_ylabels(arms), fontsize=8)
ax[1].set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
ax[1].set_xlabel("wound sensitivity  -  a DIFFERENCE, in percentage points\n"
                 "bush dwell in the heaviest wound quarter MINUS the lightest")
ax[1].grid(axis="y", visible=False)
for i, q in enumerate(v):
    ax[1].text(q + np.sign(q) * 0.06, y[i], f"{q:+.2f}", va="center",
               ha="left" if q >= 0 else "right", fontsize=7.6, color=PL.INK)
m = max(abs(v)) * 1.55
ax[1].set_xlim(-m, m)
POP = L.population()
_used = sum(int(np.asarray(D[a]["grids"]["dwt_early"], float).sum()) for a in arms)
L.record_samples("lad08_injury_dose_response", [
    dict(what="step rows in the 25-step window", used=_used, total=POP["steps"],
         note="the first 25 steps of every episode of every arm; episodes shorter than 25 steps "
              "contribute all the steps they have"),
    dict(what="episodes contributing", used=POP["episodes"], total=POP["episodes"],
         note="every episode has a randomised starting wound, so none is excluded")])

PL.assert_labels_fit(fig, ax)
PL.finish(fig, f"{L.FIG_ROOT}/lad08_injury_dose_response.png")
print(f"{'arm':22}" + "".join(f"{n:>10}" for n in L.INJ_NAMES) + f"{'slope':>10}")
for a in arms:
    print(f"{a:22}" + "".join(f"{q:>10.2f}" for q in curves[a]) + f"{slope[a]:>+10.2f}")
