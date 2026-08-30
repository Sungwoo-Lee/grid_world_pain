"""FIGURE 9 - How long the wound's effect lasts, and what the agent pays for it.

QUESTION. Figure 8 measures the wound's effect over the episode's first 25 steps. Why 25? Widen the
window and the effect shrinks: in the reference agent it runs +4.2 percentage points at 25 steps,
+0.8 at 50, and slightly negative over the whole episode. Taken at face value that looks like a
result chosen at a flattering window - so this figure asks what is actually happening in time.

WHAT IT SHOWS, in three steps.

  A - THE DOSE DISAPPEARS. The wound the environment assigned heals. The gap between the agents
      handed the heaviest wound and the lightest is 75 injury points at reset, HALF gone by step 17,
      and 90% gone by step 27. So the cause is essentially over before step 30.

  B - THE RESPONSE FOLLOWS IT. Extra hiding rises to a peak around step 12-15, then falls away on
      roughly the same schedule as the wound itself. An effect that tracks its cause through time is
      evidence FOR the causal reading, not against it - and it means the 25-step window is not a
      lucky pick but approximately the lifetime of the dose.

  C - AND THEN THE BILL ARRIVES. The extra hiding is paid for in food. An agent handed a heavy wound
      eats 4.4 items in its first 25 steps against 8.0 for one handed almost none, and by step 25 it
      is running about 18 nutrition points behind. Once the wound has healed it hides LESS than the
      unhurt agent - not noise, but the repayment of the foraging debt its early caution created.
      That is why the whole-episode number in Figure 14 is slightly negative.

So the honest statement of the finding is not "a wound makes the agent hide more" but "a wound
causes a transient increase in hiding that lasts about as long as the wound does, followed by a
compensatory decrease while the agent makes up the food it missed".

HOW IT IS COMPUTED. A separate step-by-step sweep (`build_time_course.py`) accumulates, for each
arm and each quarter of the assigned starting wound, the mean injury still carried, bush dwell,
and nutrition at every step up to 120. Panels B and C plot the heaviest quarter minus the lightest.
Episodes end at different times, so later steps average over the episodes still alive - a
survivorship effect that grows with the step number, which is why the panels stop at 100.
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

TC = f"{L.OUT_ROOT}/time_course.json"
if not os.path.exists(TC):
    raise SystemExit(f"{TC} missing - run scripts/analysis/ladder/build_time_course.py first")
T = json.load(open(TC))
D = L.load_all(); arms = [a for a in L.ARM_ORDER if a in T]
GRP = {a: L.resolves_identity(D[a]["sensory"]) for a in arms}
XMAX = 100
x = np.arange(XMAX)

def gap(arm, key, scale=1.0):
    m = np.asarray(T[arm][key], float)
    return scale * (m[3, :XMAX] - m[0, :XMAX])

fig, ax = plt.subplots(1, 3, figsize=(15.4, 5.2))

# A - the dose
ref = "V4_blur05" if "V4_blur05" in arms else arms[0]
inj = np.asarray(T[ref]["injury"], float)
for k in range(4):
    ax[0].plot(x, inj[k, :XMAX], lw=1.9, color=PL.LADDER_CMAP(0.12 + 0.26 * k),
               label=f"woke up at {L.INJ_NAMES[k]}")
g0 = inj[3, 0] - inj[0, 0]
half = int(np.argmax((inj[3, :XMAX] - inj[0, :XMAX]) < 0.5 * g0))
gone = int(np.argmax((inj[3, :XMAX] - inj[0, :XMAX]) < 0.1 * g0))
# the two markers sit only ten steps apart, so they are stacked rather than placed side by side
for s, lab, yy in ((half, f"half healed by step {half}", 70),
                   (gone, f"90% healed by step {gone}", 58)):
    ax[0].axvline(s, color=PL.MUTED, lw=0.9, ls=":")
    ax[0].annotate(lab, (s, yy), fontsize=7.4, color=PL.MUTED, ha="left",
                   xytext=(4, 0), textcoords="offset points")
ax[0].set_ylim(0, 100)
ax[0].set_title("A.  The dose disappears\nthe assigned wound heals", fontsize=9.6, loc="left", pad=8)
ax[0].set_ylabel("injury level the agent is still carrying  (0-100)")
ax[0].legend(fontsize=7.6, loc="upper right")

# B - the response, and C - the cost
for j, (key, scale, ttl, yl) in enumerate([
        ("bush", 100.0, "B.  The response follows it\nextra hiding fades as the wound does",
         "extra bush dwell  (percentage points)\nheaviest starting wound minus lightest"),
        ("nutrition", 1.0, "C.  And the bill arrives\nthe hiding was paid for in food",
         "nutrition gap  (0-100)\nheaviest starting wound minus lightest")], start=1):
    for a in arms:
        ax[j].plot(x, gap(a, key, scale), lw=0.9,
                   color=PL.GROUP_YES if GRP[a] else PL.GROUP_NO, alpha=0.30, zorder=1)
    for want, c in ((True, PL.GROUP_YES), (False, PL.GROUP_NO)):
        sel = [a for a in arms if GRP[a] == want]
        ax[j].plot(x, np.mean([gap(a, key, scale) for a in sel], axis=0), lw=2.6, color=c, zorder=3,
                   label=L.GROUP_LABEL[want].split("  (")[0] + f"  ({len(sel)} arms)")
    ax[j].axhline(0, color=PL.INK, lw=1)
    ax[j].axvline(gone, color=PL.MUTED, lw=0.9, ls=":")
    ax[j].annotate(f"wound 90% healed", (gone, ax[j].get_ylim()[1]), fontsize=7.4, color=PL.MUTED,
                   ha="left", va="top", xytext=(3, -2), textcoords="offset points")
    ax[j].set_title(ttl, fontsize=9.6, loc="left", pad=8)
    ax[j].set_ylabel(yl)
    ax[j].legend(fontsize=7.8, loc="lower right")
for a_ in ax:
    a_.set_xlabel("step within the episode")
    a_.set_xlim(0, XMAX)
PL.finish(fig, f"{L.FIG_ROOT}/lad09_injury_time_course.png")

print(f"reference arm {ref}: assigned wound gap {g0:.1f} points at reset, "
      f"half gone by step {half}, 90% gone by step {gone}\n")
print(f"{'arm':22}{'peak extra hiding':>19}{'at step':>9}{'by step 60':>12}{'worst food gap':>16}")
for a in arms:
    b = gap(a, "bush", 100.0); n = gap(a, "nutrition")
    print(f"{a:22}{b.max():>+19.2f}{int(np.argmax(b)):>9}{b[60]:>+12.2f}{n.min():>+16.1f}")
