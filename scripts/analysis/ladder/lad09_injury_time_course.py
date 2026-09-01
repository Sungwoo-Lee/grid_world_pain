"""FIGURE 9 - What the agent FEELS, not what its body is, and what that costs.

QUESTION. Everywhere else this report talks about "the wound" it means the injury level: a number
in the body. But the agent has no sensor for that number - `injury_observable` is false in all
fourteen arms. What it actually receives is one scalar from the interoceptive nociceptor: the last
twelve injury levels convolved with a normalised alpha kernel (tau=3), with the current step's slot
weighted ZERO so nothing leaks in instantaneously, and with the whole buffer ZEROED at reset.

The consequence is not a detail. An agent handed an injury of 100 at reset feels **exactly nothing**
at t=0. It feels 9% of it by step 2, 36% by step 4, and does not feel the whole wound until about
step 12 - by which time the wound itself is already healing. So the thing that drives behaviour and
the thing the earlier version of this figure plotted are two different signals with two different
time courses, and binning behaviour by the injury level bins it by a quantity the agent cannot sense.

WHAT THE PANELS SHOW.

  A - the body against the feeling. Injury level and perceived nociception, for the agents that
      woke up in the lightest and heaviest quarters. The perceived curve starts at zero and climbs
      while the injury curve is already falling.

  B - the timing, which is the argument. The three quantities as fractions of their own peak, so
      their shapes can be compared: the injury gap peaks at step 0 (it is largest the instant it is
      assigned), the perceived gap peaks around step 12 (where the kernel saturates), and the extra
      hiding peaks around step 15. Behaviour follows the FEELING, three steps behind it, not the
      body state it is a delayed trace of.

  C - the perceptual dose-response. Bush dwell against how strongly the agent is feeling hurt right
      now, pooled over every step of every episode. This is an ASSOCIATIONAL panel and is marked as
      such: an agent feels hurt because it got hurt, which depends on what it was doing. Panels A
      and B carry the causal claim, because the starting wound they are keyed to was assigned at
      random before the agent acted.

  D - the bill. The nutrition the extra hiding cost, which is what closes the loop back to survival.

HOW IT IS COMPUTED. The perceived signal is reconstructed from each episode's recorded injury
sequence exactly as the environment builds it - `sum(buffer * kernel)` with buffer slot j holding
the injury j steps earlier, and a strict `src > estart` guard so the reset row, which is never
written into the buffer, cannot leak in. Panels A, B and D are keyed to the RANDOMISED starting
wound; panel C is keyed to the contemporaneous perceived signal.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

T = L.load_time_course()
D = L.load_all(); arms = [a for a in L.ARM_ORDER if a in T]
GRP = {a: L.resolves_identity(D[a]["sensory"]) for a in arms}
REF = "V4_blur05" if "V4_blur05" in arms else arms[0]
XMAX = 100
x = np.arange(XMAX)
gap = lambda a, key, s=1.0: s * (np.asarray(T[a][key], float)[3, :XMAX]
                                 - np.asarray(T[a][key], float)[0, :XMAX])

fig, ax = plt.subplots(1, 4, figsize=(19.6, 5.0))

# --- A: the body against the feeling -----------------------------------------------------------
inj = np.asarray(T[REF]["injury"], float); noc = np.asarray(T[REF]["noci"], float)
ax[0].plot(x, inj[3, :XMAX], lw=2.2, color=PL.THREAT, label="injury level, woke up at 75-100")
ax[0].plot(x, noc[3, :XMAX], lw=2.2, color=PL.THREAT, ls="--", label="what it FEELS, same agents")
ax[0].plot(x, inj[0, :XMAX], lw=1.6, color=PL.HARMLESS, label="injury level, woke up at 0-25")
ax[0].plot(x, noc[0, :XMAX], lw=1.6, color=PL.HARMLESS, ls="--", label="what it FEELS, same agents")
ax[0].annotate("feels nothing at all for the first two steps,\nand not the whole wound until step 12",
               xy=(2, 1), xycoords="data", xytext=(0.34, 0.44), textcoords="axes fraction",
               fontsize=7.4, color=PL.MUTED, ha="left",
               arrowprops=dict(arrowstyle="->", lw=0.7, color=PL.MUTED,
                               connectionstyle="arc3,rad=0.25"))
ax[0].set_ylim(0, 100)
ax[0].set_title("A.  The body, and the feeling\nsolid = injury level, dashed = perceived",
                fontsize=9.4, loc="left", pad=8)
ax[0].set_ylabel("injury level, and the perceived signal\non the same 0-100 scale")
ax[0].legend(fontsize=7.0, loc="upper right")

# --- B: the timing -----------------------------------------------------------------------------
ig, ng, bg = gap(REF, "injury"), gap(REF, "noci"), gap(REF, "bush", 100.0)
for k, (v, c, ls, lab) in enumerate(((ig, PL.MUTED, "-", "injury gap (the body)"),
                                     (ng, PL.THREAT, "-", "perceived gap (what it feels)"),
                                     (bg, PL.ACCENT, "-", "extra hiding (what it does)"))):
    ax[1].plot(x, v / np.abs(v).max(), lw=2.2, color=c, ls=ls, label=lab)
    pk = int(np.argmax(v))
    ax[1].axvline(pk, color=c, lw=0.8, ls=":")
    # the three peaks sit within 15 steps of each other, so the labels are stacked, not inlined
    ax[1].annotate(f"peak at step {pk}", (pk, 1.34 - 0.11 * k), fontsize=7.4, color=c, ha="left",
                   xytext=(4, 0), textcoords="offset points")
ax[1].axhline(0, color=PL.INK, lw=1)
ax[1].set_ylim(-0.75, 1.42)
ax[1].set_title("B.  Behaviour follows the FEELING, not the body\n"
                "each curve as a fraction of its own peak", fontsize=9.4, loc="left", pad=8)
ax[1].set_ylabel("fraction of that curve's own maximum\n(shape only - the three have different units)")
ax[1].legend(fontsize=7.4, loc="lower right")

# --- C: the perceptual dose-response -----------------------------------------------------------
dose = {a: 100 * np.asarray(T[a]["dose_bush"], float)
             / np.maximum(np.asarray(T[a]["dose_n"], float), 1) for a in arms}
h = PL.group_lines(ax[2], np.arange(4), dose, GRP, spotlight=(REF, "A_baseline"), label_end=True)
ax[2].set_xticks(np.arange(4)); ax[2].set_xticklabels(L.INJ_NAMES, fontsize=8)
ax[2].set_xlim(-0.25, 3.95)
_dmax = max(v.max() for v in dose.values()); _dmin = min(v.min() for v in dose.values())
ax[2].set_ylim(_dmin - 3, _dmax + 9)
ax[2].set_title("C.  How hard it hides by how hurt it FEELS\n"
                "associational - a hurt agent got hurt somehow", fontsize=9.4, loc="left", pad=8)
ax[2].set_xlabel("perceived nociception right now  (0-100)")
ax[2].set_ylabel("bush dwell  (% of those steps spent in a bush)")
ax[2].legend(handles=h, fontsize=7.0, loc="upper left", framealpha=0.92, frameon=True)

# --- D: the cost -------------------------------------------------------------------------------
for a in arms:
    ax[3].plot(x, gap(a, "nutrition"), lw=0.9,
               color=PL.GROUP_YES if GRP[a] else PL.GROUP_NO, alpha=0.30, zorder=1)
for want, c in ((True, PL.GROUP_YES), (False, PL.GROUP_NO)):
    sel = [a for a in arms if GRP[a] == want]
    ax[3].plot(x, np.mean([gap(a, "nutrition") for a in sel], axis=0), lw=2.5, color=c, zorder=3,
               label=L.GROUP_LABEL[want].split("  (")[0] + f"  ({len(sel)} arms)")
ax[3].axhline(0, color=PL.INK, lw=1)
ax[3].set_title("D.  And the bill\nthe hiding was paid for in food", fontsize=9.4, loc="left", pad=8)
ax[3].set_ylabel("nutrition gap  -  a DIFFERENCE, 0-100 scale\n"
                 "heaviest starting wound MINUS lightest")
ax[3].legend(fontsize=7.4, loc="lower right")

for a_ in ax[:2]: a_.set_xlabel("step within the episode")
ax[3].set_xlabel("step within the episode")
for a_ in (ax[0], ax[1], ax[3]): a_.set_xlim(0, XMAX)
P = L.population()
_tc = sum(int(np.asarray(T[a]["n"], float).sum()) for a in arms)
_dose = sum(int(np.asarray(T[a]["dose_n"], float).sum()) for a in arms)
L.record_samples("lad09_injury_time_course", [
    dict(what="step rows in panels A, B and D", used=_tc, total=P["steps"],
         note="steps 0-119 of every episode; later steps are outside the window these panels plot"),
    dict(what="step rows in panel C", used=_dose, total=P["steps"],
         note="every step of every episode - panel C is not restricted to a window, which is "
              "why it is the only panel drawing on the entire population"),
    dict(what="episodes contributing", used=P["episodes"], total=P["episodes"], note="")])

PL.finish(fig, f"{L.FIG_ROOT}/lad09_injury_time_course.png")

print(f"reference arm {REF}\n")
print(f"{'arm':22}{'injury pk':>11}{'felt pk':>9}{'hiding pk':>11}{'r(felt)':>9}{'r(injury)':>11}"
      f"{'dose 0-25':>11}{'dose 75-100':>13}")
for a in arms:
    ig_, ng_, bg_ = gap(a, "injury"), gap(a, "noci"), gap(a, "bush", 100.0)
    d = dose[a]
    print(f"{a:22}{int(np.argmax(ig_)):>11}{int(np.argmax(ng_)):>9}{int(np.argmax(bg_)):>11}"
          f"{np.corrcoef(ng_, bg_)[0,1]:>+9.2f}{np.corrcoef(ig_, bg_)[0,1]:>+11.2f}"
          f"{d[0]:>10.1f}%{d[3]:>12.1f}%")
