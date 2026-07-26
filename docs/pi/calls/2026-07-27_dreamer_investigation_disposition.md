---
title: "PI call — what to do after the 5-perspective Dreamer investigation"
date: 2026-07-27
session: 2026-07-27_dreamer_investigation_disposition
caller: pi
status: pending-decision
trigger: "Post-analysis PI consultation after the five-way parallel Dreamer investigation (H1 speed, H2a faithfulness, H2b curves, H2c settings regime, H3 task difficulty) closed with a synthesis."
inputs:
  - docs/experiments/active/dreamer_srl_investigation/SYNTHESIS_20260727.md
  - docs/develop/active/diagnosis/dreamer_srl_h1_speed_investigation.md
  - docs/reviews/dreamer_srl_faithfulness_review.md
  - docs/experiments/active/dreamer_srl_investigation/DREAMER_SRL_INVESTIGATION.md
  - docs/project/critiques/dreamer_srl_settings_regime_critique.md
  - docs/project/critiques/gridworld_vs_dreamerv3_benchmarks_difficulty.md
  - docs/project/project_plan.md
---

# PI call — what to do after the 5-perspective Dreamer investigation

## §1 Plain-English entry point

**The question.** Five parallel investigations just finished asking why our in-house
model-based agent (DreamerV3, the "dreamer_srl" stack) looks weak on our 10×10 survival
grid world. They came back with a clean bill of health on the code and a list of things
we *could* change. This call decides **what we actually spend the next few days on** —
and, more importantly, whether "make Dreamer fast and good" is even the right thing for
the project to be spending attention on right now.

**What the investigation found, in one paragraph.** The Dreamer implementation is
faithful to the published algorithm — no bugs. Per unit of experience it actually
*learns better* than our workhorse agent (recurrent PPO, "rPPO"): 2.4× more survival
steps at the same number of episodes. Its problem is purely wall-clock — it is ~100×
slower in real time, of which roughly half is recoverable cheaply (it currently blocks
training while rendering videos, and its action-selection path isn't compiled). Two
sizes were being compared and the small one ("XS") matches the big one ("M"), so we are
paying 6.5× the parameters for nothing. One genuinely new suspect surfaced: the part of
Dreamer that predicts future value ("the critic") represents value as a histogram over
255 buckets, and *the entire survival improvement we care about spans about one
bucket* — the value signal may literally be below the resolution of the instrument
measuring it. Checking that costs one histogram from a checkpoint that already exists.

**The uncomfortable finding that isn't about Dreamer.** Buried in the difficulty
analysis: our rPPO baseline uses a discount factor of 0.95, which means a death 40 steps
in the future is worth 0.13 in the agent's own objective. **The workhorse agent that both
planned papers rest on may not be optimizing survival in any meaningful sense — it is
optimizing something like a 20-step-ahead surrogate**, while our headline metric is
survival steps. I verified this is not hypothetical: the eight rPPO runs launched
yesterday have `gamma: 0.95` recorded in their own saved configs (see §5). Whatever we
decide about Dreamer, that question sits underneath **Paper 1 and Paper 2 both**, and it
is cheap to answer.

**My recommendation in one line.** **Do not touch Dreamer training yet.** Spend the next
day on the two near-free probes that protect the papers — the discount-factor sweep on
rPPO and the critic-histogram read from an existing Dreamer checkpoint — and let the
in-flight 2×2 size study finish as designed. Confidence: high (≈85%) that retuning
Dreamer *now* is premature; moderate-high (≈75%) that the discount question outranks
everything else on this list.

## §2 Portfolio framing — where Dreamer actually sits

Per [`project_plan.md`](../../project/project_plan.md), the project is building two
papers. Paper 1 (Nature Machine Intelligence) claims a neuromodulation-inspired
modulator layer produces pain-like behavior across four behavioral categories. Paper 2
(NeurIPS) claims the same modulator architecture is one primitive unifying perceptual
modulation, hyperparameter modulation, and continual learning. **Neither paper's headline
claim names a model-based agent.** Dreamer is a second agent stack; its role is
supporting evidence ("the effect isn't an artifact of one learning algorithm"), not
load-bearing structure.

That places the whole Dreamer line in the **explore buffer**, not on a publication
track — and the explore buffer is supposed to be ≤20% of effort. Right now Dreamer is
consuming all four of the lab's best GPUs (node 114's RTX 6000 Ada cards) plus a
five-agent investigation. That is above budget for an explore-buffer thread, and it is
the strongest argument for *not* immediately opening a second round of Dreamer work.

The counter-argument, which is real: the "different in kind" analysis says our task's
achievable ceiling has **never been measured**. That is not a Dreamer question — it is a
question about our environment that a reviewer of *either* paper can ask ("how good is
good?"), and Dreamer's investigation is what surfaced it.

**Capacity note (not a constraint here).** Nodes 106, 111, and 112 are fully free (six
RTX 3090s) as of this call. GPU-days are *not* the scarce resource; analyst attention
and decision serialization are. Any option below that costs "≈4 short rPPO runs" costs
essentially nothing in cluster terms.

## §3 The three axes

**Focus vs. explore.** Currently over-exploring on the Dreamer axis and under-defending
the rPPO axis. The publication tracks rest on rPPO; the investigation just put a crack in
rPPO's objective. Correct move is to spend explore-budget on *de-risking the track*, not
on deepening the buffer thread.

**Topic / algorithm / environment / experiment.** *Algorithm* investment in Dreamer is
plateauing as a paper input (clean implementation, known fixes, no new claim).
*Environment* is the leading indicator now — the unmeasured achievable ceiling and the
discount-vs-survival mismatch are both environment-objective questions, and they should
lead. *Experiment* discipline is the real gap: we have been relaunching ladders faster
than we have been pre-registering what would falsify them.

**Pace.** The discount question is **overdue** — eight runs are burning right now under
an objective we have not defended. The Dreamer retune is **premature** — its two cheapest
diagnostics are unrun. Speed engineering is **on time but low priority** (it multiplies
future runs, but we do not yet know which future runs).

## §4 Options considered

Costs below are calendar-days-of-attention, not GPU-days (GPUs are free; see §2).

### Option 1 — Defend the baseline first *(Recommended)*
Run the rPPO discount sweep (γ ∈ {0.95, 0.99, 0.997}, ~4 runs × ~1.2 h) plus the free
critic-bin histogram from the existing Dreamer checkpoint. Let the 2×2 size study finish
untouched.
- **Costs:** ~1 day of attention; ~4 short rPPO runs on already-free 3090s. Delays every
  Dreamer change by a day.
- **Buys:** protects both papers' baseline from a "your baseline wasn't optimizing your
  metric" review; kills the cheapest Dreamer unknown at zero training cost; preserves the
  size-study data. If γ=0.99+ materially raises survival, the *entire* rPPO baseline set
  must be re-run — better to learn that today than after the comparison figures exist.
- **Risk of tunneling:** low. This is the option that most reduces the chance of a
  late-stage invalidation.

### Option 2 — Measure the ceiling (probe-first, full)
Option 1's two probes **plus** the oracle-observation ceiling probe (give an rPPO agent
privileged state; 4-cell ladder, ~4 runs × ~1.2 h) — requires a small additive change to
the environment's observation assembly in `src/`.
- **Costs:** ~2–4 days (needs a senior-developer plan, a developer change, an
  env-config-auditor pass, then designer + runner). Touches `src/` mid-flight while eight
  rPPO runs are live.
- **Buys:** an actual number for "how well can anything do on this task", which is a
  reviewer-facing asset for both papers and settles the open "is our task hard, or just
  noisy?" dispute between two of the five investigators.
- **Risk:** the additive-observation plumbing is the kind of change that quietly shifts
  the observation fingerprint and invalidates comparability if done carelessly.

### Option 3 — Retune and relaunch Dreamer now
Kill the 2×2, relaunch at the prescribed operating point: small model, replay ratio
0.25, critic bins narrowed ±20 → ±6, with the stall fixes.
- **Costs:** destroys the in-flight size-study data (~1 day already invested); bets on
  three untested prescriptions at once, so a null result is uninterpretable; the bin
  change requires a deviation-log entry against the published algorithm.
- **Buys:** the fastest route to a Dreamer baseline we would be willing to print — *if*
  the prescriptions are right.
- **Risk of tunneling:** highest of the four. It commits the best GPUs to the
  explore-buffer thread for another multi-day cycle before either cheap diagnostic has
  been read.

### Option 4 — Speed engineering first
Async video render, compile the action-selection path, batched environment reset, then
256 environments. Applies to future runs only; live runs keep their current throughput.
- **Costs:** ~2–3 developer-days plus review; produces no publishable claim by itself.
- **Buys:** ~2× (plausibly ~3× with the environment-count bump) on every Dreamer run
  afterward — real leverage, but only once we know which runs we want.
- **Risk:** classic infrastructure-without-a-deliverable. Correct *after* we decide
  Dreamer stays in the portfolio; premature before.

**Interactions to note.** Option 3 is the only one that destroys data. The critic
histogram in Options 1 and 2 needs a checkpoint that already exists — zero training cost,
so it should ride along with whatever is chosen. Option 4's changes must be scoped to
*future* runs only; applying them to live processes would change throughput mid-study and
break the size comparison. Options 1 and 4 are compatible in parallel (different agents,
no shared resource); Options 2 and 3 both serialize on `src/`/config work.

## §5 Evidence checked directly for this call

- Live rPPO ladder discount factor, read from a **saved run config** (not a source
  re-read): `results/JAX_RecurrentPPO/20260726-051209_rppo_b03_mc_dp1_n110/models/config.yaml:367`
  → `gamma: 0.95`. The synthesis' caveat ("basic-config rPPO runs inherit γ from their
  own config — verify per run") resolves **against** us: the live runs are at 0.95.
- Cluster capacity at call time: nodes 106 / 111 / 112 fully free (6 × RTX 3090); node
  114 fully occupied by the four Dreamer runs; 107 / 108 / 110 / 113 running the eight
  rPPO baselines.
- Prescriptions and probe costs as recorded in
  [`dreamer_srl_settings_regime_critique.md`](../../project/critiques/dreamer_srl_settings_regime_critique.md)
  §8 (operating-point table) and
  [`gridworld_vs_dreamerv3_benchmarks_difficulty.md`](../../project/critiques/gridworld_vs_dreamerv3_benchmarks_difficulty.md)
  §8 (probes P1–P7).
- Speed candidate ranking and effort/risk from
  [`dreamer_srl_h1_speed_investigation.md`](../../develop/active/diagnosis/dreamer_srl_h1_speed_investigation.md)
  §3.

## §6 Anchors

- **Publication tracks.** Portfolio tracks remain formally unratified
  ([`PORTFOLIO.md`](../PORTFOLIO.md) follow-up queue). Working anchors are Paper 1 (NMI)
  and Paper 2 (NeurIPS) from `project_plan.md` §4–§5. Dreamer supports neither headline
  claim → explore buffer.
- **Gates.** `project_plan.md` is a direction context and declares no phase gates (§7);
  this call therefore anchors on the two papers' claims rather than on G1/G2.
- **Hypotheses.** No H1–H5 modulator hypothesis is directly strengthened or weakened by
  any option here. Options 1 and 2 *protect* all of them by defending the baseline the
  modulator is compared against; Option 3 sidesteps them.
- **Consistency with prior calls.** The 2026-05-14 D-013 call already ruled that the
  Dreamer stack should take the *cheap, config-correcting* path over the ambitious one
  (picked "XS", rejected multi-GPU). Option 1 is the consistent continuation; Option 3
  would be the first reversal of that stance.

## §7 User decision

_Pending._

## §8 Rationale captured

_Pending._

## §9 Hand-off

_Pending user decision._ Provisional routing per option:

- **Option 1** → `experiment-designer` (pre-register the discount sweep: survival-steps
  vs. environment-steps as the verdict axis, ≥3 seeds, γ the only varied key) +
  `experiment-analyzer` (critic-bin histogram from the existing checkpoint).
  **Stop rule:** if γ=0.997 raises survival by more than the seed-noise floor, escalate
  back to PI before any further rPPO baseline is launched or cited.
- **Option 2** → `senior-developer` (additive oracle-observation block plan) then the
  Option 1 chain.
- **Option 3** → `experiment-designer` (single-change-at-a-time relaunch grid) +
  `training-runner`. **Stop rule:** if the retuned run does not beat the current 2×2 on
  survival-per-wall-clock-hour within 24 h, stop and return to probes.
- **Option 4** → `senior-developer` (speed plan, future-runs-only scope) → `developer` →
  `code-reviewer`.
