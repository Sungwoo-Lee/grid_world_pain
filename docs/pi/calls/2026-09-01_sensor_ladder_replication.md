---
title: "PI call — do we replicate the sensor ladder at multiple seeds, and when?"
date: 2026-09-01
session: 2026-09-01_sensor_ladder_replication
caller: pi
status: pending-decision
trigger: "Post-analysis PI consultation. The fourteen-arm sensor-ladder study is finished, published as an artifact, and as of today fully reproducible. Its own Limitation 1 names seed replication as the single change that would move it from exploratory to confirmatory. The cluster is idle."
inputs:
  - docs/experiments/active/sensor_ladder/sensor_ladder.md
  - docs/reviews/plan_sensor_ladder.md
  - docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md
  - docs/pi/calls/2026-07-27_dreamer_investigation_disposition.md
  - docs/project/project_plan.md
  - configs/models/recurrent_ppo/recurrent_ppo.yaml
  - scripts/lab/launch_ladder_arm.sh
---

# PI call — do we replicate the sensor ladder at multiple seeds, and when?

## §1 Plain-English entry point

**The question.** We just finished a study that trained fourteen agents in the same
10x10 survival world, each differing only in what it can sense, and measured each one
across a million test episodes. The study is good, it is honest, and it says plainly
in its own limitations that **every agent was trained from one and the same random
starting point** — so a claim like "thirteen of the fourteen agents did X" cannot be
read as thirteen independent confirmations. The study names the fix: train the whole
ladder again from two or three *different* random starting points. This call decides
whether we spend the lab on that now, spend a fraction of it on a cheaper version, or
hold.

**Why this is not simply "the cluster is free, so do it."** Three things I found while
reading around the request change the arithmetic:

1. **The trainer is about to change underneath us.** A parallel session has already
   drafted and plan-reviewed two fixes to the same reinforcement-learning trainer these
   fourteen agents were trained with, to land *before* the bigger modulation refactor.
   One of them — a units mismatch in how the agent bootstraps its value estimate at the
   edge of a training window — was measured this week and affects roughly **35% of the
   training targets in every run of this kind we have ever trained**, the fourteen ladder
   arms included. Twenty-eight new runs launched today would be twenty-eight more agents
   trained on a trainer the project has already decided is wrong.
2. **An open question from the last PI call sits directly underneath this one.** The
   2026-07-27 call is still formally undecided, and it flagged that our baseline agent
   discounts the future at 0.95 — meaning a death forty steps ahead is worth about 0.13
   in the agent's own objective, while our headline metric is survival steps. Every
   ladder arm carries that same 0.95. If that setting turns out to be wrong, a
   three-seed ladder is forty-two runs of a baseline that was not optimising the thing
   we score it on.
3. **The ladder is not a side study — it is Paper 1's control arm.** The project plan
   defines each of its four pain-like behavioural categories as a signature that
   *exceeds what a pure-nociception agent produces*. The ladder agents have their
   cognitive/neuromodulatory layer switched off entirely. So the ladder **is** the
   pure-nociception agent, and its findings on hypervigilance, recovery time-course and
   hunger-versus-injury conflict are literally the floor the modulator has to clear.
   That raises the stakes of getting it right, and it also means the definitive version
   of this ladder has to be trained on the *same* trainer and the *same* discount as the
   modulator arms — which do not exist yet.

**One correction to the framing I was handed.** The request said a second seed "does not
fix the shared-world/shared-initialisation confound entirely, only the seed part of it."
The seed part is in fact the load-bearing part. Sharing the million test worlds across
arms is a **feature** — it makes the arm-to-arm comparisons paired, which is why the
behavioural measurements are so precise. The genuine defect is that each *agent* is one
draw, and that the nine same-input-width arms plausibly share their initial weights and
data ordering. Different training seeds fix both of those. What seeds do *not* fix is
whether an effect is a quirk of that particular million-world draw, and that is
separately and much more cheaply fixed by re-collecting against a different evaluation
seed base.

**My recommendation in one line.** **Do not commit the full replication today.** Run the
cheap seed-stability probe instead — a handful of arms, two extra seeds, a fraction of
the evaluation episodes — which answers "do the headline claims survive a seed change at
all" for roughly 5% of the cost and is informative regardless of what the trainer fixes
do. Confidence: high (~85%) that the full 900 GB replication today is premature;
moderate (~65%) that the cheap probe is worth its analyst-attention cost this week
rather than next.

## §2 Portfolio framing

**The tracks are still unratified.** `PORTFOLIO.md` has carried "first-track
ratification" in its follow-up queue since May and the 2026-07-27 call flagged it again.
Working anchors remain Paper 1 (Nature Machine Intelligence — a neuromodulation-inspired
modulator produces pain-like behaviour across four categories, and does not when the
layer is absent) and Paper 2 (NeurIPS — the same modulator architecture as one primitive
unifying three ML subfields), both from `project_plan.md` §4–§5.

**Where the ladder sits against those.** It splits cleanly in two, and the two halves
have very different paper value:

- **Findings 5, 6 and 7** — the agent responds to what it *feels* rather than to the
  wound it has; a weak hypervigilance effect on the ambiguous smell channel; hunger
  outweighing the wound, with hiding as a cost rather than a good — are measurements of
  **the modulator-free control agent**, on three of Paper 1's four categories
  (Recovery, Hypervigilance, Managing conflicting needs). These are **on Track A's
  critical path**. They are also mildly uncomfortable for Paper 1's stated contrast: the
  negative control is already showing weak versions of the signatures the modulator is
  supposed to introduce. That is not fatal — a graded floor the modulator must exceed is
  a *better* experiment than an all-or-nothing one — but it is a reframing the paper has
  not yet made, and it matters more than the seed count does.
- **Findings 1 through 4** — smell direction beats sharper sight; what matters about
  vision is identity, not sharpness; sharp sight underperforms slightly blurred sight;
  identity-resolving agents stop hiding from rabbits — are a **sensory-capability
  contribution belonging to no named track**. Promoting them to a paper of their own
  would open a third track, which the portfolio's healthy default explicitly flags.

**Capacity note.** Storage is not the constraint: the NAS has ~45 TB free, so 900 GB is
noise. GPU-hours are not the constraint either — the cluster is idle. The binding
resources are **analyst attention** and **decision serialisation**, exactly as the
2026-07-27 call concluded, plus a third that call did not have: **the trainer being in a
stable state**. That third one is what makes "the cluster is free right now" a much
weaker argument than it looks.

## §3 The three axes

**Focus vs. explore.** Healthy at the moment, and tilting the right way — the parallel
session is on Track A (the modulation-site refactor) and Dreamer has correctly gone
quiet. The risk here is not over-exploring; it is **exploiting a result before deciding
what the result is for**. Committing forty-two runs and a second 900 GB collection to
make a study confirmatory, before deciding whether its findings are Paper 1 claims or
design-rationale appendix, is effort spent ahead of the decision that governs it.

**Topic / algorithm / environment / experiment.** *Algorithm* is the leading indicator
now and everything else should follow it: two measured bugs and a four-site refactor are
in flight in the rPPO trainer, and until they land, "what our baseline agent is" is not
a settled object. *Experiment* discipline remains the standing gap the last call named —
we are still producing ladders faster than we pre-register what would falsify them, and
this study says so about itself in its own first paragraph. *Environment* and *topic*
investment are both paying off; the ladder is a genuinely strong piece of work.

**Pace.** The full replication is **premature** — two of its three prerequisites (fixed
trainer, defended discount) are unresolved and one is days away. The cheap seed-stability
probe is **on time**. The ladder's paper role, and the portfolio ratification behind it,
are **overdue** — the role question has been implicit since the study was designed and
the track question since May.

## §4 Options considered

Costs below separate **GPU-hours** (abundant) from **analyst attention** (scarce) and
**re-do risk** (the probability the work is superseded).

### Option 1 — Full replication now: 2 extra seeds, all 14 arms, full 1 M-episode collection
Add a seed argument to `scripts/lab/launch_ladder_arm.sh`, launch 28 runs across the idle
cluster (~11 h wall clock for both seeds in parallel), then repeat the two-pass collection
at 1,000,000 episodes per arm for both new seeds and extend the arm-keyed figure scripts
to a seed dimension.
- **Costs:** ~28 GPU-runs (free). ~900 GB and a repeat of the multi-day, four-node
  collection effort. Unscoped figure-script work to add a seed axis. **High re-do risk:**
  trains 28 agents on a trainer with a measured bug whose fix is already drafted, at a
  discount factor the project has an open question about.
- **Buys:** converts the study to confirmatory *as it currently stands*, immediately, and
  uses a perishable idle window.
- **Honest read:** this is the option most likely to be done twice.

### Option 2 — Train the seeds now, defer the collection decision
Same 28 runs today; checkpoints only. Decide later — once the ladder's paper role is
settled — which arms to collect and at what episode count.
- **Costs:** ~11 h of otherwise-idle cluster, the launcher seed argument, small checkpoint
  storage. Zero collection cost today. **Same re-do risk as Option 1** on the training half.
- **Buys:** buys the option cheaply; if the trainer fixes invalidate the runs you lose
  eleven idle-GPU hours rather than a week of collection.
- **Honest read:** the option value is partly illusory, because the invalidating event
  (the value-bootstrap fix) is not a risk — it is *planned*, and imminent.

### Option 3 — Cheap seed-stability probe *(Recommended)*
Do not replicate the ladder. Instead take the small set of arms the headline claims
actually rest on — the omnidirectional-versus-directional smell pair behind finding 1,
the presence-channel arms behind finding 2, the sharp-versus-slightly-blurred pair behind
finding 3, and two or three arms spanning the identity split behind finding 4 — train
them at two additional seeds, and evaluate at roughly 50,000–100,000 episodes rather than
a million.
- **Costs:** ~10 runs, ~11 h (free). Tens of gigabytes, not hundreds. Modest analysis
  work — a seed axis on a handful of measures, not on the whole figure suite. Still
  carries re-do risk, but on ~5% of the stake.
- **Buys:** the single decision-relevant fact — **do the headline effects survive a
  seed change at all**. A headline that flips under a new seed is unlikely to be rescued
  by the trainer fixes, so a negative here is durable and would change the paper this
  week. A positive makes the eventual full replication worth doing properly.
- **Why the reduced episode count is defensible:** the million episodes bought precision
  on small per-step effects like the two-to-four-step lag between feeling and hiding. A
  seed-stability check asks whether a ~45-step survival gap and a group split with no
  overlap reappear — a far coarser question, needing far less data.
- **Risk of tunneling:** low. It costs little and forecloses nothing.

### Option 4 — Hold entirely; replicate once, later, on a settled trainer
Spend nothing on the ladder now. Wait for the two pre-refactor trainer fixes to land and
for the discount question from the 2026-07-27 call to be decided, then re-train the full
ladder at three seeds on fixed code — 42 runs, not 28, since seed 42 would also need
re-training — as the modulator-free control arm for Paper 1's modulator experiment,
sharing that experiment's trainer, discount and seed set.
- **Costs:** delays confirmatory status by days to weeks; spends the current idle window
  on nothing; costs more GPU eventually (which is not the scarce resource).
- **Buys:** **one** replication instead of two. A ladder that is simultaneously
  confirmatory, free of two known trainer bugs, and directly comparable to the modulator
  arms it exists to be a control for. Forces the discount decision that has been open
  since July.
- **Consistency:** this is the direct continuation of the 2026-07-27 call's own
  recommendation to defend the baseline before spending on anything built atop it.
- **When this dominates Option 3:** if analyst attention is fully committed to the
  refactor this week. The probe's cost is attention, not GPUs, and attention is what is
  scarce.

### Option 5 — Leave it exploratory, publish as-is, spend the window elsewhere
Accept Limitation 1 standing, soften every arm-count sentence, and move on.
- **Costs:** no ladder finding can be a claim in Paper 1. Given that findings 5–7 are the
  pure-nociception floor for three of Paper 1's four categories, this is a real cost, not
  a bookkeeping one.
- **Buys:** all attention on the modulator track.
- **Honest read:** viable only under the "design-rationale appendix" answer to §5's role
  question. Note that the natural alternative use of the idle window — the modulator
  experiment — **cannot consume it today**, because that experiment is not yet designed
  (seed count, environment and node allocation are all still open per today's diary).

## §5 The second question, which governs the first

**What is the sensor ladder's role in the papers?** This determines whether replication
is a must-do or a nice-to-have, and the request explicitly asked for my judgement on it.

- **(a) Paper 1's control arm.** Findings 5–7 enter Paper 1 as the pure-nociception floor
  the modulator must exceed. → Replication is **mandatory**, and must be on the post-fix
  trainer at the final discount, alongside the modulator arms. Options 3-then-4.
- **(b) Design-rationale appendix.** The ladder explains why the modulator experiments
  use the sensory configuration they use; cited as exploratory throughout. → Replication
  is **optional**. Option 5, or Option 3 as cheap insurance.
- **(c) Its own paper.** Findings 1–4 are a standalone sensory-capability contribution.
  → Replication is **mandatory**, and the PI flags this as **opening a third track**
  above the portfolio's stated explore budget. It should be an explicit decision, not a
  drift.
- **(d) Defer.** Keep it exploratory and revisit once the modulator experiment reveals
  whether the ladder baseline is load-bearing.

**My read:** (a) is already true whether or not we ratify it. The project plan defines all
four pain-like categories relative to what a pure-nociception agent produces, and the
ladder is the only measurement of that agent we have. **Exploratory-and-honestly-labelled
is not sufficient** for that role — a Nature Machine Intelligence reviewer will not accept
a single shared-seed run per configuration as the control a headline contrast rests on.
It *is* sufficient for the sensory findings if they stay an appendix.

## §6 Two things to resolve that are not this decision

- **An apparent contradiction about hiding.** The ladder reports extra hiding peaking
  fourteen to sixteen steps after injury in thirteen of fourteen arms (finding 5). Today's
  diary from the parallel session records "the target behaviour (hide more when injured)
  is currently the OPPOSITE of what the agent does, and two interventions already failed
  to reverse it." These are probably different measures over different windows — finding 7
  separately reports that arms which hide *more overall* die sooner — but if the project
  believes both sentences simultaneously, one of them is going into a paper wrong. Worth
  an hour from `experiment-analyzer` before either is cited.
- **Portfolio ratification.** Still unratified since May, flagged by two prior calls.
  §5's role question cannot be answered rigorously without it. This should be its own
  short PI call, and it is now genuinely overdue.

## §7 Anchors

- **Publication tracks.** Unratified (`PORTFOLIO.md` follow-up queue). Working anchors:
  Paper 1 (NMI) and Paper 2 (NeurIPS), `project_plan.md` §4–§5. The ladder is on Paper 1's
  path as the pure-nociception control (`project_plan.md` §3.1); its sensory findings are
  on no track.
- **Gates.** `project_plan.md` is a direction context and declares no phase gates (§7);
  this call anchors on the two papers' claims.
- **Hypotheses.** No modulator hypothesis is directly strengthened or weakened. Every
  option here bears on the *floor* those hypotheses are measured against — a weak floor
  measured on one seed is the single easiest way to lose a modulator result to a reviewer.
- **The four causes of the v8 null result.** Not directly addressed; the ladder is
  modulator-free by construction.
- **Consistency with prior calls.** The 2026-07-27 call ruled "defend the baseline before
  spending on what is built atop it" and is still formally undecided. Options 3 and 4 are
  the consistent continuation; Options 1 and 2 would spend on top of an
  acknowledged-unsound baseline and would be the first reversal of that stance.

## §8 User decision

_Pending._

## §9 Rationale captured

_Pending._

## §10 Hand-off

_Pending user decision._ Provisional routing per option:

- **Option 1 / 2** → `senior-developer` (add the seed argument to
  `scripts/lab/launch_ladder_arm.sh` and scope the seed-axis extension to the figure
  scripts) → `plan-reviewer` → `experiment-designer` (pre-register which claims the
  replication is testing and what would falsify them, *before* launch) →
  `training-runner`.
  **Stop rule:** if the value-bootstrap fix lands before collection begins, escalate back
  to PI before spending a single collection-hour on the pre-fix seeds.
- **Option 3** → `experiment-designer` (choose the minimal arm subset and episode count,
  and pre-register the per-finding seed-stability threshold — what magnitude of change
  counts as "the headline did not survive") → `plan-reviewer` → `training-runner`.
  **Stop rule:** if any headline finding reverses sign or loses its group separation under
  a new seed, stop and escalate to PI before the finding is cited anywhere else.
- **Option 4** → `senior-developer` (sequence the two pre-refactor fixes) and, in
  parallel, `experiment-designer` (design the modulator experiment with the ladder control
  arms folded in as conditions rather than as a separate study).
  **Stop rule:** escalate to PI when the trainer fixes land, to confirm the ladder is
  re-run as part of the modulator experiment rather than as a standalone repeat.
- **Option 5** → `experiment-analyzer` (soften every arm-count sentence in the report and
  the artifact to exploratory phrasing) → `artifact-format-reviewer` before republish.
- **Either way** → a short follow-on PI call to ratify `PORTFOLIO.md`, and
  `experiment-analyzer` on the hiding-direction contradiction in §6.
