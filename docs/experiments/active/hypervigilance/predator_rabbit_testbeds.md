---
title: "Held-out test environments (behavioural assays) for predator-vs-rabbit discrimination"
topic: hypervigilance
status: active
created: 2026-06-13
last_updated: 2026-06-13
phase: hypervigilance
wandb_tag: hypervigilance
develop_link: "[[20260612_1625_predator_rabbit_discrimination]]"
---

# Held-out test environments — does the agent recognise a predator, or just react to pain?

## Question / Purpose (plain-language entry point)

We have spent the whole June "predator vs rabbit" study trying to read one fact out of the
agent's everyday foraging behaviour: when a **harmful animal** (a "predator", whose touch
injures the agent) and a **harmless animal** (a "rabbit", whose touch does nothing) are made
identical on everything the agent can sense from a distance — same smell, same chase, same
approach speed, same strike-and-retreat — does the agent steer clear of the dangerous one
*before* it ever gets touched? Reading that out of free-roaming rollouts kept failing: the
apparent discrimination turned out to be riding on confounds (the predator chased harder, or
there were two rabbits and one predator, or the avoidance only showed up *after* contact, i.e.
it was a pain reflex, not danger recognition). Every clean control collapsed the signal toward
zero.

This memo proposes a different strategy, borrowed directly from the design of the project's
own EVAAA paper (Lee et al. 2025): **stop measuring discrimination in the messy training world,
and instead build small, purpose-built test worlds where the agent is physically forced to
reveal whether it discriminates.** Keep training naturalistic; make the *test environment itself
the measuring instrument* — a behavioural assay. In EVAAA's words, the testbeds "isolate
specific decision-making processes ... while minimizing potentially irrelevant extrinsic cues".
A two-resource choice task forces the agent to *choose*; a risk-taking task forces it to decide
whether to cross a hazard. We want the predator-vs-rabbit analog: a layout where a
**discriminating** agent and a **class-blind** agent (one that treats predator and rabbit the
same) produce *visibly different, unambiguous* behaviour — and where the read-out is legible
**before any contact happens**, so we can never again mistake a pain reflex for danger
recognition.

This document proposes **seven** such testbeds and ranks them. It is **design only** — no YAML
configs are written yet. Each testbed names the manipulation, the exact behaviour the
environment *forces* the agent to express, what a discriminating vs. a class-blind agent each
does, which confound it removes, whether it can run as a pure checkpoint probe or needs a
dedicated trained agent, and a rough config sketch.

> Builds on the June study summary
> ([[20260612_1625_predator_rabbit_discrimination]]) — read that for *why* the naturalistic-rollout
> approach kept producing false conclusions — and on the latest byte-identical training config,
> cell 08 ([[single_pred_rabbit_disengage]]), which most of these assays load as their trained agent.

## Design philosophy — what makes a valid assay here

Five hard constraints (the confounds that produced false conclusions before). Every testbed
below is scored against them in §3.

1. **Anticipation, not pain reflex.** The read-out must be legible *before* the agent is ever
   touched — ideally because contact is made *physically impossible* (a barrier or a track the
   agent cannot cross), so a post-contact pain reaction simply cannot contribute to the score.
2. **No motion / aggression confound.** Predator and rabbit must be matched on chase behaviour
   and approach speed; the only thing that can ever differ is whether contact injures.
3. **No out-of-distribution artifact (or flag it).** The June predator-only world was OOD (the
   agent never trained with zero rabbits). Where an assay puts the agent in an unfamiliar layout,
   that is flagged explicitly and, where possible, the *contrast is within-assay* (predator-side
   vs rabbit-side of the *same* world) so the OOD-ness cancels.
4. **No number artifact.** Never compare N rabbits to M predators with N≠M. Forced-choice assays
   use exactly one predator and one rabbit; single-lane assays present one animal per episode but
   balance predator-episodes and rabbit-episodes 50/50.
5. **Forced read-out.** The measured quantity must be a behaviour the world *forces* the agent to
   emit — a choice, a detour, an approach/avoid, a latency, a cover-dive — not a hidden statistic
   averaged out of a free rollout. (The hidden-statistic failure is the whole methodological
   lesson of the June study.)

### A shared trained agent vs. dedicated training

Most of these are **checkpoint probes**: load the cell-08 trained agent (byte-identical
single-predator/single-rabbit, strike-and-retreat, lethal contact) and evaluate it in the new
layout. A probe answers "does the agent we *already have* discriminate?" Two assays
(T6 graded-lethality, T7 generalization-by-smell-swap) are most informative with a **dedicated
trained agent**, because they ask whether discrimination *can be learned* under a richer signal,
not just whether the current agent shows it — those are flagged.

A probe's one caveat is distribution shift: a barrier maze is not the open grid the agent
trained on. We handle this three ways: (a) keep the floor, food, bushes, smell, and animal
dynamics identical to training so only the *task geometry* changes; (b) make the discriminating
vs class-blind prediction a *within-assay contrast* (predator-lane vs rabbit-lane) so any global
OOD shift hits both equally; (c) for the two assays where probe-OOD is unavoidable, list a
dedicated short fine-tune as the fallback.

---

## The seven testbeds

Each testbed: **(a)** name + one-line question; **(b)** manipulation (layout / entities /
matched-vs-differ); **(c)** the forced read-out + what discriminating vs class-blind each does;
**(d)** confound removed; **(e)** probe or dedicated agent; **(f)** config sketch (knobs to set).

### T1 — Guarded-Food Forced Choice (approach-avoidance conflict)

**(a) Question.** When the *only* path to food is guarded by an animal, does the agent pay a
detour/wait cost to avoid the predator but walk straight past the harmless rabbit?

**(b) Manipulation.** A symmetric two-lane corridor. Food sits at the far end of **both** a
left lane and a right lane. One lane's mouth is occupied by the **predator**, the other by the
**rabbit** — both *static* or *patrolling a fixed centre* so they don't chase (motion matched =
both stationary; no aggression confound at all). Predator and rabbit are byte-identical on smell,
class-visual-on-contact-only, size of patrol; they differ only in damage. Left/right assignment
is randomised per episode (counterbalanced) so the agent can't learn "left = safe".

**(c) Forced read-out.** Which lane the agent enters to reach food, and the **detour cost** it
pays (extra steps / waiting) to take the rabbit-guarded lane over the predator-guarded one.
*Discriminating agent*: preferentially routes to the rabbit lane, or waits/loops to time a
predator-lane crossing; predator-lane entry rate ≪ rabbit-lane entry rate. *Class-blind agent*:
enters the two lanes at ~50/50, eats from whichever is nearer, contacts the predator as often as
the rabbit.

**(d) Confound removed.** Both the **motion/aggression confound** (animals stationary, identical)
and the **number confound** (exactly 1 + 1). The forced choice makes "no preference" mean
something — the agent *must* go to one lane to eat, so 50/50 is a real null, not an averaging
artifact.

**(e) Probe.** Checkpoint probe on cell-08 agent. Mild geometry OOD, mitigated by the within-assay
left/right contrast (OOD hits both lanes equally).

**(f) Config sketch.** `entities`: 1 × `class:predator behaviour:static` + 1 ×
`class:neutral behaviour:static`, matched smell `[0,1,0,0,0]`, `damage:[5,120]` vs `[0,0]`,
`patrol_area` = each lane mouth (a 1×k cell band). Layout: two vertical lanes separated by a
`blocking:true` wall (use the `tree` obstacle with `blocking:true`, `count` set to build the
divider), food `count:1` at each lane's far end, `spawn_area` pinned per lane. `random_start_pos`
on the corridor mouth. Counterbalance lane assignment across the eval seed list.

---

### T2 — Barrier Anticipation Probe (contact made impossible)

**(a) Question.** With an animal visible/smellable but on the *far* side of a wall the agent
cannot cross, does the agent keep its distance from the predator-wall but approach the
rabbit-wall — when neither animal can ever touch it?

**(b) Manipulation.** The agent is in an open room. Along one edge, behind an **impassable**
(`blocking:true`) wall, an animal patrols a track parallel to the wall. Contact is *physically
impossible*. Two episode types, balanced 50/50: **predator-behind-wall** and
**rabbit-behind-wall**, byte-identical except damage. Food is placed so the agent has a reason
to spend time *near* the wall (forcing it to express a near-vs-far decision rather than ignore
the wall).

**(c) Forced read-out.** Mean agent–wall distance (and time-near-wall) in predator episodes vs
rabbit episodes, **measured over the whole episode because no contact can ever occur** — every
step is a pure anticipation step. *Discriminating agent*: keeps a larger standoff from the
predator-wall, forages closer to the rabbit-wall. *Class-blind agent*: identical wall-distance
distribution in both episode types.

**(d) Confound removed.** **The pain-reflex confound, completely** — this is the cleanest
anticipation assay because there is *no contact event in the entire episode*, so post-contact
reaction literally cannot enter the measure. (This is the constraint-1 gold standard.)

**(e) Probe.** Checkpoint probe. OOD risk: the agent never trained with an uncrossable animal —
but predator vs rabbit episodes share the identical geometry, so the *contrast* is OOD-cancelled
even if absolute behaviour is shifted.

**(f) Config sketch.** Single animal per episode, `behaviour:hunt` but confined to a patrol track
by `patrol_area` = a 1-cell-wide band on the far side of a full-width `blocking:true` wall
(`tree`, `count` sized to span the grid width). Two config files (predator-wall, rabbit-wall),
identical but for the one entity's `class`/`damage`/`nociception_intensity`. Food `spawn_area`
biased to the wall-adjacent rows. detection_range matched so the animal "notices" the agent
identically through the wall.

---

### T3 — Approaching-Threat-on-a-Track Latency (anticipatory flee timing)

**(a) Question.** As an animal advances down a lane toward the agent, does the agent start
retreating/diving for cover *earlier* for the predator than for the rabbit?

**(b) Manipulation.** A single corridor. The agent starts at one end near food; an animal starts
at the far end and advances (`hunt`, identical chase params for both classes). A bush sits
mid-corridor as available cover. Two balanced episode types: predator-approaching,
rabbit-approaching, byte-identical but damage.

**(c) Forced read-out.** **Latency to first defensive action** (first retreat step, or first
bush-dive) measured *as a function of animal-distance*, restricted to the **pre-contact window**.
*Discriminating agent*: triggers its retreat/cover at a larger animal-distance for the predator —
earlier anticipation. *Class-blind agent*: identical trigger-distance for both; or only reacts
after contact (latency ≈ contact step in both).

**(d) Confound removed.** **Motion/aggression** (both animals chase identically) and, by the
pre-contact-window restriction, the **pain reflex** (post-contact steps are dropped from the
measure). Contact is *possible* here (unlike T2) — which is exactly what lets us measure
*trigger distance*, the graded anticipation signal a barrier can't give.

**(e) Probe.** Checkpoint probe on cell-08. Low OOD — a corridor with one hunting animal is close
to the training dynamic; only the walls are new.

**(f) Config sketch.** 1×1 corridor (e.g. width 3, the centre column traversable), one animal per
episode `behaviour:hunt` matched chase fields (reuse cell-08's `detection_range:10, max_stamina:60,
...`), `spawn_area` pinned to the far end, agent `start_pos` at the near end (`random_start_pos:false`
so start-distance is fixed and latency is comparable across episodes). One bush mid-corridor
(`hides_agent:true`). Two files: predator vs rabbit.

---

### T4 — Cost-of-Avoidance / Detour Toll (how much will it pay?)

**(a) Question.** When avoiding the animal costs the agent measurable food/energy, how large a
toll will it pay to avoid the predator — and does it refuse to pay any toll for the rabbit?

**(b) Manipulation.** Like T1 but **asymmetric cost**: the animal sits on the *short* path to food;
a *longer* detour path reaches the same food animal-free. The detour length is the toll. One
animal per episode (predator or rabbit, balanced), static on the short path. Sweep detour length
across a small set of values (e.g. +2, +5, +10 steps) to get a **dose-response of willingness to
pay**.

**(c) Forced read-out.** **Detour-taking rate as a function of toll**, per class — i.e. an
indifference curve. *Discriminating agent*: pays a large toll to detour around the predator
(detour rate stays high as toll grows) but takes the short path past the rabbit at any toll
(detour rate ≈ 0). *Class-blind agent*: identical toll-vs-detour curve for both classes (it's
just trading off distance, blind to which animal blocks the path).

**(d) Confound removed.** **Pain reflex** (the *choice* is made before reaching the animal, en
route at the fork) and gives a **graded, quantitative** discrimination index (the toll at which
detour rate crosses 50%) rather than a binary "avoids / doesn't". Number-matched (1 animal).

**(e) Probe.** Checkpoint probe; sweep is over *layout files*, not retraining. Mild geometry OOD,
within-assay contrast (same toll, swap the animal class) cancels it.

**(f) Config sketch.** Fork layout: a junction where one branch is short (animal on it, static,
`patrol_area` = the choke cell) and one is long (animal-free), both ending at the same food.
N config files for N toll values × {predator, rabbit} = 2N files (or one geometry per toll with
the animal class as the only cross-file diff). Walls via `blocking:true` `tree` to enforce the
two-path topology.

---

### T5 — Predator-Only vs Rabbit-Only Matched Pair (in-distribution single-animal control)

**(a) Question.** Re-running the June "predator-only" control *correctly*: in an otherwise normal
foraging world with exactly one animal, does the agent behave differently when that one animal is
a predator vs when it is a rabbit?

**(b) Manipulation.** The cell-08 open foraging world, unchanged, but **one animal per episode**:
half the eval episodes have a single predator, half have a single rabbit — byte-identical chase,
smell, strike-and-retreat — and *nothing else changes*. The fix vs the June predator-only run is
that the **rabbit-only episodes are the matched control**, so "predator world" is compared to a
geometry-identical "rabbit world", not to a remembered two-animal training distribution.

**(c) Forced read-out.** Survival steps, pre-contact mean animal-distance, and pre-contact
bush-occupancy, **predator-only episodes vs rabbit-only episodes**. *Discriminating agent*:
larger standoff / more cover-seeking / interrupted foraging in predator episodes; survival gap.
*Class-blind agent*: the two episode sets are statistically indistinguishable pre-contact.

**(d) Confound removed.** The **OOD artifact that weakened the June predator-only verdict** — by
making the rabbit-only world the control, both sides are equally (un)familiar, so any "only one
animal" novelty cancels in the contrast. Number-matched (1 vs 1, across episodes). Motion matched.

**(e) Probe.** Checkpoint probe on cell-08 — this is the *direct, corrected* re-run of the June
control and the lowest-effort high-value assay (nearly the training world itself).

**(f) Config sketch.** Two configs cloned from `08-singlePredRabbit_disengage.yaml`: one drops the
rabbit entity (predator-only), one drops the predator and re-classes the survivor — **no**, to
keep it matched: file A = `count:1` predator + `count:0` rabbit; file B = `count:0` predator +
`count:1` rabbit, with the rabbit carrying the predator's identical chase profile. Same eval seed
list across both → paired comparison.

---

### T6 — Graded-Lethality Dose-Response (does danger *magnitude* shape avoidance?)

**(a) Question.** If we vary how dangerous the predator is (harmless → mildly damaging → lethal),
does the agent's anticipatory avoidance scale with the danger — the signature of a *learned*
danger estimate rather than a fixed class label?

**(b) Manipulation.** A graded series of *otherwise-identical* animals: damage drawn from
`[0,0]` (= rabbit), `[5,15]`, `[15,45]`, `[40,90]`, `[5,120]` (= current lethal predator). Each
level is its own episode set in a barrier or track layout (reuse T2 or T3 geometry so the read-out
is anticipatory). Smell and chase identical across all levels; **only the damage band moves.**

**(c) Forced read-out.** Anticipatory standoff distance (or flee-trigger distance) **as a function
of damage level** — a dose-response curve. *Discriminating agent that estimates danger*:
monotone increasing avoidance with damage. *Class-blind / pure-label agent*: flat (no scaling) or
a single step at "is it the predator class". A **monotone dose-response is the strongest possible
evidence** the agent represents graded danger, not just a memorised class token.

**(d) Confound removed.** Separates **"recognises the predator *class*"** from **"estimates how
dangerous this thing is"** — a distinction none of the binary assays can make. Removes the
class-label shortcut: here the smell is matched across *all* damage levels, so the only thing that
can drive a graded response is experienced danger magnitude.

**(e) Dedicated trained agent recommended.** A pure probe on cell-08 tests whether *its* learned
policy already scales — useful, but the cell-08 agent only ever saw `[5,120]`, so intermediate
levels are OOD for it. The clean version trains an agent **with the graded damage range present
during training** (e.g. damage sampled across levels), then probes the dose-response. Flag both:
run the probe first (cheap), escalate to dedicated training if the probe is flat-but-ambiguous.

**(f) Config sketch.** One barrier/track geometry (from T2/T3), K config files varying only the
predator `damage` band. For the dedicated-agent version: a training config whose single predator's
`damage` is drawn from the union range, plus the K eval-probe files. Keep `nociception_intensity`
fixed across levels (or scale it with damage — pick one; scaling it tests the *pain-signal* route,
fixing it tests the *injury-consequence* route — recommend fixing it so damage is the sole IV).

---

### T7 — Generalization Split: Smell-Swap & Novel-Layout Transfer

**(a) Question.** Has the agent learned "*this smell* / *this corner* is dangerous" (a surface
cue) or "the animal whose contact hurt me is dangerous" (a transferable danger concept)? Swap the
predator's smell with the rabbit's, or move both to a novel layout, and see if avoidance follows
the *danger* or the *cue*.

**(b) Manipulation.** Two transfer tests on a discriminating agent (one that *passed* T1/T3):
**(i) smell-swap** — give the predator the rabbit's old smell and vice-versa (so the
class↔smell mapping the agent might have latched onto is inverted; here smells are matched in
cell-08, so this variant requires a *training* config where the two smells actually differ, then
swap at test). **(ii) novel-layout** — same animals, a layout the agent never trained on
(different food positions, different wall pattern). Predator-vs-rabbit contrast preserved within
each test.

**(c) Forced read-out.** Whether the avoidance gap (predator standoff − rabbit standoff) **survives
the swap/transfer**. *Genuine danger concept*: avoidance tracks the *damaging* animal regardless
of its smell or the layout. *Surface-cue learner*: avoidance tracks the old smell / old corner,
so a smell-swap inverts or destroys the gap.

**(d) Confound removed.** Distinguishes **danger recognition** from **cue memorisation** — the
deepest construct-validity question, and the one that tells us whether a "discrimination" result
from T1–T5 is the real thing or an overfit cue. Controls for layout-specific overfitting.

**(e) Dedicated trained agent required for the smell-swap arm.** The swap is only meaningful if
the two classes had *distinct* smells in training (so there's a mapping to invert); cell-08
matched them, so this needs a sibling training config with distinct predator/rabbit smells. The
novel-layout arm is a pure probe.

**(f) Config sketch.** Smell-swap: a training config with predator smell `[0,1,0,0,0]` and rabbit
smell `[0,0,1,0,0]` (distinct), then an eval config with the two `properties` vectors exchanged.
Novel-layout: clone any T1–T5 geometry with food/wall positions perturbed, same entities.

---

## §3. Confound-coverage scorecard

How each testbed scores against the five hard constraints (✓ = fully handled, ~ = partially / by
within-assay contrast, ✗ = not addressed by this assay).

| Testbed | 1. Anticipation (no pain reflex) | 2. No motion/aggr. confound | 3. Not OOD (or cancelled) | 4. Number-matched | 5. Forced read-out | Probe or dedicated |
|---|---|---|---|---|---|---|
| T1 Guarded-Food Choice | ✓ (choice at fork, pre-contact) | ✓ (both static) | ~ (within-assay L/R) | ✓ (1+1) | ✓ (lane choice + detour) | Probe |
| T2 Barrier Anticipation | ✓✓ (contact impossible) | ✓ | ~ (contrast cancels) | ✓ (1 per ep, balanced) | ✓ (wall-distance) | Probe |
| T3 Track-Approach Latency | ✓ (pre-contact window) | ✓ (matched hunt) | ✓ (near training dyn.) | ✓ (1 per ep) | ✓ (flee-trigger dist.) | Probe |
| T4 Cost-of-Avoidance Toll | ✓ (choice at fork) | ✓ (static on path) | ~ (within-assay) | ✓ (1 per ep) | ✓ (detour-vs-toll curve) | Probe |
| T5 Pred-only vs Rabbit-only | ~ (pre-contact split) | ✓ (matched) | ✓ (rabbit-world control) | ✓ (1 vs 1 across eps) | ~ (forage/standoff) | Probe |
| T6 Graded-Lethality | ✓ (T2/T3 geometry) | ✓ | flag (intermediate OOD) | ✓ | ✓ (dose-response) | **Dedicated** (probe first) |
| T7 Generalization Split | ✓ (inherits geometry) | ✓ | flag (transfer is the point) | ✓ | ~ (gap survival) | **Dedicated** (smell arm) |

**The two strongest on constraint 1 (anticipation):** T2 (contact physically impossible) and T1/T4
(decision committed at a fork before reaching the animal). These are the assays that **cannot** be
fooled by a post-contact pain reflex — the failure mode that defeated every June naturalistic read.

## §4. Recommended assay battery (sequencing)

1. **T5 first** — it is the cell-08 training world minus one animal, so it is the lowest-OOD,
   lowest-effort assay and directly *corrects* the June predator-only control. If the agent is
   class-blind even here, that is the cleanest negative.
2. **T2 + T1 next** — the two anticipation-gold-standard assays (contact-impossible barrier;
   forced choice). Run as checkpoint probes on cell-08. These give the headline "anticipatory
   discrimination: yes/no" verdict the study has been chasing.
3. **T3 + T4** — graded refinements (trigger-distance, willingness-to-pay) that turn a yes into a
   *magnitude*, run only if T1/T2 show a non-null signal.
4. **T6 + T7** — the "is it learnable / is it a real concept" deep dives, requiring dedicated
   training; queue only if the probes (T1–T5) reveal *some* discrimination worth characterising,
   or to test whether a learnable route exists if they are all null.

## §5. Pre-registered read-out thresholds (to be finalised before launch)

Per the project's multi-seed convention and the documentation-framing rule, every assay's verdict
must be pre-registered before eval. Draft thresholds (numbers to be locked with the experts /
auditor when configs are written):

- **T1/T4 forced choice:** discrimination confirmed if predator-lane entry rate (or short-path
  rate past the predator) is below the rabbit-lane rate by a margin whose 95% CI across the eval
  seed set excludes zero; null if the CI spans zero.
- **T2 barrier:** confirmed if mean wall-distance is larger in predator episodes than rabbit
  episodes with non-overlapping 95% CIs across seeds.
- **T3 latency:** confirmed if flee-trigger animal-distance is larger for predator, CI excludes 0.
- **T5:** confirmed if any of {survival, pre-contact standoff, pre-contact bush-occupancy} differs
  predator-vs-rabbit with CI excluding 0; otherwise the corrected control replicates the June
  class-blind verdict in-distribution.
- **T6:** confirmed if avoidance is monotone non-decreasing in damage level (Spearman ρ>0, CI>0).
- **T7:** the avoidance gap from T1/T3 must persist after smell-swap / novel-layout (CI of the
  swapped gap excludes 0 and overlaps the un-swapped gap) for a "danger concept" verdict.

All assays are **multi-seed by default**: each eval uses the 200-seed list already in the cell-08
config; if a dedicated trained agent is needed (T6/T7), train ≥3 seeds.

## §6. Failure-mode catalog

- **All probes null.** If T1–T5 are all class-blind, the in-distribution corrected verdict is
  "pain-reactor, not danger-recogniser" — a clean, publishable negative (hardens the June result).
  Escalate to T6/T7 dedicated training to test whether discrimination is *learnable* at all.
- **Probe shows discrimination but T7 destroys it.** The T1–T5 signal was a surface cue
  (smell/corner overfit), not danger recognition — report as a cue-learning result, not danger
  recognition.
- **OOD geometry dominates (agent freezes / fails to forage in a maze).** If the cell-08 agent
  cannot forage in a barrier layout at all, the probe is uninterpretable; fall back to a short
  fine-tune on the assay geometry (dedicated agent) before reading discrimination.
- **Thin bins.** Forced-choice / latency measures must report per-bin n; a discrimination claim
  from a bin with n below a floor (to be set) is rejected — the exact trap the June aggregates fell
  into.
- **Contact still leaking into T2.** If any episode logs a contact in the "impossible" barrier
  assay, the wall geometry is broken — that run is refuted (config bug), not a discrimination
  result. Pre-flight via `env-config-auditor` must confirm zero reachable animal cells.

## §7. What needs new code vs. pure config

- **T1–T5, T6-probe, T7-novel-layout:** **config-only** — every knob (`behaviour:static/hunt`,
  `patrol_area`, `spawn_area`, `blocking:true` walls, per-entity `damage`/`smell`, `count:0` to
  drop an entity) is already read by `config_loader`. These are new YAML files under
  `configs/experiment/hypervigilance/` plus eval runs of an existing checkpoint.
- **Read-out scripts:** the forced read-outs (lane choice, detour toll, wall-distance,
  flee-trigger latency) are computable from the existing `.rec.gz` / `.npz` recordings (positions,
  per-class distances, actions, contact flags are all logged — see the grounding memo
  [[discrimination_measures_grounding]]). A small **offline analysis script** per assay family is a
  `feature-workflow` item, not a training-path change.
- **T6-dedicated, T7-smell-swap:** need **dedicated training configs** (graded damage during
  training; distinct predator/rabbit smells). Still config-only on the env side — no new schema.

**No schema changes are required by any testbed.** All manipulations use keys `config_loader`
already reads. (The only standing new-logging request in this line — the RNN hidden state for
internal-state decodability — is unrelated to these behavioural assays and is tracked in the
grounding memo's Metrics Requested.)

## Results / Conclusions

_To be filled after the assay battery is configured, audited, and evaluated._

## Links

- Study summary (why naturalistic reads failed): [[20260612_1625_predator_rabbit_discrimination]]
- Latest byte-identical training config (the agent most assays probe): [[single_pred_rabbit_disengage]]
- What the recordings let us measure (read-out feasibility): [[discrimination_measures_grounding]]
- EVAAA two-tier design (naturalistic training + isolated testbeds): `docs/project/references/InteroceptiveAI/sources/Lee et al. 2025 - EVAAA ... .pdf`, §5–6.
