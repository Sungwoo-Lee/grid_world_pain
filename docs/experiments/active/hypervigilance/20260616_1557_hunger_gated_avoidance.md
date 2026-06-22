---
title: "Training program: hunger-gated avoidance under olfactory uncertainty"
study: hunger_gated_avoidance
generated: 2026-06-16T15:57
last_updated: 2026-06-19T01:45
status: living-plan
---

# Training program: hunger-gated avoidance under olfactory uncertainty

> **Living plan.** This document grows by **appending training steps** over time. The Purpose and
> Carry-forward awareness sections are stable; each new training round is added as a new `### Step N`
> under **Training program**, and its one-line entry is added to the **Step log** table. Do **not**
> rewrite earlier steps — append the next one and update the log.
>
> **v3.0 note (2026-06-19).** Config references below are v3.0: the cell-08 base is now **archived**,
> new runs are authored as sparse `extends: environment/default` overrides, and the initial-state
> randomization prerequisite has **shipped**. See [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md).

## Purpose (read first)

The original motivation was never "can the agent tell a predator from a rabbit?" — it was
to find the **blurry decision point** where the agent has to *choose* between staying to eat
and fleeing. That choice only exists when the smell signal is **ambiguous**:

- If the olfactory cue is **perfectly clear** (predator and rabbit smell totally different),
  there is no uncertainty — avoid the predator, ignore the rabbit, decision trivial.
- If the cue is **perfectly matched** (the `sameProp` runs), there is no *distal* signal at
  all — the agent can only react to pain after contact. This is exactly why the discrimination
  analysis hit a wall: the agent didn't avoid the rabbit **at all**, because we had removed the
  very uncertainty the decision needs.
- The interesting regime is **in between** — a *graded* overlap where smell is suggestive but
  not conclusive, so the agent must combine the ambiguous cue with its **internal state**.

The behaviour we actually want is **avoid sometimes, not always**, gated by hunger/satiation:

- **Hungry** → the cost of not eating is high, so the agent should **take the risk** and stay to
  eat even when the animal *might* be a predator.
- **Satiated** → there is no need to gamble, so the agent should **choose avoidance** of the
  ambiguous animal.

The earlier "similar-property" (partial-overlap) training — the step *before* `sameProp` —
was the right instinct: it gave the agent a graded smell it had to reason over. `sameProp`
overshot into full ambiguity and erased the decision. This program walks the uncertainty axis
**deliberately and in steps**, eventually adding an explicit **interoceptive (hunger) gate**.

## Carry-forward awareness — how the agent beat matched smell before

The prior matched-smell **+ matched-chase** study (`sameProp`) found the agent *could* still tell
the rabbit apart — **not** by distal sensing, but through leaks and confounds we must control, or
the olfactory uncertainty we build here will simply be **bypassed**. These constraints apply to
**every** step below:

1. **Discrimination by elimination (vision-count leak).** Vision reports a per-class **count** of
   animals on the agent's *own* cell. With few animals (e.g. 1 predator + 1 rabbit), the agent
   infers an approacher's class by counting which neutrals are already "accounted for" on its cell
   — the unaccounted approacher *must* be the predator. This sidesteps the smell signal entirely.
   → For genuine smell-based uncertainty, **defeat count-tracking**: enough animals per class (and
   roaming / off-cell dynamics) that identity can't be deduced by elimination.

2. **Contact-only class channels.** At contact, the visual one-hot (predator ch5 vs rabbit ch7)
   and the extero-nociception `is_damaging` mask reveal class exactly. Discrimination is therefore
   always learnable **post-contact**. Our stay-vs-flee decision lives **pre-contact**, so smell
   ambiguity is the real variable — but expect a post-contact reactive component layered on top.

3. **Event-level, not mean-level.** The discrimination surfaced in defensive **events** (bush-dive
   gap), invisible in mean distance. Read **trajectories**, not just aggregates — and gate every
   measure on internal state (hunger), since that is exactly where averaging hid the behaviour.

4. **Spatial-encounter / geometry artifact.** A later check found the bush-dive gap can track world
   **geometry**, not the agent — a known **class-blind** agent reproduced it. Calibrate every
   behaviour measure against a class-blind agent **and** a geometry control before trusting it.

Sources: summary `docs/experiments/summaries/20260612_1625_predator_rabbit_discrimination.md`
(§2 endpoints, §3 verdict, Appendix A); insights `20260609_1719` (vision-count elimination),
`20260508_1445` (discriminating channels), `20260616_0142` (spatial-encounter artifact).

## Training prerequisite — initial-state coverage (✅ shipped in v3.0)

**Why it matters.** The agent's *starting* internal state is set once at episode reset. The eval
probes ([[experiment_environment_designs_v1]]) treat that starting state as a **dial** (start hungry,
start injured); for a probe read to be valid the trained agent must have *experienced* those starts.
Equally, **hunger-gated risk-taking can only be learned if the agent makes decisions across the full
hunger range** — a fixed full start exposes it to hunger only as a slow late-episode drift.

**Resolved (v3.0).** The start-state bounds are now **config-driven** — `body.start_nutrition_{low,high}`
and `body.start_injury_{low,high}`, required only when the matching `random_start_*` flag is `true`
(conditional-mandatory, no ripple to existing configs). The old code was locked to the upper half of
nutrition / lower half of injury; the new keys cover the **full** range, so training can start the
agent genuinely hungry or injured. Satiation stays *derived* from nutrition (nutrition is the hunger
lever; `random_start_satiation` is a documented no-op). See [[CONFIGURABLE_INITIAL_STATE_RANGES]] and
[`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md) §3.4.

**Decision (locked 2026-06-19).** Train Step 1 with **randomised initial internal state** — both
`random_start_nutrition` and `random_start_injury` on, spanning the full range (exact bounds set by
`experiment-designer` + `env-config-auditor`: nutrition kept above instant-starvation, injury below
the death line).

## Training program

### Step log

| Step | Goal | Base config | Knob(s) | Status |
|---|---|---|---|---|
| 1 | **Discrimination-onset map** — find *when* the agent starts to tell predator from rabbit, sweeping smell mean-gap × per-episode std | cell-08 scene → fresh sparse `extends:` | Δμ (mean gap) **×** σ (`properties_std`) | **DONE (results filled)** → [[20260619_hunger_gated_step1_discrimination_onset]]. Verdict: weak distal tilt (never reached threshold), survival by reactive tank-and-hide, lethal `[5,120]` likely masked any hunger-gate (40–70 % death). |
| 1b | **Hypervigilance under food scarcity + olfactory ambiguity** — add a *reason to approach* (scarce food) so the stay-vs-flee decision is forced, under the most-ambiguous separated smell + kept lethality; scarce vs abundant | cell-08 scene → fresh sparse `extends:` | **food scarcity** (scarce vs abundant), food block only | **configs + pre-registered design done** → [[20260620_hypervig_scarcity_olfactory_ambiguity]] (2 configs in `configs/environment/experiment/hypervig_scarcity/`; pending env-config-auditor pre-flight) |
| 1c | **Step-1 re-run, linear olfactory decay** — same 10-cell sweep, only change = longer-range distal smell cue (olfactory decay exponent 2.0 → 1.0); tests whether more advance warning converts the reactive tank-and-hide agent into a *pre-emptive* avoider | Step-1 cells → sparse `extends:` of each Step-1 twin | **`sensory.decay_power`** (2.0 vs 1.0), one key only | **configs + pre-registered design done** → [[20260620_hunger_gated_step1_linear_olfactory_decay]] (10 configs in `configs/environment/experiment/olfactory_ambiguity_lindecay/`; pending env-config-auditor pre-flight) |
| 2 | **Add sensory (perceptual) noise** as a further difference | from Step 1's chosen point | `perceptual_noise` block (per-step obs noise) | future |

*(further steps appended below)*

> **Mechanism note (verified in code).** Each animal's smell is drawn **once per episode** at reset
> (`src/environment/core.py` `_sample_property` inside `jax_reset`) from `N(properties,
> properties_std)` and held **fixed for the whole episode**. The agent never receives a class label
> for a smell — the *only* ground-truth class signal is **damage at contact**. So **σ = 0** gives a
> fixed, learnable smell→class mapping (**reducible** uncertainty, gone once trained); **σ > 0** makes
> the *same* smell a predator one episode and a rabbit the next (**irreducible** — only resolved by
> risking contact). The hunger-gated decision lives in the σ > 0 (irreducible) regime.

---

### Step 1 — Discrimination-onset map (mean-gap × per-episode std)

**Goal.** Find **when the agent starts to discriminate** predator from rabbit as we move *one knob at
a time* away from the known **cannot-discriminate** anchor (sameProp, σ = 0). We already know the two
ends: matched smell with no std → **no discrimination** (trained cell-08), and the old separated +
noisy smell → discrimination *but under confounds* (2-rabbit layout / spatial-count artifacts the
June work exposed). Step 1 fills the space cleanly, in the 1-vs-1 cell-08 base, over two knobs:

- **Δμ** — olfactory **mean separation** between predator and rabbit.
- **σ** — **`properties_std`**, the per-episode smell jitter.

**Base scene.** The cell-08 contrast — one predator + one rabbit, byte-identical chase (matched
aggression, `disengage_on_contact`, full-grid roam), lethal predator damage `[5,120]`. The original
config is now archived at
[`…/archive/hypervigilance/08-singlePredRabbit_disengage.yaml`](../../../../configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml)
(frozen, standalone). **Do not edit the archived file** — author each Step 1 run as a fresh **sparse
`extends: environment/default`** config (v3.0), overriding only the two animals' smell (`properties` /
`properties_std`) and the initial-state keys.

**Smell knob — symmetric (no presence/absence giveaway).** Both animals sit on olfactory channels
2 & 3, mirrored around 0.5, slid apart by a separation `s` — predator `[0, 0.5+s, 0.5-s, 0, 0]`,
rabbit `[0, 0.5-s, 0.5+s, 0, 0]`; `properties_std` (= σ) on ch 2 & 3 of both. `s=0` → identical (no
cue); `s=0.5` → orthogonal. (A pinned-predator scheme was rejected: leaving the predator at zero on
channel 3 turns that channel into a pure "rabbit present" flag, trivialising the gap at any s.)

**Using 10 GPUs — one parallel wave, then refine.** A grid is "too slow" only when runs are
sequential; with 10 GPUs the whole design runs **in one wave** (wall-clock = a single training). Spend
one GPU on a fresh **matched anchor** (`s=0`, the new baseline — we deliberately retrain rather than
reuse cell-08) and the rest across the (s, σ) design; weight the σ=0 row densely (cleanest learnability
onset) + two noise rows to see how σ shifts it:

| # | Separation `s` → pred (ch2,ch3) / rabbit | σ (`properties_std`) | d&prime; ≈ | Role |
|---|---|---|---|---|
| 1 | s = 0 → (0.5,0.5)/(0.5,0.5) | 0 | 0 | **matched anchor (fresh baseline)** |
| 2–5 | s = 0.05 / 0.1 / 0.25 / 0.5 | 0 | ∞ (reducible) | learnability onset along Δμ |
| 6–8 | s = 0.1 / 0.25 / 0.5 | 0.2 | 1.4 / 3.5 / 7.1 | onset under mild noise |
| 9–10 | s = 0.1 / 0.5 | 0.4 | 0.7 / 3.5 | onset under strong noise |

*(exact s/σ values + the 0.5 center are a proposal — adjust freely)*

- **Wave 2 (optional, +10 GPUs):** bisect around wherever Wave 1 shows the transition for a sharp
  threshold.
- **Fresh-init, not warm-start (recommended).** Warm-starting from cell-08 imports its learned prior
  that "smell is useless" and would *under-report* discrimination; 10 fresh parallel runs cost the
  same wall-clock as one. Warm-start only to make a cheap Wave 2.

**Read-out.** Per run, the **pre-contact predator-vs-rabbit gap** (distance held, flee rate,
bush-dive rate) as a surface over (s, σ). Expect a **threshold**: flat near the sameProp anchor,
rising once the smell is separable/learnable enough. On the **σ > 0** runs, additionally look for the
**hunger-gated split** (avoid when satiated, risk eating when hungry) — the irreducible regime is
where it should appear.

**Watch-outs (from Carry-forward awareness).**
- Read each run **trajectory-level**, not mean-only (item 3), and run the **class-blind / geometry
  control** on the anchor + a mid rung so a gap isn't an encounter artifact (item 4).
- 1-vs-1 keeps the **vision-count** leak (item 1) small but nonzero — confirm at the anchor.

**Status.** Design locked (2026-06-19); the 10 configs + the pre-registered design doc are written —
see [[20260619_hunger_gated_step1_discrimination_onset]] (configs under
`configs/environment/experiment/olfactory_ambiguity/`, init-state bounds nutrition `[10,100]` / injury
`[0,80]`). Next: env-config-auditor pre-flight → PI consult → training-runner.

---

### Step 2 — Add sensory (perceptual) noise (future)

**Goal.** Starting from Step 1's chosen operating point, add **per-step sensory noise** (the
`perceptual_noise` block — observation noise on the agent's sensors, distinct from the per-episode
smell std) as a further uncertainty source, and test whether the discrimination + hunger-gating
survive degraded perception. **Status.** Sketch — opens after Step 1.

## Decisions locked (2026-06-19)

- **Initial state** — randomise **nutrition + injury** across the full range (covers hunger-gating
  learnability + the forage/recovery/conflict eval probes).
- **Predator lethality** — **keep lethal `[5,120]`** (matches cell-08). *Watch-out:* risk-taking while
  hungry is often fatal, so if hunger-gated *staying* never emerges, revisit lethality first.
- **Init method** — **fresh-init** every run (no cell-08 warm-start).
- **Scope** — **full 10-run wave now**, full ~10 M-episode length per point, single seed (42) per
  point; multi-seed hardening is a follow-up.
- **Satiation contrast** — provided by the randomised nutrition start (no extra food-scarcity
  manipulation needed for the first wave).

## Links

- Base scene (archived): [`08-singlePredRabbit_disengage.yaml`](../../../../configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml);
  pre-`sameProp` distinct-smell reference: [`01-interoNocicept.yaml`](../../../../configs/environment/experiment/archive/hypervigilance/01-interoNocicept.yaml)
- v3.0 config authoring: [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md) (sparse `extends:`, init-state ranges §3.4); init-range plan [[CONFIGURABLE_INITIAL_STATE_RANGES]]
- Prior thread (why full-match was a dead end): [[sameprop_chasing_rabbit]],
  summary `docs/experiments/summaries/20260612_1625_predator_rabbit_discrimination.md`
- Latest reframe (gap tracks the world, env-as-behavior-platform): [[testbed_solo_validation_results]]
