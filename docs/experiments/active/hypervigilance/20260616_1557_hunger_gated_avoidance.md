---
title: "Training program: hunger-gated avoidance under olfactory uncertainty"
study: hunger_gated_avoidance
generated: 2026-06-16T15:57
last_updated: 2026-06-16T16:58
status: living-plan
---

# Training program: hunger-gated avoidance under olfactory uncertainty

> **Living plan.** This document grows by **appending training steps** over time. The Purpose and
> Carry-forward awareness sections are stable; each new training round is added as a new `### Step N`
> under **Training program**, and its one-line entry is added to the **Step log** table. Do **not**
> rewrite earlier steps — append the next one and update the log.

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

## Training prerequisite — initial-state coverage (needs a small code change)

**Why it matters.** The agent's *starting* internal state is set once at episode reset. The eval
probes ([[experiment_environment_designs_v1]]) treat that starting state as a **dial** (start
hungry, start injured) to induce behaviour — so for a probe read to be valid, the trained agent must
have *experienced* those starts; otherwise the probe measures out-of-distribution behaviour. Equally,
**hunger-gated risk-taking can only be learned if the agent makes decisions across the full hunger
range** — a fixed full start exposes it to hunger only as a slow late-episode drift.

**Current implementation (verified in `src/environment/core.py` `jax_reset`, ~L915–930).**
- `08` trains with `random_start_{satiation,nutrition,injury}: false` → **every episode starts
  identical** (nutrition 100 → satiation 100, injury 0).
- The randomisation mechanism exists but is **narrow and hardcoded**:
  - `random_start_nutrition: true` → nutrition ~ `U[max/2, max]` — **upper half only, never hungry**;
    satiation is *derived* from nutrition (`start_satiation` is not the lever — `start_nutrition` is).
  - `random_start_injury: true` → injury ~ `U[0, max/2]` — **≤ 50% only**.
  - `random_start_satiation` is loaded but **unused** in the reset (a no-op).
- So even flipping the flags on **cannot reach** the hungry (<50%) / badly-injured (>50%) starts the
  probes use.

**Required change (route to `senior-developer` → `developer`).** Make the start-state bounds
**config-driven** — e.g. `body.start_nutrition_{low,high}`, `body.start_injury_{low,high}` — covering
the full hungry→full and 0→high-injury range, and either wire up `random_start_satiation` or document
nutrition as the lever. Small, localised edit in `core.py` (~L917–928).

> **Plan written**: the implementation plan for this change now exists at
> [[CONFIGURABLE_INITIAL_STATE_RANGES]] (`docs/develop/active/refactors/`). It adopts the
> `body.start_nutrition_{low,high}` / `body.start_injury_{low,high}` schema, keeps satiation
> *derived* from nutrition (nutrition is the hunger lever; `random_start_satiation` left as a
> documented no-op), and makes the new range keys mandatory only when the matching
> `random_start_*` flag is `true` — so no existing config changes. Awaiting user approval of the
> three design forks before the `developer` agent implements.

**Decision for the user.** Train the Step 1 agents with **randomised initial internal state** spanning
the eval-probe range (recommended — makes the probes in-distribution *and* is arguably required for
hunger-gating to be learnable), or keep a **fixed full start** (cleaner discrimination read, but the
probes become OOD)? This gates config generation for Step 1.

## Training program

### Step log

| Step | Goal | Base config | Knob(s) | Status |
|---|---|---|---|---|
| 1 | **Discrimination-onset map** — find *when* the agent starts to tell predator from rabbit, sweeping smell mean-gap × per-episode std | `08-singlePredRabbit_disengage.yaml` (smell only) | Δμ (mean gap) **×** σ (`properties_std`) | **proposed — awaiting feedback** |
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

**Base config.** [`configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml`](../../../../configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml)
— one predator + one rabbit, byte-identical chase (matched aggression, `disengage_on_contact`,
full-grid roam), lethal predator damage `[5,120]`. **Only the two `properties` (mean) and
`properties_std` (σ) vectors change.** Knob convention: predator fixed `[0,1,0,0,0]`; rabbit moved
toward ch 2 by gap `g` → rabbit `[0, 1−g, g, 0, 0]`; σ applied to ch 1 & 2 of both animals.

**Using 10 GPUs — one parallel wave, then refine.** A grid was "too slow" only when runs are
sequential; with 10 GPUs the whole design runs **in one wave** (wall-clock = a single training). The
trained cell-08 is the σ=0/g=0 anchor — reuse it, spend **zero GPUs** there, and put all 10 on new
points. Weight the σ=0 row densely (cleanest learnability onset) + two noise rows to see how σ shifts
it:

| # | Rabbit smell `g` | σ (`properties_std`) | d&prime; ≈ | Role |
|---|---|---|---|---|
| — | g = 0 (= cell-08) | 0 | 0 | **anchor — reuse, no GPU** |
| 1–5 | g = 0.1 / 0.2 / 0.4 / 0.7 / 1.0 | 0 | ∞ (reducible) | learnability onset along Δμ |
| 6–8 | g = 0.2 / 0.4 / 1.0 | 0.2 | 1.4 / 2.8 / 7.1 | onset under mild noise |
| 9–10 | g = 0.2 / 1.0 | 0.4 | 0.7 / 3.5 | onset under strong noise (g=0.2/σ=0.4 ≈ old `5/7/4`, re-tested cleanly) |

*(exact g/σ values are a proposal — adjust freely)*

- **Wave 2 (optional, +10 GPUs):** bisect around wherever Wave 1 shows the transition for a sharp
  threshold.
- **Fresh-init, not warm-start (recommended).** Warm-starting from cell-08 imports its learned prior
  that "smell is useless" and would *under-report* discrimination; 10 fresh parallel runs cost the
  same wall-clock as one. Warm-start only to make a cheap Wave 2.

**Read-out.** Per run, the **pre-contact predator-vs-rabbit gap** (distance held, flee rate,
bush-dive rate) as a surface over (g, σ). Expect a **threshold**: flat near the sameProp anchor,
rising once the smell is separable/learnable enough. On the **σ > 0** runs, additionally look for the
**hunger-gated split** (avoid when satiated, risk eating when hungry) — the irreducible regime is
where it should appear.

**Watch-outs (from Carry-forward awareness).**
- Read each run **trajectory-level**, not mean-only (item 3), and run the **class-blind / geometry
  control** on the anchor + a mid rung so a gap isn't an encounter artifact (item 4).
- 1-vs-1 keeps the **vision-count** leak (item 1) small but nonzero — confirm at the anchor.

**Status.** Proposed — awaiting feedback before config generation.

---

### Step 2 — Add sensory (perceptual) noise (future)

**Goal.** Starting from Step 1's chosen operating point, add **per-step sensory noise** (the
`perceptual_noise` block — observation noise on the agent's sensors, distinct from the per-episode
smell std) as a further uncertainty source, and test whether the discrimination + hunger-gating
survive degraded perception. **Status.** Sketch — opens after Step 1.

## Cross-cutting open questions (program level)

- **Fresh-init vs warm-start** — run Step 1's wave fresh (clean, recommended) or warm-start from
  cell-08 to save compute (risks under-reporting discrimination)?
- **Run length** — full ~10 M-episode trainings per point, or shorter runs since we only need the
  onset threshold (discrimination may emerge late, so full is safer)?
- **Satiation read** — rely on cell 08's existing body/food dynamics to produce the hungry-vs-satiated
  contrast for the σ>0 runs, or actively manipulate food scarcity to guarantee both regimes occur?
- **Predator lethality** — keep cell 08's lethal `[5,120]` band, or soften it so risk-taking while
  hungry is survivable enough to be learnable?

## Links

- Base config: [`08-singlePredRabbit_disengage.yaml`](../../../../configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml);
  pre-`sameProp` distinct-smell reference: [`01-interoNocicept.yaml`](../../../../configs/experiment/hypervigilance/01-interoNocicept.yaml)
- Prior thread (why full-match was a dead end): [[sameprop_chasing_rabbit]],
  summary `docs/experiments/summaries/20260612_1625_predator_rabbit_discrimination.md`
- Latest reframe (gap tracks the world, env-as-behavior-platform): [[testbed_solo_validation_results]]
