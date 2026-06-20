---
title: "Hypervigilance under food scarcity + olfactory ambiguity (scarce vs abundant)"
topic: hypervigilance
status: active
created: 2026-06-20
last_updated: 2026-06-20
phase: hunger_gated_avoidance
wandb_tag: hvs
develop_link: "[[20260616_1557_hunger_gated_avoidance]]"
---

# Hypervigilance under food scarcity + olfactory ambiguity (scarce vs abundant)

> **Status**: PRE-REGISTERED — no runs launched yet.
> **Date**: 2026-06-20
> **Author**: experiment-designer
> **Builds on**: the just-completed Step-1 discrimination-onset map
> [[20260619_hunger_gated_step1_discrimination_onset]] (key finding carried forward below);
> living program [[20260616_1557_hunger_gated_avoidance]];
> base scene (archived) `configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml`.

---

## 1. Research Question

**Plain-language framing.** We put the agent in a little world with **one dangerous animal
(a predator that can kill in a single bite) and one harmless animal (a rabbit)** that behave
*identically* — same chasing, same speed, same retreat. The only distance cue that separates them
is **smell**, and we deliberately make that smell **almost useless**: the two animals smell nearly
the same, and each animal's smell **jitters wildly from episode to episode**, so the same smell is a
predator one day and a rabbit the next. The agent can only be *sure* which is which by getting close
enough to be bitten.

Into that ambiguous, lethal world we change **one thing** and ask what it does to caution: **how
much food there is.**

- **Scarce world (the hypervigilance condition):** only two food patches, in opposite corners, that
  deplete quickly and take a while to grow back. To not starve, the agent must keep **travelling
  across the open grid** — straight through the roaming ambiguous animal's territory — and **decide,
  each time, whether to push forward to food or back off**.
- **Abundant world (the reference / "can camp safely" baseline):** food everywhere, refilling
  instantly. The agent can sit on a food patch and never starve, so it can **afford to be cautious**
  with no cost.

The question: **does scarcity make the agent *pre-emptively avoid* the ambiguous animal — keep its
distance, flee before contact, dive into a bush — more than in the abundant world; and does that
caution flip with hunger (gamble and approach when starving, back off when full)?** "Pre-emptive
avoidance / hypervigilance" means acting cautiously **before** ever being bitten, on the basis of the
weak distal smell cue alone — the opposite of the **reactive tank-and-hide** survival style the
agent fell back on in the previous experiment (just getting bitten, then healing and hiding).

**Why this experiment exists (carried from the prior result).** The previous experiment (Step 1) swept
the smell from identical to fully distinct and found the agent **never learned meaningful distal
avoidance** — it survived by *reactive tank-and-hide*, not by keeping its distance, and the lethal
predator killed it in 40–70 % of episodes, which we flagged as possibly **masking** any hunger-gated
decision ("lethality masks gating"). The honest reading there was *"the lethal predator likely
prevented a hunger-gate from being reinforced,"* not *"no gate exists."* This experiment keeps the
lethality and the ambiguity but adds the missing ingredient: **a reason to approach.** In the prior
abundant world the agent could simply avoid the whole animal-rich centre and camp on food, so the
stay-vs-flee decision was never *forced*. Scarcity forces it. We pair scarce against abundant to read
the scarcity effect cleanly.

**Formal hypotheses.** (Symbolic names are introduced here once and then used in §4–§6.)

> **H₀ (null — scarcity does not change pre-contact caution):** The agent's pre-contact avoidance of
> the ambiguous animal (closest-approach distance, pre-contact flee rate, bush-dive rate) is
> statistically indistinguishable between the scarce and abundant worlds, and shows no hunger-gated
> split in either.

> **H₁ (scarcity forces a hunger-gated stay-vs-flee decision):** In the **scarce** world the agent's
> pre-contact avoidance **depends on hunger** — when it starts **satiated** it avoids the ambiguous
> animal more (greater closest-approach distance / higher flee rate), and when it starts **hungry** it
> takes the risk and pushes toward food / the animal (smaller distance / lower flee rate). This
> hunger-gated split is **larger in the scarce world than in the abundant world**, because only
> scarcity makes approaching worth the risk.

> **H₂ (baseline hypervigilance — does pre-emptive caution appear at all?):** Independent of hunger,
> does the agent in *either* world hold the lethal-but-ambiguous animal at a **greater baseline
> distance** than the reactive tank-and-hide agent did in the prior abundant-but-discriminable world?
> A yes means the high-ambiguity + lethality combination elicits genuine anticipatory caution; a no
> means contact-only distal sensing is too weak a cue to support hypervigilance regardless of food.

A fresh reader needs no other document to know what is asked and what counts as positive (H₁/H₂
supported) vs negative (H₀ retained). Symbolic / path-shaped detail lives in §2–§6.

---

## 2. Hypothesis & Predicted Outcomes (pre-registered)

State, in advance, what confirms vs refutes — and the *shape* of the predicted effect.

- **H₁ — hunger-gated stay-vs-flee, scarce > abundant. CONFIRMED if:** in the scarce world the
  satiated-minus-hungry closest-approach gap is **≥ 0.5 cell** with a 95 % CI excluding 0 (more
  avoidance when satiated), **and** this gap is **larger** in scarce than in abundant (the scarce gap
  exceeds the abundant gap by ≥ 0.3 cell, or the abundant gap is ≈ 0 / CI-includes-0 while the scarce
  gap is clean). **Shape:** a monotone hunger gradient within the scarce world (avoidance rises with
  starting nutrition), present in scarce, flat in abundant. **REFUTED if:** the scarce gap is < 0.5
  cell or its CI includes 0, **or** scarce ≈ abundant (scarcity adds no hunger gating).
- **H₂ — baseline hypervigilance vs the prior reactive agent. CONFIRMED if:** the pooled (hunger-
  collapsed) pre-contact closest-approach distance to the ambiguous animal in **either** world is
  **≥ 0.5 cell larger** than the prior Step-1 abundant-discriminable comparator (the `s=0.5, σ=0`
  orthogonal-smell run, which had the strongest distal cue yet still only ~0.2-cell gap), with
  flee-rate and bush-dive-rate agreeing (≥ 2 of 3 measures). **Shape:** a flat elevated baseline, not
  a hunger gradient (that part is H₁). **REFUTED if:** baseline distance is within 0.5 cell of the
  prior reactive agent — i.e. the agent still tank-and-hides regardless of scarcity.
- **The lethality watch-out, pre-registered as a verdict modifier (not a hypothesis).** If H₁ fails
  **and** the scarce world's hungry-start episodes show a **high death rate** (agent dies while pushing
  toward food past the animal), the verdict is **"lethality + scarcity too punishing — the gate could
  not be reinforced,"** NOT "no hunger-gate exists." Threshold: scarce-world hungry-start death rate
  ≥ 60 % (matching the prior experiment's worst rows) triggers this reading. See §6.

---

## 3. Launch Manifest

System-of-record for both runs. Designer fills the planned columns; `training-runner` fills
Node / GPU / Launched at / WandB run ID / Log path at launch. No runs launched yet.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| A | planned | scarce | `rppo_hvs_scarce_s42` | hypervig_scarcity | prod | 42 | — | — | — | — | — |
| B | planned | abundant | `rppo_hvs_abundant_s42` | hypervig_scarcity | prod | 42 | — | — | — | — | — |

Tags are unique, parseable (`rppo_hvs_<arm>_s<seed>`), and identical to the wandb-name. Both rows
share group `hypervig_scarcity` (= the topic dir). `hvs` = **h**yper**v**igilance under **s**carcity.

### 3.1 Configs to Produce

Only the food block is under test, so both runs use the **same** agent config
(`configs/models/recurrent_ppo.yaml`). The two env configs differ ONLY in the `resources:` (food)
block (diff-verified: animals, smell, obstacles, body, init-state are byte-identical).

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| A (scarce) | `configs/environment/experiment/hypervig_scarcity/01-scarce.yaml` | `configs/models/recurrent_ppo.yaml` |
| B (abundant) | `configs/environment/experiment/hypervig_scarcity/02-abundant.yaml` | `configs/models/recurrent_ppo.yaml` |

---

## 4. Experimental Design

### 4.1 Independent variable

**Food scarcity** — a single two-level factor (scarce vs abundant), realised entirely in the
`resources:` (food) block. Exact parameters and their reasoning:

| | **Scarce (A)** | **Abundant (B, = cell-08)** | Reasoning |
|---|---|---|---|
| food sources | **2** (Top-Left + Bottom-Right corners, `count:1` each) | **8** (4 quadrants × 2) | diagonal corners force the agent to **traverse the full grid** — through the animal's roam — to alternate patches; abundant is the camp-able cell-08 layout |
| `max_consumption` | **4** (each cell → 4 eats then depletes) | 12 | scarce depletes quickly so a patch can't be milked indefinitely |
| `regeneration_delay` | **40** steps | 0 (instant) | a depleted scarce patch is gone for 40 steps, so the agent must move on / wait, not camp |

**Food economics (why "scarce but survivable").** Nutrition drains at `metabolic_cost = 1.0`/step;
each eat nets `food_nutrition_gain − eating_nutrition_cost = 6 − 1 = +5` nutrition; death fires at
`nutrition ≤ 0` or `injury ≥ 100`. A 500-step episode needs ~500 nutrition of net inflow to survive on
food alone. Scarce throughput: 2 cells × 4 eats × ~12 regen-cycles over 500 steps ≈ **+480 net
nutrition** — roughly offsetting the ~500-step drain **only if the agent forages well** (visits both
corners, times the regen). This is the design target: **scarce enough to force foraging trips through
the animal's territory, survivable enough that the stay-vs-flee decision is real rather than a
death sentence.** Abundant throughput is effectively unbounded (8 cells, instant regen, 96 capacity),
so the agent can camp and never starve — the "can afford caution" reference.

### 4.2 Dependent variables

- **Primary (survival-step framing).** Headline performance is **survival steps** per run across
  training; reward is a secondary diagnostic only.
- **Behavioural read-out — the pre-contact predator-vs-ambiguous-animal avoidance signature**, the
  same toolkit (M1 closest-approach, M2 bush-dive, M7 eval-rollout) used in Step 1, measured three
  ways (definitions carried verbatim from [[20260619_hunger_gated_step1_discrimination_onset]] §5):
  1. **Closest-approach distance** — minimum agent-to-animal distance before first contact (held
     *farther* ⇒ more avoidance).
  2. **Pre-contact flee rate** — fraction of pre-contact encounters where the agent increases distance
     within the K-step (`obs_window=5`) window.
  3. **Bush-dive rate** — fraction of pre-contact encounters ending in a `hides_agent` bush cell (M2).
- **The hunger-gated split** — each of the three measures split by **starting nutrition** (satiated
  vs hungry halves, median split over the 200 eval episodes), within each world.

### 4.3 Controls / fixed factors (held IDENTICAL across A and B)

Everything except the food block is pinned and byte-identical between the two configs (diff-verified;
both load + reset correctly through `config_loader.py`):

```yaml
# Held constant across both runs:
environment:
  height: 10; width: 10; max_steps: 500; random_start_pos: true
  entities:
    - predator: class predator (is_damaging), behaviour hunt, count 1, LETHAL damage [5,120],
                nociception 0.9, full-grid spawn+patrol, disengage_on_contact true,
                detection_range 10, max_stamina 60, recovery 1.0, hunt_threshold 0.3, lose_interest 3.0
                smell [0, 0.55, 0.45, 0, 0], smell_std [0, 0.4, 0.4, 0, 0]   (HIGH AMBIGUITY)
    - rabbit:   class neutral (harmless), behaviour hunt, count 1, damage [0,0], nociception 0.0
                — BYTE-IDENTICAL chase profile; smell [0, 0.45, 0.55, 0, 0], same std (mirror)
    - hiding-predator slot: count 0 (inert schema template)
  obstacles: 12 rocks (4×3) + 12 bushes (4×3, hides_agent true) + inert tree count 0
  visual_properties: class defaults (predator ch5, rabbit ch7) — visual_sensor_range 0 => CONTACT-ONLY
body:
  random_start_nutrition: true,  start_nutrition_low: 10,  start_nutrition_high: 100
  random_start_injury:   true,   start_injury_low: 0,      start_injury_high: 80
  random_start_satiation: false  (satiation DERIVED from nutrition: S = N at scaling=1)
  metabolic_cost 1.0, food_nutrition_gain 6, eating_nutrition_cost 1.0, death at injury>=100 / nutrition<=0
sensory / perceptual_noise / behavior_measures: inherited from default.yaml unchanged
                                                (perceptual_noise.enabled: false)
# Training: agent configs/models/recurrent_ppo.yaml, fresh-init, single seed 42, ~10M episodes,
#           num-envs 128, checkpoint-frequency 100k.
```

**Smell scheme (high-ambiguity, symmetric).** This is the most-ambiguous *separated* rung from
Step 1 — `s = 0.05` separation, `σ = 0.4` per-episode jitter. The two animals sit on olfactory
channels 2 & 3, mirrored around 0.5 and slid apart by only `s = 0.05` (predator `[0, 0.55, 0.45, 0,
0]`, rabbit `[0, 0.45, 0.55, 0, 0]`), with strong per-episode jitter `[0, 0.4, 0.4, 0, 0]` on both.
The 0.1-cell mean gap is swamped by the 0.4 jitter, so smell is **irreducibly ambiguous** — exactly
the regime the program says a hunger-gate should live in. The symmetric layout carries no
presence/absence giveaway. Why this rung and not a cleaner one: the program's whole point is the
*blurry* decision; a clean smell removes the uncertainty the hunger-gate needs.

**Initial-state bounds — reasoning (carried verbatim from Step 1, env-config-auditor to confirm).**
Nutrition `[10, 100]`: `100 = max_nutrition` (full) down to a hungry floor of 10 (~10 steps of buffer
at `metabolic_cost = 1.0`), strictly above instant-starvation. Injury `[0, 80]`: death fires at
`injury ≥ 100`, so the high bound is strictly below the death line; 80 spans healthy → badly-injured
with margin and never self-triggers death at reset.

### 4.4 Seeds & sample size

- **Seeds: single training seed 42 per arm** (user-locked). This makes the experiment **provisional** —
  a single seed cannot separate a real scarcity effect from seed-specific training noise. Any
  scarce-vs-abundant difference flagged here is a **first read** that requires multi-seed hardening
  (≥ 3 seeds) before it is believed. The geometry/trajectory cross-check (§5) is therefore *mandatory*
  before trusting any mean.
- **Sample size per arm:** 200 eval episodes on the final checkpoint (fixed `eval_seeds`,
  deterministic policy) + per-checkpoint eval every 100k episodes for the temporal-evolution curve.
- **Compute (rough):** 2 runs × ~10M episodes. At the cell-08 lineage's throughput this is one
  parallel wave (2 GPUs, wall-clock = a single training); negligible relative to the Step-1 10-run wave.

### 4.5 Confounds & limitations

| Confound / limitation | Severity | Mitigation |
|---|---|---|
| **Single seed (42) per arm** — cannot separate a real scarcity effect from training noise | High | provisional first read; any effect gets multi-seed hardening before it is believed. Read trajectories, not just means. |
| **Lethality + scarcity may be too punishing** — scarcity pushes the agent *into* danger, so death rate may rise and a hunger-gated *stay/approach* may never get reinforced | High | pre-registered verdict modifier (§6): high hungry-start death rate ⇒ "too punishing," not "no gating." |
| **Vision-count (1-vs-1) leak** — vision reports a per-class on-cell count; with 1 predator + 1 rabbit the agent can sometimes infer class by elimination, bypassing smell | Medium | the Step-1 matched-smell anchor was a clean null (gap ≈ 0), so the leak was confirmed small there; re-check via the class-blind eval control (§5). Both arms share the identical 1-vs-1 layout, so the leak is **common-mode** and cancels in the scarce-vs-abundant *contrast* — it only threatens the absolute H₂ baseline, not the H₁ difference. |
| **Foraging geometry differs by construction** — scarce food sits in 2 corners, abundant in 8 cells, so encounter geometry with the animal differs between arms | Medium | this is *intrinsic* to the IV (scarcity changes where the agent must go) — not a nuisance to remove but a channel to characterise. The geometry/class-blind control (§5) separates "agent chose to avoid" from "the layout produced fewer encounters." |
| **Contact-only distal cue** — `visual_sensor_range: 0` means smell is the *only* distal class signal, and it is deliberately near-useless here | Medium (by design) | H₂ directly tests whether this is too weak for hypervigilance; a refuted H₂ is an informative null about the cue, not a bug. |

---

## 5. Analysis Plan (pre-specified)

**Primary statistic.** Per arm, each avoidance measure as **mean ± 95 % CI over the 200 eval
episodes** (fixed `eval_seeds`, deterministic policy). With a single training seed the CI is
**within-run** (episode sampling) only — it does **not** license a cross-seed claim. Survival steps is
the headline performance metric.

**Effect-size thresholds (pre-registered).**
- **H₁ (hunger-gated, scarce > abundant):** scarce satiated-minus-hungry closest-approach gap ≥ 0.5
  cell, CI excluding 0, **and** scarce gap − abundant gap ≥ 0.3 cell (or abundant gap CI-includes-0
  while scarce gap is clean). ≥ 2 of the 3 measures (distance, flee, bush-dive) must agree (event-level,
  per the carry-forward "event not mean" rule).
- **H₂ (baseline hypervigilance):** pooled pre-contact closest-approach distance in either arm ≥ 0.5
  cell larger than the Step-1 `s=0.5, σ=0` comparator (~0.2-cell distal gap, the prior strongest distal
  cue), flee + bush-dive agreeing.

**Hunger-gating test.** Within each arm, split the 200 eval episodes at the **median starting
nutrition** into hungry vs satiated halves; report each avoidance measure per half; the scarce-arm
satiated-minus-hungry difference is the H₁ statistic. Report the abundant-arm difference alongside for
the scarce > abundant contrast.

**Temporal evolution (mandatory).** Track survival steps **and** the avoidance measures (+ the
hunger-gated split) **across training checkpoints** (every 100k episodes), not just at the end — a
hunger-gate may be a slow-forming convergence phenomenon and a final-snapshot read would miss it. Do
not declare a flat/null surface until the curves have plateaued.

**Geometry / trajectory cross-check (mandatory before trusting any mean).**
- **Trajectory read.** Step-level dump (the `trajectory-story` tool) of a sample of scarce-arm
  episodes — does the agent *choose* to detour around the animal en route to food, or does it walk
  straight through and tank? Means are not trusted until the trajectory read confirms the mechanism
  (this is exactly what reclassified the Step-1 means as "tank-and-hide").
- **Class-blind control (at eval, no extra training).** Re-evaluate each trained checkpoint with the
  visual class channel masked on a sample of episodes. Any pre-contact gap the class-blind agent
  reproduces is a **geometry/encounter artifact**, subtracted from the real gap. Run on both arms so
  the scarce-vs-abundant contrast is read against the class-blind null in each.

**Cross-correlation (pre-specified window).** Time-lock pre-contact flee events to **starting
nutrition** and to **current nutrition at the encounter** within the scarce arm: predicted negative
correlation (lower nutrition ⇒ lower flee rate ⇒ approach-to-eat) with no lead/lag assumption beyond
the within-episode encounter timestamp.

---

## 6. Failure-Mode Catalog (pre-decided)

- **Training instability (NaN / value explosion).** Refutes the **run**, not a hypothesis. Re-launch
  the affected arm (same seed) once; if it recurs, flag as "no data" rather than reading it as H₀.
- **Lethality + scarcity too punishing (the carried-forward watch-out).** If H₁ fails **and** the
  scarce arm's hungry-start episodes show a death rate ≥ 60 % (the agent dies pushing toward food past
  the lethal animal), the verdict is **"lethality + scarcity too punishing — the gate could not be
  reinforced,"** NOT "no hunger-gate exists." Next step would be a sub-lethal predator (e.g. `[15,45]`)
  so a hungry approach can survive — directly testing whether the null is real or an artifact (this is
  the same lever Step 1's results flagged).
- **Scarce world unsurvivable (starvation collapse).** If the scarce arm's survival steps collapse to
  near the starvation floor regardless of behaviour (agent starves before reaching the second corner),
  the scarcity tuning is **too harsh** — this is a design flaw, not a null; loosen `regeneration_delay`
  (e.g. 40 → 20) or `max_consumption` (4 → 6) and re-run. The +480-vs-500 budget (§4.1) is sized to
  avoid this, but the single-seed read must check survival first.
- **Abundant world shows the same caution as scarce (no contrast).** A genuine, informative null for
  H₁ **only if** both arms survive well and the abundant arm is *not* starvation-pressured — i.e. the
  agent is cautious by default everywhere, scarcity adds nothing. Read against the survival + death-rate
  check before calling it.
- **Anchor leak (count-based caution).** If the class-blind control reproduces the pre-contact gap, the
  vision-count elimination leak is active; interpret gaps relative to the class-blind null, not zero.
  (Common-mode across arms, so the H₁ contrast survives even if H₂'s absolute baseline does not.)
- **Insufficient horizon.** If the avoidance / hunger-gate curve is still rising at 10M episodes
  (temporal curve not plateaued), the estimate is a lower bound — extend rather than read a null.

---

## 7. Metrics Requested

None new are required for the **primary** read-out: closest-approach distance, pre-contact flee rate,
and bush occupancy are already produced by the behaviour-measure toolkit (M1/M2 online + the M7
eval-rollout protocol, both enabled in the inherited `behavior_measures` block), and survival steps +
per-checkpoint logging already exist.

One **convenience** request, optional and non-blocking (same as Step 1): the **starting nutrition per
eval episode** logged alongside each rollout so the §5 hunger split can be computed without re-deriving
it from the reset RNG. Cheap (one scalar per episode), would live in the eval-rollout recorder in
`src/`. If not added, the split is still computable offline by replaying the fixed `eval_seeds` resets —
so this does **not** gate the launch.

---

## 8. Handoff / Next Steps

1. **env-config-auditor** pre-flight on the 2 configs (obs↔noise width, smell symmetry, food-block
   sanity, init-range bounds, list-replace completeness, diff-clean food-only contrast).
2. **PI consult** — *skipped per the prior user call* (this is a 2-run follow-up of an already-decided
   program); re-enable only if the user requests it.
3. **training-runner** launches both arms with user-supplied node + GPU. Exact per-run tags from §3:
   ```bash
   # Run A (scarce):
   python train.py --config configs/environment/experiment/hypervig_scarcity/01-scarce.yaml \
     --agent_config configs/models/recurrent_ppo.yaml --num-envs 128 --episodes 10000000 \
     --checkpoint-frequency 100000 --seed 42 --device cuda:<GPU> --tag rppo_hvs_scarce_s42
   # Run B (abundant):
   python train.py --config configs/environment/experiment/hypervig_scarcity/02-abundant.yaml \
     --agent_config configs/models/recurrent_ppo.yaml --num-envs 128 --episodes 10000000 \
     --checkpoint-frequency 100000 --seed 42 --device cuda:<GPU> --tag rppo_hvs_abundant_s42
   ```
   (The runner uses the exact per-run Tag from the §3 manifest, launched via `run_command.py`.)
4. After training, results + verdict return to this doc (§2 outcomes filled, §5 analysis run,
   trajectory cross-check first).

## Links

- Prior experiment this builds on (Step 1 + results):
  [[20260619_hunger_gated_step1_discrimination_onset]]
- Living program (source of truth): [[20260616_1557_hunger_gated_avoidance]]
- Base scene (archived, frozen):
  `configs/environment/experiment/archive/hypervigilance/08-singlePredRabbit_disengage.yaml`
- Config authoring: [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md) (sparse `extends:`,
  list-replace footgun §1, init-state ranges §3.4)
- Carry-forward confounds: summary
  `docs/experiments/summaries/20260612_1625_predator_rabbit_discrimination.md`
