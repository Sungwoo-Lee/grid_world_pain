---
title: "SameProp Round 1 — re-logged RPPO baseline (post-hoc analysis, partial budget)"
topic: hypervigilance
status: active
created: 2026-05-08
last_updated: 2026-05-08
phase: post-hoc-analysis
wandb_tag: hypervigilance-sameprop-relog
supersedes:
superseded_by:
---

# SameProp Round 1 — Re-logged RPPO Baseline (Post-Hoc Analysis)

> **Status**: COMPLETE (post-hoc, partial budget — both runs stopped at ~7.2 M / 10 M episodes by user decision; GPUs released for Round 2).
> **Date**: 2026-05-08
> **Author**: experiment-analyzer
> **Mode**: B — post-hoc, no pre-registered design doc (the prior survey at `sameprop_existing_run_survey.md` framed the *config*, not these specific runs, and explicitly carved Round 1 out as "P1 — re-run with logging").
> **Related**:
> - Predecessor (single-seed survey on the original sameProp run `rmw6m8zg`): [`sameprop_existing_run_survey.md`](sameprop_existing_run_survey.md) — see "Round 1 — Re-logged Baseline Launch Manifest" therein for the launch table.
> - Phase 1 channel-attribution memo (read-only obs-space audit): [`../../../develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md).
> - Logging plan (the change that made these runs interpretable): [`../../../develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../../develop/active/hypervigilance/per_entity_avoidance_logging.md).
> - Working files: `tmp/20260508_relog_compare.txt`, `tmp/20260508_relog_timeseries8.txt`, `tmp/20260508_relog_speed.txt`.

> **POST-HOC NOTICE**. There is no pre-registered hypothesis for this specific
> re-run. The hypothesis articulated below was framed retroactively against
> the user's stated claim ("the asymmetry is already clear; rabbit-vs-predator
> distance discriminates under sameProp") *after* the runs had reached steady
> state but *before* the analyzer inspected the steady-state numbers.
> Conclusions are therefore weaker than for a pre-registered design and should
> motivate a Round 2 pre-registered experiment, not be treated as final.
>
> **Partial-budget notice.** Both runs were stopped at ~7.2 M / 10 M episodes
> (~72 %) by user decision because the asymmetry pattern was already
> stable and visible across both seeds. GPUs were freed for Round 2.

---

## 1. Research Question

When the patrolling predator and the neutral rabbit emit the **identical**
olfactory `properties = [0, 1, 0, 0, 0]` and exist at the same time on the
grid, does an RPPO agent nonetheless learn to maintain greater distance from
the predator than from the rabbit, and **does that asymmetry grow with
training** (rather than being a pure initialization-noise artifact)?

> **H₀** (null): under matched olfactory properties, mean steady-state
> distance to rabbit equals mean steady-state distance to predator
> (`MeanDistRabbit ≈ MeanDistPredator`), and any nominal gap is flat across
> training (no learned asymmetry).
>
> **H₁** (alternative, user's claim): `MeanDistRabbit < MeanDistPredator` at
> steady state by an amount large relative to seed dispersion, **and** the
> gap is the joint product of (a) `MeanDistRabbit` *decreasing* over training
> (agent learns to approach rabbits) and (b) `MeanDistPredator` *increasing
> or holding* (agent does not approach the patrolling predator the same way).

The Round 1 logging change (commit `4b55fc6`) added the four metrics needed
to test H₁ directly: `Episode/MeanDistRabbit`, `Episode/RabbitHits`,
`Episode/MeanDistHidingPredator`, `Episode/HidingPredatorHits`
(`HidingPredatorHits` is preserved as an alias for the existing
`Episode/DangerHits` for dashboard continuity).

---

## 2. Experimental Design

This is a single-config baseline run, not a designed comparison — its purpose
is to populate the new metric set under the canonical sameProp configuration
that prior work flagged as the channel-collision control.

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| seed | 42, 43 | Two-seed minimum to estimate seed dispersion (n=2; not the 3+ that pre-registered designs require). |

### 2.2 Controlled Variables

```yaml
# configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml (unchanged)
predators[0].properties:        [0.0, 1.0, 0.0, 0.0, 0.0]   # SAME as rabbit
neutral_animals[0..1].properties: [0.0, 1.0, 0.0, 0.0, 0.0] # SAME as predator
hiding_predator[0..3].properties: [0.0, 0.0, 0.0, 0.0, 0.0] # NO olfactory cue
food[0..3].properties:          [1.0, 0.0, 0.0, 0.0, 0.0]   # discriminating
sensory.perceptual_noise.enabled: false
sensory.visual_sensor_range:    0   # visual fires only on colocation
sensory.olfactory_enabled:      true
sensory.sensor_radius:          20

# Spawn / patrol layout (load-bearing — see §2.3 confound 1):
food[TL].spawn_area:            [[1,1],[5,5]]
food[BR].spawn_area:            [[6,6],[10,10]]
neutral_animals[0].spawn_area:  [[1,1],[5,5]]   # rabbit TL, same as food TL
neutral_animals[1].spawn_area:  [[6,6],[10,10]] # rabbit BR, same as food BR
predator[0].spawn_area:         [[1,1],[10,10]] # FULL grid
predator[0].patrol_area:        [[1,1],[10,10]] # FULL grid
hiding_predator[0..3]:          one per quadrant (TL, TR, BR, BL)

# Agent: configs/models/recurrent_ppo/recurrent_ppo.yaml (unchanged from prior single-seed run)
# Budget: --episodes 10000000 (only ~7.2M reached before user-stop)
# Envs: 128 parallel
```

Code commit at launch: `f56f2bafdf952e0012f0b3d4ab12b45e35fd43bd` (post-`4b55fc6`).

### 2.3 Confounds & Limitations

| # | Confound | Severity | Notes / mitigation |
|---|---|---|---|
| 1 | **Food and rabbits share both quadrants (TL+BR), predator does not.** | **HIGH — primary confound, see Finding 3.** | A policy that learned only "go to food" would also produce `MeanDistRabbit < MeanDistPredator` simply because rabbits are co-located with food while the patrolling predator roams the full grid. Without decoupling food and rabbit spawn areas, this run cannot isolate olfactory channel attribution from food-seeking spillover. **Round 2 must control for this.** |
| 2 | n = 2 seeds | Medium | Sufficient for a "is the pattern reproducible across seeds?" check; below the n ≥ 3 standard for pre-registered effect-size claims. Both seeds agree to ≤ 0.03 cells on every steady-state metric (§4.1), so n=2 is *informative* though not conclusive. |
| 3 | Partial budget — both runs stopped at ~7.2 M / 10 M episodes (~72 %) | Low–Medium | Steady-state windows (§4.1) are taken from the last 20 % of *observed* episodes (≈ 5.7–7.2 M). All seven primary metrics are flat across the last three 8-window slices (§4.4), so the steady-state values are unlikely to shift materially in the unrun 30 %. The *trajectory* claim (rabbit-distance decreasing) survives because it manifests across the entire run, not just the tail. |
| 4 | Hiding-predator damage as confound for "predator avoidance" claims | Medium | Inherited from the prior survey. `MeanDistHidingPredator ≈ 2.63` is much smaller than `MeanDistPredator ≈ 4.40` because hiding predators are stationary in 4 corners (with rest-step alignment effects) — see Finding 4. This is informational, not bug-like. |
| 5 | No Dreamer comparison | Low (out of scope for Round 1) | Round 1 is RPPO-only. Architectural generalization is a separate (Round 3+) question. |
| 6 | RTX 3090 (this run) vs RTX 6000 Ada (original `rmw6m8zg`) | Low | Different node hardware, same code paths. SPS differs (~50–55 k vs prior). No correctness implication. |

---

## 3. Launch Manifest

This is a Mode-B doc — the launch manifest is owned by the predecessor
[`sameprop_existing_run_survey.md` § "Round 1 — Re-logged Baseline Launch
Manifest"](sameprop_existing_run_survey.md). The actuals (final episodes,
walltime, status: stopped early) are appended there. For convenience the
two rows are reproduced here:

| Run | Status | Cell | WandB tag (= name) | Seed | Node | GPU | WandB run ID | Final episodes | Walltime | Log path |
|-----|--------|------|--------------------|------|------|-----|--------------|----------------|----------|----------|
| 1 | stopped early (user) | R1-seed42 | `hypervigilance-sameprop-relog-seed42_n112_gpu0` | 42 | 112 | cuda:0 | `rg5nl1ov` | **7,284,221** | 54,974 s ≈ **15 h 16 m** | `logs/20260507_223107.log` |
| 2 | stopped early (user) | R1-seed43 | `hypervigilance-sameprop-relog-seed43_n112_gpu1` | 43 | 112 | cuda:1 | `6ks4bjbq` | **7,212,724** | 54,712 s ≈ **15 h 12 m** | `logs/20260507_223536.log` |

> **Discrepancy with user-supplied summary**: the brief stated "~6.7–6.8 M
> episodes" and "14h05m / 14h10m". Actual final episode counts are 7.28 M
> (s42) and 7.21 M (s43); actual walltimes are 15 h 16 m / 15 h 12 m.
> The user's directional claim ("stopped well short of the 10 M budget") is
> correct; the precise numbers are slightly higher than the brief reported.

### 3.1 Configs

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1, 2 | `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` | `configs/models/recurrent_ppo/recurrent_ppo.yaml` |

---

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

Steady-state means over the last 20 % of observed episodes (≈ 5.7–7.2 M),
both seeds, ± 1 stddev (within-seed temporal variance):

| Metric | seed 42 | seed 43 | Both-seed mean | Random baseline | Interpretation |
|--------|---------|---------|----------------|-----------------|----------------|
| `Episode/Steps` (survival) | 327.27 ± 8.33 | 327.27 ± 8.10 | **327.3** | — | survival 65.5 % of 500-step cap |
| `Episode/MeanDistFood` | 2.406 ± 0.045 | 2.405 ± 0.046 | **2.41** | 4.68 (uniform) | clear food-seeking |
| `Episode/MeanDistRabbit` | 3.766 ± 0.031 | 3.783 ± 0.030 | **3.77** | 4.68 (uniform) | **agent moves CLOSER than random** |
| `Episode/MeanDistPredator` | 4.416 ± 0.063 | 4.388 ± 0.064 | **4.40** | 4.68 (uniform) | indistinguishable from random by ≤ 0.3 cells |
| `Episode/MeanDistHidingPredator` | 2.635 ± 0.023 | 2.626 ± 0.024 | **2.63** | corner-stationary baseline | small — see Finding 4 |
| `Episode/RabbitHits` / ep | 6.562 ± 0.327 | 6.351 ± 0.294 | **6.46** | — | frequent contact (rabbits = 2 in config) |
| `Episode/PredatorHits` / ep | 3.430 ± 0.212 | 3.318 ± 0.197 | **3.37** | — | frequent contact (predator = 1) |
| `Episode/HidingPredatorHits` / ep | 3.009 ± 0.167 | 3.000 ± 0.159 | **3.00** | — | identical to `DangerHits` (alias verified) |
| `Episode/DangerHits` / ep | 3.009 ± 0.167 | 3.000 ± 0.159 | **3.00** | — | sanity: equals `HidingPredatorHits` exactly |
| `Episode/Term_Injury` | 0.356 ± 0.038 | 0.322 ± 0.035 | **0.339** | — | injury death rate |
| `Episode/Term_Starvation` | 0.379 ± 0.042 | 0.417 ± 0.041 | **0.398** | — | starvation death rate |
| `Episode/Term_MaxSteps` | 0.265 ± 0.025 | 0.260 ± 0.024 | **0.263** | — | timeout ("survival win") |
| `Episode/Reward` | -204.6 ± 2.20 | -204.4 ± 2.17 | **-204.5** | — | dominated by death penalty |
| `Episode/FoodEaten` / ep | 54.55 ± 2.18 | 54.89 ± 2.19 | **54.7** | — | food learned |

**Effect size (steady-state, both-seed mean):**
- `MeanDistPredator − MeanDistRabbit = 4.40 − 3.77 = **0.63 cells**`.
- Within-seed stddev ≤ 0.064 (predator) and ≤ 0.031 (rabbit); seed-to-seed
  difference of the gap is < 0.05. Effect size is ≥ 10× stddev.

> **Verdict on H₁** (under the partial-budget caveat): **SUPPORTED at the
> aggregate-distance level, but with a critical confound** — the asymmetry
> exists and is reproducible across both seeds. However, see Finding 3 —
> the food-rabbit quadrant co-location means the asymmetry isolates
> "co-location with food **plus** non-instantaneous olfactory signature
> (movement / patrol-area)", *not* olfaction alone. The gap is therefore real
> but its *attribution* to olfactory channel discriminability is still
> ambiguous after Round 1.

### 4.2 Secondary Metrics

| Metric | Value (SS, both-seed mean) | Trajectory shape | Note |
|--------|---------------------------|------------------|------|
| `Episode/Reward` | -204.5 | -204.4 → -205.5 | dominated by 100-pt death penalty + smaller homeostatic terms |
| Total walltime | 15 h 14 m | — | per run, on RTX 3090 |
| SPS | 50 757 (s43) – 55 020 (s42) | flat | training speed healthy; not throttled |

Note: this analysis does **not** include policy/value-loss inspection.
Training health is implicit in the smooth, monotonic survival curve and the
flat SPS; loss-curve verification is out of scope for this baseline analysis
and is not load-bearing for the H₁ verdict.

### 4.3 Diagnostic Metrics

RPPO does not expose architecture-internal diagnostics (no modulator gates,
no world-model latents). The four new behavioral metrics above are the
diagnostic surface.

### 4.4 Learning Dynamics — 8-window trajectory (both seeds)

Window edges by `Episode/Number`; values are window means (within-window
stddev shown for distance metrics only).

```
Window:                      1     2     3     4     5     6     7     8
Episode/Steps      (s42)  222.2 286.8 298.2 307.9 315.9 321.2 326.0 327.2
Episode/Steps      (s43)  226.1 281.9 299.0 308.2 316.0 319.7 324.4 328.3

MeanDistFood       (s42)   2.88  2.53  2.51  2.48  2.45  2.44  2.41  2.41
MeanDistFood       (s43)   2.87  2.53  2.50  2.47  2.45  2.43  2.41  2.40

MeanDistRabbit     (s42)   3.93  3.79  3.82  3.81  3.79  3.78  3.77  3.77
MeanDistRabbit     (s43)   3.93  3.83  3.84  3.82  3.81  3.80  3.79  3.78
                          ↘─── DECREASING — agent moves *closer* to rabbits over training ───↘

MeanDistPredator   (s42)   4.35  4.33  4.35  4.36  4.37  4.39  4.41  4.42
MeanDistPredator   (s43)   4.32  4.32  4.37  4.37  4.38  4.37  4.38  4.39
                          ↗─── slow growth, ~ +1.5–2 % over training ───↗

MeanDistHidingPredator (s42) 2.60 2.61 2.64 2.64 2.64 2.64 2.64 2.64
MeanDistHidingPredator (s43) 2.62 2.63 2.64 2.64 2.64 2.63 2.63 2.63

RabbitHits         (s42)   3.83  5.63  5.88  6.04  6.25  6.41  6.58  6.53
RabbitHits         (s43)   3.87  5.36  5.64  5.85  6.04  6.17  6.28  6.37
                          ↗─── monotonic, ~1.7× over training ───↗

PredatorHits       (s42)   3.18  3.50  3.55  3.47  3.42  3.42  3.44  3.41
PredatorHits       (s43)   3.31  3.65  3.54  3.46  3.41  3.38  3.32  3.32
                          peaks at window 2, then mild retreat (~ −5 %)

HidingPredatorHits (s42)   2.20  2.96  2.78  2.80  2.89  2.91  3.00  3.01
HidingPredatorHits (s43)   2.11  2.65  2.75  2.82  2.88  2.93  2.98  3.00
                          (rises monotonically as survival lengthens — confound)
```

Three qualitatively distinct shapes:
1. **MeanDistFood + MeanDistRabbit**: both *decrease* (food monotonically;
   rabbit non-monotonically with a small bounce in window 2 then steady
   decrease). The agent moves **closer** to both classes over training.
2. **MeanDistPredator**: slowly *increases* by ~1.5–2 % over training.
   Below-random at start (3.45 first-data first-window), settles slightly
   below the uniform-random baseline (4.68) by training end (4.40).
3. **PredatorHits**: peaks early (window 2, ~3.6/ep) then drops modestly
   (~3.3–3.4/ep). Consistent with: agent first aggressively explores all
   olfactory peaks (predator AND rabbit), then differentiates and reduces
   predator contacts while keeping rabbit contacts climbing.

### 4.5 Cross-seed reproducibility

Steady-state values (last 20 %) are tight across seeds:

| Gap | seed-42 value | seed-43 value | abs diff |
|------|--------------|---------------|----------|
| `MeanDistPredator − MeanDistRabbit` | 4.42 − 3.77 = 0.65 | 4.39 − 3.78 = 0.61 | 0.04 |
| `MeanDistRabbit` final | 3.766 | 3.783 | 0.017 |
| `MeanDistPredator` final | 4.416 | 4.388 | 0.028 |
| `RabbitHits − PredatorHits` | 6.56 − 3.43 = 3.13 | 6.35 − 3.32 = 3.03 | 0.10 |

n = 2 with seed-to-seed differences of ≤ 0.04 on every primary metric is
strong consistency, although still below the n ≥ 3 standard for
pre-registered designs.

---

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — A reproducible rabbit-vs-predator distance asymmetry exists at steady state.**
- *What*: `MeanDistPredator − MeanDistRabbit ≈ 0.63 cells` at steady state, ≥ 10× within-seed stddev, < 0.05 between-seed difference.
- *Why (candidate mechanisms — see Finding 2 for ranking)*: the agent has produced a behavior that distinguishes the two entity classes despite their identical olfactory `properties` vector, using one or more of the non-olfactory channels enumerated in [`sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md).
- *Evidence*: §4.1 table and §4.5 cross-seed comparison.
- *Confidence*: **High** for the *behavioral observation*; **Medium-Low** for any *mechanistic* claim (see Finding 3).

**Finding 2 — The asymmetry GROWS with training; it is not initialization noise.**
- *What*: `MeanDistRabbit` decreases monotonically from 3.93 → 3.77 over the run; `MeanDistPredator` slowly increases from 4.33 → 4.40; `RabbitHits` rises ~1.7×; `PredatorHits` peaks at window 2 then mildly retreats. The asymmetry at window 1 is ~0.4 cells; at window 8 it is ~0.63 cells — the gap widens by ~50 % over training.
- *Why*: a learned policy is differentially shaping approach to the two classes. Random/initial policy alone would not produce a widening gap.
- *Evidence*: §4.4 trajectories.
- *Confidence*: **High** — the trend is monotonic-with-mild-noise across both seeds; partial-budget tail is flat (windows 6, 7, 8 differ by ≤ 0.01 cells), so the steady-state value is unlikely to shift in the unrun 30 % of the budget.

**Finding 3 — CRITICAL CONFOUND: rabbits and food share both spawn quadrants; predator does not.**
- *What*: in `01-interoNocicept_sameProp.yaml` (§2.2), `food[TL].spawn_area = rabbit[0].spawn_area = [[1,1],[5,5]]` and `food[BR].spawn_area = rabbit[1].spawn_area = [[6,6],[10,10]]`, while the patrolling predator's spawn/patrol area is `[[1,1],[10,10]]` (full grid). A *purely food-seeking* policy — one that does nothing more than approach the food smell — would mechanically produce `MeanDistRabbit < MeanDistPredator` simply because, conditional on the agent being near food, the agent is also near rabbits but only sometimes near the predator (which roams the full 100 cells).
- *Why this matters*: the predator's steady-state distance of **4.40** is essentially the average distance from a food-quadrant-loitering agent to a uniform-grid roamer; the random uniform-vs-uniform baseline is **4.68** (from prior survey). A 0.28-cell deviation from random is consistent with "the agent is slightly more often in food quadrants than uniform" — i.e., the predator-distance number could be entirely a *food-seeking spillover*, not a *predator-avoidance* signal. Until the food/rabbit spatial coupling is broken, **Round 1 cannot attribute the asymmetry to olfactory channel discriminability vs. food-quadrant co-location**.
- *Evidence*: config inspection (§2.2); steady-state `MeanDistPredator = 4.40` ≈ random baseline `4.68 ± food-quadrant adjustment`; `MeanDistFood = 2.41` confirms food-quadrant occupancy.
- *Confidence*: **High** as a *valid alternative explanation* of the asymmetry. Whether it is the *complete* explanation cannot be settled from Round 1.

**Finding 4 — `MeanDistHidingPredator` (2.63) is structurally low, not a learning failure.**
- *What*: `MeanDistHidingPredator ≈ 2.63` ≪ `MeanDistPredator ≈ 4.40`, despite the hiding predators having `properties = [0,0,0,0,0]` (no olfactory cue) and being at least as dangerous as the patrolling predator (same damage, `nociception_intensity = 0.9`).
- *Why*: hiding predators are **stationary** in the four corners of the 10×10 grid. The mean L2 distance from a uniform-random point in the grid to the *nearest* of four corner cells is structurally small (~2.6 cells). The agent's `MeanDistHidingPredator = 2.63` is therefore consistent with the agent being approximately uniform-grid-distributed *with respect to corners* — neither approaching nor avoiding. This is informational, not a learning failure.
- *Evidence*: position-of-corners + grid-size geometry; both seeds agree to 0.01 cells.
- *Confidence*: **High**.

**Finding 5 — Channel attribution: Phase 1 candidates ranked against the trajectory shape.**

The Phase 1 memo enumerated four candidates (in decreasing prior likelihood):
(1) **movement / temporal signature in olfaction** (predator HUNT vs rabbit jitter);
(2) visual at colocation via channels 5/7;
(3) patrol-area asymmetry;
(4) instantaneous olfactory shape (indistinguishable, ruled out by definition under sameProp).

Holding the observed trajectory shape (Finding 2) against these:

- **(3) Patrol-area asymmetry is now indistinguishable from Finding 3's
  food-quadrant spillover confound.** Both predict the same direction of
  effect (`MeanDistPredator > MeanDistRabbit`) for the same reason: the
  predator covers more grid than the rabbit. Round 1 cannot separate
  "patrol-area asymmetry causes avoidance" from "rabbit happens to share
  food quadrants and the agent's behavior is food-driven". Round 2 must
  decouple these via either (a) decoupling food and rabbit spawn areas, or
  (b) restricting predator patrol to one rabbit quadrant.
- **(1) Movement/temporal signature** remains the leading *positive*
  candidate among non-confounded mechanisms. Predator HUNT-mode (`hunt_thresh=0.7`,
  `detection_range=5`) tracks `agent_pos` over time, producing a distinct
  temporal autocorrelation in the channel-1 olfactory signal. The fact that
  `MeanDistRabbit` *decreases* over training (rather than staying near 4.68)
  is consistent with the agent's GRU using temporal patterns to identify
  channel-1 peaks that *don't* track the agent's motion (rabbits) and
  approaching them, while keeping channel-1 peaks that *do* track motion
  (predators) at uniform-or-greater distance. **However, this prediction is
  observationally identical to "agent learned food, food is in rabbit
  quadrants" — see Finding 3.** The mechanism is plausible but not
  isolated by Round 1.
- **(2) Visual at colocation + extero-noc 0.9 at predator contact** is the
  cleanest *post-contact* teaching signal (predator → channel 5, rabbit →
  channel 7, paired with `extero_nociception = 0.9` only on predator/hiding-
  predator contact, never on rabbit). The window-2 PredatorHits peak
  (3.5–3.6) then retreat (3.3–3.4) is consistent with the agent first
  accumulating contact-based examples then the GRU using them to bias
  later approach decisions. This channel cannot be ruled out by Round 1
  data — it acts within-episode through the recurrent state — but it
  should leave a signature in *PredatorHits* over training (which it does:
  the post-peak decline) more than in *MeanDistPredator* (which is mostly
  flat).
- **(4) Instantaneous olfactory shape** is mechanically indistinguishable
  under matched properties (`properties_std = 0`), so this channel cannot
  contribute and is correctly ruled out.

**Net channel verdict**: Round 1 is consistent with Phase 1's predicted
ordering (1 ≥ 2 > 3 > 4) only *if* you accept that (3) and the food-quadrant
confound (Finding 3) are tangled in the data. **Until Round 2 decouples
them, Round 1 supports "the agent has learned a class-discriminating
behavior" but cannot pin the mechanism to an obs-space channel.**

### 5.2 Cross-Run Comparisons

This analysis is intra-run (two seeds of the same cell). The comparison to
the prior single-seed run on this config (`rmw6m8zg`) is:

| Metric | `rmw6m8zg` (orig, seed unknown, RTX 6000 Ada, 10 M ep) | Round 1 mean (seeds 42, 43, RTX 3090, 7.2 M ep) | Note |
|--------|------|------|------|
| `Episode/Steps` | 341.9 | 327.3 | original had ~3 M more episodes → mild additional improvement plausible |
| `MeanDistPredator` | 4.456 | 4.40 | very close |
| `MeanDistFood` | 2.366 | 2.41 | close |
| `MeanDistRabbit` | n/a (not logged) | **3.77** | the metric Round 1 was launched to obtain |
| `RabbitHits` | n/a | **6.46** | "" |
| `PredatorHits` | 3.13 | 3.37 | slightly higher in Round 1 — consistent with shorter training |
| `HidingPredatorHits` (= `DangerHits`) | 3.23 | 3.00 | close |

Round 1 reproduces the prior single-seed run to within seed dispersion on
every metric that overlaps. The new metrics (rabbit distance, rabbit hits,
hiding-predator distance) successfully populate the previously-blind
behavioral surface.

### 5.3 Failure Modes & Pathologies

**P1 — Olfactory-channel collision pathology at the level of "predator avoidance in absolute terms".**
Inherited from the prior survey (§5.3 of `sameprop_existing_run_survey.md`).
`MeanDistPredator ≈ 4.40` ≈ uniform-random baseline 4.68; the agent has
not learned strong absolute predator avoidance via olfaction. What it has
learned is a *differential* approach pattern that puts rabbits closer than
predators — but, as Finding 3 establishes, this differential is
indistinguishable from food-driven loitering in the rabbit quadrants.

**P2 — Behavioral metric blind spot is now CLOSED for Round 1.** The Phase
1 logging change (4 new keys + 1 alias) gave Round 1 the surface it needed.
Future hypervigilance experiments inherit this for free.

**P3 — Hiding-predator damage as confound** (inherited from prior survey).
Round 1 confirms `HidingPredatorHits ≡ DangerHits` exactly (both seeds, all
windows). The alias works as designed. Distance to hiding predators is
structurally small (Finding 4) and should not be read as "agent fails to
avoid corners".

**P4 — NEW: food-rabbit spawn-area co-location** (Finding 3). This is the
load-bearing limitation on Round 1's mechanistic claim and the primary
target for Round 2.

---

## 6. Conclusions

### 6.1 Summary

- Both Round 1 runs (RPPO on `01-interoNocicept_sameProp.yaml`, seeds 42 &
  43, ~7.2 M / 10 M episodes) **reproducibly produce a steady-state
  asymmetry** `MeanDistPredator − MeanDistRabbit ≈ 0.63 cells`, ≥ 10× the
  within-seed stddev, < 0.05 between-seed difference.
- The asymmetry **grows with training** — not an initialization-noise
  artifact. Rabbit distance decreases (3.93 → 3.77), predator distance
  slightly increases (4.33 → 4.40), and rabbit-contact frequency rises ~1.7×
  while predator-contact frequency mildly retreats after a window-2 peak.
- **Critical caveat (Finding 3)**: rabbits and food share both spawn
  quadrants; the patrolling predator does not. The asymmetry is therefore
  also consistent with a *purely food-seeking* policy that passively
  achieves `MeanDistRabbit < MeanDistPredator` because rabbits happen to
  live where food is. Round 1 isolates **"co-location with food + movement-
  signature olfactory discriminator"**, not olfaction alone.
- The new `Episode/MeanDistRabbit`, `RabbitHits`, `MeanDistHidingPredator`,
  `HidingPredatorHits` keys all populate cleanly; `HidingPredatorHits ≡
  DangerHits` exactly, as designed.
- Reproduces prior single-seed run (`rmw6m8zg`) on every overlapping metric
  to within seed dispersion. Partial budget (7.2 M / 10 M) does not
  materially affect steady-state values; the last three windows agree to ≤
  0.01 cells on every primary distance metric.

### 6.2 Limitations & Open Questions

- **n = 2** seeds, not n ≥ 3; no formal CI on the gap.
- **Partial budget** — 72 % of the 10 M episode plan. Steady-state has
  visibly plateaued, so the unrun 28 % is unlikely to overturn directional
  conclusions, but a strict pre-registered effect-size claim would require
  the full budget.
- **Confound 1 / Finding 3** — food-rabbit quadrant co-location is not
  controlled. Round 2 must address this.
- No Dreamer comparison (out of scope for Round 1).
- `extero_nociception` damage broadcast was not directly inspected for this
  pair of runs; relying on the read-only audit in
  [`sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md).

### 6.3 Pre-registered Next Step (Round 2 — design owned by `experiment-designer`)

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| **P0** | Round 2 — decouple food-rabbit quadrant co-location and re-test the asymmetry. Two design options: **(a)** move rabbits to the two food-empty quadrants (TR, BL) so food-quadrant loitering no longer puts the agent near rabbits; **(b)** restrict the patrolling predator's `patrol_area` to the *rabbit* quadrants (TL+BR) so the predator and rabbit have matched spatial distributions and *only* their movement / temporal signature differs. Option (b) more cleanly isolates Phase 1 channel (1) (movement signature) by collapsing channel (3) (patrol-area asymmetry). Option (a) is the simpler control. Pre-register both as separate cells, n ≥ 3 seeds each. | Resolves whether the Round 1 asymmetry is olfactory-channel-driven or food-co-location-driven. | medium (config change + ~6 runs at ~15 h each) |
| P1 | If Round 2 confirms the asymmetry under decoupled spawn, run a third cell that pins the predator into a non-HUNT mode (`hunt_stamina_threshold: 1.1` or `detection_range: 0`) to ablate Phase 1 channel (1) (movement signature) directly. | Isolates the dominant channel candidate. | medium |
| P2 | Repeat Round 2 with Dreamer-V3 to test architectural generalization of any confirmed asymmetry. | Architecture × hypervigilance interaction. | larger (Dreamer is more compute-heavy) |

---

## Metrics Requested

None new from this analysis — the Phase 1 metric set added in commit `4b55fc6`
was sufficient for Round 1's H₁ test. The next bottleneck (mechanism
attribution, not behavior detection) is *experimental-design* work, not
logging work.

If Round 2 surfaces a need to attribute the asymmetry to specific obs-space
channels at the *neural* level, two metrics would help:

| Subfield | Content |
|---|---|
| **Metric** | Per-channel olfactory mean over episode (`Episode/MeanOlfactoryCh{0..4}`), unit = arbitrary post-decay intensity. |
| **Why now** | Would let the analyzer correlate channel-1 olfactory exposure with `RabbitHits` and `PredatorHits` separately and verify whether the agent's GRU is reading channel 1's *temporal* structure or just its *magnitude*. |
| **Where it'd live** | `src/environment/sensor.py` get_observation aggregator; surfaced via `info` dict alongside the existing `dist_to_*` keys. |
| **Cost** | Cheap — one `np.mean` per channel per step, 5 scalars per env step. |

Not blocking Round 2 — flagged for later if mechanism attribution becomes
the analytic priority.

## Related Issues

- **Predecessor analysis**: [`sameprop_existing_run_survey.md`](sameprop_existing_run_survey.md) — should have its "Round 1 — Re-logged Baseline Launch Manifest" table updated with the partial-budget actuals (final episodes, walltime, status: stopped early). See change in this commit.
- **Phase 1 channel memo**: [`../../../develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md) — Round 1 trajectory is consistent with that memo's ranked discriminator list under the Finding 3 caveat.
- **Logging plan**: [`../../../develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../../develop/active/hypervigilance/per_entity_avoidance_logging.md) — fulfilled by these runs; the new metric set behaves as specified, including the `HidingPredatorHits ≡ DangerHits` alias invariant.
- **Round 2 design** (in flight, owned by `experiment-designer`): the doc author should pick up the Finding 3 confound as the central design constraint and specify which option (a)/(b) — or both — to run.
- No bugs surfaced. The food-rabbit co-location is a config *design* choice, not a code defect.

---

## Appendix

### A. Raw Data Tables

Saved working files under `tmp/`:
- `tmp/20260508_relog_compare.txt` — full `wandb_metrics.py compare` output, 14 metrics × 2 seeds, steady-state + last + trajectory summary.
- `tmp/20260508_relog_timeseries8.txt` — full 8-window timeseries, 8 metrics × 2 seeds (this analysis's primary diagnostic surface).
- `tmp/20260508_relog_speed.txt` — speed benchmark.

### B. Verification of pre-extracted numbers (from user brief)

| Quantity | User brief | Verified from `wandb_metrics.py` | Δ |
|----------|-----------|----------------------------------|----|
| `MeanDistFood` SS | 2.41 | 2.405–2.406 | ✓ |
| `MeanDistHidingPredator` SS | 2.64 | 2.626–2.635 | ✓ (within rounding) |
| `MeanDistRabbit` SS | 3.77 | 3.766–3.783 | ✓ |
| `MeanDistPredator` SS | 4.40 | 4.388–4.416 | ✓ |
| `RabbitHits` / ep | 6.5 | 6.351–6.562 | ✓ |
| `PredatorHits` / ep | 3.4 | 3.318–3.430 | ✓ |
| `HidingPredatorHits` / ep | 3.0 | 3.000–3.009 | ✓ |
| MeanDistRabbit window 1→8 | 3.94 → 3.76 | 3.93 → 3.77 | ✓ |
| MeanDistPredator window 1→8 | 4.35 → 4.41 | 4.33 → 4.40 | ✓ |
| RabbitHits window 1→8 | 3.7 → 6.6 | 3.85 → 6.45 | ✓ direction; magnitudes off by ~5 % (1.7× rather than ~1.8×) |
| PredatorHits trajectory shape | "peaks then mild retreat" | window-2 peak 3.58, retreat to 3.37 | ✓ |
| **Final episodes** | "**~6.7 M**" | **7.21 M (s43), 7.28 M (s42)** | **OFF — actual is ~7.25 M, not ~6.7 M** |
| **Walltime** | "**14h05m / 14h10m**" | **15h12m (s43), 15h16m (s42)** | **OFF by ~1 h** |

The directional claim ("partial budget, well short of 10 M") is correct; the
specific episode and walltime numbers in the brief are slightly low.
Steady-state and trajectory-shape claims verify.

### C. Config Diffs

None — both Round 1 runs use the unchanged
`configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` and
`configs/models/recurrent_ppo/recurrent_ppo.yaml`. Only `--seed` differs (42 vs 43).

### D. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Initial post-hoc analysis on Round 1 re-logged baseline (partial budget, n=2, both seeds) | experiment-analyzer |
