---
title: "SameProp Round 2 — confound control (food decoupling) + movement-signature ablation"
topic: hypervigilance
status: active
created: 2026-05-08
last_updated: 2026-05-08
phase: 1
wandb_tag: "hypervigilance-round2"
develop_link: "../../../develop/active/hypervigilance/sameprop_discriminating_channels.md"
supersedes: ""
---

# SameProp Round 2 — confound control + movement-signature ablation

> **Status**: PLANNED — Launchable after `env-config-auditor` sign-off + user authorization.
> **Date**: 2026-05-08
> **Author**: experiment-designer
> **Related**:
> - Round 1 launch + analysis (parent): [`sameprop_existing_run_survey.md`](sameprop_existing_run_survey.md)
> - Discriminating-channels memo: [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md)
> - Per-entity logging that enables this round: [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../../develop/active/hypervigilance/per_entity_avoidance_logging.md)
> - Round 1 baseline config (modified diff base): [`configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`](../../../../configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml)

---

## 0. Round 1 recap (verified, both seeds agree at ≥6.7M episodes)

Under matched olfactory `properties [0,1,0,0,0]` for predator and rabbit, RPPO appears to discriminate:

| Metric | Final (≈6.7M ep) | Trajectory |
|---|---|---|
| MeanDistRabbit | ≈3.77 | 3.94 → 3.76 (decreasing) |
| MeanDistPredator | ≈4.40 | 4.35 → 4.41 (slowly increasing) |
| RabbitHits / ep | ≈6.5 | 3.7 → 6.6 (near-doubled) |
| PredatorHits / ep | ≈3.4 | peaks then mildly retreats |

**Round-1 confound flagged in parallel by experiment-analyzer:**
food TL+BR ↔ rabbit TL+BR, while predator roams full grid `[[1,1],[10,10]]`. So the apparent class discrimination MeanDistRabbit < MeanDistPredator could be **food-seeking spillover** (agent approaches food → rabbit happens to be in the same quadrant) rather than learned olfactory class identity.

Round 2 directly attacks this confound.

---

## 1. Research Question

**Q (cell C):** When food spawn areas are decoupled from rabbit spawn areas (food → TR + BL; rabbits remain TL + BR), does the Round-1 finding `MeanDistRabbit < MeanDistPredator` persist?

**Q (cell A1):** When predator HUNT mode is structurally disabled and patrol area is shrunk to match a rabbit's quadrant — eliminating the ranked-dominant pre-contact discriminator (the predator's `agent_pos`-tracking movement signature, per `sameprop_discriminating_channels.md` §4) — does RPPO retain class-level avoidance under matched olfactory properties?

> **H₀ (C):** The Round-1 effect was driven by the food/rabbit-quadrant collinearity. With food decoupled, MeanDistRabbit and MeanDistPredator converge (|Δ| ≤ 0.1 cells) and RabbitHits / PredatorHits converge correspondingly. **There is no genuine class discrimination** under sameProp.
>
> **H₁ (C):** The Round-1 effect persists after decoupling: MeanDistRabbit < MeanDistPredator − 0.3 cells with both seeds agreeing on the sign. **Olfactory channel ambiguity does not prevent learned class avoidance** under sameProp; some other channel (movement signature, visual ch.5/ch.7 at contact, extero-noc) carries the class label.
>
> **H₀ (A1):** With HUNT disabled and patrol-area matched, predator and rabbit are kinematically identical. The agent has no remaining pre-contact discriminator and falls back on only post-contact teaching (visual + extero-noc). Predicted: PredatorHits ≈ RabbitHits within 1.0 / ep, and MeanDistPredator ≈ MeanDistRabbit within 0.3 cells. **Movement signature was the dominant discriminator.**
>
> **H₁ (A1):** Even with HUNT disabled, RPPO keeps PredatorHits < RabbitHits − 1.5 / ep at convergence. **Post-contact visual+extero-noc is sufficient** for class learning via the recurrent state.

A clean falsification of either cell does not depend on the other; the two cells together produce a 2x2 over {confound resolved? × movement-signature ablated?} when read against Round 1.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Cell C | Cell A1 |
|---|---|---|
| Food spawn quadrants | TR + BL (decoupled from rabbits) | TL + BR (Round-1 default) |
| Predator `detection_range` | 5 (Round-1 default) | **0** |
| Predator `hunt_stamina_threshold` | 0.7 (Round-1 default) | **1.1** (HUNT unreachable) |
| Predator `patrol_area` | `[[1,1],[10,10]]` (Round-1 default) | **`[[1,1],[5,5]]`** (matches rabbit TL) |
| Predator `spawn_area` | `[[1,1],[10,10]]` (Round-1 default) | **`[[1,1],[5,5]]`** |

All other fields fixed at Round-1 baseline (`01-interoNocicept_sameProp.yaml`). See §2.2 + Appendix B.

### 2.2 Controlled Variables

- Olfactory `properties` for predator + rabbit: `[0.0, 1.0, 0.0, 0.0, 0.0]` (the sameProp condition itself).
- `properties_std`: zero for all entities (no per-step olfactory noise).
- `perceptual_noise.enabled: false` (sigmas in YAML are inert).
- 4 hiding_predators in 4 quadrants, properties `[0,0,0,0,0]`.
- Rabbits: 2x count, TL + BR quadrants, move_interval 1, `properties [0,1,0,0,0]`.
- Bushes: 5 in TR, 5 in BL (hides_agent).
- Rocks: 3 in each of 4 quadrants, damage [1,5].
- Sensors: olfaction radius 20, decay 2.0, vector_size 5; visual range 0; nociception enabled; intero-noc enabled (tau 3.0, kernel 12); proprioception on; injury_observable false; nutrition_observable false.
- Body: full Round-1 homeostatic reward, max_steps 500, metabolic_cost 1.0.
- Agent: `configs/models/recurrent_ppo.yaml` (RPPO, identical to Round 1).
- Step budget: 10,000,000 episodes (full Round-1 budget; Round 1's 6.7M was an early checkpoint, not a hard ceiling).
- Parallel envs: 128 (Round-1 default).
- Per-entity metrics (`Episode/MeanDistRabbit`, `Episode/RabbitHits`, etc.) logged via the recently-merged `per_entity_avoidance_logging` (commit `4b55fc6`, branch `v1.3`).

### 2.3 Confounds & Limitations

| Confound | Affected | Severity | Mitigation |
|---|---|---|---|
| n = 2 seeds (42, 43) — same as Round 1 | both cells | Medium | Cross-round comparability with Round 1 is the priority; both seeds agreed at 6.7M in Round 1, so seed-stable signs are detectable. Add seeds 44/45 in Round 3 if Round 2 results are seed-divergent. |
| 2 GPUs, 1 cell × 1 seed each (no within-cell seed-pair this round) | both cells | Medium | This is the unavoidable consequence of running two cells in parallel with 2 GPUs. **Round-2 results are interpreted only against Round-1's 2-seed agreement** — a sign reversal in even 1 seed of Round 2 vs. consistent 2-seed Round 1 is informative. If a cell's verdict is borderline, schedule the second seed on the next GPU rotation. |
| Cell A1 keeps predator damage on contact | A1 | Low | This is intentional — we want post-contact teaching available so we can isolate "is movement signature OR is post-contact teaching what drives avoidance under sameProp." |
| Cell A1 quadrant-locks predator to TL only (one of two rabbit quadrants) | A1 | Low | Co-occupancy with TL rabbit is *desired* — it makes the discrimination maximally hard for the agent. BR rabbit then acts as an internal control: if A1 still discriminates, MeanDistRabbit_BR vs MeanDistRabbit_TL should differ if the agent uses quadrant identity rather than movement. (Per-rabbit indexing is not currently logged — see §7 Metrics Requested.) |
| Cell C still shares quadrants with hiding_predators in all 4 corners | C | Low | Hiding predators have `properties [0,0,0,0,0]` — no olfactory leak. They are a non-cue damage source, identical to Round 1. |
| `random_start_pos: true` means agent spawn distribution differs across episodes | both | Low | Same condition as Round 1; cross-round comparability preserved. |
| Round 1 total step count was reported as ~6.7M episodes; Round 2 targets 10M | both | Low | If late-training (>6.7M) shows the Round-1 effect *strengthens* in cell C, that argues against H₀(C). If it weakens, Round 1's verdict was likely premature. |

## 3. Launch Manifest

**Hardware**: Node 112, 2 GPUs (cuda:0, cuda:1). 1 cell per GPU, 1 seed per cell. Total: 2 runs.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | C-decoupleFood | `hypervigilance-round2-C-seed42_n112_gpu0` | hypervigilance | prod | 42 | 112 | cuda:0 | 2026-05-08T14:17:50 | 0u266oj5 | logs/20260508_141750.log |
| 2 | running | A1-passivePredator | `hypervigilance-round2-A1-seed43_n112_gpu1` | hypervigilance | prod | 43 | 112 | cuda:1 | 2026-05-08T14:20:15 | 27svrmhv | logs/20260508_142015.log |

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` | `configs/models/recurrent_ppo.yaml` |
| 2 | `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` | `configs/models/recurrent_ppo.yaml` |

### 3.2 Launch Commands (for `training-runner` reference)

Both invoked via `run_command.py` with the standard project Python interpreter `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. The commands the runner will issue are (modulo the `train_command-agent.sh` template):

```bash
# Run 1 — Cell C — node 112 cuda:0 — seed 42
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 42 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round2-C-seed42_n112_gpu0" \
  --tag "hypervigilance-round2-C-seed42_n112_gpu0"

# Run 2 — Cell A1 — node 112 cuda:1 — seed 43
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 43 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round2-A1-seed43_n112_gpu1" \
  --tag "hypervigilance-round2-A1-seed43_n112_gpu1"
```

The seed/GPU pairing is a deliberate choice: cell C on the same seed (42) as Round-1 R1-seed42 (`rg5nl1ov`) makes pairwise R1↔R2-C comparison cleanest under matched RNG; cell A1 takes the alternate seed (43) so any A1 verdict robust across seeds 42 and 43 (across cells) is harder to dismiss as a one-seed fluke.

## 4. Pre-Registered Analysis Plan

### 4.1 Primary metrics

Computed on the **last 10% window** of training (episodes ≈9.0M – 10.0M), mean across all 128 envs of the 1880 history records in the window.

| Metric | Cell C | Cell A1 |
|---|---|---|
| `Episode/MeanDistRabbit` (cells, L2) | primary | primary |
| `Episode/MeanDistPredator` (cells, L2) | primary | primary |
| `Episode/RabbitHits` (count / ep) | primary | primary |
| `Episode/PredatorHits` (count / ep) | primary | primary |
| `Episode/Steps` (survival) | secondary (project headline) | secondary |
| Δ ≡ MeanDistPredator − MeanDistRabbit | derived primary | derived primary |
| ΔH ≡ RabbitHits − PredatorHits | derived primary | derived primary |

### 4.2 Confirmation / refutation thresholds (cell C)

| Outcome | Thresholds | Implies |
|---|---|---|
| **H₀ (C) confirmed** (no genuine discrimination — Round 1 was the food confound) | \|Δ\| ≤ 0.1 cells **AND** \|ΔH\| ≤ 1.0 / ep | Round-1's apparent rabbit avoidance was food spillover. Sub-question for Round 3. |
| **H₁ (C) confirmed** (genuine discrimination survives) | Δ ≥ 0.3 cells (predator further than rabbit) **AND** ΔH ≥ 1.5 / ep (more rabbit hits than predator hits) | Class discrimination is real under sameProp. Movement signature / visual+extero is the channel. Pair with cell A1 to isolate. |
| **Inverted** (rabbits are now further than predators) | Δ ≤ −0.3 cells | The decoupling forced rabbits to a corner the agent learned was unreachable / unrewarding; alternative explanation needs Round 3. |
| **Ambiguous** | 0.1 < \|Δ\| < 0.3 | Add seeds 44/45 in Round 3 before drawing a verdict. |

### 4.3 Confirmation / refutation thresholds (cell A1)

| Outcome | Thresholds | Implies |
|---|---|---|
| **H₀ (A1) confirmed** (movement signature was dominant) | PredatorHits ≈ RabbitHits, \|ΔH\| ≤ 1.0 / ep **AND** \|MeanDistPredator − MeanDistRabbit\| ≤ 0.3 cells | The Round-1 discrimination collapses without HUNT. Movement signature is the dominant channel. |
| **H₁ (A1) confirmed** (post-contact teaching alone is sufficient) | PredatorHits < RabbitHits − 1.5 / ep at convergence | Visual ch.5 + extero-noc 0.9 at contact, mediated by GRU, suffices to drive class avoidance even without HUNT. |
| **Both worse than Round 1** | Survival drops below Round-1 floor (≈340 steps) | Quadrant restriction itself harmed the agent (possibly food-search interference); read as null. |

### 4.4 Temporal evolution checks (mandatory)

For each cell, partition training into 10 equal episode-count windows. Plot:
1. `Episode/MeanDistRabbit` and `Episode/MeanDistPredator` per window (overlay).
2. `Episode/RabbitHits` and `Episode/PredatorHits` per window (overlay).
3. `Episode/Steps` per window.

A cell is taken seriously only if its primary-metric verdict is **stable across the last 3 windows** (i.e., not driven by a single transient).

### 4.5 Cross-cell + cross-round contrasts

| Comparison | What it isolates |
|---|---|
| C (R2) vs Round-1 R1-seed42 (`rg5nl1ov`) | Effect of food-quadrant decoupling, holding seed = 42. |
| A1 (R2) vs Round-1 R1-seed43 (`6ks4bjbq`) | Effect of HUNT ablation + patrol-area match, holding seed = 43. |
| C vs A1 (R2) | Independent disconfirmations of two confounds. |

### 4.6 Random-policy baseline (anchor)

Round-1 analysis pre-computed: uniform-random `MeanDistPredator` ≈ 4.684 on the 10x10 grid. This is the random-policy floor for *any* MeanDist metric on this grid. A cell whose MeanDistRabbit and MeanDistPredator both sit at ≈4.68 is at the random-policy baseline and has not learned class discrimination at all.

## 5. Failure-Mode Catalog (pre-decided)

| Failure | Resolution |
|---|---|
| Training instability (NaN, value explosion) | Refutes the run, not the hypothesis. Re-launch with same seed; if it recurs, surface to `senior-developer` as a real config bug. |
| Cell A1: agent never approaches predator (because predator is in TL, food now in TL+BR, agent hangs out in BR rabbit corner) → `MeanDistPredator` looks high purely by spatial separation | Read `Episode/Steps` and `Episode/FoodEaten` together. If the agent is surviving by camping food and never visits TL, A1's verdict is moot — it tells us nothing about discrimination because the agent never had the chance. Schedule Round 3 with food forced in all 4 quadrants. |
| Cell C: agent learns to camp the TR or BL food + bush corners (where bushes hide it from hiding_predators) and rarely visits TL/BR rabbit quadrants → `MeanDistRabbit` looks artificially high | Same diagnostic as A1: cross-check with quadrant-occupancy if logged, otherwise interpret cautiously and schedule Round 3. |
| Saturation: `Episode/Steps` plateaus at ≈340 (Round-1 ceiling) without further differentiation | Null result. Insufficient horizon. Plan Round 3 at 20M episodes if Round-1's plateau is near a true ceiling rather than an under-trained one. |
| Seed-dependent noise: 1 seed per cell flips relative to Round-1's 2-seed verdict | Add a 2nd seed for the affected cell on the next GPU rotation; do **not** declare the cell verdict from a single seed unless thresholds in §4.2/§4.3 are exceeded by a wide margin (>2x). |

## 6. Predicted Outcomes (pre-registration)

Designer's prior on the outcomes — recorded so the post-hoc reading does not adapt:

- **Cell C (most likely):** Δ shrinks substantially (Round-1 0.6 cells → R2 ~0.2–0.4 cells), but does not vanish. Olfactory class identity is not the only channel — visual ch.5 + extero-noc + movement signature still carry information. Most likely outcome: **partial confound** — Round-1 verdict was overstated but not entirely an artifact.
- **Cell A1 (less likely):** ΔH (rabbit minus predator hits) shrinks but stays positive (RabbitHits > PredatorHits by ≥1 / ep), because visual ch.5 + extero-noc 0.9 at contact still teach the agent to avoid the predator quadrant via the GRU, even without HUNT. **H₁(A1)** outcome.

If both predictions are correct, the contribution decomposition under sameProp is roughly: ~30–50% food-quadrant collinearity + ~30–40% movement signature + ~20–30% post-contact visual/extero teaching. Round 3 (cell B: olfaction-only, sensors stripped) would close this.

## 7. Metrics Requested (optional — not blocking)

These would sharpen the analysis but are not required for the verdict thresholds in §4.

| Subfield | Content |
|---|---|
| **Metric (1)** | `Episode/MeanDistRabbit_TL` and `Episode/MeanDistRabbit_BR` (per-rabbit-instance distance, cells). |
| **Why now** | In cell A1, the predator is locked to TL. If the agent learns "TL is dangerous, BR is safe," MeanDistRabbit_TL would rise while MeanDistRabbit_BR falls, even though both rabbits emit identical olfactory signals. The aggregated `MeanDistRabbit` would average these and look like uniform avoidance. Per-rabbit indexing distinguishes "class-level olfactory avoidance" from "quadrant-level spatial learning." |
| **Where it'd live** | `src/environment/core.py` near the existing per-entity blocks (~line 491 of `dist_to_neutral`). Same pattern as `dist_to_food` / `dist_to_pred` per-instance variants if any exist, otherwise add `info['mean_dist_neutral_per_idx']`. |
| **Cost** | Cheap — `jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)` already computed; just don't reduce. |

| Subfield | Content |
|---|---|
| **Metric (2)** | `Episode/QuadrantOccupancy_{TL,TR,BL,BR}` — fraction of episode steps the agent spends in each quadrant. |
| **Why now** | Disambiguates "agent avoids predator class" (low PredatorHits anywhere) from "agent avoids predator quadrant" (low TL occupancy in cell A1). Critical for the failure modes in §5. |
| **Where it'd live** | `src/environment/core.py` per-step occupancy counter, surfaced via `info`. |
| **Cost** | Cheap — 4 boolean masks + reduce. |

If accepted, hand off to `senior-developer` → `developer` via `feature-workflow`. **Round 2 launches without these.**

## 8. Launchable Status

**Configs**: complete ✅
- `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml`
- `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml`

**Pre-flight gates** (must pass before runner launches):
1. ⏸ `env-config-auditor` — modality order intact, mandatory keys present, no `properties`/`property` typo, predator-side keys consistent.
2. ⏸ User authorization — explicit "go" after auditor sign-off.

**Runner handoff**: `training-runner` will fill the actual columns of §3 (Status, Node, GPU, Launched at, WandB run ID, Log path) at launch time.

---

## Appendix

### A. Cell mapping table

| Round-1 channel rank (`sameprop_discriminating_channels.md`) | Round-2 cell that ablates it | Configured how |
|---|---|---|
| §4 Movement / temporal signature in olfaction (predator HUNT tracks `agent_pos`; rabbit jitters) | Cell A1 | `hunt_stamina_threshold: 1.1` (rested_enough never true) + `detection_range: 0` (HUNT condition unreachable) |
| §4 Patrol-area asymmetry (predator full-grid; rabbit quadrant) | Cell A1 | `patrol_area: [[1,1],[5,5]]` and `spawn_area: [[1,1],[5,5]]` (predator now matches rabbit TL) |
| §1 Olfactory instantaneous shape (indistinguishable when properties match) | NOT ablated | properties remain `[0,1,0,0,0]` for both — that is the *condition under test* |
| §3 Visual ch.5 vs ch.7 at colocation | NOT ablated (intentionally, for both cells) | `visual_sensor_enabled: true`, range 0 — present in both A1 and C; isolates the *contribution* of post-contact teaching |
| §2 Extero-noc 0.9 at contact | NOT ablated | `extero_nociception` active for predator hits in both A1 and C |
| **NEW: food-quadrant collinearity (analyzer flag)** | **Cell C** | food → TR + BL only; rabbits remain in TL + BR; food and rabbits share NO quadrant |

### B. Config diff vs Round-1 baseline

**Cell C — `02-sameProp_R2_decoupleFood.yaml` vs `01-interoNocicept_sameProp.yaml`**:
```diff
-  food TL count: 2     →  food TL count: 0
-  food TR count: 0     →  food TR count: 2
-  food BR count: 2     →  food BR count: 0
-  food BL count: 0     →  food BL count: 2
   (everything else identical)
```

**Cell A1 — `02-sameProp_R2_passivePredator.yaml` vs `01-interoNocicept_sameProp.yaml`**:
```diff
   predator:
-    spawn_area: [[1,1],[10,10]]      →  [[1,1],[5,5]]
-    patrol_area: [[1,1],[10,10]]     →  [[1,1],[5,5]]
-    detection_range: 5               →  0
-    hunt_stamina_threshold: 0.7      →  1.1
   (all other predator + non-predator fields identical)
```

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Initial pre-registered design for Round 2; 2 cells × 1 seed each on node 112 cuda:0/cuda:1 | experiment-designer |
| 2026-05-08 | Launched both cells; manifest rows updated with WandB IDs and log paths; CIFS-stale duplicate noted (see §D) | training-runner |

### D. CIFS-attribute-cache stale-read incident (2026-05-08)

A second `run_command.py` invocation for Cell A1 read the stale (Cell C) version of `train_command-agent.sh` from node 112's CIFS cache and spawned a duplicate Cell C run:

- **Duplicate WandB run ID**: `f5d933ro` (`hypervigilance-round2-C-seed42_n112_gpu0`, cuda:0)
- **Duplicate log**: `logs/20260508_141905.log`

The correct Cell A1 run was subsequently launched inline (bypassing the script file) and is confirmed running as `27svrmhv`.

**Action required by user**: SIGTERM the duplicate process on node 112 that is writing to `logs/20260508_141905.log` (WandB run `f5d933ro`). It is a second Cell C instance competing on cuda:0 with the authoritative run `0u266oj5`. Delete or mark `f5d933ro` as junk in WandB.
