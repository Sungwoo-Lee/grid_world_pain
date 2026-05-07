---
title: "DreamerV3 hypervigilance diagnostic battery — 4 parallel ablations on Node 114"
topic: continual_learning
status: active
created: 2026-05-07
last_updated: 2026-05-07T22:11:41
phase: 1
wandb_tag: "dreamer_diagnostic_battery"
develop_link: "../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md"
---

# DreamerV3 hypervigilance diagnostic battery — 4 parallel ablations on Node 114

> **Status**: PLANNED (configs landed; awaiting env-config-auditor + user authorization to launch)
> **Date**: 2026-05-07
> **Author**: experiment-designer
> **Related**:
> - Failure diagnosis (the question this battery interrogates):
>   [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md)
> - Just-completed curriculum run analyzer's working file (input evidence):
>   [`tmp/20260507_194843_dreamer_curriculum_progress_check.md`](../../../../tmp/20260507_194843_dreamer_curriculum_progress_check.md)
> - Continual-learning feature contract:
>   [`docs/develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../../../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)
> - Schema-add plan for `cont_loss_weight`:
>   [`docs/develop/active/diagnosis/dreamer_cont_loss_weight_knob.md`](../../../develop/active/diagnosis/dreamer_cont_loss_weight_knob.md)
> - Reference experiments (precedents):
>   - 3-stage curriculum: [`DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md`](./DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md)
>   - rPPO 2-stage demo: [`CONTINUAL_5X5_NOPRED_TO_PREDINT3_DEMO.md`](./CONTINUAL_5X5_NOPRED_TO_PREDINT3_DEMO.md)

> **Note**: 4 single-seed runs in parallel on cuda:0..3 of Node 114. Each
> result is observational (no statistical claims). The deliverable is a
> 2x2 contrast that tells the user which of {curriculum, levers, dense
> homeostatic reward, smoother ramp} is actually doing work.

---

## 1. Research Question

The 5x5 hypervigilance task `01-5X5_PredInterval3_NutGain18` has now been
run three ways under DreamerV3:

1. **Cold-start failing baseline (`qont5dac`)** — Hafner-default agent, no
   curriculum, no levers. Survival flat at ~30 steps for 27.1 M env-steps;
   actor entropy stuck at ~ln(6); `mean_advantage` constant at -0.19. This is
   the documented hypervigilance signature ([`dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md)).
2. **Curriculum + levers (`95mpudwp`)** — 3-stage food→predator_slow→
   predator_full curriculum + entropy_scale 1e-3 + cont_loss_weight 5.0. The
   tmp progress doc shows Stages 0 and 1 produced **real learning** (mean_advantage
   diversified to -0.09 +/- 0.031, FoodEaten +2.6x, Term_Injury fell to 64% in
   Stage 1) — but Stage 2 (full predator) **collapsed bit-for-bit back to the
   `qont5dac` signature** within iterations of the boundary, and stayed there
   for the next 19.6 M env-steps.
3. **rPPO baseline** (`7t1wskpk`) — solves the same task cold-start to
   ~470/500 mean survival by ~17 M env-steps with no curriculum or knobs.

The asymmetric outcomes — rPPO solves cold; DreamerV3 fails cold; DreamerV3
fails *with* curriculum-and-levers despite Stages 0/1 unfreezing the actor —
mean the failure on the full-task transfer is structural and not yet localized
to a single factor. This battery asks:

> **Which of curriculum, lever-package, dense homeostatic reward, or step-size
> of the curriculum's difficulty cliff is responsible for DreamerV3's failure
> on the full task?**

The four parallel single-seed runs together form a 2x2-style ablation that
isolates each factor:

| Cell | Curriculum | Levers (entropy_scale=1e-3, cont_loss_weight=5.0) | Reward | Cliff (Stage→Full) |
|------|:---:|:---:|:---:|:---:|
| **A — Lever-only** | OFF (single-stage on the failing anchor) | ON | dense (homeostatic) | n/a (no curriculum) |
| **B — Curriculum-only** | ON (3-stage) | OFF (Hafner defaults) | dense (homeostatic) | 8 → 3 (large) |
| **C — No-homeostatic** | OFF (single-stage on no-homeostatic anchor) | OFF (Hafner defaults) | sparse (eat=+1, death=-100) | n/a (no curriculum) |
| **D — Smoother ramp** | ON (4-stage) | ON (entropy_scale=1e-3, cont=5.0) | dense (homeostatic) | 8 → 5 → 3 (halved) |

The matrix is **not orthogonal** by intent — it's the smallest set of cells
that pairwise discriminates the four factors against the previously-run
baselines (`qont5dac` and `95mpudwp`):

- **A vs `qont5dac`**: isolates lever effect (no curriculum in either).
- **B vs `95mpudwp`**: isolates lever effect *given* a curriculum.
- **C vs `qont5dac`**: isolates the homeostatic-reward factor.
- **D vs `95mpudwp`**: isolates the cliff-step-size factor (both have curriculum + levers).

## 2. Pre-registered Hypotheses (per experiment)

### A — Lever-only

> **H₀(A)**: With levers (entropy_scale=1e-3, cont_loss_weight=5.0) but no
> curriculum, DreamerV3 still fails — Stage 3 final-1000-ep mean survival ≈
> qont5dac floor (~30 steps).
>
> **H₁(A)**: Levers alone unlock learning — final-1000-ep mean survival > 100
> steps **without** any pre-training stages.

Significance: refutes/confirms the diagnosis's "levers are necessary but not
sufficient" framing. If H₁(A) holds, the curriculum was an unnecessary
complication. If H₀(A) holds, the curriculum + lever package's Stage-1
learning was lever-driven only when paired with the easier-task warm-up.

### B — Curriculum-only

> **H₀(B)**: Without levers (Hafner defaults: entropy_scale=3e-4,
> cont_loss_weight=1.0), the 3-stage curriculum fails to break out of the
> hypervigilance regime in any stage; Stage 0 progresses similarly to
> 95mpudwp's Stage 0 only if the un-levered actor has enough entropy to
> explore.
>
> **H₁(B)**: Curriculum alone unlocks Stage 0/1 learning (similar to 95mpudwp)
> but Stage 2 still collapses — survival < 100 in the final 1000 ep — confirming
> levers were not what produced the Stage 0/1 progress.
>
> **H₂(B) (third option, less likely)**: Curriculum-only succeeds end-to-end
> (Stage 2 final survival > 100). Would mean the levers were the source of the
> Stage-2 collapse in 95mpudwp, not the difficulty cliff.

### C — No-homeostatic

> **H₀(C)**: With `use_homeostatic_reward: false`, DreamerV3 still fails —
> survival ~30 steps. The reward-distribution change does not matter.
>
> **H₁(C)**: With sparse reward (+1 on eat, -death_penalty on death), the
> reward head's asymmetry resolves: positive_buffer fills meaningfully (the
> failing baseline saw `FoodEaten ≈ 0.2/episode`, and on the +1 sparse
> regime each eat now produces a positive-reward step), AND survival > 100
> steps in the final 1000 ep. Confirmation of the diagnosis's
> "near-deterministic dense reward distribution underfits everything except
> the modal value" framing (§4.3.2).
>
> **User-flagged caveat**: positive_buffer.size **may be very small or zero**
> if the random-walk policy never eats. This is a known consequence — the
> positive_buffer gates on per-step reward > 0 (`train.py:1356/1399`), and
> only the +1.0 eat step is positive in this regime. A useful negative
> result is "positive_buffer empty AND survival flat" which would show the
> sparse-reward regime is not even reached without a working bootstrap.

### D — Smoother ramp

> **H₀(D)**: An intermediate Stage 03 (predator_medium: move_int=5,
> attack_delay=5, damage [10,30], detection_range=2) does NOT bridge the
> Stage02→Stage04 cliff. Stage 04 (the full task) still collapses to ~30 steps
> of survival.
>
> **H₁(D)**: The intermediate stage allows transfer — Stage 04 final-1000-ep
> mean survival > 100 steps. Confirmation of the "Stage 02→Stage 03 cliff was
> too large" interpretation in the 95mpudwp tmp doc.

## 3. Launch Manifest

System-of-record for every training run in this experiment. **Owned jointly:**
- `experiment-designer` writes the planned columns (this commit).
- `training-runner` fills the actual columns at launch time.
- `experiment-analyzer` reads the table to find WandB folders.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | WandB URL | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|-----------|----------|
| 1 | running | A — lever-only | `dreamer_ablation_lever_only_s0` | `dreamer_diagnostic_battery` | `ablation` | 0 | 114 | cuda:0 | 2026-05-07T21:34:10 | `2yti69a5` | https://wandb.ai/sungwoolee/grid_world_pain/runs/2yti69a5 | `logs/20260507_213410_dreamer_ablation_lever_only_s0.log` |
| 2 | running | B — curriculum-only | `dreamer_ablation_curriculum_only_s0` | `dreamer_diagnostic_battery` | `ablation` | 0 | 114 | cuda:1 | 2026-05-07T22:05:03 | `d8fyau0c` | https://wandb.ai/sungwoolee/grid_world_pain/runs/d8fyau0c | `logs/20260507_220503.log` |
| 3 | running | C — no-homeostatic | `dreamer_ablation_no_homeostatic_s0` | `dreamer_diagnostic_battery` | `ablation` | 0 | 114 | cuda:2 | 2026-05-07T22:09:41 | `4fctrt42` | https://wandb.ai/sungwoolee/grid_world_pain/runs/4fctrt42 | `logs/20260507_220941_dreamer_ablation_no_homeostatic_s0.log` |
| 4 | running | D — smoother-ramp | `dreamer_ablation_smooth_ramp_s0` | `dreamer_diagnostic_battery` | `ablation` | 0 | 114 | cuda:3 | 2026-05-07T22:11:41 | `q3ph89rq` | https://wandb.ai/sungwoolee/grid_world_pain/runs/q3ph89rq | `logs/20260507_221141.log` |

**WandB tags (extra, not the run name)**: `["dreamer", "diagnostic", "ablation", "5x5"]` — applied via the standard `--wandb-tags` channel if available, or post-hoc on the WandB UI side. The Tag (= wandb-name) above is the unique run identifier used by `train.py`'s naming logic.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Stages / Configs-dir | Schedule | Env config | Agent config | Pre-existing inputs | NEW files written by this experiment |
|-----|---|---|---|---|---|---|
| A | n/a (single-stage) | n/a | `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml` (existing) | `configs/models/dreamer_v3_curriculum.yaml` (existing) | env + agent | none |
| B | `configs/experiment/dreamer_curriculum/` (existing dir) | `configs/continual/dreamer_curriculum_food_then_predator.yaml` (existing) | (per-stage; from configs-dir) | `configs/models/dreamer_v3.yaml` (existing) | env stages + schedule + agent | none |
| C | n/a (single-stage) | n/a | `configs/experiment/dreamer_diagnostic/03_predator_full_no_homeostatic.yaml` (**NEW**) | `configs/models/dreamer_v3.yaml` (existing) | agent | env config |
| D | `configs/experiment/dreamer_diagnostic/curriculum_smooth/` (**NEW dir, 4 NEW files**) | `configs/continual/dreamer_curriculum_smooth_4stage.yaml` (**NEW**) | (per-stage; from configs-dir) | `configs/models/dreamer_v3_curriculum.yaml` (existing) | agent | 4 env stages + schedule |

**6 NEW files** written by this experiment (paths are absolute):

1. `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_diagnostic/03_predator_full_no_homeostatic.yaml`
2. `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_diagnostic/curriculum_smooth/01_food_only.yaml`
3. `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_diagnostic/curriculum_smooth/02_predator_very_slow.yaml`
4. `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_diagnostic/curriculum_smooth/03_predator_medium.yaml`
5. `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/experiment/dreamer_diagnostic/curriculum_smooth/04_predator_full.yaml`
6. `/media/nas01/projects/Interoceptive-AI/grid_world_pain/configs/continual/dreamer_curriculum_smooth_4stage.yaml`

### 3.2 Exact launch commands

> All four runs share `--num-envs 16 --seed 0 --log-interval 50
> --wandb-group dreamer_diagnostic_battery --wandb-job-type ablation`.

#### Cell A — Lever-only (cuda:0)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml \
  --agent_config configs/models/dreamer_v3_curriculum.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group dreamer_diagnostic_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_ablation_lever_only_s0" \
  --tag "dreamer_ablation_lever_only_s0"
```

#### Cell B — Curriculum-only (cuda:1)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --configs-dir configs/experiment/dreamer_curriculum/ \
  --continual-schedule configs/continual/dreamer_curriculum_food_then_predator.yaml \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 16 \
  --seed 0 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group dreamer_diagnostic_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_ablation_curriculum_only_s0" \
  --tag "dreamer_ablation_curriculum_only_s0"
```

#### Cell C — No-homeostatic (cuda:2)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/dreamer_diagnostic/03_predator_full_no_homeostatic.yaml \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:2 \
  --log-interval 50 \
  --wandb-group dreamer_diagnostic_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_ablation_no_homeostatic_s0" \
  --tag "dreamer_ablation_no_homeostatic_s0"
```

#### Cell D — Smoother ramp (cuda:3)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --configs-dir configs/experiment/dreamer_diagnostic/curriculum_smooth/ \
  --continual-schedule configs/continual/dreamer_curriculum_smooth_4stage.yaml \
  --agent_config configs/models/dreamer_v3_curriculum.yaml \
  --num-envs 16 \
  --seed 0 \
  --device cuda:3 \
  --log-interval 50 \
  --wandb-group dreamer_diagnostic_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_ablation_smooth_ramp_s0" \
  --tag "dreamer_ablation_smooth_ramp_s0"
```

Notes (apply to all four):
- `--episodes` is set explicitly for **A** and **C** (single-stage `--config`
  mode); train.py:435 enforces that `--episodes` is incompatible with
  `--configs-dir`, so it is **omitted** for **B** and **D** (the schedule's
  last boundary is the budget). 700000 is chosen for A/C to match the
  cumulative final-stage boundary in B's and D's schedules.
- `--checkpoint-frequency 100000` is set explicitly for A and C (single-stage
  mode reads it from the CLI flag); B and D use per-stage frequencies from
  their schedule YAMLs (and train.py refuses `--checkpoint-frequency` in
  that mode).

## 4. Independent / Dependent / Controlled Variables

### 4.1 Independent variables (per experiment)

| Variable | A | B | C | D | Failing baseline (qont5dac) |
|---|:---:|:---:|:---:|:---:|:---:|
| Curriculum stages | 1 | 3 | 1 | **4** | 1 |
| `agent.entropy_scale` | **1e-3** | 3e-4 | 3e-4 | **1e-3** | 3e-4 |
| `agent.cont_loss_weight` | **5.0** | 1.0 | 1.0 | **5.0** | 1.0 |
| `body.use_homeostatic_reward` | true | true | **false** | true | true |
| Cliff (Stage→Full predator) | n/a | 8→3 | n/a | **8→5→3** | n/a |

Bold cells = the IV being uniquely tested vs the baseline that's most
informative for that experiment (see §1's pairwise table).

### 4.2 Dependent variables

| Priority | Metric | Source | Why |
|---|---|---|---|
| **Primary** | `Episode/Steps` mean over the final 1000 episodes of the run (or the final stage in B/D) | env episode log | Project headline metric (survival steps, never reward — see CLAUDE.md). |
| Secondary | `Behavior/mean_entropy`, `Behavior/mean_advantage`, `Episode/Term_Injury`, `Episode/Term_MaxSteps`, `Episode/FoodEaten` — temporal evolution across the run | wandb iteration log | Diagnostic of the qont5dac signature (entropy ~ ln(6), advantage const at -0.19, Term_Injury ~99%) — explicitly named in the failing-baseline diagnosis. |
| Diagnostic | `WorldModel/model_reward_mae_pos`, `..._mae_neg`, `WorldModel/model_cont_acc`, `WorldModel/loss_cont` | wandb iteration log | Reward-head asymmetry & continuation-class-imbalance signatures. |
| C-specific | `positive_buffer.size` over training | wandb iteration log (if exposed) or run dir analysis | Pre-registered observable per the user's note. |

### 4.3 Controlled variables (held constant across all 4 cells)

| Variable | Value | Held constant by |
|---|---|---|
| Grid size | 5x5 | All env YAMLs (anchor + stages). |
| `max_steps` | 500 | All env YAMLs. |
| `random_start_pos` | true | All env YAMLs. |
| `num_envs` | 16 | CLI: `--num-envs 16` on all four. |
| Seed | 0 | CLI: `--seed 0` on all four. |
| `agent.algorithm` | DreamerV3 | All agent configs. |
| `agent.batch_size` | 16 | Agent configs (dreamer_v3.yaml & dreamer_v3_curriculum.yaml). |
| `agent.sequence_length` | 128 | Agent configs. |
| `agent.replay_ratio` | 0.5 | Agent configs. |
| `agent.collect_interval` | 128 | Agent configs. |
| `agent.train_steps` | 64 | Agent configs. |
| `agent.model_lr / actor_lr / value_lr` | 1e-4 / 3e-5 / 3e-5 | Agent configs. |
| `agent.sampling_mode` | mixture | Agent configs. |
| `agent.encoding_mode` | hierarchical | Agent configs. |
| `agent.use_layer_norm` | true | Agent configs. |
| `agent.modulation.type` | null | Agent configs. |
| `perceptual_noise.enabled` | false | All env YAMLs. |
| Sensors (modality fingerprint) | identical 13-tuple across all stages and the C anchor (see §6) | Live-load probe verified. |
| `body.death_penalty` | 100 | All env YAMLs (including C). |
| `body.food_nutrition_gain` | 18 | All env YAMLs. |
| `body.metabolic_cost` | 1.0 | All env YAMLs. |
| `body.with_satiation/with_nutrition/with_injury` | true / true / true | All env YAMLs. |
| Hardware | Node 114 (single host, RTX 6000 Ada x4) | CLI `--device` per cell. |
| Episode budget (max) | 700000 | Schedule YAML for B (= 766k) is slightly off — discussed in §5.3 — so this is the *target* budget; A/C/D end at exactly 700k by `--episodes` or schedule. |
| Wall-clock budget per run | ~3-4 h (parallel) | User constraint. |

## 5. Confounds & Limitations

### 5.1 Single-seed limitation (all 4 cells)

- **Severity**: High for any quantitative claim.
- **Mitigation**: Pre-registered as observational. Any positive cell will
  trigger a 3-seed follow-up at seeds {7, 13, 21} as the next experiment.

### 5.2 Per-cell confounds

| Cell | Confound | Severity | Mitigation |
|------|----------|----------|------------|
| A | Cold-start with levers may be insufficient time-budget vs. failing baseline (700k ep here vs. 909k in qont5dac) | Medium | The qont5dac-equivalent ~28-step death-loop episodes consume ~25 M env-steps in 700k episodes — same order as qont5dac's 27 M. If A learns, it will inflect well within the budget; if it doesn't, it stays at the 30-step floor where each env-step is cheap. |
| A | Buffer accumulates only death-loop trajectories from step 0 (no curriculum warm-up) | High (this is the entire point of the test) | Pre-registered: A failing reproduces the diagnosis's "death-loop is self-locking" finding. Not a confound, the IV. |
| B | Hafner-default entropy_scale=3e-4 is "structurally too weak" per diagnosis §4.3.4 | High (this is the entire point of the test) | Pre-registered: B's Stage 0 might fail to leave uniform-random even on the easy food-only task. If it does fail, that **is** the result (refutes "curriculum alone is enough"). |
| C | `use_homeostatic_reward: false` changes reward distribution AND its sparsity AND eliminates drive-tracking | Medium-high | Confounded; we cannot tell whether the helpful effect (if any) is from "no dense noise" or "sparse reward signal." The user accepted this confound — refuting C cleanly is more valuable than a perfectly orthogonal C. |
| C | Predicted positive_buffer empty if random policy never eats | Medium | Pre-registered as a refutation criterion (R-C2). Diagnosis showed FoodEaten ≈ 0.2/ep under random walk — so positive_buffer will fill SOME, just much slower than under homeostatic mode. The R-criterion threshold is "positive_buffer empty AND survival ~30" (both must hold). |
| C | No homeostatic reward = no positive-reward signal until the agent eats; with predator already at full speed, episodes likely terminate before any food is found | Medium | This is the "useful negative result" the user flagged. If C survives at the floor, it's strong evidence that without a dense early signal Dreamer cannot bootstrap on this task — independent of reward-head asymmetry. |
| D | Stage 03 (predator_medium) is a designer-chosen midpoint; not validated to be "halfway" in any quantitative behavioral sense | Medium | Lever values are deliberately the integer-arithmetic midpoint of Stages 02/04 along {move_int, attack_delay, damage}. detection_range is HELD at 2 (not raised to 3) to avoid making Stage 03 strictly harder than Stage 04 along that axis — keeps the ramp monotone. If D fails, the next experiment can probe whether `detection_range=3` at the medium stage was the missing knob. |
| D | Episode budget per stage is asymmetric (16k / 50k / 134k / 500k) | Low | Per-stage budgets allow each stage time to converge before transition. Total = 700k matches A/C and is below 95mpudwp's 766k. |

### 5.3 Cross-cell confounds

- **B's schedule total is 766000** (the existing schedule from 95mpudwp;
  Stage 3 = 700k delta), while A/C/D end at 700000 episodes total. The
  ~10% difference in episode count in B is acceptable for an observational
  battery — comparison is on the *signature* (entropy plateau, advantage
  collapse, Term_Injury 99%) not on a precise survival number.
- **Cuda devices differ across cells**, but all are RTX 6000 Ada and the
  speed-profile diagnostic confirmed device choice does not affect *learning*
  (only wall-clock).
- **Replay-buffer clearing at stage boundaries** affects only B and D
  (both DreamerV3 in continual mode). A and C never trigger the clear (no
  schedule). This is a non-confound; it's a property of the experimental
  cell.

## 6. Pre-flight verification — LIVE-LOADED

All probes ran with the project conda env on this commit
(`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`). Probe script:
[`tmp/20260507_211806_dreamer_diagnostic_probe.py`](../../../../tmp/20260507_211806_dreamer_diagnostic_probe.py).

### 6.1 Experiment D — cross-stage modality fingerprint

| Stage | obs_dim | action_dim | 13-tuple fingerprint |
|---|---:|---:|---|
| `01_food_only.yaml`         | 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |
| `02_predator_very_slow.yaml`| 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |
| `03_predator_medium.yaml`   | 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |
| `04_predator_full.yaml`     | 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |

**Cross-stage match across 4 stages: OK.** All four stages produce identical
sensor dims, action dims, and the 13-tuple modality fingerprint. The startup
probe in `train.py` will pass.

Tuple decoding (in the order `_modality_fingerprint` returns):

```text
( visual_sensor_enabled = False,
  visual_sensor_range   = 0,
  local_view_size       = 5,
  olfactory_enabled     = True,
  olfactory_vector_size = 5,
  nociception_enabled   = True,
  nociception_size      = 1,
  interoceptive_nociception_enabled = True,
  location_sensor_enabled = False,
  proprioception_enabled = True,
  injury_observable     = False,
  nutrition_observable  = False,
  sensor_range          = 1 )
```

### 6.2 Experiment C — `use_homeostatic_reward: false` actually disables the homeostatic reward path

**Loaded from C's YAML:**

```text
EnvParams.use_homeostatic_reward = False  (type: bool)
```

**Cross-checked against the failing-task anchor `basic/01-5X5_PredInterval3_NutGain18.yaml`:**

```text
EnvParams.use_homeostatic_reward = True  (anchor)
EnvParams.use_homeostatic_reward = False (C config)

Other body params identical (anchor vs C):
  death_penalty=100, food_nutrition_gain=18, metabolic_cost=1.0,
  max_satiation/nutrition/injury=100, with_satiation/nutrition/injury=True
```

**Live env-loader citation (`src/environment/config_loader.py:371`)**:

```python
use_homeostatic_reward=config.get_mandatory('body.use_homeostatic_reward'),
```

— this key is mandatory and parsed via `get_mandatory`, no fallback default,
no possibility of being silently ignored.

**Behavior in `src/environment/core.py:469-477`** (the env reward branch
gated on this flag):

```python
if params.use_homeostatic_reward:
    prev_drive = calculate_drive(state.satiation, state.injury_level, params)
    curr_drive = calculate_drive(new_satiation, new_injury, params)
    reward_homeostatic = prev_drive - curr_drive
    reward_homeostatic = jnp.where(done, reward_homeostatic - params.death_penalty, reward_homeostatic)
else:
    reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
    reward_extrinsic = jnp.where(done, -params.death_penalty, reward_extrinsic)
```

— with C's `False`, the agent receives:
- `+1.0` per env-step on which `ate_food` is true,
- `-100.0` (= `-death_penalty`) on the terminal step,
- `0.0` otherwise.

The drive-tracking branch is fully bypassed. **The C experiment is loading
exactly the intended reward regime.**

### 6.3 Modality fingerprint equality between Experiment C config and failing-task anchor

```text
Anchor fingerprint: (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
C config fingerprint: (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
Match: OK
```

— so direct `qont5dac vs C` comparison on `Episode/Steps`, `Behavior/*`, and
`WorldModel/*` is well-defined (no obs-space difference).

### 6.4 Schedule YAML well-formedness (Experiment D)

`configs/continual/dreamer_curriculum_smooth_4stage.yaml` matches the
`_build_continual_schedule` validation contract:

| Check | Value | OK? |
|---|---|---|
| `len(episode_boundaries) == num_stage_files` | 4 == 4 | OK |
| `len(checkpoint_frequencies) == num_stage_files` | 4 == 4 | OK |
| boundaries strictly increasing | `[16000, 66000, 200000, 700000]` | OK |
| `boundaries[0] > 0` (Fix 2 guard in `_build_continual_schedule`) | 16000 > 0 | OK |
| all checkpoint_frequencies > 0 | `[8000, 25000, 50000, 100000]` | OK |
| Per-stage delta | `[16000, 50000, 134000, 500000]` | OK |

### 6.5 CLI compatibility

- **A & C**: `--config <single yaml>` + `--episodes 700000` is the documented
  single-config path (`train.py:435`: `--episodes` IS allowed when
  `--configs-dir` is NOT set). OK.
- **B & D**: `--configs-dir` + `--continual-schedule`, with
  `boundaries[-1]` = budget. `--episodes` and `--checkpoint-frequency`
  must NOT be passed (and aren't in our launch commands). OK.
- **B & D continual gate**: train.py guards continual mode to RecurrentPPO
  and DreamerV3 only (Fix 5 in CONTINUAL_LEARNING_CONFIG_SCHEDULE.md). Both
  cells are DreamerV3. OK.
- **A & B & D agent config**: All have `agent.cont_loss_weight` (1.0 in
  dreamer_v3.yaml; 5.0 in dreamer_v3_curriculum.yaml) — required since the
  schema-add commit. The trainer reads via `get_mandatory` and would
  ValueError if missing. OK.

## 7. Confirmation / Refutation Criteria (per cell)

Walked at the end of each run from WandB-export tables. All thresholds are
on **the final 1000 episodes** of the run (or the final stage in B/D).

### 7.1 Cell A — Lever-only

- **C-A1 (success)**: Final-1000-ep mean `Episode/Steps` > **100**.
- **C-A2 (success-strong)**: `Behavior/mean_entropy` final-1000-ep mean **< 1.5** AND `Behavior/mean_advantage` end-window std **> 0.05**.
- **R-A1 (refutation)**: Final-1000-ep mean `Episode/Steps` ≤ **35** AND `Behavior/mean_advantage` end-window std **< 0.01** AND `Behavior/mean_entropy` end-window mean **> 1.6** (qont5dac signature).
- **R-A2 (instability)**: NaN in any loss; or `loss_model` rises monotonically over any 5 M env-step window.

### 7.2 Cell B — Curriculum-only

- **C-B1 (Stage 0 unfreeze, weak)**: Stage 0 (`01_food_only`) end-window `Behavior/mean_advantage` std **> 0.02** AND `Episode/Steps` mean **> 50**. (Tighter than 95mpudwp's Stage 0 because we've removed the levers; we want to know if the easy task alone is enough.)
- **C-B2 (Stage 2 transfer)**: Stage 2 (`03_predator_full`, the failing anchor) final-1000-ep mean `Episode/Steps` > **100**.
- **R-B1 (refutation)**: Stage 0 itself fails to leave uniform — `Behavior/mean_advantage` end-window std **< 0.01** AND `Behavior/mean_entropy` end-window mean **> 1.7**. Would mean the levers were necessary even for the easy task.
- **R-B2 (Stage 2 collapse)**: Stage 2 final-1000-ep mean `Episode/Steps` ≤ **50** even though Stage 0 satisfied C-B1 (mirrors the 95mpudwp R2 pre-registration).

### 7.3 Cell C — No-homeostatic

- **C-C1 (success)**: Final-1000-ep mean `Episode/Steps` > **100**.
- **C-C2 (partial success)**: Final-1000-ep `Episode/Term_Injury` rate **< 90%** OR `Episode/Term_MaxSteps` rate **> 5%** (any reduction in the 99% predator-kill regime would itself be informative).
- **C-C3 (positive_buffer fills)**: `positive_buffer.size` final value **>= 1000** (i.e., at least ~62 sequences of `seq_len=128` get logged) — confirms the eat-event signal exists in the trajectory data.
- **R-C1 (refutation, full)**: Final-1000-ep mean `Episode/Steps` ≤ **35** AND `positive_buffer.size` final **< 100** (essentially empty). This is the user-flagged "useful negative result": the sparse-reward regime is effectively zero-shot blocked because the random-walk policy never eats often enough to feed the buffer.
- **R-C2 (refutation, partial)**: Final-1000-ep mean `Episode/Steps` ≤ **35** AND `positive_buffer.size` final **>= 1000**. This would mean the buffer fills but the agent still fails — refuting the "reward-distribution-asymmetry was the bottleneck" hypothesis even more sharply than R-C1.

### 7.4 Cell D — Smoother ramp

- **C-D1 (success, primary)**: Stage 4 (`04_predator_full`) final-1000-ep mean `Episode/Steps` > **100**.
- **C-D2 (Stage 3 bridge effect)**: Stage 3 (`03_predator_medium`) end-window `Episode/Term_Injury` rate **< 80%** AND `Episode/Steps` mean **> 50** — confirms the medium predator was a meaningful intermediate.
- **C-D3 (transfer signature)**: Stage 4's *first* 1000 ep mean `Episode/Steps` **> 50** (vs. ~30 in 95mpudwp's Stage 2 first 1000 ep). Even partial transfer is informative.
- **R-D1 (refutation, full)**: Stage 4 final-1000-ep mean `Episode/Steps` ≤ **50** despite Stage 3 satisfying C-D2 (Stage 3 worked, but transfer to Stage 4 still failed). Strongest evidence that the difficulty-cliff hypothesis is wrong.
- **R-D2 (refutation, weak)**: Stage 3 fails to satisfy C-D2 (predator_medium itself collapses). Would mean the Stage 02→Stage 03 step is itself the cliff problem, not the Stage 03→Stage 04 step.

## 8. Failure-Mode Catalog (pre-decided interpretations)

| Observation | Interpretation |
|---|---|
| All 4 cells refute (survival flat ~30 across all of A, B, C, D) | The hypervigilance failure is structural in DreamerV3 on this task — not a reward, lever, curriculum, or cliff issue. Recommend reward-head class-balanced loss (diagnosis §7.3 priority 4) AND/OR imagined-rollout termination probe (§7.3 priority 1) as next steps. |
| Only A succeeds | Curriculum was *causing* the Stage-2 collapse in 95mpudwp (paradoxical but possible: replay-buffer clearing destroys WM progress). Recommend ablating the buffer-clear behavior. |
| Only B succeeds | Levers were *causing* the Stage-2 collapse in 95mpudwp (entropy_scale=1e-3 too high once predator is at full speed). Recommend lever-magnitude sweep. |
| Only C succeeds | The dense homeostatic reward distribution was the bottleneck. Recommend reward-head fix (per-bin loss reweighting) on the original dense regime. |
| Only D succeeds | Cliff size was the issue. Recommend exploring even smoother ramps (5-stage, 6-stage). |
| A and D succeed, B fails | Lever package is necessary; curriculum is helpful but not necessary. Recommend lever sweep. |
| B and D succeed, A fails | Curriculum is necessary for the bootstrap; the smoother ramp helps but the original cliff was already enough. Recommend levers-only follow-up at smaller magnitudes. |
| All 4 succeed | Strong positive — every factor independently helps. Pick the simplest (likely A) for downstream work. |
| C's `positive_buffer.size` stays at 0 throughout | Confirms the random-walk policy never eats food in the predator-full regime; the sparse-reward bootstrap is impossible without prior food-finding capability. Pre-registered as the "useful negative result" the user explicitly flagged. |
| Any cell shows NaN or `loss_model` rising monotonically | Refute the run, not the hypothesis. Re-run with same config; if reproducible, refute the hypothesis for that cell only. |
| Stage 4 (D) collapses *immediately* (within 100 ep) to 30 steps | Same R2 pattern as 95mpudwp Stage 3 — the smoother ramp didn't help. Cliff hypothesis refuted; recommend looking at non-cliff factors (e.g., replay-buffer clearing, recurrent-state reset, hidden-state alignment between stages). |
| `Behavior/mean_entropy` < 0.5 in any cell | Premature actor commitment; would be an unusual failure mode at these lever values. Document as a side observation. |
| Wall-clock per cell exceeds 5 h | Node 114 contention or unexpected per-iteration slowdown. Halt mid-run and notify user. |

## 9. Analysis Plan (post-run)

Single seed per cell → no inferential statistics. Post-run fill-in for §10 / §11 will:

1. **Plumbing checks first** (per cell):
   - Cells A & C: single WandB run-ID, no startup ValueError, expected episode count, `agent.entropy_scale` and `agent.cont_loss_weight` show in `wandb.config` at the planned values.
   - Cells B & D: schedule banner prints at startup, modality-fingerprint probe passed, `stage/transition` events fire at expected boundaries (within drift `+ num_envs - 1`), `stage/buffer_cleared_main` logs at each transition.

2. **Per-cell temporal evolution**:
   - 20 windows of `Episode/Steps`, `Episode/Reward`, `Behavior/mean_entropy`, `Behavior/mean_advantage`, `WorldModel/model_reward_mae_pos/_neg`, `WorldModel/model_cont_acc`, `Episode/Term_*` rates.
   - Plot all on the `Episode/Number` axis with vertical lines at stage boundaries (B & D only).

3. **Cross-cell comparison vs failing baseline `qont5dac` and 95mpudwp**:
   - Side-by-side temporal trajectories on the same env-step axis.
   - Headline metric per cell: final-1000-ep mean `Episode/Steps`.
   - Signature panel: 4-cell + qont5dac + 95mpudwp Stage-2 grid showing
     `Behavior/mean_entropy` end-window vs `Behavior/mean_advantage` end-window
     std.

4. **Confirm/refute checklist**: Walk all C-X / R-X criteria from §7 with
   explicit metric values; record per-cell verdict.

5. **2x2 contrast read-off** (per §1's pairwise table):
   - A vs qont5dac: lever effect.
   - B vs 95mpudwp: lever effect given curriculum.
   - C vs qont5dac: homeostatic-reward effect.
   - D vs 95mpudwp: cliff-step effect.

6. **Wall-clock attribution**: per-cell total time + JIT-recompile blips at
   stage boundaries (for B & D).

## 10. Plumbing-check table (post-run fill-in)

> _To be filled in after the runs complete._

### 10.1 Cell A — Lever-only
- [ ] Single WandB run-ID
- [ ] No startup ValueError
- [ ] `agent.entropy_scale = 0.001` in `wandb.config`
- [ ] `agent.cont_loss_weight = 5.0` in `wandb.config`
- [ ] `body.use_homeostatic_reward = True` in `wandb.config`
- [ ] No NaN warnings
- [ ] Total wall-clock within ~4 h

### 10.2 Cell B — Curriculum-only
- [ ] Single WandB run-ID
- [ ] Schedule banner prints at startup (`Continual mode: 3 stages from configs/experiment/dreamer_curriculum/`)
- [ ] Cross-stage probe passes (no fingerprint ValueError)
- [ ] `stage/transition` events at Episode/Number ∈ `[16000, 16015]` and `[66000, 66015]`
- [ ] `stage/buffer_cleared_main` fires at both boundaries
- [ ] `agent.entropy_scale = 0.0003` in `wandb.config`
- [ ] `agent.cont_loss_weight = 1.0` in `wandb.config`
- [ ] No NaN warnings

### 10.3 Cell C — No-homeostatic
- [ ] Single WandB run-ID
- [ ] No startup ValueError
- [ ] `body.use_homeostatic_reward = False` in `wandb.config`
- [ ] `agent.entropy_scale = 0.0003` in `wandb.config`
- [ ] `agent.cont_loss_weight = 1.0` in `wandb.config`
- [ ] No NaN warnings

### 10.4 Cell D — Smoother ramp
- [ ] Single WandB run-ID
- [ ] Schedule banner prints (`Continual mode: 4 stages from configs/experiment/dreamer_diagnostic/curriculum_smooth/`)
- [ ] Cross-stage probe passes
- [ ] `stage/transition` events at `[16000, 16015]`, `[66000, 66015]`, `[200000, 200015]`
- [ ] `stage/buffer_cleared_main` fires at all 3 boundaries
- [ ] `agent.entropy_scale = 0.001` in `wandb.config`
- [ ] `agent.cont_loss_weight = 5.0` in `wandb.config`
- [ ] No NaN warnings

## 11. Results

> _To be filled in after the runs complete._

### 11.1 Per-cell signatures (table)

| Cell | Final-1000-ep `Episode/Steps` | Final `Behavior/mean_entropy` | Final `Behavior/mean_advantage` (std) | Final `Episode/Term_Injury` | Verdict |
|------|---:|---:|---:|---:|---|
| A | TBD | TBD | TBD | TBD | TBD |
| B (Stage 2) | TBD | TBD | TBD | TBD | TBD |
| C | TBD | TBD | TBD | TBD | TBD |
| D (Stage 4) | TBD | TBD | TBD | TBD | TBD |
| `qont5dac` (ref) | 30.6 ± 1.1 | 1.65 | -0.187 (0.005) | 0.99 | failing baseline |
| `95mpudwp` (ref) Stage 2 | 30.3 ± 0.84 | 1.65 ± 0.02 | -0.186 (0.009) | 0.99 | curriculum collapsed |

### 11.2 Confirm/refute walk
> _Per-cell C-X / R-X criteria from §7._

### 11.3 2x2 contrast verdict
> _Per §9 step 5._

## 12. Conclusions

> _To be filled in after the runs complete._

### 12.1 Summary
> _3-5 bullets: which factor (or factors) was responsible, what surprised us, what's next._

### 12.2 Recommended next experiments

> _Pre-populated possibilities (will be pruned post-run):_
>
> | Priority | Experiment | Rationale | Trigger |
> |---|---|---|---|
> | 1 | 3-seed repeat of any single positive cell | Required before quantitative claims. | Any C-X1 fires. |
> | 2 | Reward-head class-balanced loss (diagnosis §7.3 priority 4) | Direct fix for reward-head asymmetry if C succeeds. | C succeeds AND signature matches reward-head asymmetry recovery. |
> | 3 | Lever-magnitude sweep (entropy_scale ∈ {3e-4, 1e-3, 3e-3, 1e-2}) | If A or D succeed, finer-grained lever tuning. | A or D succeeds. |
> | 4 | Imagined-rollout termination probe (diagnosis §7.3 priority 1) | If all 4 cells refute, confirms imagination-drift is the dominant mechanism. | All 4 refute. |
> | 5 | Replay-buffer-clear ablation in continual mode | If A succeeds and B fails sharply, the buffer-clear semantics may be destroying useful WM progress. | A succeeds, B fails. |
> | 6 | Even-smoother ramp (5- or 6-stage) | If D succeeds but barely, more granularity might help. | D's C-D1 met but with low margin. |

---

## Appendix

### A. Live-load probe full output

Saved at [`tmp/20260507_211806_dreamer_diagnostic_probe.py`](../../../../tmp/20260507_211806_dreamer_diagnostic_probe.py); output verbatim in §6.1, §6.2, §6.3.

### B. Lever values across cells

| Lever | qont5dac (failing baseline) | 95mpudwp (curriculum + levers) | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|
| `agent.entropy_scale` | 3e-4 | 1e-3 | **1e-3** | 3e-4 | 3e-4 | **1e-3** |
| `agent.cont_loss_weight` | 1.0 | 5.0 | **5.0** | 1.0 | 1.0 | **5.0** |
| Curriculum stages | 1 | 3 | 1 | 3 | 1 | **4** |
| `body.use_homeostatic_reward` | true | true | true | true | **false** | true |
| Cliff (Stage→Full predator) | n/a | 8→3 | n/a | 8→3 | n/a | **8→5→3** |

### C. Per-stage delta (D's schedule)

| Stage | Name | until_ep (cum) | delta_ep | ckpt_freq |
|---:|---|---:|---:|---:|
| 0 | `01_food_only`              | 16000  | 16000  | 8000 |
| 1 | `02_predator_very_slow`     | 66000  | 50000  | 25000 |
| 2 | `03_predator_medium`        | 200000 | 134000 | 50000 |
| 3 | `04_predator_full`          | 700000 | 500000 | 100000 |

### D. Stage-3 (`03_predator_medium`) parameter derivation

| Param | Stage 02 (very_slow) | Stage 04 (full) | Midpoint | Stage 03 (medium) chosen |
|---|---:|---:|---:|---:|
| `predators[0].move_interval` | 8 | 3 | 5.5 | **5** |
| `predators[0].attack_delay` | 8 | 3 | 5.5 | **5** |
| `predators[0].damage` | [5, 15] | [15, 45] | [10, 30] | **[10, 30]** |
| `predators[0].detection_range` | 2 | 3 | 2.5 | **2** (held at Stage 02's value) |

Rationale for held detection_range: per the 95mpudwp tmp doc analysis, the
hypothesized cliff factor is speed/damage; raising detection_range at the
medium stage would *increase* difficulty along an axis that is already at
Stage 02's lower setting, conflating the smoother-ramp test.

### E. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-07 | Initial design + 5 new env configs + 1 new schedule + Launch Manifest. Cross-stage live-load probe verified. | experiment-designer |
