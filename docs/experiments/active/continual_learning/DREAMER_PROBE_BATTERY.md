---
title: "DreamerV3 imagined-rollout termination probe battery — 4-cell Node-114 launch"
topic: continual_learning
status: active
created: 2026-05-08
last_updated: 2026-05-08
phase: 1
wandb_tag: "dreamer_probe_battery"
develop_link: "../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md"
---

# DreamerV3 imagined-rollout termination probe battery — 4-cell Node-114 launch

> **Status**: PLANNED (configs landed; live-loaded clean; awaiting env-config-auditor + user authorization to launch)
> **Date**: 2026-05-08
> **Author**: experiment-designer
> **Related**:
> - **Probe instrumentation contract** (the feature this battery exercises):
>   [`docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`](../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md)
> - **Diagnostic battery refutations** that produced the surviving structural hypothesis:
>   [`DREAMER_DIAGNOSTIC_BATTERY.md`](./DREAMER_DIAGNOSTIC_BATTERY.md) — §11.4 cross-cell synthesis, §12.2 priority-1
> - **Failure-mode diagnosis** (the surviving hypothesis itself):
>   [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md)
> - **Continual-learning feature contract** (for E4):
>   [`docs/develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../../../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)
> - **Reference precedent runs** (compared against in confirm/refute walks):
>   - `qont5dac` (failing baseline) — anchor-only, Hafner defaults
>   - `95mpudwp` (curriculum + levers, Stage 2 collapse) — see `DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md`
>   - `7t1wskpk` (rPPO succeeds cold) — same task, different algo

> **Note**: 4 single-seed runs in parallel on cuda:0..3 of Node 114. Each
> result is observational (no statistical claims). The deliverable is a
> direct observation of the inferred structural mechanism — namely, whether
> the WM continuation head ever predicts the death event in imagined
> rollouts under full-predator dynamics — and whether boosting
> `cont_loss_weight` to 10 fixes it.

---

## 1. Research Question

The diagnostic battery [`DREAMER_DIAGNOSTIC_BATTERY.md`](./DREAMER_DIAGNOSTIC_BATTERY.md) refuted four candidate localizations of the DreamerV3 hypervigilance failure on `01-5X5_PredInterval3_NutGain18`: (1) levers alone, (2) levers + curriculum, (3) homeostatic-reward asymmetry, (4) curriculum-cliff size. The single mechanism that survived is **inferred but not directly observed**: the WM continuation head can't predict the death event in imagined rollouts under full-predator dynamics, so imagined trajectories never terminate, the actor optimizes against returns that miss the −100 death penalty, and the policy collapses to ~uniform.

The just-landed probe instrumentation [(probe plan)](../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md) makes that mechanism directly observable. This battery asks two falsifiable questions:

> **Q1 (mechanism confirmation)**: Under full-predator dynamics on the failing task, do imagined rollouts never terminate (`imagined_termination_fraction_h15 ≈ 0`) while real episodes terminate at step ~28?
>
> **Q2 (priority-2 fix test)**: Does setting `cont_loss_weight = 10.0` on that same anchor raise the imagined-termination fraction toward the real rate, and if so, does survival improve?

Two control cells frame the result:
- **E3 (positive control)**: Same probe-enabled config on the dramatically easier `00-5X5_NoPred` task — does the probe show a non-zero termination fraction when the WM has a tractable termination distribution? (If the probe reports ≈0 here too, either the probe is broken or the mechanism is broader than predator-specific.)
- **E4 (per-stage breakdown)**: Curriculum + levers + probe — replicates `95mpudwp` with the probe added. If the inference is right, Stage 0/1 should show non-zero termination fraction and Stage 2 should regress to ~0 within ~3,200 episodes of the boundary, directly visualising the moment WM imagination loses death events.

> **H₀ (probe-anchor)**: Imagined rollouts under full-predator dynamics terminate at the same rate as real rollouts (no per-step gap between `imagined_termination_fraction_h15` and `(real_term_step_mean ≤ 15) / N`).
> **H₁ (probe-anchor)**: Imagined rollouts almost never terminate (`imagined_termination_fraction_h15` ≪ real rate; structural hypothesis confirmed).

---

## 2. Pre-registered Hypotheses (per cell)

### E1 — Anchor + probe (mechanism confirmation)

> **H₀(E1)**: On the failing task with Hafner-default agent + probe enabled, `imagined_termination_fraction_h15` final-1000-iter median > 0.5 — imagined rollouts DO terminate, refuting the WM-imagination structural hypothesis.
>
> **H₁(E1)**: `imagined_termination_fraction_h15` final-1000-iter median < 0.1 — imagined rollouts essentially never terminate, while `imagined_real_term_step_mean` reflects real episodes ending at step ~28 (so `(real_term_step_mean ≤ 15)` is non-zero on the real side). This is the headline confirmation of the structural mechanism.

Significance: H₁ is the diagnostic battery's surviving inferred mechanism made directly observable. H₀ would mean four diagnostic cells refuted four wrong hypotheses and the right one is still hidden.

### E2 — Anchor + cont_loss_weight=10 + probe (test priority-2 fix)

> **H₀(E2)**: `cont_loss_weight=10` does not lift the imagined-termination fraction above the E1 baseline. Final-1000-iter median `imagined_termination_fraction_h15` < 0.1 (same as E1) AND survival flat at the qont5dac floor (≤35 steps).
>
> **H₁(E2) — full fix**: `imagined_termination_fraction_h15` final-1000-iter median > 0.3 AND `Episode/Steps` final-1000-ep mean > 100. The cont-weight knob both makes the WM imagine deaths AND unlocks survival. This is the cleanest fix for the structural failure.
>
> **H₂(E2) — partial / mechanism right, fix insufficient**: `imagined_termination_fraction_h15` final-1000-iter median > 0.3 (cont-weight knob does what it advertises — WM now imagines deaths) but `Episode/Steps` final-1000-ep mean ≤ 50 (survival doesn't catch up). This would mean the imagination story is correct but downstream actor/value/exploration is the next block.

Significance: H₁ resolves both Q1 and Q2 favourably. H₂ confirms Q1 but pushes the next experiment to actor-side or value-loss interventions. H₀ refutes both — would require revisiting the diagnostic battery's inferred mechanism.

### E3 — `00-5X5_NoPred` + probe (positive control)

> **H₀(E3)**: Even on the easier No-Pred task, `imagined_termination_fraction_h15` final-1000-iter median < 0.1 — refuting the probe's discriminative power, OR indicating the WM-imagines-no-death failure mode is not predator-specific (extends to MaxSteps / starvation / hiding-predator deaths).
>
> **H₁(E3)**: `imagined_termination_fraction_h15` tracks `(real_term_step_mean ≤ 15) / N` to within 30 % relative error in the final-1000-iter window — the probe correctly reports non-zero terminations when the WM has a tractable termination distribution. Survival is unimportant for this hypothesis (No-Pred is much easier; expect mean Steps > 100 anyway, but the dependent variable here is the probe ratio, not survival).

Significance: H₁ validates the probe as instrumentation. H₀ would mean either the probe doesn't work (probe-instrumentation bug) or the WM-can't-imagine-termination failure is broader than predator-specific.

### E4 — 3-stage curriculum + levers + probe (per-stage breakdown)

> **H₀(E4)**: All 3 stages show `imagined_termination_fraction_h15` final-window median < 0.1 — the curriculum's Stage 0/1 unfreeze (documented in `95mpudwp`) was NOT driven by the WM learning to imagine terminations.
>
> **H₁(E4) — staged collapse, the headline finding for this cell**:
> - Stage 0 (`01_food_only`) end-window median `imagined_termination_fraction_h15` > 0.3 (food-only is short due to starvation; WM has tractable terminations).
> - Stage 1 (`02_predator_slow`) end-window median > 0.2 (slow predator, longer episodes; some terminations).
> - Stage 2 (`03_predator_full`) end-window median < 0.1 (regresses to zero within ~3,200 ep of the boundary at episode 66 000).
>
> **H₂(E4) — uniform success**: All three stages show > 0.2 termination fraction. This would mean Stage 2's collapse in `95mpudwp` is NOT explained by the WM losing termination prediction; some other mechanism kicks in at Stage 2 even though the WM still imagines terminations (e.g., termination prediction is preserved but its temporal placement becomes wrong / values become uninformative).

Significance: H₁ provides per-stage temporal localization of the failure. The Stage-1→Stage-2 transition is the moment the WM stops imagining death events; this is the directly-observable moment that the diagnostic battery's §11.4 inferred but did not see.

---

## 3. Launch Manifest

| Run | Status  | Cell | Tag (= wandb-name)                  | wandb-group           | wandb-job-type | Seed | Node | GPU    | Launched at | WandB run ID | Log path |
|-----|---------|------|-------------------------------------|-----------------------|----------------|------|------|--------|-------------|--------------|----------|
| 1   | running | E1   | `dreamer_probe_anchor_s0`           | `dreamer_probe_battery` | ablation     | 0    | 114  | cuda:0 | 2026-05-08T05:24:13 | owz29n3t     | logs/20260508_052413.log |
| 2   | running | E2   | `dreamer_probe_cont10_anchor_s0`    | `dreamer_probe_battery` | ablation     | 0    | 114  | cuda:1 | 2026-05-08T05:26:23 | mruhypnu     | logs/20260508_052623_dreamer_probe_cont10_anchor_s0.log |
| 3   | running | E3   | `dreamer_probe_no_pred_control_s0`  | `dreamer_probe_battery` | ablation     | 0    | 114  | cuda:2 | 2026-05-08T05:28:52 | xekunmbw     | logs/20260508_052852.log |
| 4   | running | E4   | `dreamer_probe_curriculum_s0`       | `dreamer_probe_battery` | ablation     | 0    | 114  | cuda:3 | 2026-05-08T05:31:06 | udqry11l     | logs/20260508_053106.log |

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env)                                                                      | Config (agent)                                                                     |
|-----|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 1   | `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml`                    | `configs/models/dreamer_v3_probe.yaml` (NEW)                                       |
| 2   | `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml`                    | `configs/models/dreamer_v3_probe_cont10.yaml` (NEW)                                |
| 3   | `configs/experiment/basic/00-5X5_NoPred.yaml`                                     | `configs/models/dreamer_v3_probe.yaml` (NEW; same as E1)                           |
| 4   | `configs/experiment/dreamer_curriculum/{01_food_only,02_predator_slow,03_predator_full}.yaml` (via `--configs-dir`) + `configs/continual/dreamer_curriculum_food_then_predator.yaml` (existing schedule, unchanged) | `configs/models/dreamer_v3_curriculum_probe.yaml` (NEW) |

**Three new agent configs total. Zero env / schedule changes.**

### 3.2 Launch commands (verbatim, ready for `training-runner`)

#### Run 1 — E1: Anchor + probe (cuda:0)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml \
  --agent_config configs/models/dreamer_v3_probe.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group dreamer_probe_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_probe_anchor_s0" \
  --tag "dreamer_probe_anchor_s0"
```

#### Run 2 — E2: Anchor + cont_loss_weight=10 + probe (cuda:1)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml \
  --agent_config configs/models/dreamer_v3_probe_cont10.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group dreamer_probe_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_probe_cont10_anchor_s0" \
  --tag "dreamer_probe_cont10_anchor_s0"
```

#### Run 3 — E3: 00-NoPred + probe positive control (cuda:2)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/basic/00-5X5_NoPred.yaml \
  --agent_config configs/models/dreamer_v3_probe.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:2 \
  --log-interval 50 \
  --wandb-group dreamer_probe_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_probe_no_pred_control_s0" \
  --tag "dreamer_probe_no_pred_control_s0"
```

#### Run 4 — E4: 3-stage curriculum + levers + probe (cuda:3)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --configs-dir configs/experiment/dreamer_curriculum/ \
  --continual-schedule configs/continual/dreamer_curriculum_food_then_predator.yaml \
  --agent_config configs/models/dreamer_v3_curriculum_probe.yaml \
  --num-envs 16 \
  --seed 0 \
  --device cuda:3 \
  --log-interval 50 \
  --wandb-group dreamer_probe_battery \
  --wandb-job-type ablation \
  --wandb-name "dreamer_probe_curriculum_s0" \
  --tag "dreamer_probe_curriculum_s0"
```

**Notes (apply to all four):**
- `--episodes 700000` and `--checkpoint-frequency 100000` are set explicitly for **E1, E2, E3** (single-stage `--config` mode); train.py:435 enforces that `--episodes` is incompatible with `--configs-dir`, so these flags are **omitted** for **E4** (the schedule's last boundary `766000` is the budget; per-stage checkpoint frequencies `[8000, 25000, 100000]` come from the schedule YAML).
- All four use the canonical CLI flag set used by the diagnostic battery; **no `--wandb-tags` flag** (it does not exist in train.py — only `--tag`).
- All four are single-seed observational. Quantitative claims about probe values vs survival require the 3-seed follow-up under priority-1 of §6.3.

---

## 4. Independent / Dependent / Controlled Variables

### 4.1 Independent variables (per cell)

| Variable                       | E1                    | E2                    | E3                    | E4                                    | Failing baseline (`qont5dac`) | 95mpudwp (curriculum+levers) |
|--------------------------------|-----------------------|-----------------------|-----------------------|---------------------------------------|-------------------------------|-----------------------------|
| Env                            | `01-PredInterval3`    | `01-PredInterval3`    | **`00-5X5_NoPred`**   | curriculum 3-stage                    | `01-PredInterval3`            | curriculum 3-stage          |
| Curriculum stages              | 1                     | 1                     | 1                     | **3**                                 | 1                             | 3                           |
| `agent.entropy_scale`          | 3e-4                  | 3e-4                  | 3e-4                  | **1e-3**                              | 3e-4                          | 1e-3                        |
| `agent.cont_loss_weight`       | 1.0                   | **10.0**              | 1.0                   | 5.0                                   | 1.0                           | 5.0                         |
| `agent.imagined_rollout_probe` | **true**              | **true**              | **true**              | **true**                              | false                         | false                       |

The IV under primary test for each cell:
- **E1**: probe enabled on the failing baseline (Q1 mechanism observation).
- **E2**: `cont_loss_weight = 10.0` on the failing baseline (Q2 fix test).
- **E3**: env switched to No-Pred (probe positive control).
- **E4**: probe enabled on the curriculum + levers configuration (per-stage breakdown).

### 4.2 Dependent variables

| Priority           | Metric                                                                                                                     | Source                          | Why                                                                                                                                                                |
|--------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Primary**        | `WorldModel/imagined_termination_fraction_h15` final-1000-iter median (per-cell; per-stage for E4)                         | wandb iteration log (probe key) | Direct test of the WM-can't-imagine-death structural hypothesis.                                                                                                   |
| **Co-primary** (E2) | `Episode/Steps` final-1000-ep mean (project headline metric; survival steps, never reward)                                | env episode log                 | Determines whether boosting cont_loss_weight just makes the probe move (mechanism right) vs. actually unlocks survival (full fix per H₁(E2)).                       |
| Secondary          | `WorldModel/imagined_first_term_step_mean`, `imagined_term_step_p{10,50,90}`, `imagined_real_term_step_mean`               | wandb iteration log (probe)     | Distribution of imagined-rollout termination steps. For E1: expect p50 ≈ 15 (sentinel), real ≈ 28 (real episodes). For E2 H₁: p50 should drop into the rollout window. |
| Secondary          | `Behavior/mean_entropy`, `Behavior/mean_advantage`, `Episode/Term_Injury`, `Episode/Term_MaxSteps`, `Episode/FoodEaten`    | wandb iteration log             | qont5dac-signature panel — is the cell still in the failing regime?                                                                                                |
| Diagnostic         | `WorldModel/model_reward_mae_pos`, `..._mae_neg`, `WorldModel/model_cont_acc`, `WorldModel/loss_cont`                      | wandb iteration log             | Reward-head asymmetry & continuation-class-imbalance signatures (cross-reference against the 4 diagnostic-battery cells).                                          |

### 4.3 Controlled variables (held constant across all 4 cells)

| Variable                                                             | Value                                      | Held constant by                                                            |
|----------------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| Grid size                                                            | 5×5                                        | All env YAMLs (E1/E2/E3 anchors + E4 curriculum stages)                     |
| `max_steps`                                                          | 500                                        | All env YAMLs                                                               |
| `random_start_pos`                                                   | true                                       | All env YAMLs                                                               |
| `num_envs`                                                           | 16                                         | CLI: `--num-envs 16` on all four                                            |
| Seed                                                                 | 0                                          | CLI: `--seed 0` on all four                                                 |
| `body.use_homeostatic_reward`                                        | true                                       | All env YAMLs (no Cell C-style switch this round)                           |
| Sensors (modality fingerprint)                                       | identical 13-tuple across all 5 env YAMLs  | Live-load probe verified — see §5.1                                         |
| `agent.algorithm`                                                    | DreamerV3                                  | All 3 new agent configs                                                     |
| `agent.batch_size / sequence_length / replay_ratio / collect_interval / train_steps` | 16 / 128 / 0.5 / 128 / 64                  | All 3 new agent configs (verbatim from parents)                             |
| `agent.{model_lr, actor_lr, value_lr}`                               | 1e-4 / 3e-5 / 3e-5                         | All 3 new agent configs                                                     |
| `agent.sampling_mode`                                                | mixture                                    | All 3 new agent configs                                                     |
| `agent.encoding_mode`                                                | hierarchical                               | All 3 new agent configs                                                     |
| `agent.use_layer_norm`                                               | true                                       | All 3 new agent configs                                                     |
| `agent.modulation.type`                                              | null (no neuromodulation)                  | All 3 new agent configs                                                     |
| `agent.imagined_rollout_probe`                                       | **true** (the IV being introduced)         | All 3 new agent configs                                                     |
| `perceptual_noise.enabled`                                           | false                                      | All env YAMLs                                                               |
| Hardware                                                             | Node 114 (single host, RTX 6000 Ada × 4)   | CLI `--device` per cell (cuda:0 / 1 / 2 / 3)                                |
| Episode budget                                                       | 700 000 (E1/E2/E3) / 766 000 (E4 schedule) | `--episodes 700000` (E1/E2/E3); schedule's `boundaries[-1]` = 766 000 (E4)  |
| `--log-interval`                                                     | 50                                         | CLI on all four                                                             |
| Wall-clock budget per run                                            | ~3 h target, ~5 h hard ceiling             | Mirrors diagnostic battery; halt mid-run if exceeded                        |

---

## 5. Confounds & Limitations

### 5.1 Pre-flight live-load checks (verified)

Live-load probe ran with project conda env on this commit
(`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`).

**Agent-config probe** — for all 3 new agent configs, `Config.load_yaml(...).get_mandatory('agent.imagined_rollout_probe', bool)` returns `True`; mandatory-key discipline holds; `agent.algorithm == 'DreamerV3'`; `entropy_scale` and `cont_loss_weight` cast cleanly to float at the planned values:

```text
configs/models/dreamer_v3_probe.yaml
  algorithm              = 'DreamerV3'
  imagined_rollout_probe = True (type bool)
  entropy_scale          = 0.0003
  cont_loss_weight       = 1.0
configs/models/dreamer_v3_probe_cont10.yaml
  algorithm              = 'DreamerV3'
  imagined_rollout_probe = True (type bool)
  entropy_scale          = 0.0003
  cont_loss_weight       = 10.0
configs/models/dreamer_v3_curriculum_probe.yaml
  algorithm              = 'DreamerV3'
  imagined_rollout_probe = True (type bool)
  entropy_scale          = 0.001
  cont_loss_weight       = 5.0
```

**Cross-stage modality fingerprint** — for the 5 env YAMLs touched by the battery (E1/E2 anchor, E3 No-Pred, E4 curriculum's three stages), the 13-tuple `_modality_fingerprint` matches byte-for-byte:

```text
configs/experiment/basic/00-5X5_NoPred.yaml                       (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml      (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
configs/experiment/dreamer_curriculum/01_food_only.yaml           (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
configs/experiment/dreamer_curriculum/02_predator_slow.yaml       (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
configs/experiment/dreamer_curriculum/03_predator_full.yaml       (False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)
```

— so train.py's startup cross-stage-fingerprint probe (line 481, 519) will pass for E4. E1 vs E3 are also directly comparable on `Behavior/*` and `WorldModel/*` (no obs-space difference).

**Schedule YAML (E4)** — `configs/continual/dreamer_curriculum_food_then_predator.yaml` is unchanged (boundaries `[16000, 66000, 766000]`, ckpt freqs `[8000, 25000, 100000]`, 3 stages). This is the same schedule used by `95mpudwp`, so cross-run comparison on stage events / buffer-clear events / per-stage WandB sections is well-defined.

### 5.2 Single-seed limitation (all 4 cells)

- **Severity**: High for any quantitative claim about how `imagined_termination_fraction` correlates with survival.
- **Mitigation**: Pre-registered as observational. Any positive cell (E1 H₁ confirmed; or E2 H₁/H₂ confirmed) triggers a 3-seed follow-up at seeds {7, 13, 21} as the next experiment. The qualitative prediction "imagined termination fraction is essentially zero on the failing task" is robust to seed effects (the diagnostic battery's qont5dac signature was reproduced across A/B-Stage2/D-Stage3 with the same pattern).

### 5.3 Per-cell confounds

| Cell | Confound                                                                                                                | Severity         | Mitigation                                                                                                                                                          |
|------|-------------------------------------------------------------------------------------------------------------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| E1   | Probe is moderate cost per train step (O(B·H) reductions); could shift wall-clock                                       | Low              | Probe plan §Design decisions: cost is negligible vs. `ac_imagine_scan` itself. Verified zero-overhead when probe is false; expected <2% overhead when true.         |
| E2   | `cont_loss_weight=10` is the diagnosis-recommended 10×; never run before — first-time-this-knob risk                     | Medium           | `cont_loss_weight=5` was clean on `95mpudwp` (no NaN, model_loss decreased monotonically); 10× is a 2× extension along the same axis. Failure-mode catalog §6.2 covers blow-ups. |
| E3   | `00-5X5_NoPred` has only `hiding_predator` count=1 + starvation/MaxSteps as termination sources; less directly comparable to anchor's predator-driven terminations | Medium           | Pre-registered as a positive control (probe-validation), NOT a 1:1 mechanism comparison. The H₁(E3) criterion uses real_term_step_mean from THIS cell — self-consistent. |
| E4   | Replicates `95mpudwp` agent-side; only delta is `imagined_rollout_probe: true`. Probe is bit-identical when false (verified upstream) — so behavioral comparison vs. `95mpudwp` IS valid up to seed effects | Low              | Probe-disabled curriculum smoke test in the upstream plan (§Tests/verification 4) confirmed the disabled path is byte-identical. Probe-enabled adds only logging keys. |
| E4   | 3-stage schedule clears the buffer at each boundary (already documented); per-stage `imagined_termination_fraction` after a buffer clear is reading off a freshly-relearning WM | Medium           | This IS the experimental signal — we WANT to see the WM relearn (or fail to relearn) imagined terminations after each clear. Don't mitigate; describe in §6.            |

### 5.4 Cross-cell confounds

- **CUDA devices differ across cells** (cuda:0..3 on Node 114), but all are RTX 6000 Ada and the speed-profile diagnostic in `dreamer_v3_vs_rppo_speed_profile.md` confirmed device choice does not affect *learning*, only wall-clock.
- **Replay-buffer clearing at stage boundaries** affects only E4 (curriculum). E1/E2/E3 never trigger a clear (no schedule). This is a non-confound; it's a property of the experimental cell.
- **Probe key naming** — all six probe keys begin with `imagined_`, so they route under `WorldModel/` via train.py's WandB-grouping branch (probe plan Edit 2). Confirmed in upstream smoke tests.

---

## 6. Pre-registered Confirmation / Refutation Criteria (per cell)

All thresholds are on the **final 1000 logged windows of probe values** (probe is logged per-iteration per train.py log_interval=50; final 1000 iter ≈ last 50 000 iterations of training ≈ last ~3 % of the run). Survival thresholds are on the **final 1000 episodes** (project convention). For E4, "final-window" means within each stage's last 20 logged probe windows.

Probe key shorthand:
- `IMG_TF_15` ≡ `WorldModel/imagined_termination_fraction_h15`
- `IMG_TF_8`  ≡ `WorldModel/imagined_termination_fraction_h8`
- `IMG_FIRST` ≡ `WorldModel/imagined_first_term_step_mean`
- `REAL_MEAN` ≡ `WorldModel/imagined_real_term_step_mean`
- `EP_STEPS`  ≡ `Episode/Steps` final-1000-ep mean

### 6.1 E1 — Anchor + probe (mechanism confirmation)

- **C-E1a (full confirmation)**: `IMG_TF_15` final-1000-iter median **< 0.10** AND `REAL_MEAN` final-window mean **≤ 30** (real episodes are short — confirms the qont5dac signature is present on the real side) AND `IMG_FIRST` final-window mean **> 13** (essentially the HORIZON=15 sentinel — imagined rollouts almost never terminate).
- **C-E1b (qont5dac signature reproduces on this run)**: `EP_STEPS` ≤ 35 AND `Behavior/mean_entropy` end-window > 1.6 AND `Behavior/mean_advantage` end-window std < 0.01. Required co-confirmation that we're observing the failing regime — without it, a near-zero termination fraction is uninterpretable (it could just mean the agent IS surviving, which is incompatible with H₁'s premise).
- **R-E1a (refutation)**: `IMG_TF_15` final-1000-iter median **> 0.5** — imagined rollouts DO terminate. Refutes the WM-imagination structural hypothesis. Diagnostic-battery's surviving inference would be wrong; mechanism is something else (actor-side, value-loss, etc.).
- **R-E1b (instability)**: NaN in any loss; or `loss_model` rises monotonically over any 5 M env-step window. Refute the run, not the hypothesis. Re-run.
- **Verdict mapping**: C-E1a + C-E1b both fire → H₁(E1) confirmed (headline result). C-E1a fires but C-E1b fails → run is in a non-failing regime; cell uninformative for H₁; report and move on. R-E1a fires → H₀(E1) wins; new diagnosis required.

### 6.2 E2 — Anchor + cont_loss_weight=10 + probe (priority-2 fix test)

- **C-E2a (full fix, H₁(E2))**: `IMG_TF_15` final-1000-iter median **> 0.30** AND `EP_STEPS` final-1000-ep mean **> 100**. The cont-weight knob both makes the WM imagine deaths AND unlocks survival.
- **C-E2b (mechanism right, fix insufficient — H₂(E2))**: `IMG_TF_15` final-1000-iter median **> 0.30** AND `EP_STEPS` final-1000-ep mean **≤ 50**. The cont-weight knob does what it advertises but downstream pieces still fail. Recommended next step: actor-loss / value-loss probe.
- **C-E2c (cont-weight regression to E1)**: `IMG_TF_15` final-1000-iter median **< 0.10** (matches E1's expected ≈0). The knob did NOT lift the imagined-termination fraction — refutes the diagnosis's priority-2 fix. Recommended next step: revisit the priority-3/4 alternatives (class-balanced cont loss, reward-head per-bin reweighting).
- **R-E2 (instability)**: same as R-E1b. Re-run.
- **Verdict mapping**: C-E2a → H₁(E2) confirmed (priority-2 fix works). C-E2b → H₂(E2) — fix mechanism right, more work needed. C-E2c → H₀(E2) — fix doesn't even move the probe; the recommendation in diagnosis §7.3 priority 2 is wrong.

### 6.3 E3 — `00-5X5_NoPred` + probe (positive control)

- **C-E3a (probe works on a tractable termination distribution — H₁(E3))**: `IMG_TF_15` final-1000-iter median **>= 0.30** OR (more precisely, if real terminations are sparse on No-Pred too) `IMG_TF_15` tracks the empirical real-rollout termination fraction `min(1, 15 / REAL_MEAN)` to within **30 % relative error**. (E3 has only starvation / MaxSteps / hiding-predator as termination paths; expect `REAL_MEAN` higher than the full-predator anchor's ~28; we accept either an absolute or relative criterion).
- **C-E3b (DreamerV3 actually learns this task — sanity check)**: `EP_STEPS` final-1000-ep mean **> 100** (the project's "real learning" floor). Without this, the cell is uninterpretable (a failing No-Pred run is too weird to read).
- **R-E3a (probe broken or mechanism is broader)**: `IMG_TF_15` final-1000-iter median **< 0.10** AND `EP_STEPS` final-1000-ep mean **> 100**. The agent learned the task, episodes really do end (starvation / MaxSteps), but the WM still doesn't imagine terminations. Either probe instrumentation bug OR mechanism extends past predator-deaths. Either way, the failing-task probe in E1 needs reinterpretation.
- **R-E3b**: same as R-E1b. Re-run.
- **Verdict mapping**: C-E3a + C-E3b → probe validated; H₁(E1) interpretation safe. R-E3a fires → escalate (probe-instrumentation bug investigation OR widen the mechanism hypothesis).

### 6.4 E4 — Curriculum + levers + probe (per-stage breakdown)

Per-stage evaluation; "end-window" = each stage's last 20 logged probe windows (~ last 1000 train iterations of that stage). Stage-transition events (`stage/transition` 0→1 at ep~16000, 1→2 at ep~66000) are read from WandB to align analysis windows.

- **C-E4a (Stage 0 — food_only — has tractable terminations)**: Stage 0 end-window median `IMG_TF_15` **>= 0.20**.
- **C-E4b (Stage 1 — predator_slow — has tractable terminations)**: Stage 1 end-window median `IMG_TF_15` **>= 0.15**.
- **C-E4c (Stage 2 collapse — the headline finding for E4)**: Stage 2 end-window median `IMG_TF_15` **<= 0.10** AND `IMG_TF_15` first-window-after-boundary > 0.15 dropping to <= 0.10 within ~3,200 ep of the Stage 1→2 boundary (= ~50 logged probe windows). Visualises the moment WM imagination loses death events.
- **R-E4a (uniform success — H₂(E4))**: All 3 stages show end-window median `IMG_TF_15` **> 0.20**. Curriculum's Stage 2 collapse is NOT explained by the WM losing termination prediction; some other mechanism kicks in. Recommend value-loss / temporal-placement probe as next step.
- **R-E4b (uniform zero — H₀(E4))**: All 3 stages show end-window median `IMG_TF_15` **< 0.10**. Curriculum's Stage 0/1 unfreeze in `95mpudwp` was NOT driven by the WM learning to imagine terminations. Significantly weakens the diagnostic battery's inferred mechanism.
- **R-E4c (instability)**: same as R-E1b.
- **Verdict mapping**: C-E4a + C-E4b + C-E4c all fire → H₁(E4) confirmed (per-stage temporal localization established; this is the directly-observed counterpart to the diagnostic battery's §11.4 inference). R-E4a → H₂(E4); recommend follow-up. R-E4b → H₀(E4); rethink mechanism.

### 6.5 Cross-cell verdict matrix (post-run)

After all 4 cells return, fill this matrix:

| (E1 verdict, E2 verdict)            | Interpretation                                                                                                                                                                                                              |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H₁(E1), H₁(E2)                      | **Headline best case.** Mechanism confirmed (E1) + priority-2 fix works (E2). Recommend 3-seed E2 confirmation; integrate `cont_loss_weight=10` into the curriculum config; deprioritise other diagnosis-§7.3 levers.       |
| H₁(E1), H₂(E2)                      | **Mechanism confirmed; fix insufficient.** WM does imagine terminations under `cont_loss_weight=10`, but actor/value can't exploit them. Recommend value-loss + actor-loss probes as next-priority.                          |
| H₁(E1), H₀(E2)                      | **Mechanism confirmed; cont-weight knob is the wrong dial.** Recommend class-balanced cont loss (diagnosis §7.3 priority 3) and reward-head per-bin reweighting (priority 4) as alternative interventions.                   |
| H₀(E1), any E2                      | **Mechanism refuted.** Diagnostic battery's surviving inference is wrong. Re-run priority hypothesis generation. E2 results are uninterpretable in this regime (cont_loss_weight has no clear target to fix).                |

E3 and E4 modulate the above:
- E3 H₀ → reinterpret all of E1/E2/E4 with caution (probe may not be capturing what we think).
- E4 H₁(E4) per-stage collapse → the strongest observational evidence for the mechanism, regardless of E1/E2 outcomes.
- E4 H₂(E4) (uniform success) → pushes against H₁(E1); the curriculum + `cont=5` keeps the WM imagining terminations even on Stage 2. If E1 also shows H₁(E1) (`cont=1`, Stage 2 anchor), the difference between `cont=1` and `cont=5` is the structural pivot.

---

## 7. Failure-Mode Catalog (pre-decided interpretations)

| Observation                                                                                                  | Interpretation                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| All 4 cells refute (E1 R-E1a, E2 C-E2c, E3 R-E3a, E4 R-E4b)                                                  | Probe instrumentation is broken OR the mechanism is something the probe cannot measure. Spawn `code-reviewer` on the probe diff; rerun with synthetic offline checkpoints to validate.                                                  |
| Only E1 H₁ confirms (E2 C-E2c, E3 R-E3a, E4 R-E4b)                                                            | Probe works on the failing regime but reads zero everywhere else — strange. Possible explanation: `IMG_TF_15` saturates near zero whenever `EP_STEPS < some_floor`. Recommend probe re-instrumentation around floor regimes.            |
| E1 H₁ + E2 H₁ + E3 H₁ + E4 H₁ all confirm                                                                    | **Cleanest headline win.** The structural mechanism is confirmed (E1+E4-Stage2), the probe is validated (E3), the priority-2 fix works (E2). Promote `cont_loss_weight=10` to the curriculum config; design 3-seed confirmation battery.   |
| E1 H₁, E2 H₂, E3 H₁, E4 H₁ — mechanism right, cont-weight fix moves probe but not survival                    | Promote to actor/value probe diagnostics (diagnosis §7.3 priorities 5/6). Document the cont-weight knob as "necessary but not sufficient" alongside the curriculum levers.                                                              |
| E2 shows NaN / instability while `cont_loss_weight=10`                                                       | The 10× weight overpowers KL/recon terms. Refute E2 only; re-run with `cont_loss_weight=7` (intermediate). Don't confuse "knob causes blow-up" with "knob doesn't fix the mechanism".                                                  |
| E3 shows non-zero `IMG_TF_15` AND `EP_STEPS > 100` but `Behavior/mean_entropy` < 0.5 with FoodEaten saturating max | E3 trains quickly, leaves the failing regime, and might saturate before the end of the 700k-ep budget. The probe's final-1000-iter median is still meaningful, but secondary metrics may be in a converged regime. Document; not a failure.  |
| E4 Stage 0 has `IMG_TF_15` near 1.0 (not just > 0.20)                                                         | Food-only episodes are short due to starvation — ~150 steps at design — but the WM might be imagining starvation terminations that real Stage 0 episodes don't actually hit (starvation takes longer than HORIZON=15). Annotate: this is a "wrongly-imagined termination" signal, not the same as Stage 1/2's "correctly-imagined termination" signal. Doesn't refute H₁(E4); enriches it. |
| E4 Stage 2 → 0 transition is abrupt within 100 ep (faster than ~3,200)                                       | Even cleaner observation of the inferred mechanism. Annotate but no special handling.                                                                                                                                                   |
| E4 Stage 2 → 0 takes >50 000 ep                                                                              | Stage 2 collapse is slower than the diagnostic battery's `95mpudwp` 3,200-ep tolerance. Possibly probe-induced stochasticity (probe is logging-only; no behavioral coupling) — investigate post-run.                                     |
| Wall-clock per cell exceeds 5 h                                                                              | Node 114 contention or unexpected per-iteration slowdown. Halt mid-run and notify user.                                                                                                                                                  |

---

## 8. Plumbing-check table (post-run fill-in by `experiment-analyzer`)

For each cell, after launch:

- [ ] Single WandB run-ID assigned, state=finished
- [ ] No startup ValueError
- [ ] `agent.imagined_rollout_probe = True` in `wandb.config`
- [ ] `agent.entropy_scale` in `wandb.config` matches the planned value (3e-4 / 3e-4 / 3e-4 / 1e-3 for E1/E2/E3/E4)
- [ ] `agent.cont_loss_weight` in `wandb.config` matches the planned value (1.0 / 10.0 / 1.0 / 5.0 for E1/E2/E3/E4)
- [ ] All 6 probe keys present under `WorldModel/`: `imagined_termination_fraction_h{8,15}`, `imagined_first_term_step_mean`, `imagined_term_step_p{10,50,90}`, `imagined_real_term_step_mean`
- [ ] `imagined_termination_fraction_h{8,15}` ∈ [0, 1] over full run
- [ ] `imagined_first_term_step_mean` ∈ [0, 15] over full run
- [ ] No NaN warnings in any logged metric
- [ ] (E4 only) Schedule plumbing — 3 stages to ep ~766 000; cross-stage modality-fingerprint probe passed; `stage/transition` events fired at expected episodes; `stage/buffer_cleared_main` fired at both transitions; per-stage `agent.entropy_scale = 1e-3` and `cont_loss_weight = 5.0` constant in `wandb.config`
- [ ] Wall-clock < 5 h per cell

---

## 9. Analysis Plan (post-run)

Single seed per cell → no inferential statistics. Post-run fill-in for §10/§11/§12 (sections to be added by `experiment-analyzer`) will:

1. **Plumbing checks first** (per cell, against §8).
2. **Per-cell temporal evolution** of probe keys + Behavior/* + Episode/* across the run, plotted on the `Episode/Number` axis with vertical lines at stage boundaries (E4 only).
3. **Confirm/refute walk** across §6 criteria, recording per-cell verdict.
4. **Cross-cell verdict matrix** (§6.5) — fill the (E1, E2) row and annotate with E3/E4 modulation.
5. **Cross-run comparison** — overlay E4's per-stage `IMG_TF_15` against `95mpudwp`'s per-stage `Behavior/*` (`95mpudwp` does NOT have probe data, but its Stage 0/1 unfreeze and Stage 2 collapse temporal-evolution panels are aligned by episode count).
6. **Save the analysis to a stand-alone results section** in this doc (§10–§12 added in place; do not start a new doc).

---

## 10. Cross-link map (for downstream agents)

- Probe instrumentation contract → [`docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`](../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md). The 6 key names this battery reads come from there. Any change to the probe's key names or types invalidates this doc's confirm/refute thresholds.
- Surviving structural hypothesis → [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md) §6.2 / §7.3 priority 1 + 2.
- Diagnostic battery refutations that motivated this battery → [`DREAMER_DIAGNOSTIC_BATTERY.md`](./DREAMER_DIAGNOSTIC_BATTERY.md) §11.4 cross-cell synthesis, §12.2 priority-1.
- Curriculum + levers reference run (E4 replicates with probe) → [`DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md`](./DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md).
- Continual-learning feature contract (E4 schedule path) → [`docs/develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../../../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md).

---

## Appendix

### A. Lever / probe values across cells

| Lever                               | E1                  | E2                   | E3                  | E4                                    | Failing baseline (`qont5dac`) | 95mpudwp (curriculum+levers) |
|-------------------------------------|---------------------|----------------------|---------------------|---------------------------------------|-------------------------------|-----------------------------|
| `agent.entropy_scale`               | 3e-4                | 3e-4                 | 3e-4                | **1e-3**                              | 3e-4                          | 1e-3                        |
| `agent.cont_loss_weight`            | 1.0                 | **10.0**             | 1.0                 | 5.0                                   | 1.0                           | 5.0                         |
| `agent.imagined_rollout_probe`      | **true**            | **true**             | **true**            | **true**                              | false                         | false                       |
| Curriculum stages                   | 1                   | 1                    | 1                   | **3**                                 | 1                             | 3                           |
| Env                                 | `01-PredInterval3`  | `01-PredInterval3`   | **`00-5X5_NoPred`** | curriculum (3 stages)                 | `01-PredInterval3`            | curriculum (3 stages)       |
| `body.use_homeostatic_reward`       | true                | true                 | true                | true                                  | true                          | true                        |
| Episode budget                      | 700 000             | 700 000              | 700 000             | 766 000 (schedule total)              | ~909 000                      | ~766 000                    |

### B. Probe-key glossary (from probe plan)

| Key                                                | Type    | Meaning                                                                                           |
|----------------------------------------------------|---------|---------------------------------------------------------------------------------------------------|
| `WorldModel/imagined_termination_fraction_h8`      | float   | Fraction of imagined trajectories with at least one `cont < 0.5` step in the first 8 steps.       |
| `WorldModel/imagined_termination_fraction_h15`     | float   | Same, full HORIZON=15.                                                                            |
| `WorldModel/imagined_first_term_step_mean`         | float   | Mean step index of first termination across imagined batch; HORIZON=15 sentinel if never.         |
| `WorldModel/imagined_term_step_p10`                | float   | 10th-percentile first-termination step.                                                           |
| `WorldModel/imagined_term_step_p50`                | float   | Median.                                                                                           |
| `WorldModel/imagined_term_step_p90`                | float   | 90th percentile.                                                                                  |
| `WorldModel/imagined_real_term_step_mean`          | float   | Real-batch first-termination step (sentinel = `seq_len`); for parity with the imagined probe.     |

### C. Changelog

| Date       | Change                                                                                                 | Author              |
|------------|--------------------------------------------------------------------------------------------------------|---------------------|
| 2026-05-08 | Initial design + 3 new agent configs + Launch Manifest. Live-load + cross-stage fingerprint verified. | experiment-designer |
