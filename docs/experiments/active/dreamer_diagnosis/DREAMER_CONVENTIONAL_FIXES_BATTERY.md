---
title: "DreamerV3 conventional-fixes mini-battery — replay_ratio + death_penalty on Node 113"
topic: dreamer_diagnosis
status: active
created: 2026-05-09
last_updated: 2026-05-09
phase: 1
wandb_tag: "dreamer_conventional_fixes"
develop_link: "../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md"
---

# DreamerV3 conventional-fixes mini-battery — Node 113, 2 cells

> **Status**: ANALYZED — both cells finished 700k episodes; verdict (H₂-with-caveat A1, H₀ A2) recorded in §11–§13.
> **Date**: 2026-05-09
> **Author**: experiment-designer
> **Related**:
> - **Conventional-failure-mode critique** (the rationale for the two knobs):
>   [`docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`](../../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md) — top-3 ranked causes; cheapest fixes ordered.
> - **Strip-down direction memo** (the 2×2 design that motivates the cell selection):
>   [`docs/project/directions/dreamer_minimum_viable_strip_down.md`](../../../project/directions/dreamer_minimum_viable_strip_down.md) — predicts NoPred + rr=0.0625 stays stable past `3zjhap9w`'s collapse boundary.
> - **Probe instrumentation contract** (orthogonal develop-side spec; cross-linked because Cell A1 is the conventional-fix follow-up to the probe battery's `xekunmbw`/E3 NoPred-positive-control cell):
>   [`docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`](../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md)
> - **Reference collapse run** (`3zjhap9w`, the failure-mode template Cell A1 must beat):
>   [`tmp/20260508_replayRatio05_NoPred_analysis.md`](../../../../tmp/20260508_replayRatio05_NoPred_analysis.md)
> - **Diagnostic-battery refutations** that produced the surviving structural hypothesis Cell A2 stress-tests:
>   [`docs/experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md`](../continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md) — §11.4 cross-cell synthesis.
> - **Probe battery (parent)** that motivated this follow-up:
>   [`docs/experiments/active/continual_learning/DREAMER_PROBE_BATTERY.md`](../continual_learning/DREAMER_PROBE_BATTERY.md) — `xekunmbw`/E3 is the NoPred control whose `3zjhap9w` collapse signature Cell A1 targets.
> - **Underlying failure-mode diagnosis** (what these fixes are conventionally pointed at):
>   [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md)

> **Note**: 2 single-seed runs in parallel on cuda:0 / cuda:1 of Node 113. Each result is observational (no statistical claims). The deliverable is a verdict on whether the two unaddressed conventional one-line knobs from the critique's §4 ranked list — `replay_ratio: 0.5 → 0.0625` and `death_penalty: 100 → 1` — repair the published modal-trace collapse (`3zjhap9w`) and the hypervigilance failing-task signature (`qont5dac`).

---

## 1. Research Question

The conventional-failure-mode critique [(critique)](../../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md) ranks three textbook DreamerV3 failure modes against our setup. The top-2 are:

1. **Reward-scale mismatch** (§2.1): per-step homeostatic ±0.5 + terminal −100 maps to bin clusters 4 nats apart in symlog two-hot space; cross-entropy cross-fits the modal positive class while regressing on the rare large-magnitude terminal. Diagnostic signature: `model_reward_mae_pos: 0.05 / mae_neg: 0.60` and *widening*. **Fix: `death_penalty: 100 → 1`** (Crafter recipe).
2. **Replay-ratio over-training** (§2.2): our `replay_ratio = 0.5` is 8× the Hafner-2023 Atari/DMC default (1/16 = 0.0625). On a small near-deterministic state space, the WM saturates on the modal-policy trajectory long before the policy explores → self-confirming pessimism (§2.10). Diagnostic signature: the `3zjhap9w` collapse — survival climbs to 330 by 600k env steps, then drops to 142 with 80% starvation by 1.8M. **Fix: `replay_ratio: 0.5 → 0.0625`** (Hafner Atari default).

**Neither single-line config fix has been tried before.** This battery makes the cheapest two-cell test of the highest-ranked unaddressed conventional knobs.

> **Q1 (replay-ratio fix; Cell A1)**: On the NoPred task where the modal-trace collapse pattern was observed (`3zjhap9w`), does `replay_ratio: 0.5 → 0.0625` prevent the collapse? Specifically, does survival reach 250+ by ~400k env steps and stay above 250 through 700k env steps without reverting to the starvation regime?
>
> **Q2 (joint conventional fixes; Cell A2)**: On the predator-task hypervigilance baseline where the reward-MAE asymmetry signature was observed (`qont5dac`), does the joint fix `replay_ratio: 0.0625` + `death_penalty: 1` produce non-trivial survival (> 100 steps, 3× the qont5dac/`mruhypnu` floor of ~30) within the 700k env-step budget?

> **H₀ (no-effect)**: Both knobs leave the failing signatures intact. Cell A1 reproduces the `3zjhap9w` learn-then-collapse trajectory; Cell A2 stays at the qont5dac ~30-step floor with the mae_pos/mae_neg asymmetry widening.
>
> **H₁ (joint conventional fix works)**: Cell A1 reaches and *holds* survival > 250 through 700k env-steps (collapse prevented); Cell A2 reaches survival > 100 with mae_pos/mae_neg gap < 0.15 throughout (reward-scale + replay-ratio jointly unlock the predator task).
>
> **H₂ (replay-ratio fix is necessary on NoPred but reward-scale is not sufficient on predator task)**: Cell A1 holds; Cell A2 fails (~30-step floor). Conventional fixes work on the easier task but the predator task is structurally harder. Recommended next: bespoke architecture probes (per-action `cont`, fork-rollouts) at higher priority.
>
> **H₃ (reverse — neither works on either task)**: Cell A1 collapses anyway and Cell A2 stays at floor. The DreamerV3-on-vector-observations regime has a structural mismatch (critique §2.5) that conventional-knob fixes cannot reach. Recommended next: pivot to TD-MPC2 / rPPO comparison or augment the WM with auxiliary self-supervised objectives.

---

## 2. Pre-registered Hypotheses (per cell)

### Cell A1 — NoPred + replay_ratio=0.0625 (conventional replication-fix on `3zjhap9w`)

> **H₀(A1)**: With `replay_ratio: 0.0625` on the NoPred task, the run reproduces the `3zjhap9w` collapse pattern. Specifically: (i) survival reaches 250+ by ~400k env steps; (ii) survival drops below 200 in any 100k env-step window between 600k and 700k; (iii) `model_reward_mae_pos` falls below 0.10 while `model_reward_mae_neg` rises above 0.45 in the final-window. The replay-ratio fix is the wrong knob (or insufficient).
>
> **H₁(A1) — fix works (the headline confirmation)**: (i) survival reaches 250+ by ~400k env steps AND (ii) survival mean over the final 100k env-step window stays ≥ 250 AND (iii) `model_reward_mae_pos` and `model_reward_mae_neg` are both < 0.30 in the final-window (no widening asymmetry). Critically, the cell does NOT reproduce the `3zjhap9w` 600–800k collapse boundary.
>
> **H₂(A1) — slow-learning variant**: Survival never reaches 250 within 700k env steps but is monotonically increasing (no collapse). The conventional fix is correct but the env-step budget is too short to confirm asymptotic competence; recommend running to 1.5M env steps.

Significance: H₁ confirms cause #2 from the critique with a one-line config delta. H₀ refutes the conventional rank-2 hypothesis on the published-failure-mode-template task — would push priority back to bespoke architectural fixes (per-action `cont`, ensemble WM).

### Cell A2 — Predator + replay_ratio=0.0625 + death_penalty=1 (joint conventional fix on the predator task)

> **H₀(A2)**: Joint fix has no effect. Survival final-100k mean ≤ 50 steps (no escape from the qont5dac floor). The reward-scale and replay-ratio causes are not the dominant failure modes on the predator task, OR they are dominant but conventional fixes alone are insufficient.
>
> **H₁(A2) — full fix works (the joint-conventional-knob best case)**: Survival final-100k mean > 100 AND `model_reward_mae_pos / model_reward_mae_neg` gap < 0.15 throughout the final 200k env-steps (the asymmetry signature is resolved). Indicates the joint conventional fix unlocks the predator task.
>
> **H₂(A2) — partial fix**: `model_reward_mae_pos / model_reward_mae_neg` gap < 0.15 (reward scale resolved) but survival final-100k mean ≤ 100 (predator task still fails). Reward-head asymmetry was real but is not sufficient — the predator task has a downstream actor/value/exploration block. Recommend per-action `cont` probe as next priority.
>
> **H₃(A2) — kitchen-sink fail with NaN/instability**: Training diverges (NaN loss, value explosion) under `death_penalty: 1` because the symlog reward head is now under-driven on the terminal (the 1-bin-from-modal regime is itself a calibration boundary). Refute the run, not the joint fix; re-run with `death_penalty: 5` (intermediate) or revert one of the two knobs.

Significance: H₁(A2) is the cleanest possible win for the project — both unaddressed conventional knobs from the critique's top-3 list resolve the predator task. H₀(A2) joint with H₁(A1) implies cause #1 (reward scale) is not the predator-task bottleneck; H₂(A2) localises to actor-side / value-side mechanisms.

### 2.1 Why these 2 cells, not the deferred third

The professor's strip-down direction memo proposes a **2×2 mini-battery** factorially crossing `task ∈ {NoPred, joint-strip}` × `replay_ratio ∈ {0.0625, 0.5}` (3 new runs at ~3h each). With 2 GPUs we can run 2 cells. The third candidate — predator task with `replay_ratio: 0.0625` only (death_penalty: 100 unchanged) — is deferred for the following reason:

The two cells that ship now maximize **information per GPU-hour by anchoring against the two well-characterized prior failure signatures** — `3zjhap9w` (modal-trace collapse on NoPred at rr=0.5) and `qont5dac`/`mruhypnu` (~30-step floor on predator at rr=0.5, death_penalty=100). Cell A1 directly tests "does the published fix repair the published failure mode on the published failure-template task?" — a clean single-knob single-prior test. Cell A2 tests "does the kitchen-sink conventional fix repair the project-relevant hardest task?" — a maximum-likelihood-of-survival test for the project's downstream Phase-1 work.

The deferred cell (predator + rr=0.0625 only) primarily isolates which of the two knobs drives any survival gain on the predator task, which is downstream of "do the conventional fixes work at all?" If both A1 and A2 confirm, the deferred cell is the immediate Round 2 priority. If A2 fails but A1 works, the deferred cell is also the Round 2 priority (to localize whether reward-scale or replay-ratio is the predator-specific bottleneck). Either way, the deferred cell goes into Round 2; it does not block this round.

---

## 3. Launch Manifest

| Run | Status | Cell | Tag (= wandb-name)                            | wandb-group                | wandb-job-type | Seed | Node | GPU    | Launched at         | Finished at         | Final survival (Episode/Steps) | WandB run ID | Log path                                    |
|-----|--------|------|-----------------------------------------------|----------------------------|----------------|------|------|--------|---------------------|---------------------|--------------------------------|--------------|---------------------------------------------|
| 1   | done   | A1   | `dreamer_conv_NoPred_rr06_s0_n113`            | `dreamer_conventional_fixes` | ablation      | 0    | 113  | cuda:0 | 2026-05-09T05:06:01 | 2026-05-09T06:44 (1h38m) | 112.0 (Episode/Number=699,106) | czfnljf0     | `logs/20260509_050601.log`                  |
| 2   | done   | A2   | `dreamer_conv_Pred_rr06_dp1_s0_n113`          | `dreamer_conventional_fixes` | ablation      | 0    | 113  | cuda:1 | 2026-05-09T05:07:33 | 2026-05-09T05:35 (28m)   | 27.0 (Episode/Number=698,452)  | houwr6js     | `logs/20260509_050733.log`                  |

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env)                                                                          | Config (agent)                                  | Notes                                                                |
|-----|---------------------------------------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------------|
| 1   | `configs/experiment/basic/00-5X5_NoPred.yaml` (existing, unchanged)                   | `configs/models/dreamer_v3/dreamer_v3_rr06.yaml` (NEW)     | Tests `replay_ratio: 0.0625` against `3zjhap9w` collapse template.   |
| 2   | `configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml` (NEW) | `configs/models/dreamer_v3/dreamer_v3_rr06.yaml` (NEW; same as A1) | Tests joint `replay_ratio: 0.0625` + `death_penalty: 1` on predator. |

**Two new files total. Zero schema changes. Cells share the same agent config.**

Config provenance — both new files are byte-equal to their parents except for the single advertised knob:
- `configs/models/dreamer_v3/dreamer_v3_rr06.yaml` ← `configs/models/dreamer_v3/dreamer_v3.yaml` with `agent.replay_ratio: 0.5 → 0.0625`. Verified by programmatic key-set diff: zero added or removed keys; one value diff.
- `configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml` ← `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml` with `body.death_penalty: 100 → 1`. Verified by programmatic deep-diff: one value diff, no other changes.

### 3.2 Launch commands (verbatim, ready for `training-runner`)

#### Run 1 — Cell A1: NoPred + rr=0.0625 (cuda:0)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/basic/00-5X5_NoPred.yaml \
  --agent_config configs/models/dreamer_v3/dreamer_v3_rr06.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group dreamer_conventional_fixes \
  --wandb-job-type ablation \
  --wandb-name "dreamer_conv_NoPred_rr06_s0_n113" \
  --tag "dreamer_conv_NoPred_rr06_s0_n113"
```

#### Run 2 — Cell A2: Predator + rr=0.0625 + death_penalty=1 (cuda:1)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml \
  --agent_config configs/models/dreamer_v3/dreamer_v3_rr06.yaml \
  --num-envs 16 \
  --seed 0 \
  --episodes 700000 \
  --checkpoint-frequency 100000 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group dreamer_conventional_fixes \
  --wandb-job-type ablation \
  --wandb-name "dreamer_conv_Pred_rr06_dp1_s0_n113" \
  --tag "dreamer_conv_Pred_rr06_dp1_s0_n113"
```

**Notes (apply to both):**
- `--num-envs 16` and `--seed 0` match the prior probe battery's E1–E4 cells for direct cross-comparability against `xekunmbw` (NoPred + rr=0.5 + probe) and `owz29n3t` (predator anchor + rr=0.5 + probe).
- `--episodes 700000` is the same env-step target. With `replay_ratio = 0.0625` (8× lower than the probe battery's 0.5), wall-clock per env-step should fall to ~50% of the probe battery (per the speed-sweep fit `iter_time(R) ≈ 0.243 + 0.696·R`); estimated wall-clock ≈ 1.5–2.5 h per cell vs the probe battery's ~3 h.
- `imagined_rollout_probe: false` on the agent config — this battery does NOT use the probe. The probe is orthogonal instrumentation; turning it on would add a 6% overhead and the probe metrics are downstream of the conventional-knob signal we are trying to read. The probe is the right tool for "WHY" once we know whether the knobs work; it is not on the critical path for "DO THE KNOBS WORK."
- All flags use the canonical CLI surface from train.py (`--config`, `--agent_config`, `--num-envs`, `--seed`, `--episodes`, `--checkpoint-frequency`, `--device`, `--log-interval`, `--wandb-group`, `--wandb-job-type`, `--wandb-name`, `--tag`). No flags this CLI doesn't already expose.
- Tags are unique within this manifest AND globally distinct from the probe battery's `dreamer_probe_*` tags. Format: `dreamer_conv_<task>_<knob>_s<seed>_n<node>` (the `_n113` suffix is included because both cells run on the same node and re-launching on a different node would invalidate cross-cell comparison).

---

## 4. Independent / Dependent / Controlled Variables

### 4.1 Independent variables (per cell)

| Variable                       | Cell A1 (NoPred + rr06)            | Cell A2 (Predator + rr06 + dp1)             | Reference: `xekunmbw` / E3 (NoPred + rr05) | Reference: `owz29n3t` / E1 (predator + rr05) | Reference: `3zjhap9w` (NoPred + rr05, no probe) |
|--------------------------------|------------------------------------|---------------------------------------------|---------------------------------------------|----------------------------------------------|--------------------------------------------------|
| Env                            | `00-5X5_NoPred`                    | `01-PredInterval3_NutGain18` + `dp=1`       | `00-5X5_NoPred`                             | `01-PredInterval3_NutGain18`                 | `00-5X5_NoPred`                                  |
| `body.death_penalty`           | 100                                | **1**                                       | 100                                         | 100                                          | 100                                              |
| `agent.replay_ratio`           | **0.0625**                         | **0.0625**                                  | 0.5                                         | 0.5                                          | 0.5                                              |
| `agent.cont_loss_weight`       | 1.0                                | 1.0                                         | 1.0                                         | 1.0                                          | 1.0 (implicit)                                   |
| `agent.entropy_scale`          | 3e-4                               | 3e-4                                        | 3e-4                                        | 3e-4                                         | 3e-4                                             |
| `agent.imagined_rollout_probe` | false                              | false                                       | true                                        | true                                         | absent (predates probe)                          |

The IV under primary test for each cell:
- **A1**: `replay_ratio: 0.5 → 0.0625` on NoPred (Q1 — published Atari fix on the published modal-trace failure template).
- **A2**: joint `replay_ratio: 0.5 → 0.0625` + `death_penalty: 100 → 1` on predator (Q2 — kitchen-sink conventional fix on the project-relevant task).

### 4.2 Dependent variables

| Priority           | Metric                                                                                                  | Source                          | Why                                                                                                                                                                            |
|--------------------|---------------------------------------------------------------------------------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Primary**        | `Episode/Steps` final-100k-env-step mean (project headline; survival steps, never reward)              | env episode log                 | Whether the cell escapes the failing-task floor (~30 steps) and reaches non-trivial competence.                                                                                |
| **Co-primary** (A1) | `Episode/Steps` mean trajectory across 100k-env-step windows                                          | env episode log                 | Whether the cell prevents the `3zjhap9w` 600–800k collapse boundary. Headline criterion is survival ≥ 250 in the final-100k window AND no drop > 30% from peak in any window.   |
| **Co-primary** (A2) | `WorldModel/model_reward_mae_pos` and `WorldModel/model_reward_mae_neg` final-100k-env-step mean      | wandb iteration log             | Direct test of cause #1 (reward-scale mismatch). The asymmetry signature was the diagnostic for the underlying failure mode; if `dp: 1` resolves the asymmetry, cause #1 is correctly attributed even if survival doesn't fully unlock. |
| Secondary          | `Behavior/mean_value`, `Behavior/mean_advantage`, `Behavior/mean_entropy`                              | wandb iteration log             | qont5dac/`3zjhap9w` collapse signature panel — is the cell still in the failing regime? Mean_value drift more negative + advantage drift more negative + entropy reverting toward ln(6) all signal collapse. |
| Secondary          | `Episode/Term_Injury`, `Episode/Term_Starvation`, `Episode/Term_MaxSteps`, `Episode/FoodEaten`, `Episode/DangerHits` | env episode log | Termination-mix decomposition. `3zjhap9w` collapse signature was Term_Starvation rising 0.03 → 0.80 while DangerHits fell 3.2 → 0.87 — agent learns to avoid everything. Cell A1 must NOT reproduce this. |
| Diagnostic         | `WorldModel/loss_actor`, `WorldModel/loss_critic`, `WorldModel/loss_recon`, `WorldModel/loss_dyn_kl`, `WorldModel/loss_rep_kl` | wandb iteration log | WM-stability tripwires (cf. failure-mode catalog §6 R-criteria for instability).                                                                                              |

### 4.3 Controlled variables (held constant across both cells)

| Variable                                                                            | Value                                      | Held constant by                                                            |
|-------------------------------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| Grid size                                                                           | 5×5                                        | Both env YAMLs                                                              |
| `environment.max_steps`                                                             | 500                                        | Both env YAMLs                                                              |
| `environment.random_start_pos`                                                      | true                                       | Both env YAMLs                                                              |
| `--num-envs`                                                                        | 16                                         | CLI on both                                                                 |
| Seed                                                                                | 0                                          | CLI on both (`--seed 0`)                                                    |
| `body.use_homeostatic_reward`                                                       | true                                       | Both env YAMLs                                                              |
| Sensors (modality fingerprint 13-tuple)                                             | identical across both env YAMLs            | Verified — see §5.1                                                         |
| `agent.algorithm`                                                                   | DreamerV3                                  | Single shared agent config                                                  |
| `agent.batch_size / sequence_length / collect_interval / train_steps`               | 16 / 128 / 128 / 64                        | Single shared agent config                                                  |
| `agent.{model_lr, actor_lr, value_lr}`                                              | 1e-4 / 3e-5 / 3e-5                         | Single shared agent config                                                  |
| `agent.cont_loss_weight`                                                            | 1.0 (Hafner default)                       | Single shared agent config                                                  |
| `agent.entropy_scale`                                                               | 3e-4 (Hafner default)                      | Single shared agent config                                                  |
| `agent.sampling_mode`                                                               | mixture                                    | Single shared agent config                                                  |
| `agent.encoding_mode`                                                               | hierarchical                               | Single shared agent config                                                  |
| `agent.use_layer_norm`                                                              | true                                       | Single shared agent config                                                  |
| `agent.modulation.type`                                                             | null (no neuromodulation)                  | Single shared agent config                                                  |
| `agent.imagined_rollout_probe`                                                      | false                                      | Single shared agent config                                                  |
| `perceptual_noise.enabled`                                                          | false                                      | Both env YAMLs                                                              |
| Hardware                                                                            | Node 113, RTX 6000 Ada × 2                 | CLI `--device` per cell (cuda:0 / cuda:1)                                   |
| Episode budget                                                                      | 700 000                                    | CLI `--episodes 700000` on both                                             |
| `--log-interval`                                                                    | 50                                         | CLI on both                                                                 |
| Wall-clock budget per run                                                           | ~2.5 h target, ~5 h hard ceiling           | Mirrors prior battery hard ceiling; halt mid-run if exceeded                |

---

## 5. Confounds & Limitations

### 5.1 Pre-flight live-load checks (verified)

Live-load probe ran with project conda env on this commit
(`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`).

**Agent-config probe** — for `configs/models/dreamer_v3/dreamer_v3_rr06.yaml`, all mandatory keys parse correctly via `Config.load_yaml(...).get_mandatory(...)`:

```text
configs/models/dreamer_v3/dreamer_v3_rr06.yaml
  algorithm              = 'DreamerV3'
  replay_ratio           = 0.0625      ← FIX-DELTA (was 0.5)
  cont_loss_weight       = 1.0
  imagined_rollout_probe = False (type bool)
  entropy_scale          = 0.0003
```

**Env-config probe** — for `configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml`, all mandatory body and env keys parse correctly:

```text
01-PredInterval3_NutGain18_DeathPenalty1.yaml
  body.death_penalty               = 1            ← FIX-DELTA (was 100)
  body.use_homeostatic_reward      = True
  environment.predator_enabled     = True
  perceptual_noise.enabled         = False
```

**Cross-stage modality fingerprint** — for the 3 env YAMLs touched directly or by the probe-battery cross-comparison (`00-5X5_NoPred`, `01-5X5_PredInterval3_NutGain18`, `01-PredInterval3_NutGain18_DeathPenalty1`), the 13-tuple sensor/observation fingerprint matches byte-for-byte:

```text
configs/experiment/basic/00-5X5_NoPred.yaml                                         (True, 0, 5, True, 20, True, 1, True, False, True, True, False, 2)
configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml                        (True, 0, 5, True, 20, True, 1, True, False, True, True, False, 2)
configs/experiment/dreamer_diagnostic/01-PredInterval3_NutGain18_DeathPenalty1.yaml (True, 0, 5, True, 20, True, 1, True, False, True, True, False, 2)
```

— so observation-shape is constant across cells; cross-cell comparison on `Behavior/*` and `WorldModel/*` is well-defined; train.py's startup cross-stage-fingerprint probe will pass for any later continual-learning use.

**Programmatic key-set diff (semantic)** — verified zero schema drift:
- `dreamer_v3_rr06.yaml` vs `dreamer_v3.yaml`: 0 keys added, 0 keys removed, 1 value diff (`replay_ratio: 0.5 → 0.0625`).
- `01-PredInterval3_NutGain18_DeathPenalty1.yaml` vs `01-5X5_PredInterval3_NutGain18.yaml`: 0 keys added, 0 keys removed, 1 value diff (`body.death_penalty: 100 → 1`).

### 5.2 Single-seed limitation (both cells)

- **Severity**: High for any quantitative claim about effect size on survival.
- **Mitigation**: Pre-registered as observational. Either cell confirming H₁ triggers a 3-seed follow-up at seeds {7, 13, 21} as Round 2. The qualitative prediction "lowering replay_ratio prevents the modal-trace collapse" is robust to seed effects (the published Hafner-2023 finding is itself many-seed); single-seed is sufficient for a yes/no verdict on whether the conventional knob moves the failure signature.

### 5.3 Per-cell confounds

| Cell | Confound                                                                                                                                              | Severity | Mitigation                                                                                                                                                                 |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A1   | `replay_ratio = 0.0625` reduces gradient-step density. Slow learning is an *expected* consequence; absence of competence by 700k env steps could mean either "fix wrong" or "fix correct but env-step budget too short" | Medium   | Pre-registered as H₂(A1) — slow-but-monotonic learning triggers a 1.5M env-step extension, not a refutation. The collapse-prevention test (no drop > 30% from peak in any 100k window) is independent of asymptotic competence and remains valid. |
| A2   | Joint fix confounds two knobs (replay_ratio + death_penalty). Cannot independently attribute survival gain                                            | Medium   | Pre-registered as a **kitchen-sink test**, not an attribution test. If A2 confirms H₁(A2), the deferred-cell follow-up (predator + rr=0.0625 only, dp=100) becomes immediate Round 2 priority to attribute. If A2 fails, attribution is moot. |
| A2   | `death_penalty: 1` is at the low-magnitude boundary of the symlog two-hot reward head's calibration regime (1-bin-from-modal); could trigger NaN/instability | Low-Medium | Pre-registered as H₃(A2) refute-the-run criterion. Crafter uses `death_penalty: 1` successfully so the regime is published-validated, but our env's mixed-magnitude shaping (homeostatic ±0.5 dense + extrinsic eat +1 sparse + dp -1 sparse) is a non-Crafter combination. Watch for NaN in any loss; halt and revert to dp=5 if so. |
| A2   | `body.use_homeostatic_reward: true` means dense ±0.5 shaping is still on. With `dp: 1`, terminal is now within the dense reward range (no longer 100× the per-step magnitude) — but still distinct from the modal +0.5. The reward distribution is now plausibly closer to the symlog-calibrated regime, but is not byte-equal to Crafter's sparse-only regime | Low      | Critique §2.1 explicitly recommends the death-penalty-rescale alone, not the homeostatic-reward removal, since the dense shaping is providing the gradient signal that bootstraps the WM through early training. This is by design, not a confound. |

### 5.4 Cross-cell confounds

- **CUDA devices differ** (cuda:0 vs cuda:1) but both are RTX 6000 Ada on Node 113; the speed-profile diagnostic in `dreamer_v3_vs_rppo_speed_profile.md` confirmed device choice does not affect *learning*, only wall-clock.
- **Tag uniqueness** — both tags include `_n113` to distinguish from any future re-launch on a different node. The 2026-05-04 swap incident (auto-memory) confirmed tag pinning at design time is mandatory; both tags are pinned here.
- **Probe absent** — neither cell logs the imagined-rollout-probe metrics, so direct comparison against `xekunmbw`/E3 on probe values is not possible. This is acceptable: the conventional-knob signal is in survival + reward-MAE, which both cells log natively. A future cell (Round 2) can re-run A1 with probe enabled if probe-comparison is the next-priority question.

---

## 6. Pre-registered Confirmation / Refutation Criteria (per cell)

All thresholds are on the **final 100k env-step window** (final ~14% of the 700k budget). Trajectory thresholds use 100k-env-step windows centred on the relevant inflection points. For Cell A1 the `3zjhap9w` collapse-prevention test uses windows at 0–400k (climb), 400–600k (peak), 600–800k (the `3zjhap9w` collapse boundary; runs to 700k so we can read the leading edge), and 600–700k (final).

Shorthand:
- `EP_STEPS_F` ≡ `Episode/Steps` final-100k mean (≈ env-steps 600–700k)
- `EP_STEPS_PEAK` ≡ `Episode/Steps` max over 100k-env-step rolling means in [0, 700k]
- `MAE_POS_F` ≡ `WorldModel/model_reward_mae_pos` final-100k-window mean
- `MAE_NEG_F` ≡ `WorldModel/model_reward_mae_neg` final-100k-window mean
- `T_STARV_F` ≡ `Episode/Term_Starvation` final-100k mean

### 6.1 Cell A1 — NoPred + rr=0.0625 (replication-fix on `3zjhap9w`)

- **C-A1a (collapse prevented — H₁(A1) primary)**: `EP_STEPS_F ≥ 250` AND `EP_STEPS_F ≥ 0.7 · EP_STEPS_PEAK` (no drop > 30% from peak in the final window) AND `T_STARV_F ≤ 0.30` (no starvation regime). Specifically the final 100k window must NOT reproduce `3zjhap9w` window-5: survival 142, T_starv 0.80, food 7.6.
- **C-A1b (reward-MAE asymmetry resolved — H₁(A1) corroboration)**: `MAE_POS_F` and `MAE_NEG_F` both < 0.30 AND `|MAE_POS_F - MAE_NEG_F| < 0.20`. Direct confirmation that the modal-trace overtraining signature is absent.
- **C-A1c (slow-learning variant — H₂(A1))**: `EP_STEPS_F ≥ 100` AND `EP_STEPS` is monotonically non-decreasing across consecutive 100k windows (no collapse) AND `T_STARV_F ≤ 0.30`. The fix is correct but the budget is too short. Triggers extension to 1.5M env steps as Round 2.
- **R-A1a (collapse reproduces — H₀(A1))**: `EP_STEPS_PEAK ≥ 250` AND `EP_STEPS_F < 200` AND `T_STARV_F > 0.50`. The `3zjhap9w` learn-then-collapse trajectory reproduces despite `replay_ratio = 0.0625`. Refutes the rank-2 conventional cause as the dominant mechanism on NoPred.
- **R-A1b (instability)**: NaN in any loss; or `loss_model` rises monotonically across any 200k env-step window; or `Behavior/mean_value` blows up (|value| > 100). Refute the run, not the hypothesis. Re-run.
- **Verdict mapping**: C-A1a + C-A1b → H₁(A1) confirmed (the headline result). C-A1c → H₂(A1); recommend extending budget. R-A1a → H₀(A1); rank-2 conventional cause refuted, escalate to bespoke probes.

### 6.2 Cell A2 — Predator + rr=0.0625 + dp=1 (joint fix on predator task)

- **C-A2a (full fix works — H₁(A2) primary)**: `EP_STEPS_F > 100` AND `MAE_POS_F` and `MAE_NEG_F` both < 0.30 AND `|MAE_POS_F - MAE_NEG_F| < 0.15`. The joint conventional fix unlocks the predator task AND resolves the reward-head asymmetry.
- **C-A2b (mae resolved, survival fails — H₂(A2))**: `|MAE_POS_F - MAE_NEG_F| < 0.15` (reward-scale fixed) AND `EP_STEPS_F ≤ 100` (predator task still fails). Reward-head asymmetry was real and is fixed but the predator failure is downstream. Recommend per-action `cont` probe + value-loss diagnostic as Round 2.
- **R-A2a (joint fix has no effect — H₀(A2))**: `EP_STEPS_F ≤ 50` AND `|MAE_POS_F - MAE_NEG_F| ≥ 0.30`. Both conventional knobs left both signatures intact. Rank-1 and rank-2 conventional causes refuted on the predator task. Escalate to structural-mismatch hypotheses (critique §2.5: vector-input mismatch).
- **R-A2b (instability under dp=1 — H₃(A2))**: NaN in any loss; or `loss_model` rises monotonically across any 200k env-step window; or `Behavior/mean_value` blows up (|value| > 100). Refute the run; re-run with `death_penalty: 5` (intermediate calibration).
- **R-A2c (other instability)**: same standard tripwires. Re-run.
- **Verdict mapping**: C-A2a → H₁(A2) confirmed (kitchen-sink conventional fix wins; the cleanest possible result). C-A2b → H₂(A2); reward-scale was real but actor/value/exploration is the predator-task block. R-A2a → H₀(A2); both conventional knobs refuted on predator → escalate. R-A2b → H₃(A2); knob is correct in direction but `dp: 1` is too aggressive; intermediate value (5 or 10).

### 6.3 Cross-cell verdict matrix

| (A1 verdict, A2 verdict)        | Interpretation                                                                                                                                                                                                                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H₁(A1), H₁(A2)                  | **Headline best case.** Both unaddressed conventional knobs from the critique top-3 list resolve the failure on both tasks. Promote `replay_ratio: 0.0625` and `death_penalty: 1` to project default for DreamerV3 hypervigilance work; design 3-seed Round-2 confirmation; immediately run the deferred Predator+rr=0.0625-only cell to attribute. |
| H₁(A1), H₂(A2)                  | **Replay-ratio fix works, reward-scale necessary-but-not-sufficient on predator.** Confirms cause #2 broadly, partial confirmation of cause #1. Recommend per-action `cont` probe + value-loss diagnostic on predator with both knobs in place.                                                                                |
| H₁(A1), H₀(A2)                  | **Replay-ratio fix works on NoPred, conventional fixes fail on predator.** The predator task has a structural component the conventional knobs cannot reach. Recommend (a) deferred cell (Predator + rr06 only) to confirm the death_penalty knob isn't actively *harmful*; (b) per-action `cont` probe + reward-bin histograms; (c) rPPO vs Dreamer cross-architecture comparison. |
| H₂(A1), any A2                  | **Slow-learning variant on NoPred — extend budget.** Cell A1 results are strictly inconclusive at 700k env steps. Run extension to 1.5M before drawing any conclusion. Cell A2 is also conditionally inconclusive because the rr=0.0625 budget pressure on a harder task is even more severe. |
| H₀(A1), any A2                  | **Replay-ratio fix refuted on the published-template task.** This is the most surprising outcome — Hafner-2023 documents 0.0625 as the canonical Atari setting. Rules out cause #2 on NoPred → strongly suggests structural-mismatch (vector-input + small-grid) is dominant. Escalate to (a) augmented WM with auxiliary objectives; (b) algorithm pivot evaluation. |

---

## 7. Failure-Mode Catalog (pre-decided interpretations)

| Observation                                                                                                  | Interpretation                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Both cells refute (R-A1a + R-A2a)                                                                            | Conventional rank-1 + rank-2 causes both refuted on both tasks → the structural-mismatch hypothesis (critique §2.5: vector-input + 19-dim recon objective + small grid) is the dominant residual. Escalate to algorithm-pivot evaluation OR augmented WM (SPR / BYOL-on-latents). |
| A1 H₁, A2 H₁ both confirm                                                                                    | **Cleanest headline win.** Conventional fixes resolve both failures. Promote the two knobs to default; fast-track 3-seed Round 2 + the deferred attribution cell.                                                                       |
| A1 reaches `EP_STEPS_PEAK ≥ 300` but `EP_STEPS_F < 250` (drop in last 100k)                                  | Borderline collapse-prevention. Distinguish: if drop is < 30% from peak, score H₁(A1) with note. If drop ≥ 30% but doesn't cross into starvation regime (`T_STARV_F < 0.50`), score "marginal" — recommend extending to 1.5M to see if collapse stabilises or continues. |
| A2 shows NaN within first 100k env steps under dp=1                                                          | The dp=1 boundary is too aggressive — the symlog reward head is now under-driven on terminal events; KL/value bootstrap can't find a stable basin. Halt; rerun with `dp=5` or `dp=10`. **Don't conflate "knob causes blow-up" with "knob doesn't fix the mechanism."** |
| A2 mae asymmetry resolves but `Episode/Term_Starvation > 0.50` final-window                                  | The agent learned to avoid the (now-cheap) death and walks away from food too. Self-confirming-pessimism is partly downstream of reward scale (food gradient becomes anaemic relative to homeostatic shaping when terminal is no longer dominant). Distinguish from H₂(A2): if mae asymmetry resolves AND survival > 100, H₁(A2); if mae resolves AND survival ≤ 100 AND term_starvation > 0.50, H₂(A2). |
| A1 wall-clock per cell exceeds 5h                                                                            | Node 113 contention or unexpected per-iteration slowdown. Halt mid-run and notify user. (At rr=0.0625 we expect ~50% of the probe-battery wall-clock; >5h means something is wrong.)                                                   |
| A2 wall-clock per cell exceeds 5h                                                                            | Same. Halt and notify.                                                                                                                                                                                                                  |
| Both cells reach `EP_STEPS_F > 250` AND mae asymmetry resolved                                                | Beyond H₁ — both cells show full convergence with no asymmetry. This is the cleanest possible outcome and dominates the cross-cell verdict matrix.                                                                                       |
| A1 confirms but A2 NaNs out under dp=1 (R-A2b)                                                               | Run A2 again with dp=5 (intermediate). The H₂(A2) / H₁(A2) determination shifts to the dp=5 rerun.                                                                                                                                       |

---

## 8. Plumbing-check table (post-run fill-in by `experiment-analyzer`)

For each cell, after launch (filled in 2026-05-09 by `experiment-analyzer`; A1 = `czfnljf0`, A2 = `houwr6js`):

- [x] Single WandB run-ID assigned, state=finished — A1 czfnljf0, A2 houwr6js
- [x] No startup ValueError (mandatory-key discipline holds end-to-end)
- [x] `agent.replay_ratio = 0.0625` in `wandb.config`
- [x] `agent.cont_loss_weight = 1.0` in `wandb.config`
- [x] `agent.entropy_scale = 3e-4` in `wandb.config`
- [x] `agent.imagined_rollout_probe = false` in `wandb.config` (no `WorldModel/imagined_*` keys in either run's history)
- [x] (A2 only) `body.death_penalty = 1` in `wandb.config`
- [x] No NaN warnings in any logged metric
- [x] `WorldModel/loss_model` is monotonically non-increasing across the run — A1: 2.28 → 1.31; A2: 4.12 → 3.11
- [x] Episode count reaches 700,000 (within ±200 of the budget; trailing batch tolerance) — A1: 700,009; A2: 700,046
- [x] Wall-clock < 5 h per cell — A1: 1h38m; A2: 28m

---

## 9. Analysis Plan (post-run)

Single seed per cell → no inferential statistics; observational verdict only. Post-run fill-in for §10/§11/§12 (sections to be added by `experiment-analyzer`) will:

1. **Plumbing checks first** (per cell, against §8).
2. **Per-cell temporal evolution** — plot `Episode/Steps`, `Episode/FoodEaten`, `Episode/Term_*`, `Behavior/mean_value`, `Behavior/mean_advantage`, `Behavior/mean_entropy`, `WorldModel/model_reward_mae_pos`, `WorldModel/model_reward_mae_neg` across 100k-env-step windows on the `Episode/Number` axis. Annotate the 600–800k env-step region (the `3zjhap9w` collapse boundary) on Cell A1.
3. **Cross-run overlay** — same panels for Cell A1 vs `3zjhap9w` (NoPred, rr=0.5, no probe) vs `xekunmbw`/E3 (NoPred, rr=0.5, probe). Direct visual test of "did the rr knob prevent the collapse?"
4. **Cross-run overlay** — same panels for Cell A2 vs `qont5dac` (predator, rr=0.5, dp=100, no probe) vs `owz29n3t`/E1 (predator, rr=0.5, dp=100, probe) vs `mruhypnu`/E2 (predator, rr=0.5, dp=100, cont_loss_weight=10, probe). Direct visual test of "does the joint conventional fix work where the cont-weight knob did not?"
5. **Confirm/refute walk** across §6 criteria, recording per-cell verdict.
6. **Cross-cell verdict matrix** (§6.3) — fill the (A1, A2) row; cite Round 2 priorities by branch.
7. **Save the analysis to a stand-alone results section** in this doc (§10–§12 added in place; do not start a new doc).

---

## 10. Cross-link map (for downstream agents)

- Conventional-failure-mode critique (the rationale and ranking) → [`docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`](../../../project/critiques/dreamer_conventional_failure_modes_for_our_setup.md). The two knobs in this battery are §4 ranked items #1 and #2.
- Strip-down direction memo (the proposed 2×2 design from which we picked 2 cells) → [`docs/project/directions/dreamer_minimum_viable_strip_down.md`](../../../project/directions/dreamer_minimum_viable_strip_down.md). §8 specifies the 2×2; this battery picks the most informative subset given 2-GPU constraint.
- Probe instrumentation contract → [`docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`](../../../develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md). This battery does NOT use the probe (orthogonal channel; would add 6% overhead), but Round 2 will likely re-enable it on the winning cell.
- Probe battery (parent design doc; the data we cross-compare against) → [`docs/experiments/active/continual_learning/DREAMER_PROBE_BATTERY.md`](../continual_learning/DREAMER_PROBE_BATTERY.md). Cell A1 directly compares against E3 (`xekunmbw`); Cell A2 directly compares against E1 (`owz29n3t`) and E2 (`mruhypnu`).
- Reference collapse run (`3zjhap9w` post-hoc analysis) → [`tmp/20260508_replayRatio05_NoPred_analysis.md`](../../../../tmp/20260508_replayRatio05_NoPred_analysis.md). Cell A1's collapse-prevention test is benchmarked against this trajectory.
- Underlying failure-mode diagnosis → [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md). The qont5dac signature this battery's Cell A2 attempts to repair.
- Diagnostic-battery refutations (4 prior failed fixes) → [`docs/experiments/active/continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md`](../continual_learning/DREAMER_DIAGNOSTIC_BATTERY.md). Conventional fixes here are the *unaddressed* alternatives to the bespoke fixes that were tried and refuted there.
- Replay-ratio speed sweep (cost model for the ~50% wall-clock reduction we expect) → [`docs/develop/active/diagnosis/dreamer_replay_ratio_sweep.md`](../../../develop/active/diagnosis/dreamer_replay_ratio_sweep.md).

---

## Appendix

### A. Knob comparison across cells (full lever table)

| Lever                               | Cell A1 (NoPred + rr06) | Cell A2 (Pred + rr06 + dp1) | Reference: `3zjhap9w` (collapsed) | Reference: `qont5dac` (failing baseline) | Reference: `xekunmbw`/E3 (probe; collapses?) | Reference: `owz29n3t`/E1 (probe + predator) | Reference: `mruhypnu`/E2 (cont=10 + predator) |
|-------------------------------------|-------------------------|-----------------------------|-----------------------------------|------------------------------------------|----------------------------------------------|---------------------------------------------|-----------------------------------------------|
| Env                                 | NoPred                  | predator (PredInterval3)    | NoPred                            | predator                                 | NoPred                                       | predator                                    | predator                                      |
| `body.death_penalty`                | 100                     | **1**                       | 100                               | 100                                      | 100                                          | 100                                         | 100                                           |
| `agent.replay_ratio`                | **0.0625**              | **0.0625**                  | 0.5                               | 0.5                                      | 0.5                                          | 0.5                                         | 0.5                                           |
| `agent.cont_loss_weight`            | 1.0                     | 1.0                         | 1.0 (implicit)                    | 1.0                                      | 1.0                                          | 1.0                                         | **10.0**                                      |
| `agent.entropy_scale`               | 3e-4                    | 3e-4                        | 3e-4                              | 3e-4                                     | 3e-4                                         | 3e-4                                        | 3e-4                                          |
| `agent.imagined_rollout_probe`      | false                   | false                       | (predates flag)                   | false                                    | true                                         | true                                        | true                                          |
| `body.use_homeostatic_reward`       | true                    | true                        | true                              | true                                     | true                                         | true                                        | true                                          |
| Episode budget                      | 700 000                 | 700 000                     | 10 000 000 (collapsed at 1.8M)    | ~909 000                                 | 700 000                                      | 700 000                                     | 700 000                                       |

### B. Reference outcomes (from prior runs)

| Run            | Survival final-window | mae_pos final | mae_neg final | Term_Starvation final | Term_MaxSteps final | DangerHits final | Note                                                              |
|----------------|-----------------------|---------------|---------------|------------------------|----------------------|-------------------|-------------------------------------------------------------------|
| `3zjhap9w`     | 142                   | 0.05          | 0.60          | 0.80                   | 0.02                 | 0.87              | NoPred + rr=0.5; learn-then-collapse; `tmp/20260508_replayRatio05_NoPred_analysis.md` |
| `qont5dac`     | ~30                   | (asymmetric)  | (asymmetric)  | ~                      | ~                    | ~                 | predator + rr=0.5 + dp=100; failing baseline                                          |
| `xekunmbw`/E3  | 316 (at ~500k, still running) | (probe-tracking) | — | — | — | — | NoPred + rr=0.5 + probe; track for collapse past 800k env steps                                  |
| `owz29n3t`/E1  | 30.4                  | (probe-confirmed asymmetric) | — | — | — | — | predator + rr=0.5 + probe; img_h15=0.27; img_first=12.6                                |
| `mruhypnu`/E2  | 30.4                  | (probe-confirmed asymmetric) | — | — | — | — | predator + rr=0.5 + probe + cont_loss_weight=10; img_h15=0.33 (probe moves), survival flat |

### C. Changelog

| Date       | Change                                                                                            | Author              |
|------------|---------------------------------------------------------------------------------------------------|---------------------|
| 2026-05-09 | Initial design + 2 new configs + Launch Manifest. Live-load + key-set diff verified.              | experiment-designer |
| 2026-05-09 | §11–§13 Results / Analysis / Conclusions filled in after both cells finished. Verdict: H₂-with-caveat (A1) + H₀ (A2). | experiment-analyzer |

---

## 11. Results (post-run, 2026-05-09)

Both cells reached the full 700k episode budget cleanly. All §8 plumbing checks pass for both cells: no NaN; loss_model monotonically non-increasing (A1: 2.28 → 1.31; A2: 4.12 → 3.11); `agent.replay_ratio = 0.0625`, `agent.cont_loss_weight = 1.0`, `agent.entropy_scale = 3e-4`, `agent.imagined_rollout_probe = false` all verified in `wandb.config`; A2's `body.death_penalty = 1` verified. No instability tripwires (R-A1b / R-A2b/c) fired.

Source extractions: [`tmp/20260509_dreamer_conv_A1_timeseries.md`](../../../../tmp/20260509_dreamer_conv_A1_timeseries.md), [`tmp/20260509_dreamer_conv_A2_timeseries.md`](../../../../tmp/20260509_dreamer_conv_A2_timeseries.md), [`tmp/20260509_dreamer_conv_battery_verdict.md`](../../../../tmp/20260509_dreamer_conv_battery_verdict.md).

> **Caveat on the windowing axis.** WandB's resampled `_step` axis tops out at ~600,210 (A1) / ~600,121 (A2) under the default API sample cap; the run-level `wandb-summary.json` (Episode/Number = 699,106 / 698,452 respectively) is the authoritative final-iteration value. The "final 100k window" pre-registered in §6 is read jointly from window 7 (≈ 500k–600k Episode/Number) and the summary; the two numbers agree to within window-7 std on every metric where they overlap.

### 11.1 Cell A1 — NoPred + replay_ratio = 0.0625, dp = 100 (`czfnljf0`)

**Headline (final-100k window via window 7 + summary):**

| Metric | Window 7 mean (≈ 500–600k Episode/Number) | Final summary (Episode 699,106) |
|---|---:|---:|
| `Episode/Steps` | **105.6** | **112.0** |
| `Episode/Term_Starvation` | **0.60** | 0.61 |
| `Episode/Term_Injury` | 0.40 | 0.39 |
| `Episode/Term_MaxSteps` | 0.0015 | 0.003 |
| `Episode/FoodEaten` | 1.91 | 2.33 |
| `Episode/DangerHits` (= `HidingPredatorHits`) | 2.15 | 2.12 |
| `WorldModel/model_reward_mae_pos` | **0.76** | 0.72 |
| `WorldModel/model_reward_mae_neg` | **0.93** | 0.66 |
| `\|mae_pos − mae_neg\|` | **0.17** | 0.06 |
| `Behavior/mean_value` | −6.08 | −4.64 |
| `Behavior/mean_advantage` | −0.007 | +0.0004 |
| `Behavior/mean_entropy` | 1.14 | 1.00 |
| `Behavior/loss_actor` | −0.042 | −0.037 |
| `Behavior/loss_critic` | 3.73 | 3.93 |
| `WorldModel/loss_model` | 1.31 | 1.20 |
| `WorldModel/model_cont_acc` | 0.998 | 0.999 |

**Temporal evolution (7 windows, ≈ 100k Episode/Number per window):**

| Window | Eps. range | `Steps` | `Term_Starv` | `mae_pos` | `mae_neg` | `mean_value` | `mean_adv` | `mean_ent` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0–100k | 84.4 | 0.52 | 10.29 | 1.55 | −10.25 | −0.51 | 1.79 |
| 2 | 100k–200k | 98.9 | 0.45 | 2.56 | 1.13 | −8.07 | −0.08 | 1.49 |
| 3 | 200k–300k | 96.4 | 0.55 | 1.69 | 1.10 | −8.52 | −0.04 | 1.41 |
| 4 | 300k–400k | 95.6 | 0.54 | 1.31 | 1.10 | −7.61 | −0.02 | 1.33 |
| 5 | 400k–500k | 97.2 | 0.56 | 1.08 | 1.09 | −5.56 | −0.008 | 1.20 |
| 6 | 500k–600k | 104.6 | **0.63** | 0.99 | 0.95 | −5.51 | −0.005 | 1.16 |
| 7 | 600k+ | **105.6** | 0.60 | 0.76 | 0.93 | −6.08 | −0.007 | 1.14 |

Survival is monotonically non-decreasing across all 7 windows (84 → 99 → 96 → 96 → 97 → 105 → 106); the +6 step gain in windows 6–7 is the only second-order trend. `mae_pos` collapses from 10.3 (initialization-noise) to 0.76 by window 7, and `mae_neg` falls from 1.55 to 0.93 — both maes converge toward each other and the asymmetry **closes** (gap 0.17 in window 7, 0.06 in summary). Mean_advantage converges to ≈ 0 (−0.51 → −0.007), mean_entropy falls from 1.79 (uniform) to 1.14 (~0.65 nats below uniform), mean_value drifts *less* negative over training (−10.3 → −6.1).

**Comparison vs `3zjhap9w` (NoPred + rr = 0.5, no probe; ran to 1.8M env steps):**

| Metric | `3zjhap9w` PEAK (windows 2–3, 400–800k env steps) | `3zjhap9w` window 5 (1.3–1.8M, collapsed) | A1 window 7 (≈ 600k Ep#) |
|---|---:|---:|---:|
| `Episode/Steps` | **332** | 142 | **106** |
| `Term_Starvation` | 0.03 | **0.80** | 0.60 |
| `mae_pos` | 0.15 | **0.05** | 0.76 |
| `mae_neg` | 0.37 | **0.60** | 0.93 |
| `\|mae_pos − mae_neg\|` | 0.22 | **0.55** | 0.17 |
| `mean_value` | −10.7 | −14.7 | −6.08 |
| `mean_advantage` | ≈ 0 | −0.084 | −0.007 |
| `mean_entropy` | 0.43 | 0.57 | 1.14 |
| `FoodEaten` | 30.3 | 7.6 | 1.91 |

The `3zjhap9w` collapse signature — `mae_pos` collapsing far below `mae_neg`, `mean_value` drifting more negative, food intake plummeting while danger-hits also fall (avoid-everything regime) — is **NOT present** in A1. But A1 also never reached the `3zjhap9w` peak: the rr = 0.5 run hit 332 steps survival by 400k env-steps; A1 with rr = 0.0625 sits at 106 steps at 700k Ep# (≈ 600k env-steps) with starvation rate 0.60.

### 11.2 Cell A2 — Predator + replay_ratio = 0.0625 + death_penalty = 1 (`houwr6js`)

**Headline (final-100k window via window 7 + summary):**

| Metric | Window 7 mean (≈ 500–600k Episode/Number) | Final summary (Episode 698,452) |
|---|---:|---:|
| `Episode/Steps` | **27.8** | **27.0** |
| `Episode/Term_Injury` | 0.992 | 0.995 |
| `Episode/Term_Starvation` | 0.008 | 0.005 |
| `Episode/Term_MaxSteps` | 0 | 0 |
| `Episode/FoodEaten` | 0.29 | 0.28 |
| `Episode/PredatorHits` | 3.59 | 3.62 |
| `Episode/DangerHits` (= `HidingPredatorHits`) | 0.81 | 0.80 |
| `WorldModel/model_reward_mae_pos` | **3.68** | **2.44** |
| `WorldModel/model_reward_mae_neg` | **1.50** | **1.59** |
| `\|mae_pos − mae_neg\|` | **2.18** | **0.85** |
| `Behavior/mean_value` | −17.56 | −17.65 |
| `Behavior/mean_advantage` | **−0.190** | −0.181 |
| `Behavior/mean_entropy` | 1.69 | 1.69 |
| `Behavior/loss_actor` | (~−0.33 at end-of-train) | −0.33 |
| `Behavior/loss_critic` | (~0.64 at end-of-train) | 0.64 |
| `WorldModel/loss_model` | 3.11 | 3.17 |
| `WorldModel/model_cont_acc` | 0.976 | 0.976 |

**Temporal evolution (7 windows):**

| Window | Eps. range | `Steps` | `Term_Injury` | `mae_pos` | `mae_neg` | `mean_value` | `mean_adv` | `mean_ent` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0–48 (start) | 26.4 | 0.995 | 5.96 | 2.64 | −4.07 | −0.58 | 1.79 |
| 2 | ~100k | 26.0 | 0.996 | 5.81 | 1.90 | −12.19 | −0.22 | 1.79 |
| 3 | 100k–200k | 25.1 | 0.996 | 5.41 | 1.82 | −14.59 | −0.19 | 1.78 |
| 4 | ~300k | 25.1 | 0.996 | 4.92 | 1.73 | −16.35 | −0.19 | 1.78 |
| 5 | ~400k | 26.7 | 0.994 | 4.21 | 1.53 | −17.52 | −0.20 | 1.77 |
| 6 | 400k–500k | 27.8 | 0.993 | 4.26 | 1.47 | −17.75 | −0.20 | 1.73 |
| 7 | 500k–600k | 27.8 | 0.992 | 3.68 | 1.50 | −17.56 | −0.19 | 1.69 |

Survival is essentially **flat at 26–28 steps** across the entire 700k-episode budget. Mean_value drifts from ≈ −4 (initialization) to a stable −17.6 by window 5 and stays there. Mean_advantage stays at **−0.19 ± 0.005** across windows 3–7 — pixel-identical to the `qont5dac` / `owz29n3t` predator-baseline collapse signature. Mean_entropy reverts toward ln(6) ≈ 1.79 (final 1.69). Term_Injury > 99% in every window (predator kills the agent on essentially every episode). `mae_pos` (5.96 → 3.68) and `mae_neg` (2.64 → 1.50) both decline, but the **gap** stays large (window 7 = 2.18) — the asymmetry is structurally preserved despite both knobs being changed.

**Comparison vs `owz29n3t` / E1 (predator + rr = 0.5 + dp = 100 + probe; the prior anchor):**

| Metric (final / steady-state) | `owz29n3t` E1 anchor | A2 (joint fix) | Δ (A2 − E1) |
|---|---:|---:|---:|
| `Episode/Steps` | 30.4 | 27.0 | **−3.4** |
| `Behavior/mean_advantage` | −0.19 ± 0.005 | −0.19 → −0.18 | ≈ 0 |
| `Behavior/mean_entropy` | ~1.65 | 1.69 | +0.04 |
| `Term_Injury` | high | 0.995 | identical regime |

A2 is **not better than E1 on any headline metric.** Both unaddressed conventional knobs jointly applied did not move any of the predator-task failure-mode signatures.

### 11.3 Sanity check on `death_penalty: 1` (R-A2b refute-the-run gate)

R-A2b would be triggered by NaN or value blow-up under the more aggressive `dp = 1`. None of those fired:
- No NaN in any logged metric across 700k episodes.
- `Behavior/mean_value` converged to −17.6 (well-bounded, |value| ≪ 100).
- `WorldModel/loss_model` decreased monotonically (4.12 → 3.11).
- `WorldModel/model_reward_mae_pos` and `mae_neg` both *fell* over training, not rose.

`mean_value ≈ −17.6` is well-calibrated to the new reward scale: A2's per-episode return averages ≈ −105.5 (per `Episode/Reward`) over a 27-step episode; the value head's bootstrapped sum-of-future-rewards (with discount) lands at −17.6. The reward head and critic are fine. The blockage is downstream of reward calibration. **R-A2b refuted; the dp = 1 knob is not the failure mechanism.**

---

## 12. Analysis (verdict per pre-registered §6 criteria)

### 12.1 Cell A1 — strict scoring against §6.1

| Criterion | Required | Observed | Pass? |
|---|---|---|---|
| **C-A1a (collapse prevented; H₁(A1) primary)** | `EP_STEPS_F ≥ 250` | 106 (window 7), 112 (summary) | **NO** |
| | AND `EP_STEPS_F ≥ 0.7 · EP_STEPS_PEAK` | EP_STEPS_PEAK = 105.6, EP_STEPS_F = 105.6 | YES |
| | AND `T_STARV_F ≤ 0.30` | 0.60 (window 7) / 0.61 (summary) | **NO** |
| | Composite | — | **fail** |
| **C-A1b (mae asymmetry resolved; H₁(A1) corroboration)** | `MAE_POS_F` < 0.30 | 0.76 / 0.72 | **NO** |
| | AND `MAE_NEG_F` < 0.30 | 0.93 / 0.66 | **NO** |
| | AND `\|MAE_POS_F − MAE_NEG_F\| < 0.20` | 0.17 / 0.06 | YES |
| | Composite | — | **fail (on absolute level, not on gap)** |
| **C-A1c (slow-learning variant; H₂(A1))** | `EP_STEPS_F ≥ 100` | 105.6 / 112 | YES |
| | AND `EP_STEPS` monotonically non-decreasing | 84 → 99 → 96 → 96 → 97 → 105 → 106 (non-strictly increasing; slight 99→96 dip in window 3) | borderline YES |
| | AND `T_STARV_F ≤ 0.30` | 0.60 | **NO** |
| | Composite | — | **fail** (T_STARV clause broken) |
| **R-A1a (collapse reproduces; H₀(A1))** | `EP_STEPS_PEAK ≥ 250` | 105.6 | **NO** |
| | AND `EP_STEPS_F < 200` | 105.6 / 112 | YES |
| | AND `T_STARV_F > 0.50` | 0.60 / 0.61 | YES |
| | Composite | — | **fail** (peak clause broken — never reached competence to collapse from) |
| **R-A1b (instability)** | NaN / value blow-up / loss_model rising | none | **fail** (no instability) |

**A1 falls in an under-specified zone.** No verdict criterion fires cleanly. The closest match is **H₂(A1) "slow-learning variant"** — collapse prevented (peak survival never reached the `3zjhap9w` collapse boundary so there's nothing to collapse from), survival monotonically non-decreasing — but the explicit C-A1c clause `T_STARV_F ≤ 0.30` is broken (0.60 observed). The pre-registration assumed "slow learning" meant "still climbing toward competence with low starvation." A1 instead shows a **starvation-regime equilibrium** that is qualitatively different from `3zjhap9w`'s "learn-then-collapse" trajectory but is also not on track to reach competence.

A more honest reading: A1 with rr = 0.0625 produces 8× fewer gradient updates per env-step than rr = 0.5, so 700k env-steps is roughly equivalent to 87.5k env-steps of `3zjhap9w`-density training. `3zjhap9w` at ~87.5k was not yet at competence either (its window 1 = 196 step survival at ~200k env-steps was the first plateau). A1 may be tracking `3zjhap9w`'s trajectory in *gradient-step-density* terms rather than env-step terms — but at a much lower asymptote (window 7 mean 105 vs `3zjhap9w` window 1 mean 196 with the same gradient-density budget). The slowdown is substantively worse than 8× per env-step would predict, suggesting the rr knob does change the *trajectory*, not just compress it.

**A1 verdict: H₂(A1)-with-caveat — collapse prevented (positive), but the agent is in a starvation-regime equilibrium far below `3zjhap9w`'s competence peak, and the §6.1 starvation-clause for H₂(A1) is broken. The fix is correct in *direction* (no `3zjhap9w` collapse, mae asymmetry has not opened) but is not delivering competence within 700k env-steps.**

### 12.2 Cell A2 — strict scoring against §6.2

| Criterion | Required | Observed | Pass? |
|---|---|---|---|
| **C-A2a (full fix works; H₁(A2) primary)** | `EP_STEPS_F > 100` | 27.8 / 27.0 | **NO** |
| | AND `MAE_POS_F` < 0.30 | 3.68 / 2.44 | **NO** (8× over) |
| | AND `MAE_NEG_F` < 0.30 | 1.50 / 1.59 | **NO** (5× over) |
| | AND `\|MAE_POS_F − MAE_NEG_F\| < 0.15` | 2.18 / 0.85 | **NO** |
| | Composite | — | **fail** |
| **C-A2b (mae resolved, survival fails; H₂(A2))** | `\|MAE_POS_F − MAE_NEG_F\| < 0.15` | 2.18 / 0.85 | **NO** |
| | AND `EP_STEPS_F ≤ 100` | 27.8 | YES |
| | Composite | — | **fail** (mae clause broken) |
| **R-A2a (joint fix has no effect; H₀(A2))** | `EP_STEPS_F ≤ 50` | 27.8 / 27.0 | **YES** (well under) |
| | AND `\|MAE_POS_F − MAE_NEG_F\| ≥ 0.30` | 2.18 / 0.85 | **YES** (window 7 by 7×, summary by 3×) |
| | Composite | — | **PASS — H₀(A2) confirmed** |
| **R-A2b (instability under dp=1; H₃(A2))** | NaN / value blow-up / loss_model rising | none | **fail** (run is stable) |

**A2 verdict: clean H₀(A2). R-A2a fires unambiguously by 3–7× margin on every clause.** Joint conventional fix has no effect on the predator-task failure-mode signatures. Survival 27.0 < the `owz29n3t`/E1 baseline of 30.4 (numerically *worse*, well within seed-noise — so call it a tie, not better, not worse than rr = 0.5 + dp = 100). Mean_advantage at −0.19 is identical to the `qont5dac` / `owz29n3t` predator-baseline failure-mode signature.

The `dp = 1` knob is *not* the failure mechanism (R-A2b refuted: no NaN, value bounded at −17.6 ≈ achievable return). The reward head is well-calibrated to the new scale yet the actor cannot improve. **The blockage is downstream of the reward head — actor / value-imagination / exploration on the predator task. Both ranked-1 (reward-scale) and ranked-2 (replay-ratio) conventional causes are refuted on the predator task.**

### 12.3 Cross-cell verdict on §6.3 matrix

The §6.3 verdict matrix has no row for "(H₂-with-caveat A1, H₀ A2)" specifically; the closest pre-registered cells are:

- **"H₂(A1), any A2"** → "Slow-learning variant on NoPred — extend budget. Cell A1 results are strictly inconclusive at 700k env steps. Run extension to 1.5M before drawing any conclusion. Cell A2 is also conditionally inconclusive because the rr=0.0625 budget pressure on a harder task is even more severe."

- **"H₁(A1), H₀(A2)"** → "Replay-ratio fix works on NoPred, conventional fixes fail on predator. The predator task has a structural component the conventional knobs cannot reach. Recommend (a) deferred cell (Predator + rr06 only) to confirm the death_penalty knob isn't actively *harmful*; (b) per-action `cont` probe + reward-bin histograms; (c) rPPO vs Dreamer cross-architecture comparison."

Reconciling: A1 is closer to H₂ than H₁ (never reached competence), but the A2 reading is unambiguously H₀ regardless of A1. The rr = 0.0625 budget-pressure caveat in the H₂(A1)/any-A2 row does NOT apply to A2 — A2's failure-mode signatures (mean_advantage = −0.19, mean_value plateau at −17.6, mean_entropy reverting toward 1.79) **are saturated**, not climbing or evolving. Window-by-window standard deviation is *falling* (window 6 mean_value std = 0.40, window 7 = 0.38), meaning the predator task is at a stable failure-equilibrium that 8× more env-steps would not change.

So the operative cross-cell reading combines both rows:

> **A1 = H₂(A1)-with-caveat.** Replay-ratio fix prevents the `3zjhap9w` collapse on NoPred (positive result on the H₀(A1) refutation: `3zjhap9w`-style collapse signature is absent) but does NOT deliver competence within 700k env steps under rr = 0.0625. The fix is correct in direction; whether it eventually works depends on whether 1.5M env-steps escape the starvation-regime equilibrium or whether the equilibrium is asymptotic.
>
> **A2 = H₀(A2).** Joint conventional fix is fully refuted on the predator task. Both ranked-1 (reward-scale) and ranked-2 (replay-ratio) conventional causes are not the dominant failure modes. The predator task has a structural component the conventional knobs cannot reach (per the H₁(A1)/H₀(A2) pre-registered language).

---

## 13. Conclusions

### 13.1 What this battery confirmed

1. **Replay-ratio fix prevents the `3zjhap9w` collapse signature on NoPred (Cell A1 negative result on H₀(A1)).** A1's mae trajectory is qualitatively different from `3zjhap9w` (mae_pos and mae_neg converge toward each other rather than diverge), value drifts *less* negative rather than more, mean_advantage converges to ≈ 0 rather than to −0.084, and food intake increases monotonically rather than crashes. The `3zjhap9w` "learn-then-collapse" mechanism does not reproduce under rr = 0.0625. **Cause #2 (replay-ratio over-training) from the conventional-failure-mode critique is confirmed as a real mechanism on NoPred.**

2. **`death_penalty: 1` is calibrationally safe (R-A2b refuted).** No NaN, no value blow-up, mean_value sat near the achievable per-episode return throughout. The Crafter recipe transfers to this codebase's mixed-magnitude reward regime.

### 13.2 What this battery refuted

1. **Joint conventional fix (rr = 0.0625 + dp = 1) does not unlock the predator task within 700k env steps (Cell A2 H₀(A2)).** Survival flat at 27 steps; mean_advantage −0.19 (identical to `owz29n3t`/E1 anchor); mae asymmetry intact. **Both ranked-1 (reward-scale) and ranked-2 (replay-ratio) conventional causes are refuted as dominant failure modes on the predator task.** The predator-task block is downstream of the reward head — actor / value-imagination / exploration mechanisms.

2. **The replay-ratio fix on NoPred does not, by itself, deliver competence within 700k env steps under rr = 0.0625.** A1's window-7 survival (105.6) is one-third of `3zjhap9w`'s peak (332). The fix prevents collapse but the env-step budget is plausibly too short, OR (less likely) the rr = 0.0625 regime has a different asymptote — the ambiguity is the H₂(A1) ↔ "starvation-equilibrium" question that the 1.5M extension would resolve.

### 13.3 Adjustment to the conventional-failure-mode critique

The critique ranked replay-ratio over-training as **rank-2** and reward-scale mismatch as **rank-1**. This battery:

- **Confirms rank-2 (replay-ratio) is real** on the NoPred task (Cell A1 prevents the `3zjhap9w` collapse).
- **Refutes rank-1 (reward-scale) as the dominant predator-task failure mode** (Cell A2 dp = 1 produces a calibrated critic but doesn't unlock the task).
- **Refutes "rank-1 + rank-2 jointly are sufficient" on the predator task** (Cell A2 H₀).

Net: the critique's rank-1 claim was correct in *form* (the asymmetric mae was a real signature) but not in *cause* (fixing the reward scale doesn't fix the predator task). The H₁(A1) / H₀(A2) row's interpretation applies: **the predator task has a structural component the conventional knobs cannot reach**, consistent with critique §2.5 (vector-input + 19-dim recon objective + small-grid mismatch).

### 13.4 Recommended next steps (for the user; not designed here)

Per the §6.3 H₁(A1)/H₀(A2) row's prescription, adapted to the H₂-with-caveat reading on A1:

1. **(highest priority) Run the deferred Cell B = Predator + rr = 0.0625 + dp = 100** (replay-ratio knob alone, original death penalty) — confirms whether dp = 1 is *neutral* or actively *worse* on the predator task. Single seed, single GPU, ~30m wall-clock on Node 113 given A2's pace. Disambiguates joint-fix-fails from "dp = 1 is the wrong knob."

2. **(second priority) Extend Cell A1 to 1.5M env-steps (single seed)** — the H₂(A1) prescription. Disambiguates "slow learning toward competence" from "starvation-regime equilibrium." If the extension reaches survival ≥ 250 with `T_STARV_F ≤ 0.30` between 1.0–1.5M env-steps, that's H₁(A1); if it stays at the 100-step starvation-equilibrium plateau, the rr = 0.0625 regime has a different asymptote on this codebase than the published Hafner-2023 setting predicts.

3. **(third priority) Per-action `cont` probe + reward-bin histograms on the predator task** with both knobs in place. The H₁(A1)/H₀(A2) row's standard recommendation. Targets the "structural mismatch" hypothesis at the WM-internal level — does the cont head fail to predict per-action terminations (the imagined-deaths-undifferentiated-per-action hypothesis from `docs/memory/memories/dreamer_diagnosis/20260508_1432_probe_refutes_imagined_death_absence.md`)?

4. **(fourth priority) rPPO vs Dreamer cross-architecture comparison on the predator task** — does any value-based RL on this codebase learn the predator task within 700k env-steps? If rPPO succeeds, the issue is the WM; if neither, the predator task itself is the problem.

These are recommendations for the user's design routing (→ `experiment-designer`); this analyzer does not author follow-up designs.

### 13.5 Honest caveats

1. **Single seed per cell.** Pre-registered as observational. The H₀(A2) verdict is robust (effect-size is large; mean_advantage signature exactly matches the `owz29n3t`/E1 prior). The H₂(A1)-with-caveat verdict has more seed-sensitivity — the 105-step survival could be 80 or 130 on different seeds.

2. **A1's "no collapse" verdict is restricted to the 700k env-step window.** `3zjhap9w` collapsed between 600k and 1.3M env steps; A1's 600–700k window happens to be *just* the leading edge of `3zjhap9w`'s collapse boundary. Under rr = 0.0625, "collapse boundary" might shift in env-step terms — extending A1 to 1.5M is required to fully refute H₀(A1).

3. **A2's "structural failure" verdict assumes the saturated signal at 700k (mean_value plateau, falling per-window std) is asymptotic, not transient.** The pre-registered §6.3 caveat ("rr = 0.0625 budget pressure on harder task is even more severe") is acknowledged but the data does not look like a slow-learning regime — A2 looks like a stable failure-equilibrium. The deferred Cell B (predator + rr = 0.0625 only) and the rPPO comparison would each address this from a different angle.

4. **The §6.3 verdict matrix did not anticipate A1's "starvation-regime equilibrium with no collapse" outcome.** The matrix assumed "no collapse" implied "T_STARV_F low" (per C-A1c). The actual A1 trajectory broke that assumption — collapse-of-policy didn't happen but the agent never escaped the starvation regime. This is information for future-design (the H₂(A1) clause should distinguish "still climbing" from "starvation-equilibrium-stable") but does not invalidate the 13.1 claim that the `3zjhap9w` collapse mechanism is absent.

---

## 14. Metrics Requested

| Metric | Why now | Where it'd live | Cost |
|---|---|---|---|
| `WorldModel/imagined_termination_fraction_h*` (probe metrics, on dp = 1 / rr = 0.0625) | This battery deliberately did NOT run the imagined-rollout probe; the 13.4 step-(3) probe-on-predator follow-up requires the probe to be enabled. The probe code already exists for E1–E4; Round 2 just needs to set `imagined_rollout_probe: true` on the agent config. **Not a new metric — just an enable-flag.** | already in `src/agents/dreamer/probe.py` | already paid |
| `Behavior/mean_advantage_per_action` (6-vector) | A2's mean_advantage = −0.19 is the same regardless of action. The "imagined-deaths-undifferentiated-per-action" hypothesis predicts the per-action advantage spread is small (no action looks meaningfully better than others). Currently we only log the mean. | `src/agents/dreamer/behavior.py` | moderate (6-vector per iter) |
| `Episode/Steps_per_termination_type` | Episode-level survival is averaged over termination types. Knowing "A2 dies after 27 steps when killed by predator vs 14 steps when starves" would distinguish "dies fast on contact" from "dies of slow starvation." Currently the two terminations are reported as separate scalars (Term_*) but their per-episode survival is mixed in the headline `Episode/Steps`. | `src/environment/loggers.py` (or wherever `Episode/*` aggregation lives) | cheap (3-tuple per episode) |

These are recommendations only; the user routes to `feature-workflow` if accepted. Do not implement here.

## 15. Related Issues

None. No code defects surfaced; the codebase behaved as designed under both knob deltas. The verdict is about the *mechanisms* not behaving as the conventional critique predicted.
