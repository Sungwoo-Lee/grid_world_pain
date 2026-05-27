---
title: "JAX DreamerV3 matched-config SPS measurement vs sheeprl XS"
topic: sheeprl_bridge
status: active
created: 2026-05-13
last_updated: 2026-05-13
phase: pre-launch-design
wandb_tag: jax_sheeprl_matched
cross_links:
  - docs/experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md
  - docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md
  - docs/develop/active/sheeprl_bridge/JAX_VECTOR_ENV_V2_GPU_SPIKE.md
  - docs/develop/active/sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md
---

# JAX DreamerV3 matched-config SPS measurement vs sheeprl XS

## 0. Change log

- **2026-05-13** — §4.2 and §4.3 launch commands amended to add `--episodes 0`. The env YAML `configs/experiment/dreamer_curriculum/01_food_only.yaml` pins `episodes: 100`, and `train.py:1165` makes the episode-based loop the controlling stop condition whenever `episodes > 0`, silently ignoring `--total-timesteps`. Without the CLI override, Run A stopped after 100 episodes / ~110 s on WandB `pz66mhu0`, far below the §6.1 acceptance floor (`_runtime ≥ 600 s`). `--episodes 0` flips the loop to the `else` branch at line 1165 so `--total-timesteps` controls. Re-launch Run A and launch Run B with the amended commands.
- **2026-05-13** — §2 `replay_ratio` row re-pointed. The `0.0625` baseline lives in `configs/models/dreamer_v3/dreamer_v3_rr06.yaml` (the Hafner-2023 Atari/DMC default the cascade-side runs use), not in `configs/models/dreamer_v3/dreamer_v3.yaml` (which sits at `replay_ratio: 0.5`). The numerical 16× factor — 0.0625 → 1.0 — is the cascade-relevant one and stands. The structural-delta rows (rssm dim, MLP layers/widths, sequence_length, encoding_mode) are correctly baselined against `dreamer_v3.yaml` (both files share that arch).

## 1. Context (plain English)

A previous analysis showed our in-house JAX DreamerV3 trains around 2,600 times more environment steps per second than the sheeprl PyTorch DreamerV3 we adopted as the primary backend. After accounting for the fact that JAX was doing 16 times fewer gradient updates per environment step (and processing 2x larger gradient batches), the gradient-compute throughput advantage shrank to about 325-650x, depending on how you normalize. That residual gap was still large enough that the analyst recommended re-opening the question: if we rebuilt JAX with the same hyperparameters sheeprl uses (smaller recurrent state, wider but shallower MLPs, shorter sequences, and one gradient update per environment step), would JAX still be much faster, or would the apparent advantage shrink to a few times?

This design specifies the short measurement run that answers that question. We create one (or two) JAX runs with every hyperparameter pinned to sheeprl's "XS" recipe, on the same food-only no-predator survival task that sheeprl was run on. The run is short on purpose — long enough to amortize JAX-IT compilation and reach steady throughput, not long enough to train a useful policy. The single number that comes out — matched-config environment-steps-per-second on JAX — decides whether a full JAX rebuild is worth funding (greenlight at 5x advantage or more), worth a focused user conversation (1.5-5x), or definitively closed (below 1.5x).

The new model config is at `configs/models/dreamer_v3/dreamer_v3_sheeprl_matched.yaml`. The environment is the existing `configs/experiment/dreamer_curriculum/01_food_only.yaml`. The standard JAX `train.py` entrypoint is used. Nothing in `src/` is touched.

---

## 2. The matched-config recipe

Every knob audited in §2 of the SPS-comparison memo, with the override decision. "Override" means the new YAML differs from `configs/models/dreamer_v3/dreamer_v3.yaml`; "matched" means the JAX value already matches sheeprl. **One exception:** the `replay_ratio` row is baselined against `configs/models/dreamer_v3/dreamer_v3_rr06.yaml` (the Hafner-2023 Atari/DMC default the cascade-side runs use, sitting at `0.0625`), NOT against `dreamer_v3.yaml` itself (which sits at `replay_ratio: 0.5`). The numerical 16× factor — 0.0625 → 1.0 — is the cascade-relevant one and stands.

| Knob | Current JAX (`dreamer_v3.yaml`) | sheeprl XS | New value in `dreamer_v3_sheeprl_matched.yaml` | Action | Rationale |
|---|---|---|---|---|---|
| `rssm_deter_dim` (recurrent state size) | 512 | 256 | 256 | **override** | Mirror sheeprl `recurrent_state_size=256`. Halves the GRU's parameter count and compute. |
| `encoder_dim` | 128 | 256 | 256 | **override** | Mirror sheeprl `dense_units=256`. |
| `encoder_fc_layers` | `[128, 128]` (2 layers, width 128) | 1 layer, width 256 | `[256]` | **override** | Mirror sheeprl `mlp_layers=1, dense_units=256`. |
| `decoder_fc_layers` | `[128, 128]` | `[256]` (1 layer, width 256) | `[256]` | **override** | Same as encoder. |
| `reward_fc_layers` | `[128, 128]` | `[256]` | `[256]` | **override** | Same. |
| `continue_fc_layers` | `[128, 128]` | `[256]` | `[256]` | **override** | Same. |
| `actor_fc_layers` | `[128, 128]` | `[256]` | `[256]` | **override** | Same. |
| `critic_fc_layers` | `[128, 128]` | `[256]` | `[256]` | **override** | Same. |
| `sequence_length` | 128 | 64 | 64 | **override** | Mirror sheeprl `per_rank_sequence_length=64`. Halves the gradient batch (`batch_size × sequence_length`). |
| `replay_ratio` | 0.0625 (1 grad / 16 env steps) — from `dreamer_v3_rr06.yaml`; `dreamer_v3.yaml` itself is `0.5` | 1 (1 grad / 1 env step) | 1.0 | **override** | The single biggest SPS lever. Mirror sheeprl recipe. |
| `batch_size` | 16 | 16 | 16 | matched (no change) | Already aligned. |
| `encoding_mode` | `hierarchical` (per-sensor MLPs + multimodal hub) | flat MLP | `flat` | **override** | sheeprl uses a flat encoder. Hierarchical adds modest JAX-side compute that is not part of sheeprl's baseline; including it would muddy the matched comparison. |
| `rssm_stoch_dim` | 32 | 32 | 32 | matched | Aligned (categorical latent dimension). |
| `rssm_classes` | 32 | 32 | 32 | matched | Aligned (classes per categorical). |
| `model_lr / actor_lr / value_lr` | `1e-4 / 3e-5 / 3e-5` (Hafner defaults) | Same Hafner defaults | unchanged | matched | sheeprl ships the same Hafner default learning rates. |
| `buffer_device` | gpu | (sheeprl: cpu/numpy) | gpu | matched-by-architecture | We do NOT override this. The JIT-compiled-scan + GPU buffer is an *architectural* advantage that any future JAX rebuild would retain; this experiment is meant to quantify the residual JAX advantage *with* its architectural strengths. |
| `collect_interval` | 128 | (not a sheeprl knob; sheeprl does 1 env step per Python loop body) | 128 | matched-by-architecture | Same reason as `buffer_device`. The per-env grad/env ratio is governed by `replay_ratio=1`, which already enforces sheeprl-canonical density. |
| `sampling_mode` | mixture | prioritized | mixture | matched-by-purpose | Both sides ship a non-uniform sampler; we keep ours. Compute cost is comparable. |
| `apply_noise` | false (food-only env) | false (food-only env) | false | matched | Same env YAML on both sides. |
| `num_envs` | 16 (JAX vmap) | 4 (SyncVectorEnv) | see §3 | decision-point | The only knob where the matched-vs-architectural-advantage trade-off is not pre-decided. |

**Cross-reference:** every row of §2 of the SPS-comparison memo has a corresponding row in the table above. The `dreamer_v3_sheeprl_matched.yaml` deltas are exactly the rows marked **override** in the *Action* column.

---

## 3. The `num_envs` decision

The matched-config audit in §2 above pins every knob except `num_envs`. The three options the user surfaced:

| Option | `num_envs` | Pros | Cons |
|---|---|---|---|
| **A — apples-to-apples** | 4 | Pure matched-config number. Every knob mirrors sheeprl. Cleanest answer to "what is the residual JAX advantage at full parity?" | Understates what a real JAX rebuild would ship. JAX's vmap-batched parallel envs are nearly free per extra env (the env is 5x5; one parallel slot vs sixteen is < 5% throughput cost on a 4090). A rebuild would never run at 4 envs. |
| **B — JAX-architectural** | 16 | Represents what a deployed JAX rebuild would actually do. Keeps the one architectural advantage that does not depend on hyperparameter choices. | Couples the measurement to one architectural decision sheeprl doesn't make. The number reported is "JAX with sheeprl recipe + JAX's parallelism", not "matched". |
| **C — bracket** | 4 AND 16 | Gets both numbers, bracketing the answer. Lets the rebuild decision use either as appropriate (the matched number for parity arguments, the parallel number for cost-benefit). | 2x the GPU time. Two separate runs to orchestrate. |

**Recommendation: Option C (run both).**

Reasons:

1. **Cost is small.** Per the SPS-comparison memo §3.1, JAX at `replay_ratio=0.0625, num_envs=16` reached ~11,500 env-SPS. At `replay_ratio=1.0` we expect a 4-8x slowdown (since gradient updates per env-step go up 16x but grad compute is a fraction of total iter time, and the new model is roughly half-size). Expect ~1,500-3,000 env-SPS for the matched-`num_envs=16` run. For a 10-minute run that's ~1-2M env-steps — well above the warmup needed to amortize JIT compile (~30-60 s on the first iteration) and reach steady state. The `num_envs=4` run will be somewhat slower in env-SPS but the wall-clock for 10 minutes is identical. **Total GPU time: ~20-25 minutes for both runs on a single GPU.**
2. **The two numbers answer two different questions.** Option A answers "is JAX's training loop intrinsically faster than sheeprl's at matched recipe?". Option B answers "in practice, what speedup does a JAX rebuild get on this task?". The PASS / CLOSE / FAIL decision matrix in §6 below references *both* numbers — and the rebuild-or-not question depends on Option B, while the academic "is JAX faster" question depends on Option A.
3. **No coordination cost.** Both runs read the same model YAML; only `--num-envs` differs at launch.

The user can downgrade to A or B at launch time if compute is tighter than expected, but Option C is the recommended plan.

---

## 4. Launch protocol

### 4.1 Conda environment and entrypoint

- **Conda env:** `grid_world_pain` (the JAX-side env, not `sheeprl_bridge`).
- **Python interpreter:** `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` (per the project rule — no `conda run`, no `conda activate`).
- **Entrypoint:** `train.py`.
- **Launcher:** `train_command-agent.sh` (the agent-edited launch script; the `training-runner` agent rewrites the body to one of the two commands below and calls `run_command.py <node> "<cmd>"`).

### 4.2 Run A — `num_envs=4` (apples-to-apples)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/dreamer_curriculum/01_food_only.yaml \
  --agent_config configs/models/dreamer_v3/dreamer_v3_sheeprl_matched.yaml \
  --num-envs 4 \
  --episodes 0 \
  --total-timesteps 1500000 \
  --seed 0 \
  --device cuda:<gpu> \
  --log-interval 50 \
  --wandb-group sheeprl_bridge \
  --wandb-job-type sps_measurement \
  --wandb-name jax_sheeprl_matched_n4_s0 \
  --tag jax_sheeprl_matched_n4_s0
```

- **Step budget rationale.** At `num_envs=4, replay_ratio=1.0`, we expect ~500-1,200 env-SPS (sheeprl-side sps_env_interaction is ~600; JAX's flat-encoder + halved model with 1:1 grad:env should be 1-2x sheeprl's number — see §6.1 of the SPS memo for the env-collection-only floor). 1.5M env-steps = ~25 min upper bound, ~20 min lower bound. Comfortably amortizes JIT compile (one-time ~30-60 s).
- **Stability of SPS reading.** The SPS-comparison memo §3 extracted stable SPS from JAX runs after ~1M env-steps. 1.5M gives a margin.
- **`--episodes 0` rationale.** The env YAML `configs/experiment/dreamer_curriculum/01_food_only.yaml` sets `episodes: 100`, and `train.py:1165` uses the episode-based loop whenever `episodes > 0` — which means `--total-timesteps` is silently ignored. CLI override `--episodes 0` flips the loop to the `else` branch at line 1165 (`global_step < total_timesteps`), making `--total-timesteps` the controlling stop condition. Without this flag, Run A would stop after 100 episodes / ~110 s (this is exactly what happened on wandb=pz66mhu0 — far below the §6.1 acceptance floor `_runtime ≥ 600 s`).

### 4.3 Run B — `num_envs=16` (architectural)

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/dreamer_curriculum/01_food_only.yaml \
  --agent_config configs/models/dreamer_v3/dreamer_v3_sheeprl_matched.yaml \
  --num-envs 16 \
  --episodes 0 \
  --total-timesteps 3000000 \
  --seed 0 \
  --device cuda:<gpu> \
  --log-interval 50 \
  --wandb-group sheeprl_bridge \
  --wandb-job-type sps_measurement \
  --wandb-name jax_sheeprl_matched_n16_s0 \
  --tag jax_sheeprl_matched_n16_s0
```

- **Step budget rationale.** At `num_envs=16, replay_ratio=1.0`, we expect ~1,500-3,000 env-SPS. 3M env-steps = ~17-33 min. Same justification as Run A — amortizes compile, reaches steady state.
- **`--episodes 0` rationale.** Same as Run A — the env YAML pins `episodes: 100`, which silently overrides `--total-timesteps` unless CLI sets `--episodes 0`. See §4.2 note.
- **GPU pinning.** `<gpu>` is filled at launch by `training-runner`. Avoid `cuda:2` on `n114` (held by the running sheeprl replica `kfsvh1qk`). Prefer `n113` or `n114:cuda:0/1/3` (RTX 4090 / RTX 6000 Ada — same hardware class the JAX SPS numbers in the memo were measured on; do not switch GPU class or the comparison drifts).

### 4.4 WandB conventions

- **Project:** `grid_world_pain` (default; `train.py` reads from `configs/logger/wandb.yaml`).
- **Group:** `sheeprl_bridge` (so the two runs join the other sheeprl-bridge artifacts on the WandB web).
- **Job-type:** `sps_measurement` (per CLAUDE.md operational-category convention; this is not a prod training run).
- **Tag = wandb-name:** identical per row of the launch manifest below, so `train.py` does not synthesize a name and the analyzer can grep by tag.

### 4.5 Launch manifest (system-of-record)

| Run | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | num_envs | Status | Node | GPU | Launched at | WandB run ID | Log path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | matched-n4 | `jax_sheeprl_matched_n4_s0` | sheeprl_bridge | sps_measurement | 0 | 4 | completed | 113 | cuda:0 | 2026-05-13T15:43:11 | kmf1574r | logs/20260513_154311.log |
| B | matched-n16 | `jax_sheeprl_matched_n16_s0` | sheeprl_bridge | sps_measurement | 0 | 16 | completed | 113 | cuda:0 | 2026-05-13T15:54:11 | deizzwp4 | logs/20260513_155411.log |

The `training-runner` fills Node / GPU / Launched-at / WandB-run-ID / Log-path at launch.

### 4.5.1 Configs to produce

| Run | Env config | Agent config |
|---|---|---|
| A, B (both) | `configs/experiment/dreamer_curriculum/01_food_only.yaml` (existing, read-only) | `configs/models/dreamer_v3/dreamer_v3_sheeprl_matched.yaml` (**new**) |

Only one new file — the model YAML. The env YAML is reused unchanged.

---

## 5. Analysis plan (pre-specified)

The `experiment-analyzer` agent will extract these numbers and populate them into either §3 of the SPS-comparison memo (preferred) or a new dated addendum:

### 5.1 Primary metric

**Matched-config env-SPS** for each run, computed the same way as the existing JAX runs in §3.1 of the SPS memo:

```
env-SPS = (final `timesteps` WandB summary field) / (final `_runtime` WandB summary field in seconds)
```

### 5.2 Secondary metrics

- **Grad-update-SPS.** With `replay_ratio=1.0` and `collect_interval=128`, the JAX trainer does `Ratio(1.0)(global_step // collect_interval)` gradient updates per iteration (`train.py:1767`). The simpler way: `grad_update_SPS ≈ replay_ratio × env_SPS ≈ env_SPS` (since `replay_ratio=1.0`). The WandB key `Params/effective_replay_ratio` should be close to 1.0 by end of run — confirm.
- **Transitions-processed-per-grad-update.** `batch_size × sequence_length = 16 × 64 = 1024` (matches sheeprl exactly).
- **Sheeprl comparison anchor:** ~4.4 env-SPS, ~1.1 grad-SPS, ~1,024 transitions/grad-update (per §3 of the SPS-comparison memo, all three sheeprl runs).

### 5.3 Output table format

To be appended to §3 of the SPS-comparison memo:

| Run | Env steps reached | Wall-clock (s) | env-SPS | grad-update-SPS | num_envs | replay_ratio | Ratio vs sheeprl (env-SPS) | Ratio vs sheeprl (grad-SPS) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run A (matched-n4) | TBD | TBD | TBD | TBD | 4 | 1.0 | TBD | TBD |
| Run B (matched-n16) | TBD | TBD | TBD | TBD | 16 | 1.0 | TBD | TBD |
| (anchor) jzgkcep4 sheeprl | 200,000 | 45,132 | 4.43 | 1.12 | 4 | 1 | 1.0x | 1.0x |

### 5.4 Robustness checks

- **Steady-state SPS, not initial-iteration SPS.** Compute SPS using only the post-warmup window (drop the first 30 s of `_runtime`). If the summary-field reading and the steady-state reading differ by > 5%, report both.
- **Cross-check via `iteration` summary.** A second SPS reading from `iteration × collect_interval × num_envs` vs `_runtime` should match within 1%.
- **JIT-compile detection.** The first iteration's wall-clock should be ~10-30x longer than the steady-state iteration time (compile-dominated). Mention this in the report — it's expected, not a bug.

---

## 6. Acceptance criterion + decision matrix

### 6.1 Acceptance (both runs must satisfy)

1. **Run completes without crash.** No NaN, no OOM, no JAX device error. If a run crashes, treat as a failed measurement and re-launch (not a verdict on the underlying question).
2. **Stable env-SPS extractable.** Final `_runtime` ≥ 600 s. Final `timesteps` ≥ 500,000. Both `_runtime` and `timesteps` present in `wandb-summary.json`.
3. **`Params/effective_replay_ratio` ≈ 1.0** by end of run (within 5%). If it's not, the matched recipe didn't apply — investigate before treating the SPS number as the "matched" answer.
4. **Per-step memory ≤ device limit.** No OOM under `num_envs=16` on a 24 GB RTX 4090. (The smaller model — 256 vs 512 recurrent dim and 1-layer MLPs — has fewer parameters than the existing `dreamer_v3.yaml`, so this is essentially guaranteed; we list it for completeness.)

### 6.2 Decision matrix (the question this experiment answers)

The decision rides on the **larger of the two ratios** (Run A or Run B). Rationale: Run B is what a real JAX rebuild would actually achieve in production; Run A is the more conservative parity number. We greenlight on the production-relevant number, but call out both bands in the report.

| Outcome | Threshold (matched JAX env-SPS / sheeprl env-SPS) | Action |
|---|---:|---|
| **PASS — real architectural advantage** | ≥ 5x | The JAX rebuild has a real speed advantage that survives full parity. Trigger `senior-developer` to plan the rebuild scope: cascade fixes (#27/#28/#29/#30/#2) + sheeprl-matched recipe + any silent code-fidelity gaps. PI consultation per the protocol in `CLAUDE.md` before a multi-run launch. |
| **CLOSE — advantage exists, cost-benefit unclear** | 1.5x ≤ ratio < 5x | User decision needed. The advantage is real but not overwhelming. Surface to the user with the question: "is a 2-3x speedup worth the engineering cost of porting the cascade fixes into a clean JAX rebuild?" Do not unilaterally trigger the rebuild plan. |
| **FAIL — apparent JAX advantage was a config artifact** | < 1.5x | The 2,600x raw ratio in the SPS memo was driven by unmatched configs, not by JAX's compute model. **Stop here.** sheeprl is the right backend on both axes (survival quality AND speed at matched recipe). Update §4 of the SPS-comparison memo to retract the "GO (qualified)" verdict; mark JAX-rebuild as closed. |

**Sanity-check anchor.** The SPS-comparison memo §4 estimated the matched-config ratio would be **~20-40x**, dropping from 2,600x as the 16x replay_ratio mismatch and the 2x sequence-length mismatch washed out. If the actual measured ratio comes in at < 5x, that estimate was significantly wrong and the §3.5 "Where the JAX advantage is coming from" diagnosis (compounding JIT + vmap + GPU-buffer factors) needs revisiting — possibly the env-collection-only ~20x number in §3.4 also has a parity-confound.

---

## 7. Failure-mode catalog

Pre-decided so post-run interpretation is not ad-hoc.

| Failure mode | Refutes hypothesis or refutes run? | Action |
|---|---|---|
| **OOM at `num_envs=16`** (we don't expect this — the matched-config model is smaller than the existing one) | Refutes the run | Retry at `num_envs=8`; report the lower-envs SPS with caveat. |
| **JIT compile dominates (final `_runtime` < 60 s)** | Refutes the run | Increase `--total-timesteps`. The acceptance criterion §6.1.2 prevents reporting on too-short runs. |
| **`Params/effective_replay_ratio` ends at < 0.9 or > 1.1** | Refutes the run | The recipe didn't apply correctly. Investigate (likely a `Ratio` clamp or warmup interaction) before reporting. |
| **One run crashes, one completes** | Refutes the crashed run, not the question | Report the completed run with a caveat; relaunch the crashed run. |
| **Both runs complete but Run B (n=16) is SLOWER per env-step than Run A (n=4)** | Probably reflects num_envs scaling cost on the matched-replay-ratio recipe (interesting but unexpected) | Report both. The PASS/CLOSE/FAIL decision uses the LARGER of the two ratios, so this scenario would default to Run A's number. |
| **Both runs land in CLOSE band (1.5-5x)** | Designed-for outcome | Surface the cost-benefit question to the user via `AskUserQuestion`. Do not auto-trigger rebuild planning. |
| **Both runs land in FAIL band (< 1.5x)** | Hypothesis refuted | Update the SPS-comparison memo §4 to retract the "GO (qualified)" verdict. JAX-rebuild question is closed. |
| **Both runs land in PASS band (≥ 5x)** | Hypothesis confirmed (matched-config JAX is still much faster than sheeprl) | Trigger `senior-developer` rebuild-planning per §6.2. PI consultation before any subsequent multi-run launch. |

---

## 8. Open questions (for post-run analysis)

These do NOT block the launch — they are flagged for the `experiment-analyzer` to address in the post-run report.

1. **Does the matched-config JAX env-collection-only number (sps_env_interaction-equivalent) agree with sheeprl's ~600?** Not directly observable from current WandB keys (per §5/§6 of the SPS memo, JAX-side never logged `Time/sps_env_interaction`). If the matched-config full-loop env-SPS ends up close to sheeprl's ~600 floor, that argues the env-collection layer itself wasn't 20x faster — it was JIT-fused-with-gradient that gave the throughput advantage.
2. **Does the residual JAX advantage have a fixed source, or does it scale with `num_envs`?** Comparing Run A (n=4) and Run B (n=16) directly tells us how much of the JAX advantage is "free parallelism" vs intrinsic compute-graph speed.
3. **Is the matched-config policy learning anything?** Not the point of this experiment (it's an SPS measurement, not a survival measurement). But the `Episode/survival_steps` curve will be logged anyway. A casual check ("is the agent moving past random?") is fine, but no survival conclusion should be drawn from these short runs.

---

## 9. Pre-launch gate

Before `training-runner` is invoked:

1. `env-config-auditor` reviews `configs/models/dreamer_v3/dreamer_v3_sheeprl_matched.yaml` against `configs/experiment/dreamer_curriculum/01_food_only.yaml`. Standard schema + obs/noise consistency check. The new agent config does not introduce new YAML keys, so this should pass cleanly.
2. User authorization to launch.
3. `training-runner` invoked with: node + GPU per run, env config, agent config, the exact launch commands above.

After both runs complete, control returns to the `experiment-analyzer` to fill the §5.3 table and update either §3 of the SPS-comparison memo or a new dated addendum.

---

## 10. Results

_To be filled by `experiment-analyzer` after both runs complete._

## 11. Conclusions

_To be filled by `experiment-analyzer` after both runs complete. The conclusion should reference §6.2 and route to one of PASS / CLOSE / FAIL._
