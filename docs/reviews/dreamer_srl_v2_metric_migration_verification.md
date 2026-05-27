---
title: "dreamer-srl v2 metric migration — verification report"
topic: dreamer_srl_v2
status: active
created: 2026-05-15
last_updated: 2026-05-15
---

# dreamer-srl v2 metric migration — verification report

## Headline / Verdict (plain-language entry point)

**Verdict: PASS. Authorize relaunch of tasks #28 (5×5 + predator) and #29
(10×10 hypervigilance).**

The developer's 7-chunk metric migration — landed across five commits
`e917187` → `f90a183` on branch `v1.4` — ports the standalone dreamer-srl
JAX trainer onto the master project's unified WandB metric schema. Before
this migration, dreamer-srl logged ~25 keys to its own WandB project
(`grid_world_pain_dreamer_srl_v2`) under names like `Game/ep_len_avg` and
`Loss/world_model_loss`. After this migration, dreamer-srl logs **49 unified
keys** to the **shared `grid_world_pain` WandB project** under the schema
the master `train.py` (rPPO and JAX-DreamerV3) drivers already use:
`Episode/*` (survival-step + behavior-event family, 24 keys),
`WorldModel/*` (model losses + reward-MAE + continue-head accuracy probes,
12 keys), `Behavior/*` (actor / critic / imagined-rollout statistics,
10 keys), and 3 cross-cutting keys (`Time/sps_env`,
`Params/effective_replay_ratio`, `Diagnostic/moments_invscale`).

The 49-key check on the smoke run is exact: **49 expected, 49 present, 0
missing, 0 unexpected extras**. The 49 dreamer-srl pytest tests pass after
every commit; the 17 offline parity checks pass after every commit. The
two new `Time/sps_env` log-sites in master `train.py` (rPPO branch L1485,
JAX-DreamerV3 branch L1791) compute env-steps-per-second as
`global_step / elapsed_seconds` with a small denominator guard. The
deviation the developer flagged — `load_behavior_measure_cfg` lives in
`src.environment.config_loader`, not `src.behavior.config` as the plan
sketched — is consistent across all 7 import sites in the codebase.

With the schema unified, all 49 keys from the relaunched runs (#28 and
#29) will land in the same WandB project as the rPPO / JAX-DreamerV3
master runs and feed the existing dashboards.

## What was checked (6 verification steps)

| # | Step | Verdict | Evidence |
|---|------|:------:|----------|
| 1 | All 5 commits land on branch `v1.4` in order | ✅ | `git log --oneline` shows `e917187` → `d1da49f` → `81aa414` → `a0df3d1` → `f90a183`, capped by `0e6ba5e` (doc-only Implementation Report close-out) |
| 2 | pytest `tests/algorithms/dreamer_srl/` PASS | ✅ | **49 passed in 76.39s** |
| 3 | offline_check 17/17 PASS | ✅ | `Result: 17/17 checks passed; Maximum tensor drift: 4.768e-07 in [neg_lp2]` (under budget 5e-05) |
| 4 | Smoke run `91xxe64y` summary keys match manifest | ✅ | 49 expected metric keys, 49 actual metric keys, 0 missing, 0 unexpected (parser script below) |
| 5 | rPPO + JAX-Dreamer `Time/sps_env` log-site spot-check | ✅ | `train.py:L1485` (rPPO branch, inside `wandb_logs.update`) and `train.py:L1791` (Dreamer branch, inside `wandb_logs` dict literal) — both compute `global_step / max(elapsed, 1e-9)` |
| 6 | `load_behavior_measure_cfg` import site consistency | ✅ | Sole definition at `src/environment/config_loader.py:55`; 7 import sites in `src/` `scripts/` `tests/` all use the canonical path |

## 49-key manifest — per-key PASS/FAIL on smoke run `91xxe64y`

Smoke run: <https://wandb.ai/sungwoolee/grid_world_pain/runs/91xxe64y>
(local mirror: `wandb/run-20260515_142455-91xxe64y/files/wandb-summary.json`).

### Episode/* — 24 fixed keys (Commit 2 + 3)

| Key | PASS |
|---|:---:|
| `Episode/Reward` | ✅ |
| `Episode/Reward_Min` | ✅ |
| `Episode/Reward_Max` | ✅ |
| `Episode/Steps` | ✅ |
| `Episode/Number` | ✅ |
| `Episode/FoodEaten` | ✅ |
| `Episode/PredatorHits` | ✅ |
| `Episode/DangerHits` | ✅ |
| `Episode/RestCount` | ✅ |
| `Episode/Collisions` | ✅ |
| `Episode/TotalDamage` | ✅ |
| `Episode/DamagePredator` | ✅ |
| `Episode/DamageDanger` | ✅ |
| `Episode/DamageObstacle` | ✅ |
| `Episode/MeanDistFood` | ✅ |
| `Episode/MeanDistPredator` | ✅ |
| `Episode/MeanDistRabbit` | ✅ |
| `Episode/MeanDistHidingPredator` | ✅ |
| `Episode/RabbitHits` | ✅ |
| `Episode/HidingPredatorHits` | ✅ |
| `Episode/Term_MaxSteps` | ✅ |
| `Episode/Term_Starvation` | ✅ |
| `Episode/Term_Overeating` | ✅ |
| `Episode/Term_Injury` | ✅ |

### WorldModel/* — 12 keys (Commit 6, including aliases of legacy sheeprl-style keys)

| Key | PASS |
|---|:---:|
| `WorldModel/loss_model` | ✅ |
| `WorldModel/loss_recon` | ✅ |
| `WorldModel/loss_rew` | ✅ |
| `WorldModel/loss_kl` | ✅ |
| `WorldModel/loss_cont` | ✅ |
| `WorldModel/loss_dyn_kl` | ✅ |
| `WorldModel/loss_rep_kl` | ✅ |
| `WorldModel/model_reward_mae` | ✅ |
| `WorldModel/model_reward_mae_pos` | ✅ |
| `WorldModel/model_reward_mae_neg` | ✅ |
| `WorldModel/model_latent_entropy` | ✅ |
| `WorldModel/model_cont_acc` | ✅ |

### Behavior/* — 10 keys (Commit 6; the sheeprl `Loss/*` prefix has been replaced — no `Loss/*` keys present in summary)

| Key | PASS |
|---|:---:|
| `Behavior/loss_critic` | ✅ |
| `Behavior/loss_actor` | ✅ |
| `Behavior/loss_actor_policy` | ✅ |
| `Behavior/loss_actor_entropy` | ✅ |
| `Behavior/mean_return` | ✅ |
| `Behavior/mean_norm_return` | ✅ |
| `Behavior/mean_value` | ✅ |
| `Behavior/mean_advantage` | ✅ |
| `Behavior/mean_entropy` | ✅ |
| `Behavior/value_mae` | ✅ |

### Cross-cutting — 3 keys (Commits 6 + 7)

| Key | PASS |
|---|:---:|
| `Time/sps_env` | ✅ |
| `Params/effective_replay_ratio` | ✅ |
| `Diagnostic/moments_invscale` | ✅ |

**Total: 49/49 PASS. 0 missing. 0 unexpected extras.**

The 5 keys observed beyond the 49 metric keys (`_runtime`, `_step`,
`_timestamp`, `_wandb`, `timesteps`) are WandB internal book-keeping
fields, expected on every run regardless of schema.

## Methods

### Verifier script (manifest check, step 4)

```python
import json
expected = set()
# Commit 2 — 5 base Episode keys
expected |= {"Episode/Reward", "Episode/Reward_Min", "Episode/Reward_Max",
             "Episode/Steps", "Episode/Number"}
# Commit 3 — 19 Episode extras
expected |= {"Episode/FoodEaten", "Episode/PredatorHits", "Episode/DangerHits",
             "Episode/RestCount", "Episode/Collisions", "Episode/TotalDamage",
             "Episode/DamagePredator", "Episode/DamageDanger", "Episode/DamageObstacle",
             "Episode/MeanDistFood", "Episode/MeanDistPredator", "Episode/MeanDistRabbit",
             "Episode/MeanDistHidingPredator", "Episode/RabbitHits",
             "Episode/HidingPredatorHits", "Episode/Term_MaxSteps",
             "Episode/Term_Starvation", "Episode/Term_Overeating", "Episode/Term_Injury"}
# Commit 6 — 12 WorldModel/* + 10 Behavior/* + 2 cross-cutting
expected |= {"WorldModel/loss_model", "WorldModel/loss_recon", "WorldModel/loss_rew",
             "WorldModel/loss_kl", "WorldModel/loss_cont",
             "WorldModel/loss_dyn_kl", "WorldModel/loss_rep_kl",
             "WorldModel/model_reward_mae", "WorldModel/model_reward_mae_pos",
             "WorldModel/model_reward_mae_neg", "WorldModel/model_latent_entropy",
             "WorldModel/model_cont_acc"}
expected |= {"Behavior/loss_critic", "Behavior/loss_actor",
             "Behavior/loss_actor_policy", "Behavior/loss_actor_entropy",
             "Behavior/mean_return", "Behavior/mean_norm_return",
             "Behavior/mean_value", "Behavior/mean_advantage",
             "Behavior/mean_entropy", "Behavior/value_mae"}
expected |= {"Params/effective_replay_ratio", "Diagnostic/moments_invscale"}
# Commit 7 — 1 cross-cutting
expected |= {"Time/sps_env"}
# Result: |expected| = 49

with open("wandb/run-20260515_142455-91xxe64y/files/wandb-summary.json") as f:
    s = json.load(f)
actual = {k for k in s.keys() if not k.startswith("_") and k != "timesteps"}
assert len(actual) == 49
assert expected == actual  # zero missing, zero extras
```

### `Time/sps_env` log-site spot-check (step 5)

- **rPPO branch — `train.py:L1481–L1488`**: `wandb_logs.update({…, "Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9), …})` followed by `wandb.log(wandb_logs)`. Inside the correct `wandb.log` dict.
- **JAX-DreamerV3 branch — `train.py:L1786–L1792`**: `wandb_logs = {"timesteps": global_step, "iteration": iteration, "Params/effective_replay_ratio": …, "Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9)}`. Inside the correct `wandb_logs` dict literal.
- **dreamer-srl branch — `src/algorithms/dreamer_srl/dreamer_srl_main.py:L716, L778`**: `sps_env = policy_step / max(time.time() - t_start, 1e-9)` computed at L716, logged at L778. Same formula; same denominator guard.

All three throughput sites use `steps / max(elapsed, eps)` with `eps=1e-9`. Consistent.

### `load_behavior_measure_cfg` import audit (step 6)

| Site | File | Path used |
|---|---|---|
| Definition | `src/environment/config_loader.py:55` | (declared here) |
| Import 1 | `src/algorithms/dreamer_srl/dreamer_srl_main.py:400` | `from src.environment.config_loader import load_behavior_measure_cfg` |
| Import 2 | `scripts/eval_rollout.py:50` | `from src.environment.config_loader import …, load_behavior_measure_cfg` |
| Import 3 | `scripts/motif_cluster.py:34` | `from src.environment.config_loader import …, load_behavior_measure_cfg, …` |
| Import 4 | `tests/environment/test_behavior_measures.py:21` | `from src.environment.config_loader import …, load_behavior_measure_cfg, …` |

All 4 import sites use the canonical path. No stale imports from a hypothetical `src.behavior.config` module. The plan's path sketch was incorrect; the developer corrected to the canonical path in `dreamer_srl_main.py`. No scope impact.

## Findings

| Class | Item | Note |
|---|---|---|
| Nit | Plan sketched `src.behavior.config` for `load_behavior_measure_cfg` | Canonical location is `src.environment.config_loader`. Developer corrected at the import site; all import sites in the codebase agree. No remediation needed. |

No P-class (blocker), no F-class (fix-before-merge) findings.

## Relaunch authorization

The two trainings killed at the start of this metric migration cycle are authorized to relaunch under the unified `grid_world_pain` WandB project. Both runs will land all 49 unified keys.

### Task #28 — 5×5 + predator (node 114 GPU 0)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
WANDB_RUN_GROUP=dreamer_srl_v2_extension_2026-05-15 WANDB_JOB_TYPE=5x5_pred \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  src/algorithms/dreamer_srl/dreamer_srl_main.py \
  --env-config configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml \
  --agent-config configs/models/dreamer_srl/01_food_only.yaml \
  --total-steps 200000 --num-envs 1 --seed 42 \
  --wandb-project grid_world_pain \
  --wandb-name dreamer_srl_v2_5x5_pred_XS_s42
```

### Task #29 — 10×10 hypervigilance (node 114 GPU 1)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 \
WANDB_RUN_GROUP=dreamer_srl_v2_extension_2026-05-15 WANDB_JOB_TYPE=10x10_intero \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  src/algorithms/dreamer_srl/dreamer_srl_main.py \
  --env-config configs/experiment/hypervigilance/01-interoNocicept.yaml \
  --agent-config configs/models/dreamer_srl/01_food_only.yaml \
  --total-steps 200000 --num-envs 1 --seed 42 \
  --wandb-project grid_world_pain \
  --wandb-name dreamer_srl_v2_10x10_intero_XS_s42
```

Per the agent-team contract, this verification report does **not** spawn the
training-runner. The launch invocations above are returned to the parent
(top-level Claude) for handoff to `training-runner`.

## Links

- [`MIGRATION_PLAN.md`](../develop/active/dreamer_srl_v2/MIGRATION_PLAN.md) — the 7-commit plan the developer executed
- [`IMPLEMENTATION_PLAN.md`](../develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md) — the v2 master plan; Post-CP12 work section logs this migration
- [`PARITY_LAUNCH_V2.md`](../experiments/active/dreamer_srl_v2/PARITY_LAUNCH_V2.md) — the v2-CP11 PASS-OUTPERFORM verdict
- Smoke run with 49 unified keys: <https://wandb.ai/sungwoolee/grid_world_pain/runs/91xxe64y>

**Verified by: senior-developer**
