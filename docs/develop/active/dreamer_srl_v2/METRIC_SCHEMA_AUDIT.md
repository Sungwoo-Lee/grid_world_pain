---
title: "Metric-schema audit: JAX-Dreamer vs rPPO vs dreamer-srl"
topic: dreamer
status: active
created: 2026-05-15
last_updated: 2026-05-15
---

# Metric-schema audit: JAX-Dreamer vs rPPO vs dreamer-srl

> **Companion plan:** the migration plan that closes the gaps surfaced below
> lives at [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) (to be authored by
> `senior-developer` after the user dispositions the open questions in §9).
> Implementation will be executed by the `developer` agent.

---

## 1. Context (plain-language entry point)

**What this doc is.** A read-only inventory of every metric the three training
drivers in this repo currently log to **Weights & Biases** (WandB — the cloud
dashboard the project uses for run comparison; each driver opens a "project" =
top-level folder of runs). Across the three drivers we found **~120 distinct
WandB keys**; the new dreamer-srl driver logs only **5 of them**. This audit
makes the gap visible so the user can decide, key by key, which to bring across.

**The three drivers compared.**

| Short name | What it is | Lives at |
|---|---|---|
| **JAX-Dreamer** | The original Dreamer-V3 implementation in this repo. Logs under WandB project `grid_world_pain` (the "master" project) with the `Episode/*` + `Behavior/*` + `WorldModel/*` schema described below. | `train.py` (Dreamer branch, lines 1497–1827) |
| **rPPO** | The original Recurrent-PPO baseline in this repo. Same project (`grid_world_pain`), same `Episode/*` schema, but a different `loss/*` namespace (PPO-specific: policy / value / entropy losses). | `train.py` (RecurrentPPO branch, lines 1286–1495) |
| **dreamer-srl** | The freshly rebuilt sheeprl-faithful Dreamer-V3 port (parity-validated at "✅ PASS-OUTPERFORM" on commit `0f50877`). **Currently logs into its own separate project `grid_world_pain_dreamer_srl_smoke` with sheeprl conventions** (`Game/ep_len_avg`, `Rewards/rew_avg`, `Loss/*`, plus a handful of `Diagnostic`/`Params`/`Time` keys). | `src/algorithms/dreamer_srl/dreamer_srl_main.py` (lines 302–615) |

**Three layers of metric.** A WandB key is logged at one of three cadences:
**(a) per-episode** (averaged across the episodes that finished in this
logging window — the survival-step toolkit), **(b) per-iteration** (one
gradient-cycle's worth of losses + diagnostics — the toolkit's heartbeat), or
**(c) per-eval** (every-N-iterations rollouts at greedy-action mode against a
held-out seed). JAX-Dreamer has all three; rPPO has (a) and (b) but no (c);
dreamer-srl has only (b) plus a stripped-down (a) (raw per-episode rather than
window-averaged).

**The user's two asks.** **(1)** Make dreamer-srl log into the master
`grid_world_pain` project with the JAX-Dreamer key schema so the project's
existing dashboards keep working. **(2)** Add `Time/sps_env` (steps-per-second
throughput, currently only in dreamer-srl) to rPPO so all three drivers report
training speed in the same key. Everything else (which `Episode/*` keys, which
behavioral-toolkit keys, whether to port the eval-level rollouts) is
"audit-then-decide" — see §9 Open Questions.

---

## 2. Audit methodology

**Files walked, fully.**

1. [`train.py`](../../../../train.py) — every `wandb.log(...)` and
   `wandb.define_metric(...)` call surveyed via `grep -n`. Sites: L590
   (login + `define_metric` block at L625–L634), L1238 (`stage/buffer_cleared_*`),
   L1262 (`stage/index` + `stage/transition`), L1397–L1437 (rPPO episode-aggregate
   log dict), L1450–L1486 (rPPO step-aggregate log dict including modulator
   block), L1710–L1753 (Dreamer episode-aggregate log dict — same schema as rPPO
   episode block), L1784–L1812 (Dreamer step-aggregate log dict including
   `Behavior/*` + `WorldModel/*` + `Modulator/*` fan-out from the trainer's
   loss metrics), L2002–L2004 (DQN paths — out of scope for this audit but
   noted), L2463 (Eval/MeanReward + Eval/MeanLength at eval-rollout boundary).
2. [`src/models/dreamer_v3_trainer.py`](../../../../src/models/dreamer_v3_trainer.py) —
   the trainer's internal metrics dict at L277–L290 (`loss_*` + `model_*`),
   L301–L323 (`mod_*` under `modulation_enabled`), L468–L490 (behavior-loss
   metrics: `loss_critic`/`loss_actor*` + `mean_return`/`mean_norm_return`/
   `mean_value`/`mean_advantage`/`mean_entropy`/`value_mae`), and IMG_PROBE
   block at L481–L489 (`imagined_*` keys, gated on
   `agent.imagined_rollout_probe`). The driver at L1797–L1810 sorts these by
   prefix into the `Behavior/*` / `WorldModel/*` / `Modulator/*` namespaces.
3. [`src/algorithms/dreamer_srl/dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py) —
   two `wandb.log` sites: L455 (per-episode `Game/ep_len_avg` + `Rewards/rew_avg`
   logged raw, one row per done env, no iteration averaging) and L604 (per-log-window
   `Loss/*` + `Params/replay_ratio` + `Time/sps_env` + `Diagnostic/moments_invscale`).
   `wandb.init` at L310 — no `define_metric` calls (sheeprl-style: WandB picks
   its own step axis from `step=policy_step` kwarg).
4. [`src/behavior/accumulators.py`](../../../../src/behavior/accumulators.py) —
   the `BMState` + `bm_finalise_episode` toolkit; the full ordered list of
   keys it emits is given by `bm_wandb_keys()` at L459–L487.
5. [`configs/logger/wandb.yaml`](../../../../configs/logger/wandb.yaml) —
   current `wandb.project: grid_world_pain` (the master project the user
   wants dreamer-srl to share).

**Coverage caveats.** The DQN, DRQN, and vanilla-PPO branches of `train.py`
also call `wandb.log` (L2002/L2208/L2361) — these were **not** inventoried
because they are out of scope (the user's two asks are about Dreamer and
rPPO only). The DQN/DRQN/PPO branches reuse the same `Episode/*` keys via
the shared `_bm_log_wandb` / `_append_per_tag_means` helpers (this is the
project's universal episode schema), but introduce their own algorithm-specific
loss/exploration keys (e.g. `train/epsilon` for DQN) that are not part of this
audit.

**Per-tag fan-out caveat.** The exact number of `Episode/MeanDistRabbit_<tag>`
/ `Episode/MeanDistPredator_<tag>` / behavioral-toolkit per-tag keys depends
on the environment config (`neutral_tags` and `predator_tags` from
`params`) — so the totals below count "1 key per family + per-tag fan-out
schema" rather than expanding the per-tag families.

---

## 3. Master metric table

**Legend.** ✅ = logged. ❌ = NOT logged. 🟡 = logged under a different name
(alias given). All Dreamer + rPPO entries cite `train.py` line ranges; all
dreamer-srl entries cite `src/algorithms/dreamer_srl/dreamer_srl_main.py`.

### 3a. Episode-level keys (step_metric = `Episode/Number`)

JAX-Dreamer and rPPO both compute these as **iteration-level
`np.mean(iteration_episodes)`** — the buffer `iteration_episodes` (rPPO L1394,
Dreamer L1710) collects every episode finished since the last log row and
averages once per `log_interval` iterations. dreamer-srl currently logs raw
per-episode under aliases (one WandB row per done env, no averaging).

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `Episode/Number` | ✅ L1718 | ✅ L1402 | ❌ | counter | step_metric for all `Episode/*`. |
| `Episode/Reward` | ✅ L1714 | ✅ L1398 | 🟡 `Rewards/rew_avg` (L455, raw-per-ep) | iter-mean | |
| `Episode/Reward_Min` | ✅ L1715 | ✅ L1399 | ❌ | iter-min | |
| `Episode/Reward_Max` | ✅ L1716 | ✅ L1400 | ❌ | iter-max | |
| `Episode/Steps` | ✅ L1717 | ✅ L1401 | 🟡 `Game/ep_len_avg` (L455, raw-per-ep) | iter-mean | **the project's survival-step KPI.** |
| `Episode/FoodEaten` | ✅ L1724 | ✅ L1408 | ❌ | iter-mean | gated on `ate_food in iteration_episodes[0]`. |
| `Episode/PredatorHits` | ✅ L1725 | ✅ L1409 | ❌ | iter-mean | |
| `Episode/DangerHits` | ✅ L1727 | ✅ L1411 | ❌ | iter-mean | alias for `hit_hiding_predator`, kept for dashboard history. |
| `Episode/RestCount` | ✅ L1728 | ✅ L1412 | ❌ | iter-mean | |
| `Episode/Collisions` | ✅ L1729 | ✅ L1413 | ❌ | iter-mean | |
| `Episode/TotalDamage` | ✅ L1730 | ✅ L1414 | ❌ | iter-mean | |
| `Episode/DamagePredator` | ✅ L1731 | ✅ L1415 | ❌ | iter-mean | |
| `Episode/DamageDanger` | ✅ L1732 | ✅ L1416 | ❌ | iter-mean | aliases `damage_hiding_predator`. |
| `Episode/DamageObstacle` | ✅ L1733 | ✅ L1417 | ❌ | iter-mean | |
| `Episode/MeanDistFood` | ✅ L1734 | ✅ L1418 | ❌ | iter-mean of per-ep-mean | |
| `Episode/MeanDistPredator` | ✅ L1735 | ✅ L1419 | ❌ | iter-mean of per-ep-mean | |
| `Episode/MeanDistRabbit` | ✅ L1736 | ✅ L1420 | ❌ | iter-mean of per-ep-mean | |
| `Episode/MeanDistHidingPredator` | ✅ L1737 | ✅ L1421 | ❌ | iter-mean of per-ep-mean | |
| `Episode/RabbitHits` | ✅ L1738 | ✅ L1422 | ❌ | iter-mean | aliases `hit_neutral`. |
| `Episode/HidingPredatorHits` | ✅ L1739 | ✅ L1423 | ❌ | iter-mean | (duplicate of `DangerHits` under new label.) |
| `Episode/Term_MaxSteps` | ✅ L1744 | ✅ L1428 | ❌ | iter-mean (fraction) | termination-reason fraction. |
| `Episode/Term_Starvation` | ✅ L1744 | ✅ L1428 | ❌ | iter-mean (fraction) | |
| `Episode/Term_Overeating` | ✅ L1744 | ✅ L1428 | ❌ | iter-mean (fraction) | |
| `Episode/Term_Injury` | ✅ L1744 | ✅ L1428 | ❌ | iter-mean (fraction) | |
| `Episode/MeanDistRabbit_<tag>` | ✅ L1747 | ✅ L1431 | ❌ | iter-mean of per-ep-mean | fan-out, 1 key per `neutral_tag` in env config. |
| `Episode/MeanDistPredator_<tag>` | ✅ L1748 | ✅ L1432 | ❌ | iter-mean of per-ep-mean | fan-out, 1 key per `predator_tag` in env config. |

**Episode/* family total:** 24 fixed keys + 2 per-tag fan-out families.
**dreamer-srl currently logs 0 of these 24 fixed keys** (the two aliases
`Game/ep_len_avg` and `Rewards/rew_avg` are sheeprl names for `Episode/Steps`
and `Episode/Reward` respectively, but logged at raw cadence not iter-mean).

### 3b. Step-level / Loss keys (step_metric = `iteration` or `timesteps`)

rPPO and JAX-Dreamer use lowercase `loss/*`; dreamer-srl uses uppercase
`Loss/*` (sheeprl convention). **The two namespaces are different WandB folders
on the dashboard.**

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `loss/total` | ❌ | ✅ L1452 | ❌ | iter-mean | rPPO only. |
| `loss/policy` | ❌ | ✅ L1453 | ❌ | iter-mean | PPO-specific. |
| `loss/value` | ❌ | ✅ L1454 | ❌ | iter-mean | PPO-specific. |
| `loss/entropy` | ❌ | ✅ L1455 | ❌ | iter-mean | PPO-specific. |
| `loss/grad_norm` | ❌ | ✅ L1456 | ❌ | iter-mean | PPO-specific (model grad-norm). |
| `WorldModel/loss_model` | ✅ L1801,trainer L278 | ❌ | 🟡 `Loss/world_model_loss` (L592) | last grad-step | Dreamer total WM loss. |
| `WorldModel/loss_recon` | ✅ trainer L279 | ❌ | 🟡 `Loss/observation_loss` (L593) | last grad-step | |
| `WorldModel/loss_rew` | ✅ trainer L280 | ❌ | 🟡 `Loss/reward_loss` (L594) | last grad-step | |
| `WorldModel/loss_cont` | ✅ trainer L281 | ❌ | 🟡 `Loss/continue_loss` (L596) | last grad-step | |
| `WorldModel/loss_kl` | ✅ trainer L284 | ❌ | 🟡 `Loss/state_loss` (L595) | last grad-step | sheeprl calls KL "state_loss". |
| `WorldModel/loss_dyn_kl` | ✅ trainer L282 | ❌ | ❌ | last grad-step | dynamic-prior KL component. |
| `WorldModel/loss_rep_kl` | ✅ trainer L283 | ❌ | ❌ | last grad-step | representation KL component. |
| `Behavior/loss_actor` | ✅ trainer L470 | ❌ | 🟡 `Loss/policy_loss` (L598) | last grad-step | |
| `Behavior/loss_critic` | ✅ trainer L469 | ❌ | 🟡 `Loss/value_loss` (L597) | last grad-step | |
| `Behavior/loss_actor_policy` | ✅ trainer L471 | ❌ | ❌ | last grad-step | actor-loss policy term (separate from entropy). |
| `Behavior/loss_actor_entropy` | ✅ trainer L472 | ❌ | ❌ | last grad-step | actor-loss entropy term. |
| `Behavior/mean_return` | ✅ trainer L473 | ❌ | ❌ | last grad-step | imagined-rollout lambda-return mean. |
| `Behavior/mean_norm_return` | ✅ trainer L474 | ❌ | ❌ | last grad-step | normalized version. |
| `Behavior/mean_value` | ✅ trainer L475 | ❌ | ❌ | last grad-step | |
| `Behavior/mean_advantage` | ✅ trainer L476 | ❌ | ❌ | last grad-step | |
| `Behavior/mean_entropy` | ✅ trainer L477 | ❌ | ❌ | last grad-step | actor entropy. |
| `Behavior/value_mae` | ✅ trainer L478 | ❌ | ❌ | last grad-step | |
| `WorldModel/model_reward_mae` | ✅ trainer L285 | ❌ | ❌ | last grad-step | |
| `WorldModel/model_reward_mae_pos` | ✅ trainer L286 | ❌ | ❌ | last grad-step | reward-MAE conditional on positive reward. |
| `WorldModel/model_reward_mae_neg` | ✅ trainer L287 | ❌ | ❌ | last grad-step | reward-MAE conditional on negative reward. |
| `WorldModel/model_latent_entropy` | ✅ trainer L288 | ❌ | ❌ | last grad-step | |
| `WorldModel/model_cont_acc` | ✅ trainer L289 | ❌ | ❌ | last grad-step | continue-head classification accuracy. |
| `WorldModel/imagined_termination_fraction_h8` | ✅ trainer L483 | ❌ | ❌ | last grad-step | gated on `IMG_PROBE`. |
| `WorldModel/imagined_termination_fraction_h15` | ✅ trainer L484 | ❌ | ❌ | last grad-step | |
| `WorldModel/imagined_first_term_step_mean` | ✅ trainer L485 | ❌ | ❌ | last grad-step | |
| `WorldModel/imagined_term_step_p10` | ✅ trainer L486 | ❌ | ❌ | last grad-step | |
| `WorldModel/imagined_term_step_p50` | ✅ trainer L487 | ❌ | ❌ | last grad-step | |
| `WorldModel/imagined_term_step_p90` | ✅ trainer L488 | ❌ | ❌ | last grad-step | |
| `WorldModel/imagined_real_term_step_mean` | ✅ trainer L299 | ❌ | ❌ | last grad-step | |

### 3c. Step-level / Time keys

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `Time/sps_env` | ❌ | ❌ | ✅ L600 | running (`policy_step / wall_clock`) | **only dreamer-srl logs this — user wants it in rPPO too.** |

JAX-Dreamer and rPPO have **no throughput metric** at all in the WandB log
dict — wall-clock comes from WandB's own internal `_runtime` only.

### 3d. Step-level / Behavior keys

The `Behavior/*` namespace in JAX-Dreamer is populated by the driver at
L1797–L1800: any metric the trainer returns whose key starts with `loss_actor`,
`loss_critic`, `mean_`, or `entropy` gets re-prefixed `Behavior/`. The full
list is enumerated in §3b above (rows tagged `Behavior/*`).

### 3e. Step-level / Params keys

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `Params/effective_replay_ratio` | ✅ L1787 | ❌ | 🟡 `Params/replay_ratio` (L599) | running ratio | sheeprl alias = JAX-Dreamer's `effective_replay_ratio`. |
| `Params/positive_buffer_blocks` | ✅ L1793 | ❌ | ❌ | snapshot | gated on `positive_buffer is not None`. |
| `Params/positive_buffer_utilization` | ✅ L1794 | ❌ | ❌ | snapshot | |
| `Params/main_buffer_blocks` | ✅ L1795 | ❌ | ❌ | snapshot | |

### 3f. Step-level / Diagnostic keys

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `Diagnostic/moments_invscale` | ❌ | ❌ | ✅ L601 | last grad-step | dreamer-srl-only — return-normalizer scale guard. |

### 3g. Step-level / Modulator keys (gated on `modulation_enabled`)

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `modulator/grad_norm` | ❌ | ✅ L1462 | ❌ | iter-mean | rPPO-only. |
| `modulator/gamma_uni_mean` | ❌ | ✅ L1463 | ❌ | iter-mean | rPPO modulator z-unimodal. |
| `modulator/gamma_uni_std` | ❌ | ✅ L1464 | ❌ | iter-std | |
| `modulator/gamma_multi_mean` | ❌ | ✅ L1465 | ❌ | iter-mean | |
| `modulator/gamma_multi_std` | ❌ | ✅ L1466 | ❌ | iter-std | |
| `modulator/z_memory_mean` | ❌ | ✅ L1467 | ❌ | iter-mean | |
| `modulator/z_memory_std` | ❌ | ✅ L1468 | ❌ | iter-std | |
| `modulator/temperature_mean` | ❌ | ✅ L1469 | ❌ | iter-mean | |
| `modulator/temperature_min` | ❌ | ✅ L1470 | ❌ | iter-min | |
| `modulator/temperature_max` | ❌ | ✅ L1471 | ❌ | iter-max | |
| `modulator/beta_uni_mean` | ❌ | ✅ L1475 | ❌ | iter-mean | gated on PreActivation / FiLM. |
| `modulator/beta_uni_std` | ❌ | ✅ L1476 | ❌ | iter-std | |
| `modulator/beta_multi_mean` | ❌ | ✅ L1477 | ❌ | iter-mean | |
| `modulator/beta_multi_std` | ❌ | ✅ L1478 | ❌ | iter-std | |
| `Modulator/mod_z_unimodal_mean` | ✅ trainer L310 | ❌ | ❌ | last grad-step | Dreamer-side modulator. |
| `Modulator/mod_z_unimodal_std` | ✅ trainer L311 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_z_multimodal_mean` | ✅ trainer L312 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_z_multimodal_std` | ✅ trainer L313 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_memory_mean` | ✅ trainer L314 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_memory_std` | ✅ trainer L315 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_z_reward_mean` | ✅ trainer L316 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_z_reward_std` | ✅ trainer L317 | ❌ | ❌ | last grad-step | |
| `Modulator/mod_beta_unimodal_mean` | ✅ trainer L321 | ❌ | ❌ | last grad-step | gated on PreActivation / FiLM. |
| `Modulator/mod_beta_multimodal_mean` | ✅ trainer L322 | ❌ | ❌ | last grad-step | |

**Note:** rPPO uses lowercase `modulator/*`; JAX-Dreamer uses uppercase
`Modulator/*` (driver L1807). Same underlying signal, different folder on the
dashboard.

### 3h. Eval-level keys (every-N-iterations eval rollouts)

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `Eval/MeanReward` | ✅ L2463 | ✅ L2463 | ❌ | per-eval | eval-rollout summary. |
| `Eval/MeanLength` | ✅ L2463 | ✅ L2463 | ❌ | per-eval | survival steps in greedy-action eval. |

(JAX-Dreamer and rPPO share the same eval block — single `wandb.log` call.)

### 3i. System / WandB-meta keys

| Key | JAX-Dreamer | rPPO | dreamer-srl | Aggregation | Notes |
|---|---|---|---|---|---|
| `timesteps` | ✅ L1786 | ✅ L1482 | ❌ (uses WandB `step=` kwarg) | counter | global step. |
| `iteration` | ✅ L1786 | ✅ L1483 | ❌ | counter | iteration counter. |
| `stage/index` | ✅ L1263 | ✅ L1263 | ❌ | snapshot | continual-learning current stage. |
| `stage/name` | ✅ via _stage_tag | ✅ via _stage_tag | ❌ | snapshot | |
| `stage/transition` | ✅ L1264 | ✅ L1264 | ❌ | snapshot | spike to 1 on transition. |
| `stage/buffer_cleared_main` | ✅ L1239 | ❌ | ❌ | snapshot | Dreamer-only buffer wipe on stage boundary. |
| `stage/buffer_cleared_positive` | ✅ L1240 | ❌ | ❌ | snapshot | |

---

## 4. Episode-level deep dive

The user's primary KPI is **survival in steps**. The Episode/* family is the
dashboard the project lives by. JAX-Dreamer and rPPO populate it
**identically** via the same code block (rPPO L1397–L1437, Dreamer L1713–L1753
— byte-for-byte the same dict-construction block plus the conditional
`_bm_log_wandb` + `_append_per_tag_means` calls). The `_stage_tag()` helper
at L1155–L1160 appends `stage/index` + `stage/name` when continual-learning is
enabled.

**Aggregator pattern.** Both drivers buffer every episode that finished in
the current iteration into a list `iteration_episodes` (rPPO L1377, Dreamer
L1650), then once per `log_interval` iterations (`log_interval = args.log_interval
or config.get('training.log_interval', 1)`, train.py:L447) compute one
`np.mean(...)` per WandB key. **dreamer-srl currently does not buffer-then-average**
— it logs each done env raw at the env-step boundary inside the action loop
(L451–L456), producing one WandB row per episode rather than one per logging
window. This mismatch is the central decision point of §9 Q3.

**Termination-reason fractions** (`Episode/Term_*`) are mean-of-indicator
(L1428, L1744): fraction of episodes in this window that ended for each cause
(MaxSteps=1, Starvation=2, Overeating=3, Injury=4 — matches the
`infos['termination_reason']` env field). dreamer-srl already reads
`termination_reason` to split `terminated`/`truncated` (L433–L435 of
`dreamer_srl_main.py`) — the signal is available, just not aggregated.

**Per-tag fan-out.** `_append_per_tag_means` (L1046–L1059) groups the raw
per-instance distance scalars (`mean_dist_rabbit_<tag>_raw`,
`mean_dist_predator_<tag>_raw`) by tag, computes per-episode mean across
matching instances, then iteration-mean across episodes, writes back as
`Episode/MeanDistRabbit_<tag>` / `Episode/MeanDistPredator_<tag>`. The
**number of per-tag keys depends on env-config `params.neutral_tags` and
`params.predator_tags`** — common values are 0–3 tags per family.

---

## 5. Behavior-measure toolkit v1 (the `BM_*` keys)

**What it is.** The "Behavior-measure toolkit v1" is a per-episode online
accumulator at [`src/behavior/accumulators.py`](../../../../src/behavior/accumulators.py)
that tracks three behavioral measures the project's downstream experiments
depend on:

1. **M1 — Interrupted feeding rate.** Fraction of episodes in which the
   agent started to feed (`ate_food`) within `cue_radius` of a predator/rabbit
   but failed to complete the feeding bout. Per class (`predator`, `rabbit`)
   AND per tag (`<predator_tag>`, `<neutral_tag>`).
2. **M2 — Bush dive rate.** Fraction of "onsets" (entering `cue_radius` of
   threat) on which the agent dived into a bush within K observation windows.
   Per class and per tag.
3. **M5 — Eat-under-threat ratio.** Ratio of `P(eat | threat-step)` to
   `P(eat | safe-step)`. Both per-rate and the ratio itself are logged. Per
   class and per tag.

**Keys logged** (the full ordered list comes from `bm_wandb_keys()` in
`accumulators.py` L459–L487, and is emitted by `_bm_log_wandb` at train.py
L1008–L1044 when `bm_enabled` is True):

Per-class (2 classes × 8 keys = 16 keys):

- `Episode/InterruptedFeedingRate_<class>`
- `Episode/InterruptedFeedingDenominator_<class>`
- `Episode/BushDiveRate_<class>`
- `Episode/BushDiveDenominator_<class>`
- `Episode/EatUnderThreatRatio_<class>`
- `Episode/EatUnderThreatRate_<class>`
- `Episode/EatSafeRate_<class>`
- `Episode/EatUnderThreatSafeSteps_<class>`

Per-tag fan-out (4 keys per predator-tag + 4 keys per neutral-tag):

- `Episode/InterruptedFeedingRate_predator_<tag>` / `_rabbit_<tag>`
- `Episode/BushDiveRate_predator_<tag>` / `_rabbit_<tag>`
- `Episode/EatUnderThreatRatio_predator_<tag>` / `_rabbit_<tag>`
- `Episode/EatUnderThreatRate_predator_<tag>` / `_rabbit_<tag>`

**Total:** 16 fixed + 4 × `(len(predator_tags) + len(neutral_tags))` per-tag
keys. For a typical 2-predator-tag + 2-neutral-tag env: 16 + 16 = 32 keys.

**Trigger.** `bm_enabled = bm_cfg is not None and bm_cfg.enabled`
(train.py:L963), where `bm_cfg` is loaded from
`config['behavior_measures']`. When the YAML lacks this block (backwards
compat), `bm_enabled = False` and **no `BM_*` keys are emitted**.

**Audit verdict for dreamer-srl.** dreamer-srl **does not call any of**:
`bm_reset_env`, `bm_step_update`, `bm_finalise_episode`, `_bm_log_wandb`.
The toolkit's accumulators live entirely in train.py's driver scope — they
are not encapsulated in a reusable mixin. Porting requires either calling
the shared `src.behavior.accumulators` API at each of the four hook points
(per-step update, per-episode finalise, per-env reset, per-iter WandB fan-out)
or, simpler, building a thin wrapper that exposes the same interface as
`_bm_state` and ports the helpers. See §9 Q1 for the decision.

---

## 6. rPPO specifics (PPO-only loss namespace)

What rPPO logs that JAX-Dreamer doesn't (other than the modulator family
already covered in §3g):

- `loss/total` (L1452) — total PPO loss (policy + value × clip_coef +
  entropy × ent_coef).
- `loss/policy` (L1453) — clipped policy loss.
- `loss/value` (L1454) — value-function loss.
- `loss/entropy` (L1455) — negative-entropy bonus.
- `loss/grad_norm` (L1456) — model gradient norm pre-clipping.

What rPPO **does NOT** log that should be added per the user's ask:

- **`Time/sps_env`** — wall-clock throughput. Insertion point: the
  step-aggregate `wandb_logs` dict at train.py L1481–L1485, immediately
  before `wandb.log(wandb_logs)`. Compute from the existing `start_time`
  (L1152: `start_time = datetime.now()`) and `global_step` (L1318:
  `global_step += steps_this_iter`). One-line addition:
  ```python
  wandb_logs["Time/sps_env"] = global_step / max((datetime.now() - start_time).total_seconds(), 1e-9)
  ```

**What rPPO does NOT log but PPO baselines typically do** (out of scope —
the user did not ask for these, listing for completeness):

- `Train/approx_kl` — KL divergence between old and new policy.
- `Train/clip_fraction` — fraction of samples that hit the clip boundary.
- `Train/explained_variance` — value-function explained variance.
- `Train/learning_rate` — current LR (relevant if LR-scheduled).

These signals are not currently computed in `src/models/recurrent_ppo_trainer.py`
either (only `policy_loss`, `value_loss`, `ent_loss`, `grad_norm`,
`mod_grad_norm` are returned per `losses[1]` at L1443–L1447). Adding them
is a separate plan — out of scope for this audit.

---

## 7. Recommended unified schema for dreamer-srl

**Senior-developer recommendation** for each metric category. The user's
final decisions in §9 will override; this is the baseline.

### 7a. Adopt as-is (1:1 key rename from sheeprl → JAX-Dreamer)

These exist in both schemas under different names; the rename is mechanical.

| dreamer-srl current | → JAX-Dreamer key | Trivially renameable? |
|---|---|---|
| `Game/ep_len_avg` | `Episode/Steps` | Yes, plus switch aggregator (see 7d). |
| `Rewards/rew_avg` | `Episode/Reward` | Yes, plus switch aggregator. |
| `Params/replay_ratio` | `Params/effective_replay_ratio` | Yes, identical formula. |
| `Loss/world_model_loss` | `WorldModel/loss_model` | Yes, mechanical. |
| `Loss/observation_loss` | `WorldModel/loss_recon` | Yes, mechanical. |
| `Loss/reward_loss` | `WorldModel/loss_rew` | Yes, mechanical. |
| `Loss/state_loss` | `WorldModel/loss_kl` | Yes, sheeprl's `state_loss` IS the KL. |
| `Loss/continue_loss` | `WorldModel/loss_cont` | Yes, mechanical. |
| `Loss/value_loss` | `Behavior/loss_critic` | Yes, mechanical. |
| `Loss/policy_loss` | `Behavior/loss_actor` | Yes, mechanical. |

### 7b. Adapt (compute additional signals from existing trainer state)

The dreamer-srl trainer returns a `losses` dict (see
[`src/algorithms/dreamer_srl/train.py`](../../../../src/algorithms/dreamer_srl/train.py)
`make_train_step`). To match JAX-Dreamer's `WorldModel/*` and `Behavior/*`
families, additional fields would need to be exposed in that dict — every
key in §3b column "JAX-Dreamer" that is **not** already aliased above.
This means the trainer must be edited to return:

- `loss_dyn_kl`, `loss_rep_kl` — split the current `state_loss` into its
  two KL components.
- `model_reward_mae`, `model_reward_mae_pos`, `model_reward_mae_neg`,
  `model_latent_entropy`, `model_cont_acc` — world-model quality probes.
- `loss_actor_policy`, `loss_actor_entropy` — actor-loss decomposition.
- `mean_return`, `mean_norm_return`, `mean_value`, `mean_advantage`,
  `mean_entropy`, `value_mae` — imagined-rollout statistics.

This is a **non-trivial trainer-side edit**. Scope estimate: ~40 lines added
to `make_train_step`; no algorithmic change (these are read-out computations
inside the existing loss functions). See §9 Q4.

### 7c. Add new (dreamer-srl keeps, JAX-Dreamer ports back)

Two dreamer-srl-only keys that are useful — recommendation is to **keep them
under the unified schema** (i.e. they go into JAX-Dreamer too, not just
dreamer-srl):

- `Time/sps_env` — also adding to rPPO per user's ask. Recommend adding to
  JAX-Dreamer's Dreamer branch as well (L1786 area) for symmetry. **The
  user's ask is only rPPO**; this is a senior-developer suggestion — flag
  in §9 Q5.
- `Diagnostic/moments_invscale` — return-normalizer floor probe. Already
  caught the §S7 bug once at parity time. **Strongly recommend retaining**
  in dreamer-srl. JAX-Dreamer does not currently use a sheeprl-style
  moments normalizer (it has its own `Moments` module), so no port needed
  on the JAX-Dreamer side.

### 7d. Episode-aggregation: switch to per-iteration averaging

**Recommended.** Switch dreamer-srl from raw-per-episode to per-iteration
`np.mean(iteration_episodes)`. This matches JAX-Dreamer's dashboard rhythm
(one row per `log_interval` iterations) and makes the resulting WandB
panels visually comparable. The trade-off: dreamer-srl currently logs
~episode-per-row (very high resolution); the switch reduces resolution to
one row per N×B env-steps (N = log_interval iterations × B envs). For the
project's typical 200k-step runs at N=1 (default), this is **one row per
num_envs env-steps** — still adequate for the standard dashboards.

### 7e. Defer (Behavior-measure toolkit v1)

The 16+ fixed + per-tag `BM_*` keys (§5) are the **largest scope-creep risk**
in this migration. Two reasons:

1. The toolkit is wired into the rollout collector (`_bm_step_update` is
   called per-step), not just at episode boundary — non-trivial to graft
   onto dreamer-srl's sheeprl-style step loop.
2. The toolkit consumes per-step extras (`agent_in_bush`, `ate_food`,
   `dist_per_predator`, `dist_per_neutral`) that dreamer-srl's env-step
   call site currently does not extract from `infos`.

**Recommendation:** **Defer to a follow-up plan.** Get dreamer-srl into the
master project with the §7a–§7d migration first; surface BM toolkit support
as a separate scoped plan after the user confirms the migration is healthy.
See §9 Q1.

### 7f. Skip (rPPO-only)

The entire `loss/*` namespace (`loss/total`, `loss/policy`, `loss/value`,
`loss/entropy`, `loss/grad_norm`) is PPO-specific and does not apply to
dreamer-srl. Skip.

### 7g. Eval-level (`Eval/MeanReward` / `Eval/MeanLength`)

dreamer-srl does not have an eval loop at all. JAX-Dreamer's eval lives at
train.py L2440–L2495 and calls `evaluate_jax_checkpoint`. Porting requires
plumbing through the dreamer-srl `Player` to support greedy-action mode +
running eval against a fresh `ParallelEnv` every `eval_frequency` iterations.
This is a **medium-scope feature add**, ~80 lines including config keys.
See §9 Q4 (bundled with §7b decision).

### 7h. Continual-learning stage tags

`stage/*` (5 keys in §3i) come from the schedule loader and `_stage_tag()`.
**Defer** unless the user plans to run dreamer-srl under continual
learning — currently dreamer-srl is single-config-only.

---

## 8. Concrete migration plan checklist

The bullets below are the developer-facing line-item edits. Pulled out into
a separate `MIGRATION_PLAN.md` document once the user dispositions §9.

**WandB-init changes** (`src/algorithms/dreamer_srl/dreamer_srl_main.py`):

- L176: change argparse default
  `--wandb-project default="grid_world_pain_dreamer_srl_smoke"` →
  `default="grid_world_pain"` (matches `configs/logger/wandb.yaml:L5`).
- L310 area (`wandb.init` call): add `entity=` from
  `configs/logger/wandb.yaml:L4` (`"sungwoolee"`) — currently the dreamer-srl
  driver omits entity, defaulting to user's WandB default. Make it explicit
  for parity.
- After `wandb.init` (insert ~L329): add JAX-Dreamer's `define_metric` block
  from `train.py:L625–L634` verbatim so the `Episode/*` keys plot against
  `Episode/Number` on the dashboard instead of WandB's auto-step.

**Episode-aggregation rewrite** (decision-gated on §9 Q3):

- Hoist the rPPO/Dreamer iteration-level aggregation block (`train.py:L1394–L1437`)
  into a reusable helper in `src/utils/wandb_utils.py`, e.g.
  `compute_episode_log_dict(iteration_episodes, neutral_tags, predator_tags,
  bm_enabled, bm_helpers) -> dict`. Both `train.py` (Dreamer branch + rPPO
  branch) and `dreamer_srl_main.py` then call this helper.
- In `dreamer_srl_main.py`:
  - Add `iteration_episodes: list[dict] = []` at L364 area.
  - Replace L455 raw-log with `iteration_episodes.append({'r': ep_rew, 'l': ep_len, ...})`.
  - Add iteration-level log block at L588 area (same cadence as `Loss/*`):
    if `iteration_episodes`, call the helper, merge into `log_dict`, clear
    `iteration_episodes`.

**Key renames** (mechanical, §7a):

- `dreamer_srl_main.py:L455`: drop raw `Game/ep_len_avg` + `Rewards/rew_avg`
  log (replaced by the iter-aggregated dict above).
- `dreamer_srl_main.py:L592–L598` (log_dict): rename keys per §7a table.
- `dreamer_srl_main.py:L599`: `Params/replay_ratio` → `Params/effective_replay_ratio`.

**New WorldModel/* + Behavior/* keys** (decision-gated on §9 Q4):

- Edit `src/algorithms/dreamer_srl/train.py`'s `make_train_step` to expose
  the additional keys listed in §7b (currently it returns ~10 keys; new
  total ~25 keys).
- In `dreamer_srl_main.py:L591–L602` (log_dict construction): switch from
  flat `Loss/*` mapping to JAX-Dreamer-style prefix sort (mirror
  `train.py:L1797–L1810`): if key starts with `loss_actor` / `loss_critic`
  / `mean_` / `entropy` → `Behavior/*`; if starts with `loss_model` /
  `loss_recon` / `loss_kl` / `loss_rew` / `loss_cont` / `loss_dyn` /
  `loss_rep` / `model_` / `imagined_` → `WorldModel/*`; else passthrough.

**rPPO change** (Time/sps_env addition — `train.py`):

- L1481 area (rPPO step-aggregate `wandb_logs.update`): add
  `"Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9)`.
- **Also recommended (senior-developer suggestion, flag in §9 Q5):** same
  edit at L1786 area (Dreamer step-aggregate) for symmetry.

**Eval-level port** (decision-gated on §9 Q4):

- Out of scope for the migration plan unless the user takes Q4 = "full port".
  If so, separate plan needed.

**Behavior-measure toolkit port** (decision-gated on §9 Q1):

- Out of scope for migration plan; separate follow-up plan if Q1 = "yes".

---

## 9. Open questions for the user

Five decisions are needed before `developer` can execute. Each lists a
recommendation + a one-line cost/benefit so the user can answer in a single
`AskUserQuestion` round.

**Q1 — Behavior-measure toolkit v1 port.** dreamer-srl currently logs **none**
of the 16+ fixed + per-tag `Episode/InterruptedFeedingRate_*` /
`BushDiveRate_*` / `EatUnderThreatRatio_*` keys (§5). These power the
downstream hypervigilance + sameprop experiments.

- **Recommendation: Defer.** Get the migration in first; do BM toolkit as a
  scoped follow-up plan.
- Full port: ~200 lines, requires the per-step env-info extraction graft.
- Minimal subset (M5 only — `EatUnderThreatRatio_*`): ~80 lines.
- Defer entirely: ~0 lines now, decision later.

**Q2 — Per-tag fan-out.** Even the basic `Episode/MeanDistRabbit_<tag>` /
`Episode/MeanDistPredator_<tag>` per-tag families require extracting
`dist_per_neutral` / `dist_per_predator` per-instance arrays from the env's
`infos` dict at every step in dreamer-srl's collector. Currently dreamer-srl
extracts only `termination_reason` from `infos`.

- **Recommendation: Skip for v1 of the migration; port if/when the user
  needs per-tag dashboards for dreamer-srl experiments.**
- Full per-tag fan-out: ~30 lines added to dreamer-srl env-step block.
- Top-level only (`Episode/MeanDistRabbit` / `..Predator` — no per-tag):
  ~10 lines.
- Skip entirely: ~0 lines.

**Q3 — Episode aggregation: per-iteration mean (Dreamer) vs raw-per-episode
(sheeprl).** Currently dreamer-srl logs one WandB row per done env at the
env-step boundary; JAX-Dreamer logs one row per `log_interval` iterations
with iteration-level mean across all episodes finished in the window.

- **Recommendation: Switch to per-iteration mean.** Matches the project's
  existing dashboards; lower row volume on WandB (cheaper); makes runs
  visually comparable to JAX-Dreamer runs in the same project.
- Trade-off: drops episode-level resolution. The user can always re-derive
  raw-per-episode from `Episode/Number` if needed.

**Q4 — `WorldModel/*` + `Behavior/*` family + eval-level port.** Currently
dreamer-srl exposes ~10 loss keys; matching JAX-Dreamer requires ~25 keys
plus per-eval `Eval/MeanReward` + `Eval/MeanLength`.

- **Recommendation: Yes — port the additional keys** (the trainer-side edit
  is mechanical, ~40 LOC) **but defer the eval loop** (separate ~80 LOC
  feature).
- Full port (extra keys + eval loop): ~120 LOC across 3 files.
- Extra keys only (no eval): ~40 LOC.
- Skip entirely (current ~10 keys frozen): ~0 LOC.

**Q5 — Should `Time/sps_env` also go into JAX-Dreamer (not just rPPO)?**
The user's explicit ask is rPPO only, but symmetric logging across all
three drivers would be cleaner.

- **Recommendation: Yes, add to JAX-Dreamer's Dreamer branch too** at
  train.py L1786 area (one-line edit, same as the rPPO L1481 edit).
- Trade-off: tiny additional scope, but introduces a per-iteration
  wall-clock read to the Dreamer hot path (negligible).

---

## 10. Manifest

**Files cited.**

- [`train.py`](../../../../train.py) — original Dreamer + rPPO driver.
- [`src/models/dreamer_v3_trainer.py`](../../../../src/models/dreamer_v3_trainer.py) — Dreamer trainer's metrics dict.
- [`src/algorithms/dreamer_srl/dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py) — dreamer-srl driver.
- [`src/behavior/accumulators.py`](../../../../src/behavior/accumulators.py) — Behavior-measure toolkit v1.
- [`configs/logger/wandb.yaml`](../../../../configs/logger/wandb.yaml) — master WandB project config.

**Tallies.**

- Unique WandB keys across the three drivers: **~120** (24 fixed Episode/*,
  ~30 BM toolkit, ~16 trainer-side WorldModel/* + Behavior/*, 14 Modulator/*
  (combined rPPO+Dreamer), 5 PPO loss/*, 4 Params/*, 1 Diagnostic/*, 1 Time/*,
  7 stage/*, 2 Eval/*, plus per-tag fan-out families counted at family-level).
- Dreamer keys not yet in dreamer-srl: **all 24 Episode/* fixed keys**, **all
  ~30 BM toolkit keys**, **~16 of ~25 WorldModel/* + Behavior/* keys** (the
  9 that ARE present are aliased to sheeprl names per §7a), **0 of 2 Eval/*
  keys**, **0 of 7 stage/* keys**.

**Companion plan to be authored after §9 dispositions:**
[`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) (file does not exist yet; created
when the user has answered Q1–Q5).
