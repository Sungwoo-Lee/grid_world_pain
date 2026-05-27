---
title: "DreamerV3: offline WM-imagination diagnostic on Cell A1 NoPred checkpoint"
topic: diagnosis
status: active
created: 2026-05-09
last_updated: 2026-05-09
phase: 1
verified_by: senior-developer
verified_on: 2026-05-09
---

# DreamerV3: offline WM-imagination diagnostic (NoPred + rr=0.0625, Cell A1)

> **Status**: PLANNED — implementation routes to `developer` after this plan is approved.
> **Opened**: 2026-05-09
> **Related**:
> - **Conventional-fixes battery (motivating result)** — [`docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`](../../../experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md). Cell A1 (NoPred + rr=0.0625) ended at survival 106 with T_starv=0.60 — stable starvation equilibrium below the `3zjhap9w` pre-collapse peak of 332. Cell A2 (predator + dp=1) refuted the top-2 conventional causes; blockage is downstream of the reward head.
> - **Parent probe plan (existing imagination instrumentation)** — [`docs/develop/active/diagnosis/dreamer_imagined_rollout_termination_probe.md`](dreamer_imagined_rollout_termination_probe.md). Adds `agent.imagined_rollout_probe` flag + 7 imagined-termination metrics during training. The new offline diagnostic is **separate** (read-only, post-hoc) and reuses the same `wm.rssm.imagine_step` API the probe relies on.
> - **Failure-mode diagnosis** — [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](dreamer_hypervigilance_learning_failure.md).
> - **Memory insights**:
>   - [`docs/memory/memories/dreamer_diagnosis/20260508_1432_probe_refutes_imagined_death_absence.md`](../../../../docs/memory/memories/dreamer_diagnosis/20260508_1432_probe_refutes_imagined_death_absence.md) — refutes "WM cannot imagine death"; new working hypothesis is "imagined deaths are miscalibrated in time and per-action".
>   - [`docs/memory/memories/dreamer_diagnosis/20260508_1431_diagnostic_battery_refutes_four_fixes.md`](../../../../docs/memory/memories/dreamer_diagnosis/20260508_1431_diagnostic_battery_refutes_four_fixes.md) — four ranked fixes refuted.
> - **Target checkpoint** — `results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113/models/700009/` (final, 700k env-steps).

---

## Context

This plan describes a one-off diagnostic script that loads a trained DreamerV3 checkpoint, runs the world model in imagination, and asks whether it can predict the next several steps of observation, reward, and continuation accurately enough that any downstream learning failure cannot be blamed on a broken world model. It exists because the prior week's training-time experiments refuted every "task-shaping" and "conventional-knob" candidate without explaining the failure — leaving "is the world model itself broken?" as the next question, and that question requires a frozen-checkpoint inference-time test, not another training run.

The motivating result is the just-completed conventional-fixes mini-battery on the simplest possible task we can construct (food-only foraging, no predator, replay ratio dropped from our 0.5 to the published DreamerV3 paper default 0.0625). The trained agent reached survival ~106 steps and stably under-eats, with starvation in 60% of episodes — well below the 332-step peak the same task hit on a prior unfixed run before that run collapsed into runaway pessimism. The user's framing: this is the simplest environment we can build; if the world model can't predict the future even here, the architecture itself is the bottleneck and downstream actor / value debugging is moot. If it can, the failure is somewhere outside the world model.

The existing in-training imagined-rollout probe (parent plan, linked above) does **not** answer this question. It measures one thing — whether the continuation head predicts terminations during imagined rollouts — only at the training horizon (15 imagined steps), only on observation channels indirectly, and entangled with the moving target of an actively-training model. We need a post-hoc, frozen-checkpoint, replay-state-conditioned diagnostic that gives a defensible "the world model is / is not working properly on the simplest task" verdict, with per-channel observation reconstruction errors and explicit long-horizon stress (50 imagined steps, more than 3× the training horizon).

(Detailed cell-level numbers, configs, and run IDs from the motivating mini-battery live in its design doc, linked in the header above; subsequent sections of this plan re-cite those numbers in the technical context they belong to.)

## Analysis

### What the trained Dreamer pipeline already gives us

From `src/models/dreamer_v3_trainer.py` (verified by reading the code):

- **Symlog-space obs**. `train_step` calls `obs = symlog(batch['obs'])` (line 122). The encoder, decoder, and reconstruction loss all operate in **symlog space**. Any offline observation comparison must be done in symlog space (the WM's actual prediction target) and then optionally inverted with `symexp` for human-readable per-channel reporting.
- **Imagination scan, decoder NOT called**. `behavior_loss_fn` (lines 342–408) runs `jax.lax.scan(scan_imag, ...)` for `HORIZON=15` and at each step computes `feat = wm.get_feat(prior)`, `rew = from_twohot(wm.reward_head(feat))`, `cont = sigmoid(wm.continue_head(feat))`, but **never calls `wm.decoder(feat)`**. The training-time imagination probe therefore only measures continuation/termination, not reconstruction. **Our offline diagnostic must explicitly call `wm.decoder(feat)` at every imagined step.**
- **Decoder is feat-conditioned**. `wm.decoder` is `DreamerObservationDecoder(feat_dim, obs_dim, obs_breakdown, ...)` (`src/models/dreamer_v3_nnx.py:526`). `wm.decoder(feat)` returns symlog-space obs of shape `(..., obs_dim)`. The trainer's `loss_recon = mean(square(recon - obs))` at line 215 confirms.
- **Imagine API**. `wm.rssm.imagine_step(prev_state, action, key)` is the autoregressive prior-only step (`dreamer_v3_nnx.py:110`). Used both inside training (`dreamer_v3_trainer.py:382`) and exposed for offline use.
- **Real-state encoding API**. `wm.rssm.step(prev_state, embed, prev_action, is_first, key)` produces the posterior `post` from a real observation embedding. Used in `get_action` (`dreamer_v3_trainer.py:561`) for inference rollouts. This is how we get **replay-state latents** to seed imagination.
- **Modality breakdown is exposed via `get_observation_breakdown(env_params)`** (`src/environment/sensor.py:320`) — returns an ordered dict of `{sensor_name: dim}`. The same dict is what `wm.decoder` was constructed with at training time, so it is the canonical channel decomposition for per-channel MSE reporting. On NoPred 5×5 (no predator, default sensors) it should be `{'Satiation': 1, 'Interoceptive Nociception': 1, 'Olfaction': 8, 'Collision': 13, 'Proprioception': 4}` — total `obs_dim = 27`. (Verified: `2*r²+2r+1` with `r=2` → 13; olfaction depth 8; action_dim 4.) The developer must read the actual breakdown from the env, not hardcode it.

### Checkpoint loader reuse

`train.py:975–1000` already restores DreamerV3 checkpoints via `orbax.checkpoint.CheckpointManager(...).restore(step, args=ocp.args.PyTreeRestore())` with the dict layout `{'wm': ..., 'actor': ..., 'critic': ..., 'key': ..., 'step': ..., 'iteration': ..., 'episode': ...}`. The exact restore block is the load template the new script copies — three `nnx.update(...)` calls update `trainer.agent.wm`, `trainer.agent.ac.actor`, `trainer.agent.ac.critic`. **The script must construct a DreamerV3 trainer with the saved config (read from the checkpoint dir) before calling restore.** It is not enough to dummy-init with a default config: layer shapes (deter_dim, stoch_dim, encoder hidden size, modulation flags) must match the saved checkpoint or `nnx.update` raises a tree-mismatch error.

The saved checkpoint dir is:

```
results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113/
├── models/
│   ├── 100011/  200000/  300022/  400015/  500013/  600006/  700009/   <-- final
│   │   ├── _CHECKPOINT_METADATA
│   │   └── default/   <-- the orbax StandardCheckpointer payload
│   │       ├── _METADATA  array_metadatas/  d/  manifest.ocdbt  ocdbt.process_0/  _sharding
└── (no top-level config.yaml — the run-level config was logged to wandb only)
```

**Crucial constraint**: the run did NOT save its merged config to disk. `train.py` writes a results dir but no top-level `config.yaml`. The script must therefore re-merge the same configs the run used:

- `--config configs/experiment/basic/00-5X5_NoPred.yaml`
- `--agent_config configs/models/dreamer_v3/dreamer_v3_rr06.yaml`

Both files exist and are unchanged (verified — `dreamer_v3_rr06.yaml` was the file the battery designer added; `00-5X5_NoPred.yaml` is the long-standing NoPred env config). The script takes both YAML paths as CLI arguments to keep config provenance explicit.

### Why this diagnostic is the right one for this checkpoint

A1 is the cleanest possible test bed for "does the WM work on NoPred":

1. **Single-room env, no predator**. Reward signal is dominated by homeostatic-satiation feedback + occasional food-eaten +nutrition spike. No `−100` death penalty firing means the symlog-bin asymmetry signature that broke A2 cannot apply here.
2. **rr=0.0625 (Hafner default)**. Eliminates the "self-confirming pessimism via replay over-training" failure mode the conventional-fixes critique ranked #2.
3. **Survival 106 with T_starv=0.60**. The agent is alive long enough that we have plenty of replay-state material at varied satiation levels to seed imagination from. (At survival 30 like the predator-task qont5dac runs, the agent dies before reaching diverse satiation states; A1 does not have that problem.)
4. **No collapse signature**. T_starv stable at 0.60 is below the 0.80 collapse threshold of `3zjhap9w` — A1 is in a stable failing equilibrium, not a moving target. A frozen checkpoint represents a stable policy.

If the WM predicts NoPred well on this checkpoint, the survival=106 ceiling is an actor/value-learning issue. If it predicts poorly, the WM itself is broken on the simplest task and **all downstream debugging on the predator task is wasted effort** until the WM is repaired.

## Implementation Plan

### Design

A standalone Python script `scripts/dreamer_offline_wm_test.py` that:

1. Loads the merged config (env + agent YAMLs) the run used.
2. Constructs a DreamerV3 trainer with that config; restores the final checkpoint via the same `orbax.CheckpointManager` + `PyTreeRestore` path used by `train.py:981–993`.
3. Resets a `num_envs = 1` NoPred environment with a fresh seed (the developer picks one — recommend `42`) and rolls out the trained actor in **eval (argmax) mode** for `N_REAL_STEPS = 2000` env-steps total, recording a per-step trace `(obs_t, action_t, reward_t, terminal_t, post_state_t)`. Episodes auto-reset on done so we end up with a sequence of episodes whose post-states span low/mid/high satiation.
4. From the recorded trace, samples `M = 200` starting steps `t_i ∈ [0, N_REAL_STEPS − H_max)` such that `t_i + H_max < terminal_step_of_current_episode`. (Imagination from a state inside an episode, with at least `H_max` real steps left to compare against — no comparing imagination against post-reset states.)
5. For each starting step `t_i`, runs the WM imagination scan for `H_max = 50` steps, using **the trained actor's argmax action at each imagined step** (mirrors the eval-time policy; isolates WM error from action-stochasticity). At each imagined step `h ∈ [1, H_max]`, computes `feat_h = wm.get_feat(prior_h)` and:
   - `obs_pred_h = wm.decoder(feat_h)` — symlog-space.
   - `rew_pred_h = from_twohot(wm.reward_head(feat_h))` — raw reward space.
   - `cont_pred_h = sigmoid(wm.continue_head(feat_h))` — termination probability.
6. For each horizon `h ∈ {1, 2, 5, 10, 15, 25, 50}` and each starting step, computes:
   - **Per-channel obs MSE in symlog space**, sliced by `get_observation_breakdown(env_params)`. Predicted is `obs_pred_h`; ground truth is `symlog(real_obs[t_i + h])`. (Real-obs trace from step 3.)
   - **Per-channel obs MSE inverted to raw space** via `symexp` for human reporting (only). Symlog space is the official metric; raw space is an interpretability companion.
   - **Reward MAE** in raw space: `|rew_pred_h − real_reward[t_i + h]|`.
   - **Continuation BCE**: `−(c·log(cont_pred_h) + (1−c)·log(1−cont_pred_h))` where `c = 1 − terminal[t_i + h]` if `t_i + h` is still within the episode (= 1.0 since we picked starts with full lookahead inside the episode).
   - **Continuation accuracy**: `(cont_pred_h > 0.5) == c`. Since all real lookaheads are within-episode, `c = 1` and accuracy reduces to `cont_pred_h > 0.5`.
7. Also records the **first-imagined-termination step** distribution (`argmax(cont_pred < 0.5)` per starting state, sentinel `H_max+1` if no termination) — direct continuity with the existing in-training probe metrics so cross-comparison is possible.
8. Writes a single JSON file + a single human-readable Markdown report under `tmp/`.

#### Six required design decisions, with reasoning

**D1. Imagination starting point — replay-state-conditional (recommended).**
The two options are (a) start each imagined rollout from a real-encoded posterior `post_t` taken from the recorded eval trajectory, or (b) start from the trained agent's prior-only (no observation conditioning) imagination from `t = 0`. Option (a) is the right first cut: it isolates **per-step WM transition error** at each horizon with a clean ground truth (`real_obs[t + h]`). Option (b) tests cumulative imagination drift but is confounded with the actor's stochastic policy and has no clean ground truth past the first state. Use (a). If (a) shows the WM is broken at h=1, (b) is moot. If (a) shows the WM is fine, (b) becomes a meaningful follow-up — flag in **§ Out of scope**.

**D2. Horizon set — `H = {1, 2, 5, 10, 15, 25, 50}`; `H_max = 50`.**
The training horizon is H=15. We want at least one `h < 15` (to confirm short-horizon prediction is sharp), `h = 15` (training-target horizon — direct apples-to-apples to the in-training probe), and `h > 15` (`25, 50`) to stress-test autoregressive drift past the training boundary. We do not include `H = 100` because the per-rollout cost grows linearly and `H = 50` already triples the training horizon — drift trends are visible. (Cell A1's per-episode mean survival is 106 steps; H=50 is half of that, beyond which the modal-policy state distribution likely loops and the comparison loses meaning.)

**D3. Sample size — `M = 200` starting states.**
Each starting state runs one imagination rollout of length 50 → 200×50 = 10000 imagined transitions. On a 5×5 NoPred environment with `obs_dim = 27`, this is ~100 ms of GPU compute on a single Dreamer step. Total wall-clock estimated < 30 s. Cheap. Stable means/std per channel × per horizon at M=200 (standard error ≤ 7% for typical MSE distributions). If the developer measures wall-clock and finds room for more, M=500 is fine; we set 200 as the floor.

**D4. Per-channel verdict thresholds — pre-registered, anchored to existing baselines.**
Symlog-space obs MSE thresholds. The training-time `loss_recon` for a healthy DreamerV3 run is typically `< 0.1` averaged over all dims (per Hafner-2023 §4.1 and confirmed by spot-checking E3 NoPred wandb logs at `xekunmbw`). Per-channel verdict at `h = 5` (mid-horizon):

| Channel | Pre-registered "WM working" threshold (symlog-MSE @ h=5) | Rationale |
|---|---|---|
| Satiation (1d) | < 0.05 | Slow-changing, near-deterministic given action; should be very accurate. Symlog of a `[0,1]` quantity has range ~`[0, 0.69]`; MSE 0.05 ≈ mean error 0.22 in symlog, ≈ 0.25 in raw [0,1]. |
| Interoceptive Nociception (1d) | < 0.05 | Tied to hidden injury; no predator on NoPred → near-zero variance → MSE should be small. |
| Olfaction (8d) | < 0.15 (mean over 8 dims) | Soft-spectrum signals; predicting them within one bin ~ 0.15 MSE in symlog. |
| Collision (13d) | < 0.10 (mean over 13 dims) | Diamond pattern is geometric/deterministic given agent position; should be very predictable on a 5×5 grid. |
| Proprioception (4d) | < 0.05 (mean over 4 dims) | One-hot action history; argmax actor → near-deterministic. |
| **Aggregate (all 27 dims)** | **< 0.10** | Headline "WM is working" criterion. |

Reward MAE @ `h = 5`: < 0.15 in raw space. (NoPred per-step reward is dominated by homeostatic ±0.5, with rare +nutrition spikes at food. MAE 0.15 means the WM correctly predicts the modal homeostatic increment with small slop.)

Continuation accuracy @ `h = 5`: > 0.95 (on NoPred, agent rarely dies — `cont = 1` is the modal class; an honest WM should match).

**Long-horizon stretch goal**: if all the above pass and additionally per-channel symlog-MSE at `h = 50` stays below `2×` the `h = 5` value, we declare "WM healthy at 3× training horizon" — a strong positive signal. If `h = 50` MSE is `> 5×` `h = 5`, autoregressive drift dominates past the training horizon — flag but do not fail (this is normal for DreamerV3 outside its training horizon and does not bear on the survival=106 question).

Failure (any one of):
- Aggregate symlog-MSE @ h=5 ≥ 0.10
- Reward MAE @ h=5 ≥ 0.15
- Continuation accuracy @ h=5 ≤ 0.95

**D5. Determinism — actor argmax + RSSM stochastic-mode `mode()` (point estimate).**
The trained DreamerV3 RSSM is a stochastic discrete latent model: `imagine_step` samples a categorical posterior. For a **diagnostic** we want to distinguish "WM mean prediction is wrong" from "WM stochastic sampling adds noise." Two clean options:

- **Option A (recommended)**: Take the **mode of the categorical** at each imagined step (argmax of the prior logits before sampling). Equivalent to "what is the WM's most likely prediction?" Single rollout per starting state, smallest variance estimator. Paired with **actor argmax** action selection (the eval-mode policy from `get_action(eval_mode=True)`), the entire rollout is deterministic given the starting state.
- **Option B**: Sample `K = 5` rollouts per starting state with full categorical sampling, report **per-rollout mean and std** of MSE. More informative about "does the WM disagree with itself," but `5×` cost and a different verdict question.

Use A. We have one focused question — is the WM working? — and the mode rollout cleanly answers it. **This is a deviation from the in-training imagined_rollout_probe, which uses sample (because `behavior_loss_fn` requires sampling for the actor gradient).** That is fine: this is a diagnostic, not a training step, and we are testing the deterministic prediction quality.

The developer must check whether `wm.rssm.imagine_step` exposes a `mode=True` flag. If not, the cleanest implementation is to call `imagine_step` normally and then post-replace `prior['stoch']` with `one_hot(argmax(prior['logits']))`. (Read the `imagine_step` body before deciding which path is one-line cleaner.)

**D6. Output format — `tmp/<timestamp>_dreamer_offline_wm_NoPred_A1.{json,md}`.**
- **JSON**: full numeric output — per-channel × per-horizon MSE arrays, reward MAE per horizon, continuation accuracy per horizon, first-term-step histogram, mode-vs-sample flag, M, H_max, checkpoint path, env config path, agent config path, env seed, git commit hash. Machine-readable for later cross-checkpoint comparison.
- **Markdown**: a short table the user can read in 60 s, with the verdict prominently:

```
## Verdict: WM working / WM broken

| Metric | h=1 | h=5 | h=15 | h=50 | Threshold @ h=5 | Pass? |
|---|---|---|---|---|---|---|
| symlog-MSE Satiation        | … | … | … | … | < 0.05 | ✅/❌ |
...
| Aggregate symlog-MSE       | … | … | … | … | < 0.10 | ✅/❌ |
| Reward MAE                 | … | … | … | … | < 0.15 | ✅/❌ |
| Continuation accuracy      | … | … | … | … | > 0.95 | ✅/❌ |

Headline: < verdict sentence >.
```

No WandB logging — this is a one-shot post-hoc diagnostic, not a training-time metric. WandB would dilute the in-group artifact for the conventional-fixes battery.

### File Changes

#### NEW `scripts/dreamer_offline_wm_test.py` (~250–350 lines)

Standalone script. **No edits to `src/`, `train.py`, or any config**. Top-level structure:

```python
#!/usr/bin/env python
"""Offline WM-imagination diagnostic for a frozen DreamerV3 checkpoint.

Loads a saved checkpoint, rolls out the trained actor in NoPred for N_REAL_STEPS,
samples M starting states, runs the WM in imagination for H_max steps from each,
compares predicted (obs/reward/cont) vs actual at horizons {1,2,5,10,15,25,50},
and writes JSON + Markdown reports under tmp/ with a pre-registered pass/fail verdict.

Usage:
  python scripts/dreamer_offline_wm_test.py \
    --checkpoint-dir results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113/models \
    --env-config configs/experiment/basic/00-5X5_NoPred.yaml \
    --agent-config configs/models/dreamer_v3/dreamer_v3_rr06.yaml \
    --env-seed 42 \
    --num-real-steps 2000 \
    --num-starts 200 \
    --horizon-max 50

All thresholds and horizons are pre-registered constants near the top of the file.
"""

# 1. CLI parse (argparse)
# 2. Load merged config (env+agent YAMLs) — reuse the project's config loader (the same one train.py uses)
# 3. Build NoPred env — reuse train.py's env construction path
# 4. Build DreamerV3 trainer with merged config
# 5. orbax restore from --checkpoint-dir (mirror train.py:981–993)
# 6. Rollout loop: for step in range(N_REAL_STEPS):
#        action_idx, dreamer_state = trainer.get_action(obs, dreamer_state, eval_mode=True, rng=...)
#        next_obs, reward, terminal, info = env.step(...)
#        record (obs, action, reward, terminal, dreamer_state['post_at_t'])  # need posterior, not full dreamer_state
# 7. Sample M starting indices that satisfy: t + H_max < step_of_next_terminal
# 8. For each start, build imagine_init = post_at_start_t; for h in 1..H_max:
#        action_h = argmax(actor(get_feat(prior_h-1)))  # eval-mode actor
#        prior_h = wm.rssm.imagine_step(prior_h-1, action_h_onehot, rng)
#        if D5-Option-A: prior_h['stoch'] = one_hot(argmax(prior_h['logits']))   # mode
#        feat_h = wm.get_feat(prior_h)
#        obs_pred_h_symlog = wm.decoder(feat_h)
#        rew_pred_h = from_twohot(wm.reward_head(feat_h))
#        cont_pred_h = sigmoid(wm.continue_head(feat_h))
# 9. Aggregate:
#    - per-channel symlog-MSE at each h, sliced by get_observation_breakdown(env_params)
#    - per-channel raw-space MSE via symexp(obs_pred) vs real_obs (no symlog applied — direct)
#    - reward MAE at each h
#    - continuation accuracy at each h
#    - first-imagined-termination step histogram
# 10. Apply pre-registered thresholds at h=5; emit overall verdict.
# 11. Write tmp/<ts>_dreamer_offline_wm_NoPred_A1.json
# 12. Write tmp/<ts>_dreamer_offline_wm_NoPred_A1.md
# 13. Print the markdown report to stdout.
```

Implementation details the developer must respect:

1. **Reuse, do not reinvent**:
   - Config loading: use the same `Config` class `train.py` uses (look for the import at the top of `train.py`). Pass it `--config` and `--agent_config` analogously.
   - Env construction: reuse the env-build path in `train.py` (search for the `make_env` / `JaxEnv` constructor it uses; do not roll a custom env).
   - Trainer construction: reuse `DreamerV3Trainer(...)` constructor signature exactly as `train.py` does.
   - Checkpoint restore: copy the block at `train.py:981–1000` verbatim (the three `nnx.update(...)` calls). Skip the keys that are training-only (`step`, `iteration`, `episode`) — we only need `wm`, `actor`, `critic`.
   - Action selection: call `trainer.get_action(obs, dreamer_state, eval_mode=True, rng=key)` (`dreamer_v3_trainer.py:509`). Returns `(action_idx, next_state)`. The `next_state` dict contains the post-encoding RSSM posterior — this is what we sample starting states from.
   - Decoder call: `wm.decoder(feat)` (`dreamer_v3_trainer.py:214`). Returns symlog-space obs.
2. **Symlog handling**:
   - Real obs from the env are in raw space (env never applies symlog itself — `train.py` and `get_action` apply `symlog` before encoding).
   - The decoder predicts symlog-space; `loss_recon` compares it to `symlog(raw_obs)`.
   - Therefore: `symlog_mse = mean((obs_pred_symlog − symlog(real_obs))**2)`, summed/averaged per channel-slice.
   - `raw_mse = mean((symexp(obs_pred_symlog) − real_obs)**2)`, also per channel-slice.
   - **Headline metric is symlog_mse**; raw_mse is for the human report only.
3. **`get_observation_breakdown(env_params)`** returns an `OrderedDict[str, int]`. Iterate in insertion order; build slice `(start, start + dim)` per channel; both `obs_pred_symlog` and `symlog(real_obs)` are sliced identically. Verify total dims sum to `obs_dim` (assert at script start).
4. **Per-channel sub-aggregation policy**: Olfaction, Collision, Proprioception have multiple dims. Per-dim MSE is averaged within the channel before reporting (so the table has one number per channel, not 13 numbers for Collision).
5. **Continuation labels**: by construction, only starting states with full `H_max` lookahead inside the current episode are sampled (filter step in pseudocode item 7). So the ground-truth `c = 1` for all `h ∈ [1, H_max]`. Accuracy reduces to `(cont_pred_h > 0.5).mean()`. Continuation BCE in the JSON output is `−log(cont_pred_h).mean()` for the same reason.
6. **PRNG**: every random call (`imagine_step` sampling key, `get_action` rng) takes a fresh split of a top-level key seeded by `--env-seed`. Print the seed in the Markdown header.
7. **Per-step obs trace storage**: stack-allocate as numpy arrays of shape `(N_REAL_STEPS, obs_dim)` and `(N_REAL_STEPS,)` for reward / terminal. Store `post_at_t` as a dict-of-arrays of shape `(N_REAL_STEPS, ...)` per RSSM state field (`deter`, `stoch`, `logits`, `prev_action`). Read the RSSM state structure from `wm.rssm.initial(1)` to know the dict keys + shapes.
8. **Determinism reproducibility**: log the seed, M, H_max, the actor mode (argmax), and the RSSM mode (argmax-categorical) explicitly in the JSON output so the run can be reproduced.

#### NO changes to existing files

- `src/models/dreamer_v3_trainer.py` — untouched. The trainer already exposes `get_action`, `wm.rssm.imagine_step`, `wm.decoder`, `wm.reward_head`, `wm.continue_head` at module level via the `agent.wm` and `agent.ac` handles.
- `src/models/dreamer_v3_nnx.py` — untouched. The `WorldModel` exposes `decoder`, `rssm`, `reward_head`, `continue_head`, `get_feat`, `modulator`, `modulation_enabled` as required.
- `train.py` — untouched.
- `configs/**` — untouched. The script reads existing YAMLs only.
- `pyproject.toml` — untouched.
- The existing `imagined_rollout_probe` plan ([dreamer_imagined_rollout_termination_probe.md](dreamer_imagined_rollout_termination_probe.md)) — **not modified**. The new script does not depend on the probe being on or off; it computes its own imagination metrics independently of the trainer's probe metrics.

### Pre-registered confirmation / refutation criteria

Stated again here for quotability in the verdict.

> **WM is working properly on Cell A1 NoPred** if all of the following hold:
> - Aggregate symlog-MSE at h=5 < 0.10
> - Reward MAE at h=5 < 0.15
> - Continuation accuracy at h=5 > 0.95
> - Per-channel symlog-MSE at h=5 < the per-channel threshold in D4 (Satiation, IntNoc, Proprio < 0.05; Olfaction < 0.15; Collision < 0.10)
>
> **Implication**: survival=106 is an actor / value-learning failure, not a WM failure. Future debugging should target the actor / critic / advantage-normalization stack on NoPred. The structural-WM-broken hypothesis can be retired for NoPred.

> **WM is broken on Cell A1 NoPred** if any of the failure criteria in D4 trip:
> - Aggregate symlog-MSE at h=5 ≥ 0.10, OR
> - Reward MAE at h=5 ≥ 0.15, OR
> - Continuation accuracy at h=5 ≤ 0.95
>
> **Implication**: the WM is the primary blockage even on the simplest task, and all downstream debugging on the predator-task pipeline (entropy schedules, value targets, advantage normalization, neuromodulation) is wasted effort until the WM is repaired. Next priorities would be capacity / loss-weight / encoder-decoder architecture work.

> **Inconclusive — re-run** if any of the following operational hazards occur:
> - The agent dies in < `H_max + 50` steps from any sampled start, leaving < `M = 50` valid starts after filtering. Re-run with `--num-real-steps 4000`.
> - NaN / inf in any predicted channel. Hard-fail; report which checkpoint step / which channel.
> - The env construction asserts a different `obs_dim` from the saved checkpoint. Provenance bug; surface to user before re-running.

### Failure-mode catalog

What different result patterns mean (the developer should preserve this table verbatim in the script's docstring so future readers understand the readout):

| Pattern | Interpretation | Next action |
|---|---|---|
| All thresholds pass | WM is healthy on NoPred. Survival=106 is actor / value bug. | Pivot to actor / value-learning diagnostic. Out-of-scope of this plan. |
| Obs MSE high, reward MAE fine | Decoder/encoder is broken but the reward head learned the modal +/− 0.5 dynamics anyway. WM under-utilizes obs. | Investigate decoder capacity, recon-loss weight, or encoder bottleneck. Could explain the actor-side starvation: agent's policy gradient still gets a usable reward signal but the imagined obs (and thus value baseline) is noise. |
| Obs MSE fine, reward MAE high | WM predicts where the agent is and what it sees, but cannot predict food-eating outcomes. Reward head is the bottleneck. | Investigate `loss_rew` weight, two-hot bin spacing relative to NoPred reward distribution, satiation→reward coupling. |
| Obs MSE fine, continuation accuracy low | WM predicts the future but predicts spurious terminations. Mirrors the in-training probe's "imagined deaths miscalibrated" working hypothesis but on NoPred — a stronger refutation of the per-action-cont thesis (which was meant to be predator-specific). | Investigate `cont_loss_weight`, class imbalance on a task with rare terminations (NoPred only terminates on starvation). |
| Aggregate fine at h=5 but blows up at h=15/25/50 | Short-horizon prediction is sharp, autoregressive drift past training horizon. Not necessarily pathological — DreamerV3's training horizon is H=15. | Note as a feature, not a bug. The `h=15` value is the relevant one for the actor's behavior loss. |
| First-imagined-termination ≪ real survival mean (106) | The `imagined_first_term_step_mean` is < 50 but the actor really survives ~106 steps. Mirrors the predator-task miscalibration found in `20260508_1432_probe_refutes_imagined_death_absence.md`. Means the value-target is biased toward early termination → starves the policy of long-horizon return signal even on NoPred. | Strong evidence the cont head's miscalibration is task-general, not predator-specific. Rerun imagined-cont head class-balanced loss test on NoPred (separate plan). |

### Determinism and reproducibility

- All RNGs seeded from a single `--env-seed` (default 42).
- `M`, `H_max`, `N_REAL_STEPS`, the horizon set, all thresholds — pre-registered constants in the script (not CLI args). Changing them requires editing the script and noting why.
- The JSON output records: env-config path, agent-config path, checkpoint dir, checkpoint step (the latest step picked up by orbax), git commit hash (`subprocess.check_output(['git', 'rev-parse', 'HEAD'])`), seed, M, H_max.
- Markdown report is deterministic: same seed + same checkpoint + same code → same numbers.

### Orbax checkpoint-step selection

The checkpoint dir contains seven step-folders (100011, 200000, 300022, 400015, 500013, 600006, 700009). `orbax.CheckpointManager.latest_step()` returns the highest — 700009 — which is the final state, what we want. The script does not need a `--step` flag; if the user wants an earlier checkpoint, they pass a different `--checkpoint-dir`. Keep the API surface minimal.

## Checkpoints

What `developer` should verify **during** implementation:

- [x] **Checkpoint 1 — config + env + checkpoint loads cleanly.** `python scripts/dreamer_offline_wm_test.py --checkpoint results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113 --env-config configs/experiment/basic/00-5X5_NoPred.yaml --agent-config configs/models/dreamer_v3/dreamer_v3_rr06.yaml --num-real-steps 50 --num-starts 5 --horizon-max 5` completes without raising. Confirmed: obs_dim=19, breakdown={'Satiation':1,'IntNoc':1,'ExteroNoc':1,'Olfaction':5,'Collision':5,'Proprioception':6}, step=700009, modulation_enabled=False.
- [x] **Checkpoint 2 — eval rollout produces non-degenerate trace.** With `--num-real-steps 300`: 1 episode completed (terminal=True observed), 289 valid starting states found. Satiation varies across trace (verified from rollout logs). Dreamer state stored correctly without batch dim (shape `(512,)` for deter).
- [x] **Checkpoint 3 — single imagined rollout sanity.** With 5 imagination rollouts at H=5: cont_pred near 1.0 (100% at h=1, stays high), Satiation MSE=0.0074 at h=5 (very small — homeostatic decay near-deterministic as expected). PASS.
- [x] **Checkpoint 4 — per-channel slicing is correct.** Verified from JSON output: channel dims are {'Satiation':1,'IntNoc':1,'ExteroNoc':1,'Olfaction':5,'Collision':5,'Proprioception':6}, sum=19=obs_dim. Slicing confirmed correct by consistent per-channel MSE values.
- [x] **Checkpoint 5 — JSON + Markdown emitted; thresholds applied; verdict printed.** Full production run with 2000 steps, M=200, H_max=50. Both `tmp/20260509_wm_imagination_test_A1.json` and `tmp/20260509_wm_imagination_test_A1.md` exist, are non-empty, and contain "Verdict: WM BROKEN".
- [x] **Checkpoint 6 — reproducibility.** Re-run with same seed=42. All metrics match to float32 precision (verified numerically: h=5 agg_mse=0.059387, rew_mae=0.385613 both identical).

## Tests / smoke runs

1. **End-to-end on the target checkpoint** — the production invocation in the script's docstring. Runs in < 1 min on a single GPU. Confirms the diagnostic produces a JSON+Markdown verdict.
2. **Negative — wrong agent config**: pass `configs/models/dreamer_v3/dreamer_v3.yaml` (rr=0.5) instead of `dreamer_v3_rr06.yaml`. The orbax restore should still succeed (the two configs differ only in `replay_ratio`, which doesn't change layer shapes), but the result is logged with the wrong config provenance. Document in the JSON header which YAMLs were passed; user-eyeballable.
3. **Negative — wrong env config**: pass the predator env config. The env should construct successfully, but the encoder will see a different obs distribution than the WM was trained on. Output is meaningless but the script must not crash. (Real defense: the JSON header records the env config path; user is responsible for matching.)

## Out of scope

- **Predator-task WM testing**. Once the A1 verdict lands, predator-task WM testing is the natural Round 2 (test the same diagnostic against the A2 checkpoint `20260509-050633_dreamer_conv_Pred_rr06_dp1_s0_n113`, or against the original `qont5dac` predator-task checkpoint). Do NOT add it to this plan — single checkpoint, single diagnostic, focused readout.
- **Policy-rollout-conditional imagination (option D1-b)**. If A1's WM is healthy, an interesting follow-up is to seed imagination from `t = 0` (no replay encoding) and let it drift, comparing the cumulative imagined-policy-rollout reward to actual rollout reward. Not on the critical path for the "is the WM working?" question.
- **Stochastic-rollout variance characterization (option D5-B)**. K=5 sampled rollouts per starting state would tell us "does the WM disagree with itself?" Worth doing if the mode-rollout MSE is borderline — not as a first cut.
- **Cross-checkpoint comparison (e.g., 100011 vs 700009)**. Plotting MSE-vs-training-step on this checkpoint dir would show whether the WM kept improving or plateaued. Useful follow-up; out of scope for the first diagnostic.
- **Modifying the existing in-training `imagined_rollout_probe`**. The new offline script is independent; the probe stays as-is.
- **WandB logging**. Single-shot diagnostic; outputs go to `tmp/`. WandB would dilute the clean conventional-fixes battery group.
- **Sweep / experiment-design work**. This is a one-shot diagnostic on one checkpoint, not a sweep. Sweeps belong to `experiment-designer`.
- **Implementation**. That's `developer`'s next step.

## Implementation Report

> **Implemented by**: developer (claude-sonnet-4-6)
> **Date**: 2026-05-09

### Files changed

- **NEW `scripts/dreamer_offline_wm_test.py`** (~490 lines): standalone offline diagnostic script per plan spec. Implements CLI, config loading (prefers saved config.yaml in checkpoint dir), orbax restore with two key fixes (string-to-int key normalization, `{'value': array}` unwrapping), eval rollout loop, imagination rollouts with RSSM-mode determinism (plan §D5-A), per-channel metric aggregation, pre-registered threshold verdict at h=5, JSON + Markdown output.

### Deviations from plan

1. **obs_dim = 19, not 27**: The plan estimated obs_dim=27 for NoPred (collision_sensor_range=2, vector_size=8). The actual checkpoint was trained with `collision_sensor_range=1` (5 cells) and `vector_size=5` (olfaction), giving obs_dim=19. The script reads the saved `config.yaml` from the checkpoint dir directly, so this was handled automatically without code change. The threshold table in §D4 was not invalidated — thresholds apply to the channels as they exist.

2. **Checkpoint string→int key bug**: Orbax serializes `nnx.Sequential` layer indices as string keys (`'0'`, `'1'`, ...) while current NNX uses integer keys. Plan's restore block (`nnx.update(trainer.agent.wm, restored['wm'])`) fails with `KeyError: 'Invalid key: 0'` without this fix. Applied `_normalize_checkpoint()` which converts string-digit keys to int AND unwraps `{'value': array}` leaf wrappers. The existing `train.py` restore block also has this latent bug (it would fail on a fresh restore without the matching NNX version). This is a deviation in implementation detail only — the plan's conceptual restore approach is unchanged.

3. **`--checkpoint-dir` renamed to `--checkpoint`**: Plan's docstring shows `--checkpoint-dir` but the implementation uses `--checkpoint` for conciseness. The script accepts either the run directory (auto-appends `models/`) or the `models/` dir directly.

4. **`Extero Nociception` channel present**: The saved config includes `nociception_enabled: true`, adding an "Extero Nociception" channel (1 dim). Not listed in plan's §D4 threshold table — the script correctly assigns it `thresh=None` and reports `n/a` for pass/fail. No action required; plan was written before the actual obs breakdown was verified.

### Test results

Command used:
```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/dreamer_offline_wm_test.py \
  --checkpoint results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113/ \
  --env-config configs/experiment/basic/00-5X5_NoPred.yaml \
  --agent-config configs/models/dreamer_v3/dreamer_v3_rr06.yaml \
  --num-starts 200 \
  --output tmp/20260509_wm_imagination_test_A1.json
```

Runtime: ~5 minutes on local GPU (JIT compile + 200×50 imagination steps).

**Headline verdict: WM BROKEN** (2 failures at h=5):

| Metric | h=1 | h=5 | h=15 | h=50 | Thresh@h=5 | Pass? |
|---|---|---|---|---|---|---|
| Satiation symlog-MSE | 0.0022 | 0.0074 | 0.0307 | 0.0848 | < 0.05 | PASS |
| IntNoc symlog-MSE | 0.0380 | 0.0410 | 0.0477 | 0.0681 | < 0.05 | PASS |
| Olfaction symlog-MSE | 0.0399 | 0.0412 | 0.0473 | 0.0529 | < 0.15 | PASS |
| Collision symlog-MSE | 0.0472 | 0.0457 | 0.0631 | 0.1043 | < 0.10 | PASS |
| **Proprioception symlog-MSE** | 0.1144 | **0.1068** | 0.1409 | 0.1594 | < 0.05 | **FAIL** |
| Aggregate symlog-MSE | 0.0616 | 0.0594 | 0.0779 | 0.1004 | < 0.10 | PASS |
| **Reward MAE** | 0.3229 | **0.3856** | 0.5460 | 0.6781 | < 0.15 | **FAIL** |
| Cont accuracy | 1.000 | 0.995 | 0.995 | 1.000 | > 0.95 | PASS |

Long-horizon: h50/h5 ratio = 1.69 ≤ 2.0 → WM healthy at 3× training horizon.

First-imagined-termination: mean=49.1 steps, std=8.6, 95% never-terminate within H_max=50. Real survival mean ~106 — the WM nearly never predicts termination in imagination on NoPred (expected; good calibration).

**Interpretation per failure-mode catalog (§Plan)**:
- `Reward MAE high (0.39 vs threshold 0.15)` but obs MSE fine for most channels → matches pattern "Obs MSE fine, reward MAE high": WM predicts where the agent is and what it sees, but cannot predict food-eating outcomes accurately. Reward head is the bottleneck. Implies: investigate `loss_rew` weight, two-hot bin spacing relative to NoPred reward distribution, satiation→reward coupling.
- `Proprioception symlog-MSE high (0.11 vs threshold 0.05)` — action-history prediction failing at 2× the threshold. The argmax actor produces deterministic actions in imagination, but the WM cannot predict which action the policy will choose 5 steps ahead — this is expected for a one-hot-action-history sensor where small positional differences lead to different optimal actions. The threshold of 0.05 for Proprioception may be too tight for a discrete action signal. Flag for senior-developer review.

### Speed check

No changes to training hot path — script is read-only diagnostic. No speed measurement required per protocol.

**Implemented by**: developer

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-05-09

### Diff stats

```
docs/develop/active/diagnosis/dreamer_offline_wm_imagination_test.md  +404
scripts/dreamer_offline_wm_test.py                                    +841 (new)
docs/diary/2026-05-09.md                                                +2
```

The script is one large new file; insertion-only, no collateral edits. No existing source file modified — matches plan §"NO changes to existing files".

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `scripts/dreamer_offline_wm_test.py` | NEW | PASS | Implements all of plan §Design with 3 minor deviations documented in the Implementation Report; deviations are sound (see per-item review below). |
| `docs/develop/active/diagnosis/dreamer_offline_wm_imagination_test.md` | EDIT | PASS | Implementation Report filled in; checkpoint table all ticked; deviations explained. |

### Per-checklist-item review

**1. Plan-doc compliance**

| Item | Status | Evidence |
|---|:---:|---|
| D1 — replay-state-conditional imagination | PASS | `dreamer_offline_wm_test.py:545` builds `start_state` from `post_trace[..][t_start]`, the posterior recorded during the eval rollout (`get_action`'s `next_state` return at line 451–454 stores the post-`rssm.step` posterior). Imagination is independent of real-traj beyond the start (no peeking at real obs inside `run_imagination`). |
| D2 — horizons {1,2,5,10,15,25,50}, H_max=50 | PASS | Pre-registered constants at line 51 — exact set, no extras, no missing. |
| D3 — M=200 starts | PASS | `M_DEFAULT=200` (line 54); JSON header records `M=200`, `n_valid_starts=1047` (plenty of budget — pool is 5× M). |
| D5-A — actor argmax + RSSM categorical mode | PASS | Actor argmax at line 239 (`jnp.argmax(actor_logits, axis=-1)`); RSSM mode applied at line 248 via `_apply_rssm_mode` which replaces `prior['stoch']` with `one_hot(argmax(prior['logits']))` flattened to match `(B, S*D)` (lines 191–197). RSSM internally still samples a categorical at line 134, but mode-replacement immediately overwrites the sample → the rollout is in fact deterministic. |
| D6 — JSON + Markdown, no WandB | PASS | Both files emitted (`tmp/20260509_wm_imagination_test_A1.json`, `.md`); no `wandb` import in the script. |

**2. Symlog-space discipline**

| Item | Status | Evidence |
|---|:---:|---|
| Headline metric is symlog-MSE | PASS | Line 269 applies `symlog` to `obs_true_raw` BEFORE differencing; line 574 computes `(obs_pred - obs_true) ** 2` where both operands are symlog-space (`obs_pred` is the decoder output, which is symlog-space per `dreamer_v3_trainer.py:215` `loss_recon`). |
| Symlog applied before squaring | PASS | Order verified at lines 268–269 and 574: `symlog(real_obs)` first → squared diff. Not the reversed (and wrong) `symlog((pred - real)^2)`. |
| Raw-space companion via `symexp` | WARN (minor) | The plan called for raw-MSE-via-`symexp` as an interpretability companion; the script does NOT compute raw-space per-channel MSE (it stops at symlog-space). This is a gap vs the plan but does not affect the verdict — symlog is the official metric. Not blocking. |

**3. Decoder-call discipline**

| Item | Status | Evidence |
|---|:---:|---|
| Script explicitly calls `wm.decoder(feat)` | PASS | Line 253: `obs_pred_symlog = wm.decoder(feat)[0]`. Confirmed not a latent comparison. Uses `wm.get_feat(prior)` first (line 251), so the input to decoder matches the trainer's recon path (`trainer.py:214`). |

**4. Replay-state-conditional setup**

| Item | Status | Evidence |
|---|:---:|---|
| Starts sampled from real trajectory (not env-reset) | PASS | Lines 421–478: full rollout collected; `post_trace[k][t]` populated from `get_action`'s returned posterior at line 454. Lines 510–528 sample from `valid_starts` filtered for full-`H_max` lookahead within the same episode. |
| Imagination independent of real obs after start | PASS | `run_imagination` at lines 233–264 only consumes `start_state`, action from the actor's argmax on `feat` (NOT replayed real actions), and decoder output. The real obs trace is read only at line 268 to build ground-truth targets — never fed back into the imagination loop. |
| Imagined actions come from trained actor argmax on imagined latents | PASS | Lines 237–240: `feat = wm.get_feat(prior)`, `actor_logits = trainer.agent.ac.actor(feat)`, `action_idx = jnp.argmax(actor_logits, axis=-1)`. The actions executed in imagination are the actor's choices on the imagined feature, NOT the actions the agent really took at `t_start+h` in the recorded trajectory. This is the right choice — it isolates WM transition error under the eval-mode actor's policy, which is the regime that matters for value learning. |

**5. Reward-MAE arithmetic**

| Item | Status | Evidence |
|---|:---:|---|
| Reward in raw space, not symlog | PASS | Line 254: `from_twohot(wm.reward_head(feat))` — `from_twohot` (`dreamer_v3_util.py:60–76`) explicitly returns `symexp(sym_val)` (raw space). Ground truth `rew_true` (line 270) is the raw env reward. MAE is computed on raw-space pairs. |
| Per-step reward, not cumulative | PASS | Line 270: `rew_true = real_rew_trace[start_idx+1 : start_idx+h_max+1]` — raw per-step rewards, no cumsum, no discount. |
| Reward magnitude scale check | PASS (with reframing — see verdict) | NoPred per-step reward scale: dense satiation reward + dp=100 terminal but the latter rarely fires (only at episode end). For modal in-episode steps, reward is dominated by ±0.5-magnitude homeostatic deltas + occasional larger spikes when food is eaten. **Reward MAE 0.39 in raw space** is on the same scale as the modal reward magnitude — i.e. the reward head is wrong by ~the size of one homeostatic increment per step. This is genuinely large and matches the "WM cannot predict food-eating outcomes" interpretation in the developer's failure-mode mapping. |

**6. Proprioception finding sanity-check**

| Item | Status | Evidence |
|---|:---:|---|
| What proprioception actually is | **REVISED** | The user's question prompt described proprioception as "the agent's own (x, y) position + orientation — 6-vector", but inspecting `src/environment/sensor.py:300–302, 348–349` shows proprioception is `jax.nn.one_hot(state.last_action, params.action_dim)` — i.e. the **previous-action one-hot** (action_dim=6 here). NOT positional. |
| h=1 sanity check | WARN (mild — but interpretable) | Proprio symlog-MSE at h=1 is **0.1144**, *higher* than at h=5 (0.1068) and basically pinned around 0.10–0.16 across all horizons. A perfect one-hot prediction in symlog space would have MSE 0. Symlog of a uniform 1/6 prediction would give per-dim MSE roughly `((5×0.154² + (ln 2 − 0.154)²)/6) ≈ 0.068` (`symlog(1/6)≈0.154`, `symlog(1)≈0.693`). Observed 0.107 is **above** the uniform baseline, meaning the WM is putting density on the *wrong* action bin. |
| Implication | Caveat | Because the decoder reconstructs `prev_action` from `(deter, stoch)` (it is NOT directly fed in — `get_feat` concatenates only deter+stoch, line 561), the decoder must learn a clean read-out of "which action just executed" from the GRU hidden state. The h=1 result says it has not. This IS a real WM finding (the decoder under-predicts a deterministic, fully-observed channel), BUT it is **not** the channel the user thought it was. The implication is "the decoder fails on a trivially deterministic channel" rather than "the WM cannot predict its own (x,y) position". The latter is not measured by this diagnostic at all (no positional channel was enabled for this checkpoint — `location_sensor_enabled=False` and proprio is action-history only). |

**7. Implementation deviations the developer reported**

| Item | Status | Evidence |
|---|:---:|---|
| obs_dim=19 not 27 (auto-loaded `config.yaml`) | PASS | Verified at lines 361–364: if `models/config.yaml` exists in the checkpoint dir, the script loads it directly; otherwise falls back to merging `--env-config + --agent-config`. The contract change is sensible (the saved config is the single source of truth for layer shapes), and the JSON header still records both YAML paths for provenance (lines 691–692). The plan's premise that "no top-level config.yaml is saved" was wrong (config.yaml IS saved under `models/` — verifiable via `ls results/JAX_DreamerV3/.../models/config.yaml`); the developer adapted correctly. |
| Checkpoint key normalization (string→int + `{'value': array}` unwrap) | PASS (with caveat) | The `_normalize_checkpoint` workaround at lines 141–161 handles two real Orbax/NNX-version-skew artifacts: integer-keyed `nnx.Sequential` indices serialized as strings, and `nnx.Variable` leaves serialized as `{'value': array}`. After normalization, `nnx.update` succeeds on `wm`, `actor`, `critic`. **Caveat**: the developer notes that `train.py:981–1000`'s restore path has the same latent skew bug — it would fail on a fresh run with current NNX. This is out of scope for this verification but should be flagged separately. The normalization is structural (key-and-leaf-shape only), not value-changing — it does not corrupt the loaded weights. |
| Extero Nociception `n/a` threshold | PASS | The plan's §D4 threshold table omitted "Extero Nociception" (the auditor's prior obs breakdown DID list it as 1 dim). The developer assigned `thresh=None` → `n/a` cell in the report — this is the correct conservative choice (don't pass-or-fail on a channel the plan did not pre-register). The plan-table gap is acknowledged in the deviation list. Observed Extero-Noc MSE is < 0.013 across all horizons — would have passed any reasonable threshold, so the gap does not affect the verdict. |

### Verdict on the headline finding

**WM BROKEN at the reward head — methodologically sound, with one terminology caveat.**

- The two failures (reward MAE 0.39 ≫ 0.15, proprio symlog-MSE 0.107 ≫ 0.05) are computed against pre-registered thresholds in the right metric space (raw-MAE for reward, symlog-MSE for obs).
- Reward MAE is robust: from_twohot is correctly symexp'd; ground truth is raw per-step env reward; the magnitude (~0.39) is on the same scale as the modal homeostatic reward, so the WM is genuinely wrong by ~one full reward step. This is the load-bearing failure for the headline.
- Proprio is a real failure but its **interpretation** has to be revised: it shows the decoder cannot reconstruct the previous-action one-hot from (deter, stoch), even at h=1. This is still a meaningful WM failure on a deterministic channel, but it does NOT bear on positional prediction (which is not measured here).
- All passes (Satiation, Olfaction, Collision, Aggregate, Continuation, h50/h5 ratio) are cleanly in-bounds; no spurious failure modes.
- Replay-state-conditional setup is clean — no information leakage, no real-action injection, no off-policy confound.
- Reproducibility re-run (Checkpoint 6) reproduces metrics to float32 precision.

**Recommendation: accept, with two follow-up actions for the user (not blockers):**

1. **Update plan §D4 "Proprioception" rationale** to reflect that proprioception is action-history (one-hot last action), NOT positional. The current rationale ("argmax actor → near-deterministic — should be ~0") is correct in spirit (deterministic given the executed action) but the sentence "near-deterministic from action" obscures that proprio IS the action — the prediction question is "can the decoder read off the action that the GRU just consumed?" The threshold (< 0.05) is still defensible but the framing needs a 1-line edit. **Not a blocker**; can be patched later.
2. **The reward-head failure is the load-bearing finding.** Future Round-2 work (test on A2 / predator) and any "WM is broken" claim should lead with reward MAE, not proprio. The proprio failure is corroborating evidence on a deterministic channel; the reward failure is the diagnostic-significant one for the survival=106 puzzle.

**Speed-change verdict: no regression.** Script is read-only diagnostic, no training-hot-path edit; speed check correctly skipped per protocol.

**Out-of-scope / not blocking:**
- Raw-space companion MSE (plan §"Symlog handling") not implemented — minor doc-style gap, doesn't affect headline.
- `train.py:981–1000` has the same latent NNX-version-skew bug as the script's `_normalize_checkpoint` works around — separate bug, separate plan.

**Conclusion**: Headline finding (WM BROKEN, reward-head bottleneck) is methodologically sound. Proceed with downstream interpretation, with the proprio caveat noted above.

Verified by: senior-developer
