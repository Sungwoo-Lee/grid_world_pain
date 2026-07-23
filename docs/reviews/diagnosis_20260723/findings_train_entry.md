# Diagnosis findings — train.py (recurrent-PPO entry point), 2026-07-23

Reviewer: JAX/Flax correctness pass, report-only. Unit: `/media/nas01/projects/Interoceptive-AI/grid_world_pain/train.py` (2189 lines, branch v3.0 HEAD).

Known-bug OPEN rows were NOT re-reported as new findings; root-cause depth added in a dedicated section.

---

## NEW FINDINGS

### F1 — [P1] train.py:1293-1295 + 1612 / 1811 / 1985 — DQN/DRQN/PPO episode metrics are lifetime cumulative means (never-cleared `iteration_episodes`)

**Claim.** The per-window clearing of `iteration_episodes` is gated on `logging_cfg is None` (line 1293), but `logging_cfg` is *never* None in practice: `configs/train/default.yaml` is merged unconditionally for every algorithm (train.py:309-314) and declares a full `logging:` block, so `resolve_logging_cfg` (src/utils/rolling_logging.py:53-77) always returns a dict. Only the RecurrentPPO branch was migrated to the two-level `RollingWindow` path; the DQN (1612), DRQN (1811), and PPO (1985) branches still append every finished episode to `iteration_episodes` and, at every `iteration % log_interval == 0`, log `np.mean(...)` over the *entire since-start list* (1647-1690, 1853-1898, 2005-2049).

**Failure scenario.** Any DQN/DRQN/PPO run launched through train.py today: `Episode/Steps` (the survival-steps headline metric), `Episode/Reward`, and all `Episode/*` behavior metrics converge to the lifetime average — a learning agent's curve flattens artificially and late-training improvement is invisible. Secondary: the list grows without bound (memory leak over long runs).

**Evidence.** Trace: train/default.yaml `logging:` block -> `resolve_logging_cfg` returns non-None (the all-None check at rolling_logging.py:67 fails) -> line 1293 `if logging_cfg is None:` skips the reset forever -> `iteration_episodes = []` only ever executes at init (line 1083). rPPO is unaffected (episodes go into `ep_window`; legacy append at 1387-1389 and legacy emit at 1427 are both gated on `logging_cfg is None`, hence dead).

**Fix direction.** Migrate the three branches to the shared `RollingWindow`/`_emit_episode_row` path, or clear `iteration_episodes` per log interval unconditionally in those branches.

**Corollary (same root).** The legacy `log_interval`/`log_accumulate` machinery and the DEPRECATION print at 512-516 are dead code for every train.py run; `--log-interval` is always ignored-with-warning.

---

### F2 — [P2] train.py:709-712 vs 1455-1461, 1479-1496, 1644-1645, 1851 — WandB define_metric case "fix" matches none of this file's keys

**Claim.** The comment (707-708) says actual logged keys use uppercase prefixes (Loss/*, Modulator/*) and defines uppercase patterns — but every loss/modulator key train.py actually logs is *lowercase* (`loss/total`, `loss/policy`, `modulator/grad_norm`, `loss/dqn`, `loss/drqn`, `train/epsilon`). The uppercase keys belong to dreamer_srl_main.py. So in train.py the lowercase keys still fall through to the `"*"` catch-all (`step_metric="timesteps"`) — the exact fall-through the fix was meant to remove persists for this file's keys.

**Failure scenario.** rPPO loss/modulator panels default to a timesteps x-axis instead of iteration. Cosmetic: `iteration` is logged in the same row (1500), so panels can be re-axised manually.

**Fix direction.** Add lowercase `loss/*`, `modulator/*`, `train/*` define_metric patterns (keep both cases).

---

### F3 — [P2] train.py:2075-2081 — resume triggers an immediate redundant checkpoint + eval because `main.last_checkpoint_save` is not restored

**Claim.** `main.last_checkpoint_save` initializes to 0 (function-attribute hack, 2075-2076). After `--load-checkpoint` restores e.g. `total_episodes_completed = 1,400,000`, the first iteration evaluates `1400000 >= 0 + checkpoint_freq` -> True -> saves a checkpoint of the just-restored weights into the *new* results dir and runs the full eval-video/stats pass before any new training has happened.

**Failure scenario.** Every resume wastes GPU minutes on a redundant save+eval and produces a duplicate checkpoint/eval artifact at the restored episode count. Not corrupting (the subsequent snap `(total // freq) * freq` re-aligns cadence), but wasteful and confusing in the artifact trail.

**Fix direction.** After restore, seed `main.last_checkpoint_save = (total_episodes_completed // checkpoint_freq) * checkpoint_freq`.

---

### F4 — [P2] train.py:1198-2185 — no final/exit checkpoint: sub-frequency runs produce zero checkpoints; graceful stop discards the tail

**Claim.** Checkpoints are saved only when `total_episodes_completed` crosses a `checkpoint_freq` milestone inside the loop. There is no save after the loop exits — neither at normal budget completion (tail beyond the last milestone is lost), nor on `stop_requested` (SIGINT/SIGTERM graceful path, 1202-1203), nor in the KeyboardInterrupt handler (2178-2179).

**Failure scenario.** rPPO's default `checkpoint_frequency` is 200,000 episodes (configs/train/recurrent_ppo.yaml). Any run stopped (or budgeted) below 200k episodes ends with NO checkpoint at all — `models/` contains only config.yaml; a graceful Ctrl-C loses up to 200k episodes of training. Smoke runs (`episodes: 100`) never checkpoint.

**Evidence.** `should_checkpoint` (2078) is the only save gate; after `while` exit the code goes straight to `wandb.finish()`/`checkpointer.close()` (2181-2185). The signal handler's promise "Finishing current iteration..." finishes the iteration but persists nothing.

**Fix direction.** Unconditional final `checkpointer.save(total_episodes_completed, ...)` after loop exit (guard against duplicate step id).

---

### F5 — [P2] train.py:1502 — `Time/sps_env` inflated after resume

**Claim.** `Time/sps_env = global_step / (now - start_time)` where `global_step` includes all restored steps but `start_time` (1144) is the fresh process start. A resume at 100M steps reports an absurdly high SPS for hours, corrupting speed comparisons across runs.

**Fix direction.** Use `(global_step - restored_step)` in the numerator (or persist elapsed seconds in the checkpoint).

---

### F6 — [P2] train.py:672 — `wandb.name` from YAML is silently ignored

**Claim.** `wandb_kwargs["name"] = args.wandb_name or tag` — the config key `wandb.name` is only ever *written* (line 404, mirroring the CLI flag) and never read back. A user setting `wandb: {name: ...}` in a config silently gets the tag as run name. Inconsistent with the sibling keys (`wandb.project`/`entity`/`group` are read via `get_mandatory`, 669-671).

**Fix direction.** `args.wandb_name or config.get('wandb.name', None) or tag`.

---

### F7 — [P2] train.py:1226-1267 — stage transition wipes ep_window but not `step_window` / `ep_info_buffer` (depth on the OPEN stage-swap accumulator row)

**Claim.** The continual stage-swap block clears per-env accumulators, BM state, and `ep_window` (Buffer B), but NOT: (a) `step_window` — the loss RollingWindow — so the first post-swap loss rows average up to `smoothing_iters` (100) pre-swap iterations into the new stage's loss curve; (b) `ep_info_buffer` (deque 100) — tqdm's "Rew" postfix mixes stages (cosmetic); (c) legacy `iteration_episodes` (moot while the legacy path is dead per F1, but latent). This is likely the concrete content of the OPEN "continual stage swap keeps stage-0 metric accumulators" row; recording the exact surviving accumulators here.

**Fix direction.** Clear `step_window.buf`/`count` (and optionally `ep_info_buffer`) inside the stage-transition block, symmetric with ep_window.

---

## ROOT-CAUSE DEPTH ON OPEN ROWS (not new findings)

- **lr_critic dead** (train.py:825-832, 538-540): the single `optax.adam(lr)` with `lr = agent.lr_actor` covers ALL `nnx.Param` — actor, critic, AND neuromodulator. `agent.lr_critic` is read nowhere in train.py or recurrent_ppo_trainer.py (grep-confirmed zero hits). Fix direction when tackled: `optax.multi_transform` with a param-label fn over the NNX path prefix.
- **Single-config resume pairs restored memory with fresh worlds** (train.py:1097-1116 vs 759-764, 2085-2094): two compounding causes. (1) `env_state` is absent from `ckpt_data`, so resume cannot reconstruct the worlds the restored `h_state` was mid-episode in. (2) The fresh worlds come from `env_key` split off `PRNGKey(seed)` BEFORE restore replaces `key` — so a resumed run replays the exact same initial world sequence the original run already trained at t=0 (the restored `key` fixes the training stream going forward but not the already-consumed env reset). The continual path (1123-1142) correctly resets `h_state` after its env rebuild; the single-config path resets nothing — that asymmetry is the bug.
- **CLI overrides not saved to config (OPEN row) — appears ALREADY FIXED, row likely stale**: `--num-steps` -> `config.set('agent.sequence_length', ...)` (533), `--hidden-size` (539), `--lr` -> `agent.lr_actor` (540), `--no-satiation`/`--no-overeating-death` -> `body.*` (412-413, plus per-stage propagation 365-370), `episodes` (494), `num_envs`/logging (517-525) — all set before the dump at 638-640. Recommend bug-curator re-verify and close/update the row.

---

## FIXED-BUG REGRESSION CHECK (all fixes still present, no regressions)

| Fixed bug | Status | Evidence |
|---|---|---|
| Resume silently restored nothing (H1) | STILL FIXED | `restore_rppo_training_state` (src/utils/checkpoint_restore.py) is restore-to-target, fatal on missing steps (FileNotFoundError, lines 38-42) and on any orbax structure/shape mismatch; no blanket except. Consumed at train.py:1102-1112. |
| Continual resume ran wrong stage's world (H2) | STILL FIXED | train.py:1123-1142 — schedule-derived stage, unconditional env rebuild, fresh reset, h_state re-init. |
| `extends:` inheritance ignored | STILL FIXED | `load_env_config` used at train.py:389 (--config) and :198 (per continual stage). |
| Typo'd --config path silently used default (H3) | STILL FIXED | `Config.load_yaml` raises FileNotFoundError on a missing path (src/utils/config.py:30-38); `load_env_config` -> `_resolve_extends` -> `Config.load_yaml`. Note: `base_config_path` at train.py:304 is now vestigial (debug print only) — harmless. |
| CLI satiation flag not persisted (G3/L4) | STILL FIXED | train.py:408-413 sets `body.with_satiation`/`body.overeating_death` (the keys `load_env_params` reads) before the config dump at 638-640; continual stages get the same at 365-370 before their per-stage dumps. |

Bonus: `--num-envs` mismatch on resume fails loudly (orbax shape mismatch against the live `h_state` restore template) — good.

## REVIEWED BUT CLEAN

- rPPO rollout state threading: `env_state`, `h_state`, `key` all round-trip through `jit_train` each iteration (1300-1302); PRNG never reused across iterations (trainer advances the carried key; commit b8eb286 covers the auto-reset key draw).
- Episode accounting: per-env return/length counters reset exactly on the done step; `termination_reason` read at the correct `[t][i]`; distance means divide by `max(ep_length,1)`; done-step reward included before reset. No off-by-one found in `global_step`/`iteration`/episode counting.
- Two-level rPPO episode path: push-per-episode into `ep_window` correctly avoids any num_envs cap; emission gate (full window AND count % interval) matches the documented design; loss samples kept as JAX scalars until emission (no per-iteration host sync).
- rPPO checkpoint payload: `key`, `iteration`, `step`, `episode`, `stage`, optimizer state (incl. Adam moments via `nnx.state(optimizer)`) all saved; `h_state` saved (its pairing problem is the known open row). Model has no non-Param state to lose.
- Checkpoint scheduler: no duplicate step ids possible (snap-then-compare); Orbax `max_to_keep=null` -> keep-all for rPPO per config.
- DQN/DRQN auto-reset: `select_done` tree_map picks reset-state on done; next_obs stored pre-reset (terminal obs) — correct; DRQN hidden state zeroed on done (1748-1754).
- Continual guards: obs-dim + action-dim + modality-fingerprint validation across stages (581-610); `--episodes` vs schedule mutual exclusion; boundary monotonicity checks; per-stage CLI satiation propagation; only-rPPO restriction enforced (479-483, 1113-1116).
- `model_key` split at 761 is unused (dead split) — harmless, not a correctness issue.
- DQN/DRQN/PPO never checkpoint (`ckpt_data = {}` stays empty) — intentional per the explicit resume error at 1113-1116; noted, not filed.
- Eval-at-checkpoint uses the fixed run `seed` each time -> deterministic, repeated eval worlds across checkpoints; intentional determinism, flagged only.
