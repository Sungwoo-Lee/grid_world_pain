# Diagnosis findings — offline eval tooling (eval_rollout.py + evaluation_core.py)

Date: 2026-07-23. Reviewer: eval-tooling deep-diagnosis pass.
Unit: `scripts/eval/eval_rollout.py`, `src/utils/evaluation_core.py` (plus alignment spot-checks of
`src/utils/eval_recording.py`, `scripts/eval/trajectory_story.py`, `src/environment/sensor.py`, `train.py`, `evaluation.py`).
Report-only: no code modified.

Recent history on the unit: `c023145` (batched Dreamer eval), `766938d` (unify Dreamer into eval_rollout),
`fa9ada0` (batched rPPO rollout), `332ce9f` (H8/H9 fixes) — all on eval_rollout.py; only `332ce9f` on evaluation_core.py.

---

## New findings

### F1 [P1] eval_rollout.py:908 — rPPO step-dir invocation collapses run identity; cross-run output collision

**Claim.** `run_tag = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.parent.parent.name`
special-cases only the Dreamer container dir name (`checkpoints`), not the rPPO one (`models`, created by
train.py:623 `models_dir = results_dir/"models"`). Invoking with the documented step-dir form
(`--help`: "the CheckpointManager root or a <root>/<step> dir") — i.e. `--checkpoint <run>/models/<step>` —
yields `run_tag == "models"`, so `out_dir = <out_root>/models/<step>`.

**Failure scenario.** Two different runs evaluated at the same checkpoint step write to the *same* directory;
the second silently overwrites the first's `episodes/*.npz`, `online_replay.json`, `metadata.json`. Even without
same-step overlap, run attribution is lost — an analysis pointed at `results/eval/models/<step>` cannot tell which
run produced the data.

**Evidence.** This has fired in reality: `results/eval/models/` exists and contains 7 step dirs from evidently
different runs (`819`, `10000003`, `10000021`, `10000022`, `1810043`, `3900028`, `8900007`) alongside the
properly-tagged siblings (`avoidance/`, `predatorOnly/`, ...). No proven same-step overwrite yet (steps are
distinct), which keeps this P1 rather than P0 — but any past analysis that consumed `results/eval/models/<step>`
without independent provenance should be re-attributed.

**Fix direction.** Treat `"models"` like `"checkpoints"` in the run_tag derivation (walk up one more level), or
derive run_tag from the CheckpointManager root's parent generically.

### F2 [P1] evaluation_core.py:565-581 vs 694-706 — parallel-env eval: first-generation episodes off-by-one in survival steps (and malformed recordings) when record_stats is off

**Claim.** In `_run_parallel_env_eval`, the step-0 seeding of the per-slot buffers (`slot_states/infos/actions/
rewards/obs`, with `action=-1, reward=0.0`) is gated on `if record_stats:` (line 565), but the refill-time seeding
after an episode completes (lines 694-706) is unconditional, and the per-step appends (609-627) are unconditional.

**Failure scenario.** Run `evaluation.py --num_envs > 1` with `testing.record_stats: false` (the default when the
key is absent; `record_stats=None` -> `config.get('testing.record_stats', False)` at evaluation_core.py:276-278):
- The first `effective_num_envs` episodes have no step-0 entry, so `episode_lengths.append(len(slot_rewards[i]) - 1)`
  (line 637) undercounts their survival by exactly 1 step; refilled episodes are correct. `mean_length` — the
  project's headline survival-steps metric — is biased low by `effective_num_envs/num_episodes` steps.
- With `render_video=True`, the first-generation `.rec.gz` files are also missing the initial frame: `snapshots[0]`
  is the post-step-1 state and `actions[0]` is a real action instead of the `-1` sentinel, while refilled episodes
  follow the correct convention. Any per-step analysis (trajectory_story) of those episodes is shifted by one step
  relative to the others in the same directory.

**Evidence.** Code trace above; the train.py in-training path is dormant for this bug (video pass uses
`num_envs=1`, stats pass uses `record_stats=True` — train.py:2117-2138), so only the standalone `evaluation.py`
parallel path is exposed.

**Fix direction.** Seed the slot buffers at initial reset unconditionally (mirror the refill block), gating only
`slot_true_obs` on `record_true_obs`.

### F3 [P1] eval_rollout.py:98, 201, 1073-1081 — stochastic eval mode reuses one fixed PRNG key for every step's action sample

**Claim.** `_run_episode` / `_run_episode_with_recording` pass the *episode's* `rng_key` (which is also the
`jax_reset` key, `PRNGKey(seeds[ep])`) unchanged into `policy_fn` on every iteration; `policy_fn` forwards it as
`key=key if not deterministic else None` into `get_action_and_value_nnx`, which does
`jax.random.categorical(key, logits)`. The key is never split or folded per step.

**Failure scenario.** Set `behavior_measures.eval_policy_mode: "stochastic"` (an allowed, validated mode —
config_loader.py:144) — every step of an episode samples with the identical key, so the per-step uniform draws are
perfectly correlated across steps (and correlated with the reset stream). "Stochastic eval" silently produces a
degenerate, near-deterministic policy whose sampling statistics are meaningless. Dormant today: every config in
`configs/` sets `deterministic`, and the `--batched` path explicitly refuses non-deterministic mode.

**Evidence.** Code trace; `get_action_and_value_nnx` at recurrent_ppo_network.py:396-405 consumes the key raw.

**Fix direction.** Thread a split key through the loop (`key, sub = jax.random.split(key)` per step) inside
`_run_episode*`, or fold in the step index.

### F4 [P1-latent] eval_rollout.py — `behavior_measures.eval_obs_noise` is validated and reported but never enforced

**Claim.** `eval_obs_noise` ("training" | "zero" | "custom", validated at config_loader.py:218) is consumed
nowhere in the rollout: `get_observation(state, params)` always applies the env config's training noise
(sensor.py:291-294, keyed off `fold_in(state.key, 999)`). Yet `metadata.json` records
`"eval_obs_noise": bm_cfg.eval_obs_noise` (eval_rollout.py:1364) as if it had been honored.

**Failure scenario.** A user sets `eval_obs_noise: "zero"` expecting a noise-free probe eval; episodes run with
full training noise, and the output metadata *confirms* "zero" — a self-certifying wrong record. All current
configs set "training", so no existing result is wrong; the trap is armed for the first person who flips the knob.

**Fix direction.** Either implement the mode (pass `apply_noise=False` for "zero"; raise NotImplementedError for
"custom") or fail loudly on any value other than "training" at eval_rollout startup.

### F5 [P2] eval_rollout.py:113-114, 216-217, 423-426 — npz `nociception` field is structurally always 0.0

**Claim.** `info.get("nociception", info.get("exteroception_nociception", 0.0))` always hits the final fallback:
`jax_step`'s info dict (core.py:679-797) never carries either key. Every `episodes/*.npz` therefore contains an
all-zeros `nociception` array in both the legacy and batched paths (the batched path documents this and writes
`np.zeros` outright, preserving parity with a dead field).

**Failure scenario.** Any offline analysis that reads `nociception` from the npz dumps silently concludes
"no nociceptive signal" regardless of the run.

**Fix direction.** Either populate from the actual sensed value (e.g. recompute `sense_extero_nociception` from
state) or drop the field so consumers fail loudly.

### F6 [P2] eval_rollout.py:896 + 403-407 — `eval_max_steps` dead; batched-path length-1 silent-truncation hazard

**Claim.** `bm_cfg.eval_max_steps` (validated >= 1 at config_loader.py:216) is never used; the rollout horizon is
always `environment.max_steps`. Separately, `_run_episodes_batched` computes `T = np.argmax(done_seq, axis=0) + 1`,
which returns T=1 for an all-False done column — correct only because the caller passes
`max_steps == params.max_steps` so env truncation is guaranteed inside the scan window. If the function is ever
called with a shorter horizon (e.g. someone wires up `eval_max_steps`), every non-terminating episode silently
records length 1.

**Fix direction.** Assert `np.all(done_seq.any(axis=0))` before the argmax; either honor or remove
`eval_max_steps`.

### F7 [P2] scripts/eval/trajectory_story.py:136, 196 — step-0 sentinel action `-1` renders as the LAST action name

**Claim.** Recordings written by both eval_rollout and evaluation_core set `actions[0] = -1` ("no action yet").
`cmd_dump` and `cmd_obs` print `AM[int(acts[t])]` — Python negative indexing turns `-1` into the last action-map
entry ("Eat" when the eat action is enabled), so every episode's first row claims the agent ate/rested at t=0.

**Fix direction.** `"None" if acts[t] < 0 else AM[acts[t]]` (mirrors `_write_episode_stats`'s handling).

### F8 [P2] eval_rollout.py:1229-1236, 1257-1259 — misleading provenance metadata

**Claim.** (a) The Dreamer branch's `metadata.json` writes `"seeds": seeds` — the rPPO per-episode seed list —
though the Dreamer rollout derives all reset keys from the single `--seed` master key and ignores that list.
(b) `write_run_meta` in the rPPO record path receives `args.config`, not the stage-resolved `config_path`, so a
continual run's `run_meta.pkl` names the stage-0 config even when a later stage's env was (correctly) used
(`metadata.json`'s `config_resolved` is right; `run_meta.pkl` is not).

**Fix direction.** Write `{"seed": args.seed}` for Dreamer; pass `config_path` to `write_run_meta`.

### F9 [P2] evaluation_core.py:437-439 — dead per-episode closure

`get_sensory_viz` is defined inside the episode loop and never called (sensory viz now happens at render time via
`build_sensory_viz` in the renderer). Harmless; delete when next touching the file.

### F10 [registry note] KNOWN_BUGS OPEN row stale

"eval_rollout.py raises NotImplementedError for dreamer_srl checkpoints (capability gap)" is no longer true:
commit `766938d` added a full Dreamer branch (restore + rollout + recordings). The `NotImplementedError` at
line 1246 is now dead code (agent_type is always "rppo" or "dreamer"). Suggest bug-curator closes/updates the row.

---

## Fixed-bug regression check

| Fixed bug | Status | Evidence |
|---|---|---|
| H8 — stats CSV columns mislabeled/off-by-one | **Fix intact** | `build_stat_headers` (evaluation_core.py:198-239) is the single source of the header layout; `_write_episode_stats` writes values in the same order and carries fail-loud drift checks (lines 84-93: obs_*-header count vs obs dim, true_* count vs true-obs dim, with `obs_entity_` correctly excluded from the obs count); `_sensor_stat_columns` raises on unknown sensors and on name-count != dim. Header order vs row-append order verified column-for-column (16 fixed cols, obs_*, true_*, res r/c/active, pred r/c, neutral r/c, obs_entity r/c, termination_reason, max_satiation, max_injury). |
| H9 — interrupted-feeding rate structurally zero | **Fix intact** | `_compute_online_replay` (eval_rollout.py:571-590) updates `steps_since_eat` BEFORE candidate aging/resolution, so a candidate eat followed by K eat-free steps resolves with `steps_since_eat >= K` -> interrupted increments. The known resolution-time-denominator divergence from the online accumulator is deliberately preserved and documented in-code (lines 574-577). |
| Continual-run eval rebuilding stage-0 env | **Fix intact** | `_resolve_continual_stage_config` (eval_rollout.py:633-731) fires only when `schedule.yaml` exists AND `--config` is exactly the run's stage-0 `config.yaml`; reads the stage index from the checkpoint's own saved `stage` field (not recomputed from boundaries); fails loudly on missing stage config. Wired into `main()` at lines 846-849 before env/params construction. |
| Noise invisible in eval video | **Fix intact** | `record_true_obs` gate (evaluation_core.py:284-286) includes `render_video and params.perceptual_noise_enabled`; true obs computed per step with `apply_noise=False` and threaded into `EpisodeRecorder.append` in both single (line 491) and parallel (lines 656-658) paths. eval_rollout's own recording paths record true_obs unconditionally. |
| Noise painted on wrong sensory channel | **Fix intact** | `build_sensory_viz` (sensor.py:392+) iterates `get_observation_breakdown()` in order with paired `ptr`/`t_ptr` advanced per sensor — obs and true_obs slices stay aligned per modality. |

## Reviewed but clean

- **Checkpoint->network reconstruction fidelity (rPPO):** eval reads the same `models/config.yaml` train.py dumps
  (CLI overrides for hidden_size/lr/num_steps are persisted into it before dump, train.py:537-548 + 638-640);
  `ActorCriticRNN` is constructed with kwarg-for-kwarg the same arguments as train.py:811-821, including
  `encoding_config=config.to_dict().get('agent', {})` and the same `modulation_config` None-normalization.
  Silent partial restore is defeated: `partial_restore=True` is followed by a strict per-leaf presence-and-shape
  check over the *current* model's flattened state (eval_rollout.py:1043-1061) that raises on any missing key or
  shape mismatch.
- **Batched rPPO parity path (`--batched`):** reset-key parity guard (lines 360-370) is a real runtime check;
  vmap-axis argmax assertion (388-394); per-episode slicing `[:Ti, i]`, `termination_reason` at `Ti-1`, and length
  `T = argmax(done)+1` all match legacy semantics (terminal step included, same as `step_count`); recorder
  reconstruction (SimpleNamespace) covers exactly the 9 fields `_snapshot_state` reads; forward pass correctly
  goes through `nnx.jit` (documented stale-view hazard).
- **Seed handling / fake-low-variance check:** rPPO legacy and batched paths use one distinct `PRNGKey(seed)` per
  episode from `bm_cfg.eval_seeds` (default `range(10)`, extension logic produces distinct ints); evaluation_core
  splits a master key per episode reset and per refill — no identical-seed episodes anywhere.
- **Deterministic-vs-stochastic assumption:** the default and only-used path is argmax (`eval_mode=True`
  everywhere in evaluation_core; `bm_cfg.eval_policy_mode == "deterministic"` in all configs), matching what the
  behavior-measure analyses assume. (Stochastic mode itself is broken — F3 — but never exercised.)
- **Survival-steps off-by-one (eval_rollout):** legacy `_run_episode` counts the terminal step (`step += 1` after
  the death/truncation step); batched matches; `length` and `episode_lengths` (single-env evaluation_core path)
  agree. Only the parallel-path first-generation case (F2) deviates.
- **Recomputed-noise consistency:** `get_observation` derives noise from `fold_in(state.key, 999)` (sensor.py:294),
  so the true-obs/obs recomputations in recording paths see exactly the noise the policy saw — no divergence.
- **`_write_episode_stats` info keys:** all eight keys exist in core.py's `jax_step` info dict (679-797) — the
  `.get(ik, 0)` fallback fires only for the intentional empty step-0 dict.
- **`_detect_threat_onsets` / M2 / M5 logic**, `_pad_ragged`, ragged-to-(T,0) parity for zero-predator configs,
  Dreamer env-config merge chain (deliberately mirrors dreamer_srl_main.py's layering and is documented as such),
  Dreamer restore-target completeness vs `save_checkpoint`'s key set, and `EpisodeRecorder.write`'s
  all-None/all-set `true_obs` invariant (mixed None can't occur from any current writer).
