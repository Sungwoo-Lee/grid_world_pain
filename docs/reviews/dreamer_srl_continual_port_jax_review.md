---
title: "Code review — dreamer-srl continual-learning port (stage-swap JAX/Flax correctness)"
topic: continual_learning
status: active
created: 2026-06-09
last_updated: 2026-06-09
---

# Code review: dreamer-srl continual-learning port — stage-swap correctness

## Verdict (plain language)

This change teaches the dreamer-srl trainer to walk through a *curriculum*: a
sequence of progressively harder environments (a 5×5 grid with food and a hiding
rock, then a 10×10 grid, then versions with more animals), switching from one to
the next at fixed cumulative-episode milestones while **keeping the agent's
learned weights** and **throwing away the replay buffer** so the world model does
not learn from a mix of two different physics. The risky part is the moment of
the switch: rebuilding the environment with a different grid size and a different
number of entities, re-pointing the trainer at the new environment, clearing the
buffer, and resetting the agent's short-term memory — all mid-rollout.

**Bottom line: no correctness *blocker*, but one genuine latent bug.** The six
hazards the brief asked me to scrutinise — stale JIT traces, a stale
`env_params` reference, replay-buffer reset leaking old data, broken PRNG
threading, pytree/shape drift, and vmap axis mismatch — are all handled
correctly **except** that the first transition record written into the
freshly-cleared buffer after each stage swap pairs the *new* stage's
"this-is-a-fresh-episode" flag with the *old* stage's observation, reward, and
termination flags. The cause is a NumPy array-aliasing subtlety: the stage block
rebinds the Python name `next_obs` to a new array but leaves the buffer's staged
observation pointing at the pre-swap array. It self-heals as the buffer refills,
so the blast radius is small (one anchored sequence per stage boundary, four
boundaries), but it is real and trivially fixable. I rate it a 🟡 concern, not a
blocker. Detail, the exact lines, and the fix are in the findings table.

The single most-feared risk — that the rebuilt 10×10 environment silently reuses
the 5×5 compiled trace and produces wrong-shaped or wrong-physics rollouts — does
**not** occur: see Finding F1.

## Scope

Working-tree (uncommitted) diff, three files:

- `src/algorithms/dreamer_srl/dreamer_srl_main.py` — `ContinualSchedule`
  dataclass, `_build_continual_schedule`, `_load_stage_env_cfg`, CLI flags +
  mutual-exclusion guard, modality-fingerprint pre-flight, and the
  stage-transition block inside the training loop.
- `src/algorithms/dreamer_srl/buffers.py` — `SequentialReplayBuffer.reset()`.
- `src/algorithms/dreamer_srl/checkpoint.py` — `stage` kwarg on `save_checkpoint`.

Plan (context, not under review):
`docs/develop/active/continual_learning/DREAMER_SRL_CURRICULUM_PORT.md`.

## Findings

| Sev | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟡 concern | `dreamer_srl_main.py:1118-1127` (interacts with `:991`) | **Stale `step_data` row at each stage boundary.** Line 991 sets `step_data["obs"] = next_obs[np.newaxis]`, a NumPy **view** aliasing `next_obs`. The stage block rebinds `next_obs = np.array(env.reset(...))` (line 1122) to a brand-new array, so `step_data["obs"]` keeps pointing at the *pre-swap* array. The block re-syncs `step_data["is_first"][:] = 1.0` (1127) but not `step_data["obs"]`/`["rewards"]`/`["terminated"]`/`["truncated"]`. At the next iteration `buffer.add(step_data)` (line 937) runs *before* line 991 re-writes obs, so the first row of the freshly-cleared buffer = (old-stage obs/reward/term, is_first=1). The RSSM treats that row as the new stage's latent-init anchor → wrong initial observation for the new stage. Self-heals as the buffer cycles, but every sampled sequence whose window starts on that row trains on a corrupt first obs. | In the stage block, after `next_obs = np.array(_next_obs_jax)`, re-sync the staged row to the fresh env: set `step_data["obs"] = next_obs[np.newaxis]` and zero `step_data["rewards"]`/`["terminated"]`/`["truncated"]` (matching the startup init at `:783-789`). This makes the first post-swap buffer row a clean `(new_obs_0, reward=0, terminated=0, is_first=1)` anchor, identical in semantics to the episode-0 row. |
| 🟢 ok | `wrapper.py:7-33` + `dreamer_srl_main.py:1118-1121` | **F1 — No stale JIT trace.** `ParallelEnv.step`/`reset` use `jax.vmap` (not a cached `jax.jit`) and read `self.params` fresh each call. The stage block builds a *new* `ParallelEnv(load_env_params(...))` whose vmap closures capture the new `EnvParams`. `load_env_params` constructs a fresh `EnvParams` pytree (`config_loader.py:828`); since grid `height`/`width`, `num_entities`, etc. are `pytree_node=False` static fields, XLA keys its trace cache on the new static signature and **re-traces** for 10×10. No JIT'd function in the hot loop closes over `env`/`env_params` (the world-model `train_step` operates on replay batches, not env dynamics; `env_params` at `:1222/:1262` is checkpoint *data*). | — |
| 🟢 ok | `dreamer_srl_main.py:1118`, `:942`, `:1088-1099` | **F2 — `env_params` not stale.** The name `env_params` is rebound at line 1118 *before* it is next read. The per-step path uses `env.step` (which reads the new `ParallelEnv.self.params`); the per-env autoreset loop (`:1093-1094`) reads the module-level `env_params`, but it runs *before* the stage block and is then fully overwritten by the authoritative full-env `env.reset` (`:1121`), as the comment at `:1103-1106` notes. Wasted old-shape compute, no correctness impact. | — |
| 🟢 ok | `buffers.py:reset()` + `:316-402` | **F3 — Buffer reset cannot leak stale rows.** `reset()` sets `_pos=0, _full=False` and retains backing arrays. In the non-full branch, `sample()` draws `start ∈ [0, _pos-seq_len+1)` and `idxes = (start+chunk) % buffer_size`, so every index is `< _pos`; retained rows beyond `_pos` are unreachable until the buffer naturally refills and flips `_full`. The early-sample guards (`:316`, `:323`) plus the loop's `buffer._pos >= seq_len` train gate (`:1305`) prevent reading the still-empty region. | — |
| 🟢 ok | `dreamer_srl_main.py:1120`, `:1086`, `:922` | **F4 — PRNG threading clean.** Stage reset splits and advances the main key (`key, _k_stage_reset = jax.random.split(key)`); autoreset and player each split independently; no sub-key reuse and the main key is always advanced. `player.init_states()` consumes no key. Reproducibility property preserved. | — |
| 🟢 ok | `dreamer_srl_main.py:533-563`, `state.py:184-245` | **F5 — Pytree/shape consistency enforced.** Obs is 27-dim invariant across the 5×5→10×10 swap; the pre-flight `_modality_fingerprint` check validates obs_dim, action_dim, and 13 obs-layout fields are identical across all stages, failing fast otherwise. It correctly *excludes* `height`/`width`/`num_entities` (which change but don't affect obs layout). All 13 fields + `rest_action_enabled`/`eat_action_enabled` exist on `EnvParams`, so the check won't `AttributeError`. Agent weights are never rebuilt → weight pytrees stay structurally identical. | — |
| 🟢 ok | `dreamer_srl_main.py:1108-1175` | **F6 — vmap axis correct.** The stage block runs once *after* the per-env autoreset loop and counts the whole batch's dones before transitioning. `env.reset(_k_stage_reset, num_envs)` returns batched states/obs with leading axis-0 = env index, matching the step/autoreset convention. Per-tag accumulators and BM state are rebuilt with leading `num_envs`. No batch-axis mismatch. | — |
| 🟢 ok | `buffers.py:reset()` docstring | **Minor: docstring path reference.** Cites `train.py:1230-1231` (rPPO `buffer.idx=0; buffer.size=0`) as the analogue — accurate; the rPPO reference clears via direct counter assignment, the port encapsulates it in `reset()`. No action. | — |

## Why the 🟡 is a concern and not a blocker

The rPPO reference (`train.py:1193-1195`) assigns the post-swap reset obs straight
to `obs`, and rPPO adds to its buffer at the *end* of the iteration using that
`obs` directly — so the reference has no separate staged-obs view to fall out of
sync. dreamer-srl's loop is structurally different: it carries a `step_data`
dict whose `"obs"` is a **view** set at end-of-iteration and consumed by
`buffer.add` at the *top* of the next iteration (the sheeprl "write the row one
step late" pattern). The port faithfully copied the rPPO stage block but did not
account for this structural difference, so the obs/reward/term re-sync that rPPO
gets for free is missing here. The defect is bounded (one anchor row per stage
boundary × 4 boundaries, self-healing) and does not corrupt the policy's online
actions (line 1284 `obs = next_obs` uses the correct fresh obs) — hence concern,
not blocker. It should be fixed before the curriculum run because the corrupted
rows land exactly on the `is_first=1` latent-init anchors the world model is most
sensitive to.

## Conventions audit

- Pytree / immutability: ✅ (`load_env_params` builds fresh `EnvParams`; no in-place mutation)
- JIT recompilation: ✅ (new `ParallelEnv` re-traces under new static signature; no stale-trace reuse)
- vmap & batch axis: ✅ (axis-0 = env index preserved across rebuild)
- PRNG threading: ✅ (keys split + advanced; no reuse)
- Sensor / obs-breakdown sync: ✅ (fingerprint pre-flight enforces obs-layout invariance across stages)
- Config protocol: ✅ (`get_mandatory("continual.episode_boundaries")` / `checkpoint_frequencies`; mutual-exclusion guard; no silent defaults)
- step_data / buffer-row consistency: ❌ (stale obs/reward/term anchor row at each stage boundary — Finding F1-row / 🟡)

## Conclusion

No correctness blocker; one 🟡 concern — re-sync `step_data["obs"]` (and zero
`rewards`/`terminated`/`truncated`) inside the stage-transition block so the first
post-swap buffer row is a clean fresh-stage anchor. Fix before launching the
curriculum run.

Reviewed by: code-reviewer

---

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-06-09
> **Tree state**: left dirty (uncommitted), as requested.

### Summary of changes

**Only one file was modified**: `src/algorithms/dreamer_srl/dreamer_srl_main.py`.

**Location of change**: inside the stage-transition block, immediately after `next_obs = np.array(_next_obs_jax)` (line 1122), four new lines were inserted before the existing `# 2. Reset player recurrent...` comment:

```python
step_data["obs"]        = next_obs[np.newaxis]                    # [1, B, obs_dim]
step_data["rewards"]    = np.zeros((1, num_envs, 1), dtype=np.float32)
step_data["terminated"] = np.zeros((1, num_envs, 1), dtype=np.float32)
step_data["truncated"]  = np.zeros((1, num_envs, 1), dtype=np.float32)
```

This mirrors the startup init at lines 783–789 exactly. Shapes are `[1, num_envs, obs_dim]` and `[1, num_envs, 1]` — identical to the existing field layouts. The existing `step_data["is_first"][:] = 1.0` line was left unchanged. No other files were touched.

### Test results

| Test | Command | Result |
|---|---|---|
| Unit: buffer reset | `pytest tests/algorithms/dreamer_srl/test_buffer_reset.py -v` | **5/5 passed** |
| Unit: continual schedule | `pytest tests/algorithms/dreamer_srl/test_continual_schedule.py -v` | **26/26 passed** |
| Regression probe (pre-fix = fails, post-fix = passes) | `python tmp/20260609_step_data_resync_verify.py` | Pre-fix obs=0.5/reward=7.0 (old stage — bug confirmed); Post-fix obs=9.9/reward=0.0/term=0.0/trunc=0.0 (new stage — CORRECT); ALL ASSERTIONS PASSED |
| Integration smoke (boundaries [3,6,9], 2 stage transitions) | `python src/algorithms/dreamer_srl/dreamer_srl_main.py --configs-dir tmp/smoke_stages/ --continual-schedule tmp/tiny_schedule.yaml --agent-config configs/models/dreamer_srl/01_food_only_smoke.yaml --num-envs 1 --seed 0 --no-wandb --results-dir tmp/20260609_smoke_bug_fix` | **Exit 0. Two transitions: `[STAGE] 0->1 at ep=3`, `[STAGE] 1->2 at ep=6`. 9/9 episodes completed. All losses finite.** |

### Speed check

Skipped. This change adds four `np.zeros` / `np.newaxis` lines inside the `if _new_stage != current_stage:` branch, which fires at most twice per run (at stage boundaries, not per-step). The hot path (every-iteration env-step / buffer-add / train-step) is entirely unaffected. Zero regression by construction.

### Deviations from plan

None. Exactly the fix specified by the reviewer — four lines, one file. No scope expansion.

### Blockers / follow-up

None. Tree is dirty; ready for `senior-developer` verification and commit.
