# Code Review — Continual Learning Config Schedule

**Reviewer:** code-reviewer
**Plan:** [`docs/develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)
**Sibling review:** [`config_continual_learning.md`](config_continual_learning.md)
**Diff scope:** `train.py` (uncommitted), supporting code in `src/models/dreamer_v3_trainer.py` (read-only context).
**Date:** 2026-05-06

---

## Summary

The continual-learning patch is, on balance, JAX-correct. The author asked the right questions about pytree mutation (`buffer.idx = 0` is safe because `ReplayBuffer` is a plain Python class, **not** a `@struct.dataclass`), about JIT recompilation triggers (the per-iteration stage check is purely Python-side and never enters a traced region), and about in-place wipes (`episode_returns[:] = 0.0` is fine because the accumulators are NumPy arrays). PRNG threading is consistent with existing code: one `split` consumed at transition, no per-env desync.

There are however **two correctness concerns the plan acknowledges as intentional but should be revisited** (asymmetric agent-state handling, and `dreamer_state['is_first']` not being raised after the hard reset), plus **one silent behavioral bug** that is not in the plan: CLI overrides `--no-satiation` / `--no-overeating-death` are applied to the stage-0 `EnvParams` but are **not** re-applied at stage transitions, so they silently revert at the first boundary. There are also a couple of WandB metric-axis nits.

**Verdict: WARNINGS.** No 🔴 blockers, but one 🟡 silent correctness bug (`-no-satiation` revert) and three 🟡 design concerns worth surfacing before this is committed.

---

## Findings

| Sev | File:Line | Issue | Suggested Fix |
|---|---|---|---|
| 🟡 | `train.py:1024` (and `train.py:468`) | **SILENT BEHAVIORAL BUG.** `params = load_env_params(schedule.stage_configs[new_stage])` at the transition does not re-apply the CLI overrides at `train.py:457-458` (`--no-satiation`, `--no-overeating-death`). Same for the per-stage validation probe `p_i = load_env_params(schedule.stage_configs[i])` at `train.py:468`. So a user running `--configs-dir X --no-satiation` will silently get `with_satiation=True` after the first stage transition (whatever the stage YAML says). The validation probe also won't notice, because both stage 0 and stage 1+ probes load straight from YAML. **Silent** because nothing crashes — `EnvState` shapes match, `obs_dim` matches, training continues, but the env semantics flipped. | Wrap the CLI override in a helper applied to every newly-loaded `params`: `def _apply_param_overrides(p): if args.no_satiation: p = p.replace(with_satiation=False); if args.no_overeating_death: p = p.replace(overeating_death=False); return p`. Call it after the initial `load_env_params` (line 454) AND inside the transition block (line 1024) AND inside the validation loop (line 468). |
| 🟡 | `train.py:1024-1027` | **Asymmetric agent-state handling at transition.** Plan explicitly states "Model, optimizer, RNG key, hidden state... persist across the transition", but Dreamer's replay buffer IS cleared (lines 1043-1054) to prevent "cross-stage dynamics contamination of the world model." The same contamination logic applies to the **agent's recurrent state**: PPO's `h_state` and Dreamer's `dreamer_state` (including `dreamer_state['is_first']`) carry across the boundary unchanged. After `env.reset(reset_key, num_envs)` the new env reports no `done`, so PPO's `_h_reset_on_done` (`recurrent_ppo_trainer.py:91-99`) is never triggered, and Dreamer's `dreamer_state['is_first']` keeps whatever value it had at end of last iteration's scan (`dreamer_v3_trainer.py:605`) — **not** `1.0` as it should be for a fresh-start observation. Either accept this asymmetry (and explain in the plan why buffer-clear matters but recurrent-state-clear doesn't) or also reset hidden state and force `is_first := 1` on all envs at transition. | Decision needed. Minimum-impact fix: add `h_state = model.initial_state(num_envs)` (PPO) / `dreamer_state['is_first'] = jnp.ones((num_envs, 1))` and `dreamer_state['prev_action'] = jnp.zeros(...)` (Dreamer) inside the transition block. |
| 🟡 | `train.py:1239-1243`, `train.py:1505` | **WandB step-axis inconsistency for `stage/index`.** `wandb.define_metric("stage/index", step_metric="Episode/Number")` declares `Episode/Number` as the step axis, but the iteration-log dicts at `train.py:1239` (PPO) and `train.py:1505` (Dreamer) include `**_stage_tag()` (which contains `stage/index`) without `Episode/Number`. WandB will fall back to the auto step counter for those `stage/index` points, producing a non-monotone series when plotted against episode. The Episode-log dicts (`train.py:1169`, `train.py:1426`) include both, so they're fine. | Remove `**_stage_tag()` from the iteration-log dicts (it's redundant — Episode log already carries it), or add `"Episode/Number": total_episodes_completed` to the iteration-log dicts (preferred, since iteration logs already need the episode tag for stage filtering). |
| 🟡 | `train.py:1011` (via `ContinualSchedule.stage_for_episode`) | The check fires once per training iteration. Drift is documented and accepted. Just confirm: `total_episodes_completed` is a Python int (incremented by `+= num_completed` at PPO/Dreamer accumulator updates), so the comparison `new_stage != current_stage` stays Python-side and never traces into JIT. ✅ Verified — no JIT recompilation hazard from the gate itself. | None — just confirming. |
| 🟢 | `train.py:925`, `train.py:980` | After `restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())`, `restored['stage']` is a **NumPy 0-d array** (`np.int64`), not a Python `int`. Subsequent assignment `current_stage = new_stage` (Python int from `stage_for_episode`) re-types it back to int. List indexing `schedule.stage_names[current_stage]` works for numpy scalars, so no crash, but the type is briefly heterogeneous. Pre-existing pattern (`iteration`, `global_step`, `total_episodes_completed` have the same drift), so not a regression. | Optional: cast on restore: `current_stage = int(restored.get('stage', 0))`. |
| 🟢 | `train.py:1024-1025` | `env = ParallelEnv(params)` rebuilds the wrapper, but `env.step` is **never called from `train.py`** — both `jit_train` (`recurrent_ppo_trainer.py:255`) and `collect_sequence` (`dreamer_v3_trainer.py:548`) re-vmap `jax_step` over `params` internally. So the new `env` instance is used **only for its `.reset` method** at the transition. This is correct, but slightly misleading: a future maintainer might think `env` participates in the hot loop. | Optional: comment `# `env` is used only for `.reset()` here; the per-iteration step is JIT'd inside jit_train / collect_sequence and takes `params` directly.` |
| 🟢 | `train.py:1027` | `jax.random.split(key)` at the transition: one split, returns `(new_key, reset_key)`. `reset_key` is consumed by `env.reset`; `key` advances forward to the next iteration. Inside `env.reset`, `jax.random.split(reset_key, num_envs)` produces unique sub-keys per env (`wrapper.py:18`). No reuse, no per-env desync. ✅ | None. |
| 🟢 | `dreamer_v3_trainer.py:875` | `class ReplayBuffer:` is a **plain Python class**, not `@struct.dataclass` and not `nnx.Module`. Therefore `buffer.idx = 0; buffer.size = 0` (and the same on `positive_buffer`) at `train.py:1043-1050` mutates the actual object in place — exactly what the plan needs. `sample()` gates on `self.size <= self.sequence_length` (`dreamer_v3_trainer.py:941`), so old array contents are unreachable until refilled. ✅ Confirmed not a silent-pytree-copy bug. The big GPU arrays (`buffer.obs` etc.) stay allocated, which is correct (they'll be overwritten in place by `add_batch`'s scatter). | None. |
| 🟢 | `train.py:1031-1036` | `episode_returns`, `episode_lengths`, `episode_behavior[k]`, `episode_dist_sums[k]` are all **NumPy** arrays (`np.zeros` at `train.py:885,886,894,895`). `[:] = 0.0` is correct in-place semantics for NumPy. ✅ | None. |
| 🟢 | `train.py:466-471` | The action-dim check `4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)` mirrors the live computation at `train.py:636`. Catches sensor-toggling-induced obs_dim drift cleanly at startup. ✅ | None. |
| 🟢 | `train.py:1010` | `if schedule is not None:` Python-side guard, `stage_for_episode` is a Python list scan, `new_stage != current_stage` is Python int comparison — none of this enters a JIT trace. ✅ No recompilation hazard from the gate. | None. |

---

## Conventions Audit

| Convention | Status | Notes |
|---|:---:|---|
| Pytree immutability (`@struct.dataclass` not mutated) | ✅ | `EnvParams`/`EnvState` only flow via `params = ...` rebind or `state.replace(...)`. `ReplayBuffer` is plain Python, mutation is intentional and safe. |
| JIT recompilation triggers (static fields, no Python control flow in JIT'd code) | ✅ | New params trigger recompile of `jit_train`/`collect_sequence` — accepted cost per plan. Per-iteration gate is Python-only. |
| vmap conventions (`EnvParams` broadcast not batched) | ✅ | Validation probes call `env_i.reset(probe_key, 1)` — uses `_v_reset` with `in_axes=(None, 0)`, broadcasting params correctly. No vmap regression. |
| PRNG key threading (no reuse, main key advances) | ✅ | One `split` at transition, sub-key consumed by reset, main key carried forward into the next iteration. Consistent with existing reset semantics at `train.py:628`. |
| Sensor / observation breakdown sync | N/A | No new sensors added; the validation probe at `train.py:466-471` will catch `obs_dim` drift across stages, which is the right guard. |
| Configuration protocol (`get_mandatory`, no silent defaults) | 🟡 | `_build_continual_schedule` correctly uses `get_mandatory("continual.episode_boundaries")`. **One silent default**: CLI overrides `--no-satiation` / `--no-overeating-death` are not propagated to per-stage params (see 🟡 finding above). |

---

## Conclusion

**WARNINGS.** No blockers; the patch is structurally sound and does not commit any of the JAX-pytree-mutation footguns flagged in the prompt. The single most important finding is the **silent revert of `--no-satiation` / `--no-overeating-death` at stage transitions** (`train.py:1024`, with collateral at `train.py:468`), because it changes env semantics without any error. Also recommend a decision on the asymmetric agent-state handling (Dreamer buffer cleared but Dreamer/PPO recurrent state preserved). The WandB step-axis nit is cosmetic.

**Reviewed by:** code-reviewer

---

## Re-Review (2026-05-06)

**Reviewer:** code-reviewer
**Scope:** verify the developer's fixes for the two correctness bugs flagged above plus the auxiliary Fixes 2/3/5.
**Verdict:** **FAIL on Fix 1.** Fixes 2, 3, 4, 5 PASS.

### Fix 1 — CLI override re-application — 🔴 **FAIL (does not fix the bug)**

The developer added a loop at `train.py:333-342`:

```python
if args.no_satiation:
    for _sc in schedule.stage_configs:
        _sc.set('environment.with_satiation', False)
if args.no_overeating_death:
    for _sc in schedule.stage_configs:
        _sc.set('environment.overeating_death', False)
```

**This is at the wrong key path.** `load_env_params` (`src/environment/config_loader.py:370,372`) reads:

```python
overeating_death=config.get_mandatory('body.overeating_death'),
with_satiation=config.get_mandatory('body.with_satiation'),
```

The override sets `environment.with_satiation` / `environment.overeating_death`, while the loader consumes from `body.with_satiation` / `body.overeating_death`. `Config.set` and `Config.get` use literal dotted-path indexing into the YAML tree (`src/utils/config.py:27-35,16-25`), so the override silently lands in a never-read subtree.

Concretely, the bug behavior the original review flagged is **unchanged**:
- Stage 0 still works only because the redundant `params = params.replace(...)` at `train.py:474-475` patches the live `EnvParams` after `load_env_params` returns — but that patch operates on `params`, not on `schedule.stage_configs[i]`.
- At every stage transition `params = load_env_params(schedule.stage_configs[new_stage])` (`train.py:1083`) re-reads from `body.*`, picks up the YAML's value, and the CLI flag is silently lost.
- The startup probe loop at `train.py:508-526` likewise reads from `body.*` for every stage `> 0`, so a `--no-satiation`-vs-stage-YAML mismatch will not be caught at startup either.

A user running `--configs-dir X --no-satiation` still gets `with_satiation=True` after the first transition. This is **the same silent semantics flip as the original bug**, with the additional concern that a fix is now claimed to be in place — making it more likely to evade further review.

**Required fix:** change the path in both `set` calls to `body.with_satiation` and `body.overeating_death`. Optionally also propagate at the transition itself by extracting a `_apply_param_overrides(params)` helper that calls `.replace(...)` and apply it everywhere `load_env_params` is invoked (line 472, line 509 inside the probe loop, line 1083 inside the transition block) — defense in depth so a future renamed YAML key cannot silently regress this again.

### Fix 4 — Recurrent state reset — 🟢 PASS

- `model.initial_state(num_envs)` exists on `RecurrentPPONetwork` (`src/models/recurrent_ppo_network.py:367-384`) and returns the same zero pytree shape as the startup init at `train.py:766`. Pure `jnp.zeros`, no PRNG consumption — deterministic, no key desync.
- Dreamer reset at `train.py:1119-1124` is a verbatim copy of the startup init at `train.py:821-829`. Same shapes, same dtypes, same modulator branch. JIT cached graph for `collect_sequence` will be reused (no recompile, no vmap axis error).
- `is_first = jnp.ones((num_envs, 1))` is **not permanently latched**: `dreamer_v3_trainer.py:605` overwrites it from `done` after every step (`next_d_state['is_first'] = done[..., None].astype(jnp.float32)`), so on the second post-transition step it falls back to 0 unless the env terminates. Correct dynamics.
- `trainer.agent.ac.actor.net.layers[-1].out_features` for `prev_action` shape: same expression as startup, so no shape mismatch risk.

One minor note (🟢 nit): `model.initial_state(num_envs)` and `rssm.initial(num_envs)` allocate small fresh arrays at every transition — fine, but worth knowing if a future variant uses learned initial states. None today.

### Fix 2 — `boundaries[0] <= 0` guard — 🟢 PASS

`train.py:181-185` raises `ValueError` with a clear message when the first boundary is non-positive. Fires inside `_build_continual_schedule` before any model init.

### Fix 3 — Modality fingerprint — 🟢 PASS

`_modality_fingerprint(p)` at `train.py:483-499` covers visual (enabled, range, local-view size), olfactory (enabled, vector size), nociception (enabled, size, interoceptive flag), location, proprioception, injury/nutrition observability, and `sensor_range`. The cross-stage fingerprint check at `train.py:519-526` runs alongside the obs/action-dim probe and raises before `wandb.init`, so a same-total-dim modality swap is caught at startup. Adequate.

### Fix 5 — Algorithm guard — 🟢 PASS

`train.py:430-434` raises `ValueError` if `schedule is not None and algorithm not in ("RecurrentPPO", "DreamerV3")`. Fires before model init and before `wandb.init`. Correct location.

### Conventions Audit (delta only)

| Convention | Status | Notes |
|---|:---:|---|
| Pytree immutability | ✅ | Reset uses fresh objects (`model.initial_state`, `rssm.initial`); no in-place mutation of `EnvState`/`EnvParams`. |
| JIT recompilation triggers | ✅ | Reset arrays match training-startup shapes/dtypes; `jit_train` and `collect_sequence` reuse cached traces. |
| vmap conventions | ✅ | New `params` is broadcast (single object), not batched. |
| PRNG key threading | ✅ | One `jax.random.split` per transition; main key advances. `model.initial_state` consumes no RNG. |
| Sensor / observation breakdown sync | ✅ | Fingerprint check enforces cross-stage modality identity. |
| Configuration protocol (`get_mandatory`, no silent defaults) | 🔴 | **Regression of the original silent-default bug:** Fix 1 sets `environment.with_satiation` / `environment.overeating_death`, but the loader reads `body.with_satiation` / `body.overeating_death` — silent path mismatch. |

### Conclusion

**Blocker on Fix 1.** The wrong-path `set` call means the original silent correctness bug (`--no-satiation` / `--no-overeating-death` reverting at the first stage transition) is still present. The fact that the fix exists makes the bug *worse*, not better — the next reviewer will skim the new code, see "override applied to every stage", and trust it. Fixes 2, 3, 4, 5 are correct and can land independently.

**Reviewed by:** code-reviewer
