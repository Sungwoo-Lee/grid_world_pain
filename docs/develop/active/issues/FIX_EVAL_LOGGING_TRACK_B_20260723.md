---
title: "Track B — Eval/Logging fix plan (5 P1 bugs, 2026-07-23 diagnosis)"
topic: issues
status: active
created: 2026-07-23
last_updated: 2026-07-23
---

# Track B — Eval/Logging fix plan (5 P1 bugs from the 2026-07-23 diagnosis)

> **Status**: PLANNED
> **Opened**: 2026-07-23
> **Related**: [[review_full_diagnosis_20260723]] (P1 table rows 6–9 + the two dormant/latent rows) · [[findings_eval_tooling]] (F1–F4 full evidence) · [[findings_train_entry]] (F-wandb context) · [[KNOWN_BUGS]] rows for eval step-dir collision, parallel-eval undercount, and "zero"-noise non-enforcement

---

## Context

The 2026-07-23 full-codebase diagnosis found ten P1-level defects (things that produce wrong results in realistic use). This plan fixes the five that live in the **offline evaluation and experiment-logging** layer — the "Track B" cluster — none of which touch the learning math itself. In plain terms:

1. **Eval outputs from different training runs collide on disk.** When you evaluate a recurrent-PPO checkpoint by pointing at its step folder (`<run>/models/<step>`), the eval script mislabels the output folder as just `models` and drops the run's identity. Seven step folders from different runs already sit collided under `results/eval/models/` on disk; two runs evaluated at the same step would silently overwrite each other.
2. **Multi-environment eval undercounts the first episodes' survival by one step.** With more than one parallel eval environment and per-step stat recording turned off (the default), the very first episode in each environment slot is missing its step-0 seed entry — so its logged survival-steps count (the project's headline metric) is one short, and its saved video is missing the first frame. Later episodes are fine.
3. **The WandB "job type" facet is never actually set, and its default value is misspelled.** The config exposes a `job_type` knob (meant to tag runs as prod vs. debug) but the training entry point never passes it to WandB, so every run's job-type is unset; the fallback string in the config is literally `"defualt"`.
4. **(Latent) Stochastic eval sampling is degenerate.** If anyone selects the "stochastic" eval policy mode, every step of an episode samples its action with the *same* frozen random key — the draws are perfectly correlated, so "stochastic eval" is a near-deterministic lie. Dormant today because every config uses deterministic (argmax) eval.
5. **(Latent) The "evaluate with noise-free observations" knob is a silent lie.** Setting `eval_obs_noise: "zero"` passes validation and gets written into the run's saved metadata as if honored, but the rollout still feeds the policy full training-time sensory noise. No config sets it today, but the trap is armed.

Bugs 4 and 5 are **latent** (no current config exercises them). They are folded into this single plan **only because they co-locate in `scripts/eval/eval_rollout.py`** — fixing them here avoids a second round-trip through the same file. This is a deliberate scope note, **not** scope creep: the verifier should expect edits to bugs 4–5's lines in `eval_rollout.py` and should not flag them as out-of-plan.

**Out of scope** (other tracks / other agents): the config-loader silent-failure cluster (P1 #1–3), the GRU init confound (P1 #4), the MC window-edge units mismatch (P1 #5), the `05-sensory_noise_10x10.yaml` header bug (P1 #8), and the DQN/DRQN/PPO lifetime-cumulative-metrics bug (P1 #10). No learning-math or config-schema change is made here.

---

## Analysis

All five root causes were traced to exact lines during planning (code read on branch `v3.0`, 2026-07-23). Evidence is condensed here; full evidence in [[findings_eval_tooling]] F1–F4.

### Bug 1 — run_tag collapses for rPPO step-dir invocations (`eval_rollout.py:908`)

```python
# current (line 908):
run_tag = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.parent.parent.name
```

`ckpt_path = Path(args.checkpoint).resolve()`. The documented step-dir invocation (`--help`: "the CheckpointManager root or a `<root>/<step>` dir") for rPPO is `<run>/models/<step>`, created by `train.py:623` (`models_dir = results_dir/"models"`). For that path:

- `ckpt_path.name == "<step>"`, `ckpt_path.parent.name == "models"`, `ckpt_path.parent.parent.name == "<run>"`.
- The special-case only catches `"checkpoints"` (the Dreamer container dir name), so `run_tag == "models"` and `out_dir = <out_root>/models/<step>` — run identity lost.

Dreamer's container dir is `checkpoints`, already handled. The asymmetry is that rPPO's container dir (`models`) was never added to the special-case. Seven collided step dirs already exist under `results/eval/models/` (steps `819`, `10000003`, `10000021`, `10000022`, `1810043`, `3900028`, `8900007`).

### Bug 2 — parallel-eval first-generation off-by-one (`evaluation_core.py:565–581 vs 694–706`)

In `_run_parallel_env_eval` the **initial** step-0 seeding of the per-slot buffers is gated `if record_stats:` (line 565), but the **refill** seeding after an episode completes (lines 694–706) is unconditional, and the per-step appends (609–627) are unconditional. `episode_lengths.append(len(slot_rewards[i]) - 1)` (line 637) subtracts one assuming a step-0 sentinel entry is present.

With `record_stats=False` and `num_envs>1` (the default `testing.record_stats` is `False`, resolved at evaluation_core.py:276–278): the first `effective_num_envs` episodes have **no** step-0 entry, so their length is undercounted by exactly 1; refilled episodes are correct. `mean_length` (headline survival metric) is biased low by `effective_num_envs / num_episodes` steps. With `render_video=True`, those first-generation `.rec.gz` files also lack the initial frame (`actions[0]` is a real action instead of the `-1` sentinel).

The `train.py` in-training path is dormant for this (video pass uses `num_envs=1`; stats pass uses `record_stats=True`) — only the standalone `evaluation.py --num_envs>1` path is exposed.

### Bug 3 — `wandb.job_type` never wired; typo default (`configs/logger/wandb.yaml:7` + `train.py:668–699`)

`train.py:403` writes `config.set('wandb.job_type', args.wandb_job_type)` when the CLI flag is given, but `wandb_kwargs` (668–674) never reads `wandb.job_type` and never passes `job_type=` to `wandb.init` (699). The config default is `job_type: "defualt"` (misspelled). Sibling keys (`project`/`entity`/`group`) are read via `get_mandatory`. Net effect: every run's job-type facet is unset in WandB.

**Dreamer co-location check:** `dreamer_srl_main.py:853` also omits `job_type=` from its `wandb.init`, **but** the Dreamer entry point has no `--wandb-job-type` arg and does not load the `configs/logger/wandb.yaml` layer (it builds `wandb_kwargs` from argparse, hardcoding `entity="sungwoolee"`). Wiring job-type there would require adding a new CLI arg and a config source — that is **not trivially co-located**, so it is deferred (noted in File Changes as an explicit non-change). Only the typo and the `train.py` wiring are in scope.

### Bug 4 (latent) — stochastic eval reuses one frozen PRNG key (`eval_rollout.py:98, 201, 1077`)

`_run_episode` (line 98) and `_run_episode_with_recording` (line 201) pass the **episode-level** `rng_key` (which is also the `jax_reset` key, `PRNGKey(seeds[ep])`) unchanged into `policy_fn` on **every** loop iteration. `policy_fn` (1073–1081) forwards it as `key=key if not deterministic else None` into `get_action_and_value_nnx`, which does `jax.random.categorical(key, logits)` (recurrent_ppo_network.py). The key is never split per step, so with `eval_policy_mode: "stochastic"` every step samples with the identical key → perfectly correlated per-step draws. Dormant: all configs are deterministic and the `--batched` path refuses non-deterministic mode.

### Bug 5 (latent) — `eval_obs_noise` validated + echoed but never enforced (`eval_rollout.py:1364`)

`eval_obs_noise` ("training" | "zero" | "custom", validated at config_loader.py:218) is consumed nowhere in the rollout: `policy_fn` calls `get_observation(state, params)` (line 1074) which always applies the env's training noise. Yet `metadata.json` records `"eval_obs_noise": bm_cfg.eval_obs_noise` (line 1364) as if honored — a self-certifying wrong record for any value other than `"training"`.

**Decision (recorded):** *Fail loud, do not implement enforcement.* Honoring `"zero"` (and defining `"custom"`) would require threading `apply_noise=False` through **four** independent obs-computation code paths (legacy `_run_episode`, `_run_episode_with_recording`, `_run_episodes_batched`, and `evaluation_core`'s single- and parallel-env paths) — a multi-path change that violates the "surgical, no adjacent refactors" constraint of this plan and carries real regression risk in the batched path. Since **no config sets anything but `"training"`** today, the honest, minimal, project-idiomatic fix (fail loud, no fallback defaults) is to **hard-error at eval startup** on any `eval_obs_noise != "training"`. Full enforcement is left as a follow-up feature plan if a noise-free probe eval is ever actually needed. This mirrors the guidance in [[findings_eval_tooling]] F4 ("fail loudly on any value other than 'training'").

---

## Implementation Plan

### Design

Five independent, surgical edits. No shared helper is introduced (each fix is a few lines at one site). Order is irrelevant; grouping by file below. Every fix ships with a regression test that fails on pre-fix code (where feasible) and passes after.

### File Changes

#### `scripts/eval/eval_rollout.py` (line 908) — Bug 1: run_tag derivation

Derive `run_tag` by walking up past **either** known checkpoint-container dir name (`checkpoints` for Dreamer, `models` for rPPO), instead of special-casing only `checkpoints`.

**Derivation rule (recorded):** *If the checkpoint's immediate parent directory name is a known checkpoint-container name (`{"checkpoints", "models"}`), the run tag is the grandparent directory name (the run directory); otherwise the run tag is the parent directory name.* This leaves the CheckpointManager-root invocation form (`<run>/models`, where `parent.name == "<run>"`) unchanged, fixes the step-dir form (`<run>/models/<step>`), and preserves the existing Dreamer behavior.

```python
# BEFORE (line 908):
run_tag = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.parent.parent.name

# AFTER:
# Checkpoint-container dir names: "checkpoints" (Dreamer), "models" (rPPO, train.py:623).
# A step-dir invocation is <run>/<container>/<step>; the run identity is the grandparent.
_CKPT_CONTAINER_DIRNAMES = {"checkpoints", "models"}
run_tag = (
    ckpt_path.parent.parent.name
    if ckpt_path.parent.name in _CKPT_CONTAINER_DIRNAMES
    else ckpt_path.parent.name
)
```

**Do NOT migrate** the seven existing collided step dirs under `results/eval/models/` (steps `819`, `10000003`, `10000021`, `10000022`, `1810043`, `3900028`, `8900007`). They are **legacy artifacts of unknown provenance** — their run attribution is already lost and cannot be reconstructed safely. Leave them in place; any downstream analysis that consumed `results/eval/models/<step>` must treat that data as provenance-unknown (documented in the diagnosis F1).

#### `scripts/eval/eval_rollout.py` (lines 98, 201) — Bug 4: per-step PRNG split (latent)

Thread a fresh split key per step inside both episode loops, so stochastic sampling is decorrelated across steps. Deterministic mode is unaffected (key is unused when `deterministic=True`).

```python
# _run_episode — BEFORE (line 96-98):
    while step < max_steps and not done:
        obs = state  # obs = full state (policy sees sensory obs internally)
        action, carry = policy_fn(obs, carry, rng_key, deterministic=deterministic)

# _run_episode — AFTER:
    step_key = rng_key
    while step < max_steps and not done:
        obs = state  # obs = full state (policy sees sensory obs internally)
        step_key, act_key = jax.random.split(step_key)
        action, carry = policy_fn(obs, carry, act_key, deterministic=deterministic)
```

```python
# _run_episode_with_recording — BEFORE (line 200-201):
    while step < max_steps and not done:
        action, carry = policy_fn(state, carry, rng_key, deterministic=deterministic)

# _run_episode_with_recording — AFTER:
    step_key = rng_key
    while step < max_steps and not done:
        step_key, act_key = jax.random.split(step_key)
        action, carry = policy_fn(state, carry, act_key, deterministic=deterministic)
```

Note: `rng_key` must remain the `jax_reset` key at the top of each function (do not overwrite it) — the split is derived into a separate `step_key`/`act_key` so reset-world reproducibility is untouched.

#### `scripts/eval/eval_rollout.py` (near line 895, in `main()` before the rollout loop) — Bug 5: fail loud on unenforced eval_obs_noise (latent)

Add a hard guard immediately after `bm_cfg` is available and before episodes run (e.g. right after `params = load_env_params(config)` at line 895). Do NOT silently proceed.

```python
# AFTER (new guard, inserted after line 895 params load):
if bm_cfg.eval_obs_noise != "training":
    raise NotImplementedError(
        f"behavior_measures.eval_obs_noise={bm_cfg.eval_obs_noise!r} is not enforced by the "
        "rollout — the policy always sees the env's configured (training) noise. Only "
        "'training' is currently honored; 'zero'/'custom' would require threading "
        "apply_noise through every obs path (see docs/develop/active/issues/"
        "FIX_EVAL_LOGGING_TRACK_B_20260723.md, Bug 5). Set eval_obs_noise: 'training'."
    )
```

This makes `metadata.json` incapable of certifying a mode it did not honor. The metadata write at line 1364 is then always truthful (only `"training"` can reach it) — leave line 1364 as-is.

#### `configs/logger/wandb.yaml` (line 7) — Bug 3a: fix typo

```yaml
# BEFORE:
  job_type: "defualt"
# AFTER:
  job_type: "default"
```

#### `train.py` (lines 668–674) — Bug 3b: wire job_type into wandb.init

Read `wandb.job_type` from config (already set from the CLI flag at line 403 when provided; otherwise the YAML default) and pass it to `wandb.init`. Use `get_mandatory` for parity with the sibling keys, since `configs/logger/wandb.yaml` always provides the key.

```python
# BEFORE (lines 668-674):
        wandb_kwargs = {
            "project": args.wandb_project or config.get_mandatory('wandb.project'),
            "entity": args.wandb_entity or config.get_mandatory('wandb.entity'),
            "group": args.wandb_group or config.get_mandatory('wandb.group'),
            "name": args.wandb_name or tag,
            "reinit": True
        }

# AFTER:
        wandb_kwargs = {
            "project": args.wandb_project or config.get_mandatory('wandb.project'),
            "entity": args.wandb_entity or config.get_mandatory('wandb.entity'),
            "group": args.wandb_group or config.get_mandatory('wandb.group'),
            "job_type": config.get_mandatory('wandb.job_type'),
            "name": args.wandb_name or tag,
            "reinit": True
        }
```

(No change to `train.py:403` — the CLI-flag→config write already lands in `wandb.job_type`, which `get_mandatory` now reads.)

#### `src/utils/evaluation_core.py` (lines 565–581) — Bug 2: seed step-0 unconditionally

Remove the `if record_stats:` gate on the initial step-0 seeding so the first-generation episodes get the same step-0 sentinel entry the refill block (694–706) already writes unconditionally. Keep only the `record_true_obs` gate on the `slot_true_obs` append (mirroring the refill block at 704–706).

```python
# BEFORE (lines 565-581):
    if record_stats:
        for i in range(effective_num_envs):
            slot_states[i].append({
                'agent_pos': states.agent_pos[i], 'satiation': states.satiation[i], 'nutrition': states.nutrition[i],
                'injury_level': states.injury_level[i], 'rest_streak': states.rest_streak[i],
                'res_pos': states.res_pos[i], 'res_active': states.res_active[i],
                'animal_pos': states.animal_pos[i], 'obs_pos': states.obs_pos[i],
            })
            slot_infos[i].append({})
            slot_actions[i].append(-1)
            slot_rewards[i].append(0.0)
            slot_obs[i].append(obs[i])
            if record_true_obs:
                true_obs_i = get_observation(
                    jax.tree_util.tree_map(lambda x: x[i], states), params_ref, apply_noise=False
                )
                slot_true_obs[i].append(true_obs_i)

# AFTER (drop the `if record_stats:` wrapper; the loop body is now unconditional):
    for i in range(effective_num_envs):
        slot_states[i].append({
            'agent_pos': states.agent_pos[i], 'satiation': states.satiation[i], 'nutrition': states.nutrition[i],
            'injury_level': states.injury_level[i], 'rest_streak': states.rest_streak[i],
            'res_pos': states.res_pos[i], 'res_active': states.res_active[i],
            'animal_pos': states.animal_pos[i], 'obs_pos': states.obs_pos[i],
        })
        slot_infos[i].append({})
        slot_actions[i].append(-1)
        slot_rewards[i].append(0.0)
        slot_obs[i].append(obs[i])
        if record_true_obs:
            true_obs_i = get_observation(
                jax.tree_util.tree_map(lambda x: x[i], states), params_ref, apply_noise=False
            )
            slot_true_obs[i].append(true_obs_i)
```

This makes first-generation slots identical in structure to refilled slots; `episode_lengths.append(len(slot_rewards[i]) - 1)` (line 637) then subtracts the now-present sentinel for every episode.

### Docs / maintenance side-effects

- **No new file under `scripts/`, no rename/move** → `SCRIPTS_DEPENDENCY_MAP.md` needs **no** update (in-place edit of an existing script only).
- **No config-schema change** → `CONFIG_GUIDE.md` / `02_config_schema.md` need **no** update. `wandb.job_type` already exists in the schema; only its value (typo) and its consumption change. (Developer: sanity-check this assumption — if `wandb.job_type` is *not* currently documented in `02_config_schema.md`, add the one-line row in the same change; it is not a new key, so this is at most a doc-completeness fix.)
- After landing, ask **`bug-curator`** to flip the KNOWN_BUGS rows for these five bugs from OPEN/LATENT to FIXED with the fix commit. Do not hand-edit `KNOWN_BUGS.md`.

## Regression tests (one per bug — must fail pre-fix where feasible)

Add under `tests/` following the existing patterns (`tests/scripts/test_eval_rollout_online_replay.py` imports `eval_rollout as er` via `sys.path`; `tests/environment/test_behavior_measures.py` builds env params from an inline YAML via `load_env_params(Config.load_yaml(...))`).

1. **Bug 1 — run_tag derivation** (unit, pure): `tests/scripts/test_eval_rollout_run_tag.py::test_step_dir_invocation_preserves_run_identity`. Refactor the derivation into a tiny importable pure helper `_derive_run_tag(ckpt_path: Path) -> str` (or test the module-level constant + inline logic if kept inline — preferred: extract the 4-line logic into a named helper so it is unit-testable without a checkpoint). Assert: `_derive_run_tag(Path("/x/runA/models/8900007")) == "runA"`, `_derive_run_tag(Path("/x/runB/checkpoints/500")) == "runB"` (Dreamer unchanged), `_derive_run_tag(Path("/x/runC/models")) == "runC"` (root form unchanged), and a bare-dir form falls back to parent name. **Pre-fix:** the first assertion returns `"models"` → fails.

2. **Bug 2 — parallel-eval step-0 undercount** (integration, tiny env): `tests/scripts/test_parallel_eval_step0_seeding.py::test_first_generation_episodes_not_undercounted`. Build a tiny env from inline YAML with **death disabled** (`body.with_injury: false`, `body.with_satiation: false`) and small `environment.max_steps` (e.g. 8) so **every** episode truncates at exactly `max_steps`. Call `_run_parallel_env_eval` with `model=None` (random-action path), `effective_num_envs=2`, `num_episodes=4` (2 first-gen + 2 refilled), `record_stats=False`, `render_video=False`, into fresh `episode_lengths=[]`. Assert `all(L == max_steps for L in episode_lengths)`. **Pre-fix:** the first 2 lengths are `max_steps-1` → `min(episode_lengths) < max_steps` → fails. (If constructing `_run_parallel_env_eval`'s full arg list is heavy, the developer may instead assert the weaker-but-still-failing invariant that all four lengths are equal.)

3. **Bug 3 — job_type wiring** (unit, kwargs inspection): `tests/training/test_wandb_job_type_wiring.py::test_job_type_passed_to_wandb_init`. Monkeypatch `wandb.init` to capture `**kwargs`, run `train.py`'s wandb-init block (or the smallest callable that builds `wandb_kwargs`) with a config whose `wandb.job_type` is a sentinel string, and assert the captured kwargs contain `job_type == "<sentinel>"`. If invoking the block standalone is impractical, a lighter assertion: load `configs/logger/wandb.yaml`, assert `wandb.job_type == "default"` (typo gone) AND grep-assert that `train.py`'s `wandb_kwargs` literal contains a `"job_type"` key. **Pre-fix:** kwargs lack `job_type` (and/or config value is `"defualt"`) → fails.

4. **Bug 4 — per-step PRNG distinctness** (unit): `tests/scripts/test_eval_stochastic_key_distinct.py::test_policy_fn_receives_distinct_keys_per_step`. Pass a stub `policy_fn` into `_run_episode` (with a tiny env params, `deterministic=False`, `max_steps>=3`) that records every `key` it is handed; assert the recorded keys are pairwise distinct (`len({tuple(np.asarray(k)) for k in seen}) == len(seen)`). **Pre-fix:** all keys identical → set size 1 → fails.

5. **Bug 5 — eval_obs_noise enforcement** (unit / startup guard): `tests/scripts/test_eval_obs_noise_guard.py::test_non_training_noise_mode_raises`. Build a `bm_cfg` (SimpleNamespace or real `BehaviorMeasureCfg`) with `eval_obs_noise="zero"` and assert the startup guard raises `NotImplementedError`; assert `eval_obs_noise="training"` does not raise. Test at the granularity the developer wires the guard (either the `main()` guard via a thin invocation, or extract the guard into a checkable `_assert_eval_obs_noise_supported(bm_cfg)` helper — preferred for testability). **Pre-fix:** no raise → fails.

Run the full suite after: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/scripts tests/training tests/environment -q`.

## Checkpoints

- [ ] Bug 1: `_derive_run_tag` returns the run dir for `<run>/models/<step>`; existing Dreamer `<run>/checkpoints/<step>` still resolves to `<run>`; no existing `results/eval/models/` dir was moved or deleted.
- [ ] Bug 2: on a death-disabled tiny env with `record_stats=False`, `num_envs=2`, all four episode lengths equal `max_steps` (print them).
- [ ] Bug 3: a real (or dry-run) `train.py` wandb-init shows `job_type` present in the `wandb.init` kwargs; `configs/logger/wandb.yaml` no longer contains `"defualt"`.
- [ ] Bug 4: keys handed to the stub `policy_fn` across steps are pairwise distinct; deterministic mode (`deterministic=True`) still passes `key=None` and is byte-identical to pre-fix output on a fixed seed (no behavior change for the only exercised path).
- [ ] Bug 5: `eval_obs_noise="zero"` raises at startup before any episode runs; `"training"` runs normally.
- [ ] Each of the 5 new tests **fails on the pre-fix code** (stash the fix, run the test, confirm red) and **passes after**.

## Speed note

None of these edits touch a hot loop in a way that should change throughput: Bug 1/3/5 are one-time startup/label logic; Bug 2 adds one step-0 append per env slot (negligible); Bug 4 adds one `jax.random.split` per eval step (eval-only, not training). No training-speed measurement required — but the developer should confirm the reasoning holds and note it in the Implementation Report.

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- developer fills: what changed, deviations, test red→green evidence per bug, speed reasoning confirmation -->

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `scripts/eval/eval_rollout.py` | Bug 1 run_tag + Bug 4 PRNG + Bug 5 guard | | |
| `src/utils/evaluation_core.py` | Bug 2 step-0 seeding | | |
| `configs/logger/wandb.yaml` | Bug 3a typo | | |
| `train.py` | Bug 3b job_type wiring | | |
| tests (×5) | one regression test per bug | | |

**Conclusion**: [one-line summary]
