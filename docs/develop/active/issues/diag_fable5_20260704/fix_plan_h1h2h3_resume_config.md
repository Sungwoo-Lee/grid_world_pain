---
title: "Fix plan — WP-A: checkpoint resume + strict config load (H1, H2, H3)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-05
---

# Fix plan — WP-A: checkpoint resume + strict config load (H1, H2, H3)

> **Status**: PLANNED
> **Opened**: 2026-07-04
> **Related**: [[01_train_entry_config]] (Findings 1, 2, 5 — the source of record), [[00_combined_diagnosis]] §2 rows H1–H3
> **Implements**: work package WP-A of the Fable 5 re-diagnosis (user-approved 2026-07-04)

---

## Context

This plan fixes the three High-severity bugs the 2026-07-04 re-diagnosis found in the code that starts and resumes a training run (`train.py` and the low-level YAML loader `src/utils/config.py`):

1. **H1 — Resuming a RecurrentPPO run from a saved checkpoint never actually restores the weights.** The restore code compares the checkpoint against the live model using two textually different key formats, concludes every weight is missing, raises an error — and a blanket error-swallower then prints one line (nothing at all under `--quiet`) and **training silently continues from freshly random weights** while the operator believes it resumed. A second, independent defect on the same path breaks the optimizer restore even when the weights would match. Both were demonstrated empirically, not just read from the code.
2. **H2 — Resuming a multi-stage (continual-curriculum) run puts the agent back in the first stage's world** while all counters and logs claim it is in the later stage: restoring the saved stage number suppresses the very check that would have rebuilt the environment for that stage.
3. **H3 — A misspelled `--config` path silently trains on the default environment.** The YAML loader prints a one-line warning and returns an empty config when the file does not exist, so a typo'd experiment config vanishes and the run proceeds on defaults with no error — bypassing the project's strict no-fallback config rule.

All three are in the "operator believes X, the run did Y" family — the project's worst failure class. Evidence, empirical proof scripts, and full mechanism traces are in the diagnosis report [[01_train_entry_config]] (Findings 1, 2, 5).

**Registry status** (already cross-referenced by the diagnosis): H1 confirms and upgrades the latent KNOWN_BUGS row "Checkpoint restore may not map onto model" (memory `20260509_1536_train_py_checkpoint_restore_nnx_skew`); H2 and H3 are new. After this fix lands and is verified, ask `bug-curator` to update/record the three rows.

## Scope fence — ONLY H1, H2, H3

The same diagnosis report lists other findings in these same files. They are **explicitly OUT of scope** for this plan; do not touch them even where the diff is adjacent:

- Finding 3 — DreamerV3 `--lr` / `--hidden-size` persisted-but-not-applied inversion.
- Finding 4 — `--total-timesteps` accepted but ignored.
- Finding 6 — perceptual-noise typo silently means "no noise".
- Finding 7 — DreamerV3 checkpoint omissions (optimizer states, target critic).
- Finding 8 — `wandb.job_type` / `wandb.name` kwargs drift.
- Finding 9 — SIGINT save-on-exit, DQN/DRQN never checkpoint, and all other hygiene items.

If implementation surfaces a new, independent issue: note it in the Implementation Report and stop — do not expand scope.

---

## Analysis

### H1 — root cause and mechanism choice

**Where:** `train.py:1141-1200` (strict-match restore), `train.py:1184` (optimizer restore), `train.py:1198-1200` (blanket `except`), `train.py:2450-2460` (save side, unchanged by this plan).

Save writes `nnx.state(model, nnx.Param)` via `ocp.args.StandardSave` (`train.py:2475`). Restore reads it back with `ocp.args.PyTreeRestore()` (`train.py:1127`), which returns a **raw nested dict**. The strict-match block string-compares tree paths of that dict against live `nnx.state(model)`: under the installed flax/orbax versions the restored leaf path ends in `DictKey(key='value')` while the live one ends in `GetAttrKey(name='value')` — 100 % of leaves mismatch, the architecture-mismatch `ValueError` fires, and the blanket handler at `train.py:1198-1200` swallows it. Independently, `nnx.update(optimizer, restored['optimizer'])` at `train.py:1184` fails with `ValueError: Cannot set key '1' on immutable node of type tuple` (orbax serializes the optax-chain tuple as a dict with string integer keys).

Empirical proof: `tmp/20260704_diag_ckpt_restore_test.py` (44/44 path mismatches → `ValueError`, exactly as live code) and `tmp/20260704_diag_ckpt_restore_test2.py` (direct `nnx.update(model, raw_dict)` restores weights bit-exactly; optimizer `nnx.update` with the raw dict fails on the tuple). Regression test 1 below reuses this recipe.

**Chosen mechanism: restore-to-target** (`ocp.args.StandardRestore(item=<target tree built from live state>)`), the canonical flax-NNX + orbax pattern, over "direct `nnx.update` with the raw dict + manual assertions":

- Passing a target tree makes orbax reconstruct the **exact live pytree structure** — `nnx.State` objects for model and optimizer, real tuples inside the optax chain — so both `nnx.update(model, …)` and `nnx.update(optimizer, …)` work. This kills both H1 defects with one mechanism; the raw-dict route fixes only the model half and leaves the optimizer needing fragile string-int-key reconstruction (lexicographic `'10' < '2'` ordering hazard).
- Orbax validates structure, shape, and dtype against the target and **raises on any mismatch** — the strict-architecture check the ~60-line hand-rolled block was trying (and failing) to implement comes for free, correctly.
- **Backward compatible with existing checkpoints**: the on-disk payload is untouched (`StandardSave` of the same dict since the continual feature landed); only the read side changes. Accepted limitation: any ancient checkpoint saved before the `'stage'`/`'h_state'` keys existed in the payload will fail structure validation — acceptable, since H1 means no rPPO checkpoint has *ever* successfully resumed anyway.

**Fatality:** the blanket `try/except` is deleted. Any restore failure — missing directory, no checkpoint steps, structure/shape mismatch, corrupted arrays — must raise and kill the process. `latest_step() is None` becomes `FileNotFoundError` instead of a warning. This applies to the DreamerV3 branch too (its restore code is otherwise untouched — Finding 7 is out of scope, but its *failures* must no longer be swallowed).

The restore logic moves into a small new module `src/utils/checkpoint_restore.py` so the regression test can exercise the **exact production code path** without importing the heavy `train.py`.

### H2 — root cause and mechanism choice

**Where:** `train.py:1138-1139` and `1193-1194` (restore `current_stage`), `train.py:1224 ff.` (stage-transition check), `train.py:523` / `729` / `734` (env built once from stage 0 *before* restore).

The environment is constructed from stage 0's config before checkpoint restoration. Restore sets `current_stage = restored['stage']`. The per-iteration transition check only rebuilds when `schedule.stage_for_episode(total_episodes_completed) != current_stage` — after a resume these are already equal, so the rebuild never fires: stage-N counters and WandB tags on a stage-0 world.

**Chosen mechanism: unconditional post-restore rebuild block** (diagnosis "Suggested direction", option 1) rather than a `current_stage = -1` sentinel. The sentinel would make the existing transition block fire, but `-1` indexes `schedule.stage_names[-1]` / `episode_boundaries[-1]` in the transition print (wrong-but-plausible output) and would leak into `_stage_tag()` if anything logs before the first iteration — a trap for future readers. An explicit block right after restore is self-describing:

- `current_stage` is **derived from the schedule** (`schedule.stage_for_episode(total_episodes_completed)`), not trusted from the checkpoint — stage is a pure function of the episode count, which is exactly what the in-loop transition check uses, so counters, `_stage_tag()`, and the transition check stay mutually consistent by construction. A mismatch with the checkpoint's `'stage'` field is printed as a warning (kept as a cross-check).
- The env is rebuilt **unconditionally** (even for a stage-0 resume) — one code path, idempotent, resume is rare so the extra stage-0 rebuild cost is irrelevant.
- rPPO recurrent state is reset to `model.initial_state(num_envs)` — the rebuilt env is freshly reset, so episodes restart; pairing a restored mid-episode hidden state with fresh episodes would be wrong. This mirrors the existing stage-transition block (`train.py:1289-1291`).
- Nothing else needs wiping at this point: the block runs **before the training loop**, where episode accumulators are still zero (`train.py:988-989`), the Dreamer replay buffer is still empty, `dreamer_state` is still at its fresh init, and BM state is fresh — unlike the mid-loop transition block, which must wipe all of those.
- A `[RESUME] Stage …` marker line is printed (not gated on `--quiet` state used by tests); regression test 2 asserts on it.

### H3 — root cause and caller audit

**Where:** `src/utils/config.py:30-33` (`Config.load_yaml`: missing file → print warning, return empty config), reached from `train.py:381` via `load_env_config` → `_resolve_extends` → `Config.load_yaml` (`src/environment/config_loader.py:57`). No existence check on `args.config` anywhere in `train.py`; every mandatory key then resolves from `configs/environment/default.yaml`, so no `ValueError` ever fires.

**Chosen mechanism: raise `FileNotFoundError` inside `Config.load_yaml` itself** (diagnosis "Suggested direction", option 1) — closing the hole at the root protects *every* entry point (`train.py`, `evaluation.py`, `main.py`, eval/analysis scripts, the dreamer_srl pipeline) in one line, instead of guarding one call site in `train.py` and leaving the same soft-fail live everywhere else.

**Caller enumeration** (every `Config.load_yaml` call site in the main repo — excludes `.claude/worktrees/`, `legacy/`, `tmp/` — verified 2026-07-04). Legend: **guarded** = call is inside an `os.path.exists` / `Path.exists` check, so the optional-load semantics are preserved by construction; **tracked** = the path is a repo-tracked file that exists; **hard-fail desired** = user-supplied path where raising is exactly the wanted new behavior.

| Call site | Path loaded | Effect of the change |
|---|---|---|
| `src/utils/config.py:103` (`get_default_config`) | `configs/environment/default.yaml` | tracked — no change |
| `src/environment/config_loader.py:57` (`_resolve_extends`) | top-level env config + `extends:` bases | **this is the H3 bug path** — top-level path now raises; bases were already existence-checked at `config_loader.py:68-72` |
| `train.py:166` (schedule) | `--continual-schedule` YAML | hard-fail desired (today a missing file dies later on `get_mandatory` with a worse message) |
| `train.py:305, 320, 327, 335, 343` | train/rPPO/eval/logger/vis defaults | all guarded |
| `train.py:315, 389` | `--agent_config` | hard-fail desired (today: soft-empty, then dies on `agent.algorithm`) |
| `main.py:47, 53` | train/eval defaults | guarded |
| `main.py:58` | `--config` (legacy entry point) | hard-fail desired |
| `evaluation.py:138, 140` | eval/vis defaults | guarded |
| `evaluation.py:179` | `schedule.yaml` at ckpt root | reached only after a `'stage' in restored` continual check; file written by train.py — hard-fail desired if missing |
| `hyperparameter_search.py:243` | `configs/logger/wandb.yaml` | guarded |
| `scripts/eval/eval_rollout.py:443` | `schedule.yaml` | guarded (`.exists()` at line 436) |
| `scripts/eval/eval_rollout.py:625` | `--agent_config` | hard-fail desired |
| `scripts/eval/eval_rollout.py:631` | saved `config.yaml` | guarded |
| `scripts/eval/motif_cluster.py:221` | `--config` | hard-fail desired |
| `scripts/dreamer/dreamer_offline_wm_test.py:125, 371` | defaults / saved config | guarded |
| `scripts/dreamer/dreamer_offline_wm_test.py:133` | agent config | hard-fail desired |
| `scripts/dreamer/dreamer_srl_offline_wm_test.py:142, 159` | defaults / saved config | guarded |
| `scripts/dreamer/dreamer_srl_offline_wm_test.py:146` | agent config | hard-fail desired |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py:113, 455` | defaults | guarded |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py:153, 460` | schedule / agent config | hard-fail desired |
| `pytorch_agents/…/grid_world_pain.py:79`, `…/jax_vector_env.py:75` | caller-supplied `config_path` | hard-fail desired |
| `test_c4.py:9`, `test_predator_count.py:10,19,43` (root scratch tests) | `configs/environment/experiment/archive/labmeeting/basic-00-predator.yaml` | tracked (existence verified) — no change |
| `tests/**` (extends_layering, evaluation_model_rebuild, dreamer_srl suite, environment suite) | repo-tracked configs / tmp files the tests create | tracked/self-created — no change |

**Conclusion: no caller anywhere depends on the soft-return-empty behavior.** Every intentional optional load already guards with an existence check; every unguarded load is either a repo-tracked file or a user-supplied path where hard failure is the correct behavior. No call-site changes are required.

Because this changes the config **system's** behavior, the Maintenance Contract in [docs/environment/CONFIG_GUIDE.md](../../../../environment/CONFIG_GUIDE.md) applies: the guide must be updated in the same change (see File Changes). No file under `scripts/` is added/moved/renamed/deleted, so `SCRIPTS_DEPENDENCY_MAP.md` is untouched.

---

## Implementation Plan

### Design summary

| Bug | Mechanism | New/changed surface |
|---|---|---|
| H1 | Restore-to-target (`StandardRestore(item=live-state target)`) in a new helper module; delete strict-match block + blanket `except`; `latest_step() is None` → raise | `src/utils/checkpoint_restore.py` (new), `train.py:1114-1200` |
| H2 | Unconditional post-restore env rebuild from `schedule.stage_configs[stage_for_episode(episode)]`; stage derived from schedule, not checkpoint; rPPO h-state reset; `[RESUME]` marker | `train.py` insert after restore block (before current line 1202) |
| H3 | `Config.load_yaml` raises `FileNotFoundError` on missing path | `src/utils/config.py:30-33`, `docs/environment/CONFIG_GUIDE.md` |

No new config keys. No changes under `scripts/` or `configs/`.

### File Changes

#### 1. `src/utils/checkpoint_restore.py` (NEW)

```python
"""Fatal-on-failure checkpoint restore for train.py's RecurrentPPO resume path.

Fixes H1 of docs/develop/active/issues/diag_fable5_20260704/01_train_entry_config.md:
the old strict-match restore compared DictKey vs GetAttrKey path strings (0 leaves
ever matched) and a blanket except swallowed the failure. This module uses the
canonical flax-NNX + orbax restore-to-target pattern: orbax reconstructs the exact
live pytree structure (nnx.State, real optax tuples) and raises on any
structure/shape/dtype mismatch. Failure is ALWAYS fatal — never print-and-continue.
"""
import os

import jax
import orbax.checkpoint as ocp
from flax import nnx


def restore_rppo_training_state(load_path, model, optimizer, h_state, key, *, quiet=False):
    """Restore model/optimizer in place; return the restored counters.

    Args:
        load_path: CheckpointManager root (the run's ``models/`` dir).
        model, optimizer: live NNX model + optimizer — used as restore targets
            and updated in place via ``nnx.update``.
        h_state, key: live recurrent state and PRNG key — used only as structure
            templates for the restore target.
        quiet: suppress the success print.

    Returns:
        dict with keys ``h_state, key, step, iteration, episode, stage``
        (counters as saved; caller casts to int).

    Raises:
        FileNotFoundError: no checkpoint steps under ``load_path``.
        Whatever orbax raises on structure/shape/dtype mismatch or corruption.
    """
    mngr = ocp.CheckpointManager(os.path.abspath(load_path))
    step = mngr.latest_step()
    if step is None:
        raise FileNotFoundError(
            f"--load-checkpoint given but no checkpoint steps found under "
            f"{load_path!r} — refusing to silently train from scratch."
        )
    # Target mirrors the exact StandardSave payload written at train.py
    # checkpoint-save time (rPPO branch).
    target = {
        'model': nnx.state(model, nnx.Param),
        'optimizer': nnx.state(optimizer),
        'h_state': h_state,
        'key': key,
        'iteration': 0,
        'step': 0,
        'episode': 0,
        'stage': 0,
    }
    restored = mngr.restore(step, args=ocp.args.StandardRestore(item=target))
    nnx.update(model, restored['model'])
    nnx.update(optimizer, restored['optimizer'])
    if not quiet:
        n_leaves = len(jax.tree_util.tree_leaves(restored['model']))
        print(f"  -> Restored {n_leaves} model param leaves + optimizer state "
              f"(checkpoint step {step}).")
    return {k: restored[k] for k in
            ('h_state', 'key', 'step', 'iteration', 'episode', 'stage')}
```

Notes for the implementer:
- If the installed orbax version restores the scalar counters as 0-d arrays rather than ints, that is fine — the train.py call site casts with `int(...)`.
- Verify against a real checkpoint payload that `StandardRestore(item=...)` accepts plain-int template leaves; if it insists on array templates, wrap the four counters in `np.asarray(0)` in `target` — behavior otherwise identical. Record whichever variant was needed in the Implementation Report.

#### 2. `train.py` — restore block (current lines 1114–1200)

Replace the whole `if args.load_checkpoint:` block. Key changes: **no `try/except` at all**; DreamerV3 branch body unchanged except gaining the fatal no-step check; rPPO branch delegates to the new helper; the old `'model' in locals()` catch-all is replaced with an explicit else-raise (PPO/DQN/DRQN never save checkpoints — see diagnosis Finding 9 — so restoring into them was always dead/undefined).

```python
    # --- Checkpoint Restoration (Continual Learning / Transfer) ---
    # H1 fix (diag_fable5_20260704/01 Finding 1): restore-to-target, FATAL on
    # any failure. The old strict-match block never matched a single leaf and
    # a blanket except silently trained from random weights.
    if args.load_checkpoint:
        if args.debug: print(f"[DEBUG] Phase 6.5: Restoring Checkpoint from {args.load_checkpoint}...", flush=True)
        if not args.quiet:
            print(f"Restoring checkpoint from {args.load_checkpoint}...")

        if algorithm == "DreamerV3":
            restore_mngr = ocp.CheckpointManager(os.path.abspath(args.load_checkpoint))
            step = restore_mngr.latest_step()
            if step is None:
                raise FileNotFoundError(
                    f"--load-checkpoint given but no checkpoint steps found under "
                    f"{args.load_checkpoint!r} — refusing to silently train from scratch.")
            restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())
            nnx.update(trainer.agent.wm, restored['wm'])
            nnx.update(trainer.agent.ac.actor, restored['actor'])
            nnx.update(trainer.agent.ac.critic, restored['critic'])
            key = restored['key']
            global_step = restored['step']
            iteration = restored['iteration']
            total_episodes_completed = restored['episode']
            if schedule is not None:
                current_stage = restored.get('stage', 0)
            if not args.quiet: print(f"  -> DreamerV3 Model fully restored (Step: {step}).")
        elif algorithm == "RecurrentPPO":
            restored_meta = restore_rppo_training_state(
                args.load_checkpoint, model, optimizer, h_state, key,
                quiet=args.quiet)
            h_state = restored_meta['h_state']
            key = restored_meta['key']
            global_step = int(restored_meta['step'])
            iteration = int(restored_meta['iteration'])
            total_episodes_completed = int(restored_meta['episode'])
            if schedule is not None:
                current_stage = int(restored_meta['stage'])
        else:
            raise ValueError(
                f"--load-checkpoint is not supported for algorithm {algorithm!r} "
                f"(only RecurrentPPO and DreamerV3 save checkpoints).")
```

Add the import near the other `src.utils` imports at the top of `train.py`:

```python
from src.utils.checkpoint_restore import restore_rppo_training_state
```

The DreamerV3 direct-`nnx.update`-with-raw-dict style is empirically sound for model weights (proof script 2) and its payload omissions are Finding 7 — out of scope. Only its silent-failure and silent-no-step behaviors are corrected here.

#### 3. `train.py` — post-restore continual rebuild (insert immediately after the block above, i.e. before the current line 1202 `start_time = datetime.now()`)

```python
    # --- H2 fix (diag_fable5_20260704/01 Finding 2): continual resume must
    # rebuild the env for the restored stage. The env above was built from
    # stage 0 BEFORE restore, and restoring current_stage makes the in-loop
    # transition check compare equal — so without this block a stage-N resume
    # trains stage-N counters on a stage-0 world.
    if args.load_checkpoint and schedule is not None:
        resumed_stage = schedule.stage_for_episode(total_episodes_completed)
        if resumed_stage != current_stage and not args.quiet:
            print(f"[RESUME] Checkpoint 'stage' field ({current_stage}) != "
                  f"schedule-derived stage ({resumed_stage}) for "
                  f"ep={total_episodes_completed}; trusting the schedule.")
        current_stage = resumed_stage
        # Unconditional rebuild (idempotent for stage 0; resume is rare).
        params = load_env_params(schedule.stage_configs[current_stage])
        env = ParallelEnv(params)
        key, reset_key = jax.random.split(key)
        env_state, obs = env.reset(reset_key, num_envs)
        # Fresh env episodes -> fresh recurrent state (mirrors the in-loop
        # stage-transition block at the 'Fix 4' comment).
        if algorithm == "RecurrentPPO":
            h_state = model.initial_state(num_envs)
        if not args.quiet:
            print(f"[RESUME] Stage {current_stage}:"
                  f"{schedule.stage_names[current_stage]} environment rebuilt "
                  f"at ep={total_episodes_completed}.")
```

Consistency argument (why counters/tags stay correct):
- `current_stage` is now `stage_for_episode(total_episodes_completed)` — the same function the in-loop transition check uses, so the first loop iteration compares equal and does **not** fire a spurious transition; the next genuine boundary fires normally.
- `_stage_tag()` reads `current_stage` → WandB `stage/index` / `stage/name` now match the actual world.
- Episode accumulators (`episode_returns`, `episode_lengths`, behavior arrays), the Dreamer replay buffer, `dreamer_state`, and BM state are all still at their fresh pre-loop init at this point — the mid-loop transition block's wipes are unnecessary here, so this block intentionally omits them.
- The restored `h_state` is deliberately discarded on the continual path (episodes restart in the rebuilt env). Single-config resume keeps the restored `h_state` — existing behavior, untouched.
- The exact `[RESUME] Stage` marker string is load-bearing: regression test 2 asserts on it. Do not reword without updating the test.

#### 4. `src/utils/config.py` (lines 29–35) — H3

```python
# BEFORE:
    @classmethod
    def load_yaml(cls, path):
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found. Using defaults.")
            return cls()
        with open(path, 'r') as f:
            return cls(yaml.safe_load(f))

# AFTER:
    @classmethod
    def load_yaml(cls, path):
        if not os.path.exists(path):
            # Strict Config (H3, diag_fable5_20260704/01 Finding 5): a missing
            # config file must be a hard error, never a silent empty config —
            # a typo'd --config used to train silently on default.yaml.
            # Intentional optional loads must guard with os.path.exists at the
            # call site (all existing ones already do).
            raise FileNotFoundError(
                f"Config file not found: {path!r}. "
                f"Optional loads must check os.path.exists before calling load_yaml.")
        with open(path, 'r') as f:
            return cls(yaml.safe_load(f))
```

No call-site changes needed (see caller enumeration in Analysis).

#### 5. `docs/environment/CONFIG_GUIDE.md` — Maintenance Contract compliance

Add one short paragraph (in the loading/`extends:` section) documenting: `Config.load_yaml` raises `FileNotFoundError` on a missing path as of this change; optional loads must guard with `os.path.exists` at the call site. Check `docs/environment/02_config_schema.md` for any mention of the old warn-and-default behavior and update if present (grep for "Using defaults" / "not found").

#### 6. `tests/training/test_checkpoint_restore_roundtrip.py` (NEW) — regression test 1 (H1)

Reuses the recipe of the diagnosis proof scripts `tmp/20260704_diag_ckpt_restore_test.py` / `..._test2.py`: tiny `ActorCriticRNN` (`input_dim=10, action_dim=4, hidden_size=16`, LSTM, `observation_breakdown={"Olfaction": 5, "Collision": 4, "Satiation": 1}`, `encoding_config={'encoding_mode': 'flat', 'use_layer_norm': False}`), optimizer built exactly as `train.py` does (`optax.chain(clip_by_global_norm, adam)`, `wrt=nnx.Param`). Force CPU (`os.environ["JAX_PLATFORMS"] = "cpu"` before jax import).

- `test_rppo_restore_roundtrip_bit_exact` (tmp_path): build model A (seed 0) + optimizer + `h_state = model.initial_state(2)` + key + counters `{iteration: 7, step: 1234, episode: 42, stage: 1}`; save with `ocp.CheckpointManager` + `ocp.args.StandardSave` using the **exact payload dict shape from train.py's rPPO save branch**; take one optimizer step first so the optimizer state is non-trivial (non-zero Adam moments). Then build model B (seed 99) + fresh optimizer, call `restore_rppo_training_state(...)`, and assert:
  - every `nnx.Param` leaf of B is **bit-exact** equal to A's (`jax.tree_util.tree_map(jnp.array_equal, ...)` all true);
  - every optimizer-state leaf equal;
  - returned counters == saved counters; returned `h_state`/`key` equal to saved.
- `test_rppo_restore_architecture_mismatch_raises` : save from a `hidden_size=16` model, restore into a `hidden_size=32` model → must raise (pin the exact exception type orbax throws — expected `ValueError` or an orbax error class; record the observed type in the Implementation Report) and must **not** return normally.
- `test_rppo_restore_missing_checkpoint_raises`: `restore_rppo_training_state(<empty tmp dir>, ...)` → `pytest.raises(FileNotFoundError)`.

**Red→green contract:** pre-fix these fail with `ImportError` (`src.utils.checkpoint_restore` does not exist). The *behavioral* pre-fix proof of the bug is the two tmp diagnosis scripts, whose assertions this test replicates against the production helper.

#### 7. `tests/training/test_continual_resume_rebuild.py` (NEW) — regression test 2 (H2)

Copy the subprocess harness of `tests/training/test_continual_bm_transition.py` (same tiny 5×5 two-stage scene, same `_PYTHON` interpreter constant, `--no-wandb`, `tempfile` stage dir + `--results-dir`). Single test, two `train.py` subprocess runs:

1. **Run 1 (create the checkpoint):** 2-stage schedule, `episode_boundaries: [3, 8]`, `checkpoint_frequencies: [2, 2]`, `--num-envs 2`, `--seed 42`, `--quiet` off. Runs to completion (8 episodes); checkpoints land at episode-numbered steps ≥ boundary 3, i.e. some saved **while in stage 1**.
2. **Doctor the checkpoint dir** so the latest step is a **mid-stage-1** step: list the numeric step subdirs under `<results>/models/`, delete every step dir `> 4` (leaving latest = 4, which is in stage 1 since `4 >= boundaries[0]=3`). This forces the resume to land mid-stage-1 with episodes still remaining.
3. **Run 2 (resume):** same `--configs-dir` / `--continual-schedule` / agent config / seed, plus `--load-checkpoint <run1_results>/models` and a fresh `--results-dir`. Assert:
   - `returncode == 0` and no `Traceback` in output (pre-fix H1 alone would silently pass this — hence the marker assert);
   - stdout contains `[RESUME] Stage 1:` (the H2 marker — proves the env was rebuilt for stage 1, not left at stage 0);
   - stdout does **not** contain a `[STAGE] 1:` → `0:` back-transition or a spurious `[STAGE] 0:` transition line after the resume marker.

**Red→green contract:** pre-fix, the `[RESUME]` marker does not exist anywhere in `train.py` → the assert fails (and behaviorally, the run really is on the stage-0 world per Finding 2). Post-fix it passes. Mark with `@pytest.mark.slow` if the suite uses that convention (check `pyproject.toml` markers; the sibling BM-transition test is the precedent to follow).

#### 8. `tests/env/test_config_strict_load.py` (NEW) — regression test 3 (H3)

Co-located with the other config-loading tests (`tests/env/test_extends_layering.py`).

- `test_load_yaml_missing_path_raises`: `Config.load_yaml('/nonexistent/definitely_missing.yaml')` → `pytest.raises(FileNotFoundError)`. **Fails pre-fix** (currently returns an empty `Config` after a print — a genuine behavioral red).
- `test_load_env_config_missing_path_raises`: `load_env_config('/nonexistent/typo.yaml')` (the exact `train.py:381` path through `_resolve_extends`) → `pytest.raises(FileNotFoundError)`. **Fails pre-fix.**
- `test_load_yaml_existing_path_still_works`: `Config.load_yaml('configs/environment/default.yaml')` returns a non-empty config (guards against over-tightening).

### Summary of touched files

| File | Kind | Change |
|---|---|---|
| `src/utils/checkpoint_restore.py` | NEW | restore-to-target helper (H1) |
| `train.py` | edit | replace restore block (~1114–1200); insert post-restore rebuild block; add one import (H1+H2) |
| `src/utils/config.py` | edit | `load_yaml` raises on missing path (H3) |
| `docs/environment/CONFIG_GUIDE.md` | edit | document the hard-fail (Maintenance Contract) |
| `docs/environment/02_config_schema.md` | edit (conditional) | only if it documents the old soft-fail |
| `tests/training/test_checkpoint_restore_roundtrip.py` | NEW | regression test 1 |
| `tests/training/test_continual_resume_rebuild.py` | NEW | regression test 2 |
| `tests/env/test_config_strict_load.py` | NEW | regression test 3 |

**New config keys: none.** No files under `scripts/` or `configs/` are touched (→ no `SCRIPTS_DEPENDENCY_MAP.md` update needed).

### Test plan

Interpreter: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` (never system python, never `conda run`).

```bash
# The three new regression tests (run RED first on pre-fix code where applicable, then GREEN):
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
    tests/env/test_config_strict_load.py \
    tests/training/test_checkpoint_restore_roundtrip.py \
    tests/training/test_continual_resume_rebuild.py -v

# Neighborhood: config + training regression suites
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
    tests/env/ tests/training/ -v

# Full suite
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/ -x -q
```

Run test 3 (H3) **before** touching `src/utils/config.py` to record the genuine red; tests 1–2 are red pre-fix via ImportError / missing-marker as documented above.

**Known-red baseline (do NOT fix, not your breakage):** `tests/env/test_unified_parity.py` `observability_gates_S1`–`S4` are RED for an unrelated known reason (finding A1 of the same re-diagnosis). Leave them red; do not count them against this change. Any *other* newly red test is on this change and must be resolved.

**Speed check:** all three fixes are outside the training hot loop (one-time startup/restore code + a file-existence check at config-load time). No throughput impact is possible; a before/after speed run is not required. State this in the Implementation Report.

## Checkpoints

What the `developer` agent should verify **during** implementation:

- [x] 1. Record the pre-fix RED runs: test 3 fails behaviorally; tests 1–2 fail (ImportError / marker absent). Paste the failure lines into the Implementation Report. — done, see Implementation Report.
- [x] 2. After writing `checkpoint_restore.py`: run the round-trip test alone; confirm **bit-exact** weight equality (not allclose) and that the optimizer-state leaves (Adam moments) are non-zero and equal — proving the optimizer half of H1 is really fixed. — 3/3 pass; test asserts non-zero Adam moments + `jnp.array_equal` on every leaf.
- [x] 3. Confirm the exception type orbax raises on the hidden-size-mismatch case and pin it in the test; note it in the Implementation Report. — `builtins.ValueError`, "Requested shape … not compatible with the stored shape"; pinned with `pytest.raises(ValueError, match=...)`.
- [x] 4. After the train.py edit: `grep -n "except Exception" train.py` in the restore region — the blanket handler must be gone; `--load-checkpoint` with an empty dir must raise `FileNotFoundError` (quick manual check is fine). — no `except Exception` between lines 1050–1250; empty-dir raise covered by `test_rppo_restore_missing_checkpoint_raises` on the exact production helper.
- [x] 5. After the H2 block: run the continual-resume subprocess test; confirm stdout shows `[RESUME] Stage 1:` and that the first training iteration does **not** print a `[STAGE]` transition line. — test asserts both; PASSED.
- [x] 6. Verify old-checkpoint compatibility if any real rPPO checkpoint is handy under `results/JAX_RecurrentPPO/*/models/` (read-only restore smoke via the helper in a throwaway script); if none is available, note that and rely on test 1. — attempted (April run, step 100069): even the raw `PyTreeRestore` read fails on CPU ("sharding … should be specified, concrete"); checkpoint predates the current payload. Relying on test 1 per plan's accepted limitation.
- [x] 7. Full `tests/env/ + tests/training/` pass, modulo the known-red `test_unified_parity.py observability_gates_S1-S4` (unrelated, finding A1). — see Implementation Report: 3 additional reds are PRE-EXISTING (stale config paths from commit b093023), empirically shown red pre-fix too.
- [x] 8. CONFIG_GUIDE.md paragraph added; `grep -rn "Using defaults" docs/environment/` returns nothing stale. — paragraph added in §1; grep clean.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-05 (session started 2026-07-04)

### What was implemented (file-by-file)

| File | Change |
|---|---|
| `src/utils/checkpoint_restore.py` (NEW) | Restore-to-target helper exactly per File Change 1. **Plain-int counter templates work** with the installed orbax 0.11.33 — the `np.asarray(0)` variant was NOT needed. Counters come back as Python ints. |
| `train.py` | (a) Added `from src.utils.checkpoint_restore import restore_rppo_training_state` next to the other `src.utils` imports. (b) Replaced the whole `if args.load_checkpoint:` block (old lines 1114–1200) per File Change 2: no `try/except`, DreamerV3 branch keeps `PyTreeRestore` + direct `nnx.update` but gains the fatal no-step `FileNotFoundError`; rPPO branch delegates to the helper; explicit else-raise for PPO/DQN/DRQN. (c) Inserted the H2 unconditional post-restore rebuild block before `start_time = datetime.now()` per File Change 3, verbatim (schedule-derived stage, unconditional env rebuild, rPPO h-state reset, `[RESUME] Stage k:` marker). |
| `src/utils/config.py` | `load_yaml` raises `FileNotFoundError` on missing path per File Change 4, verbatim. |
| `docs/environment/CONFIG_GUIDE.md` | Added a "**Missing config file = hard error**" paragraph in §1 (after the `load_env_config` chokepoint paragraph). |
| `docs/environment/02_config_schema.md` | NOT touched — grep for "Using defaults" found no mention of the old soft-fail (conditional change not needed). |
| `tests/training/test_checkpoint_restore_roundtrip.py` (NEW) | Regression test 1 (3 tests) per File Change 6. |
| `tests/training/test_continual_resume_rebuild.py` (NEW) | Regression test 2 (1 subprocess test, `@pytest.mark.slow` following the sibling BM-transition test) per File Change 7. Minor addition: the stage YAMLs set `training.video_during_training/stats_during_training: false` so the checkpoint saves don't trigger the video-eval pass (keeps the test <1 min; no effect on what is being tested). |
| `tests/env/test_config_strict_load.py` (NEW) | Regression test 3 (3 tests) per File Change 8. |

### Pre-fix RED evidence (recorded before any fix was applied)

- **Test 1** (H1): `ModuleNotFoundError: No module named 'src.utils.checkpoint_restore'` (ImportError red as planned; behavioral proof = the two tmp diagnosis scripts whose assertions the test replicates).
- **Test 2** (H2): `AssertionError: No '[RESUME] Stage 1:' marker — the post-restore rebuild did not fire` — and the captured subprocess output shows the resumed run training **8/8 episodes from scratch** (H1's silent restore no-op, live).
- **Test 3** (H3): `Failed: DID NOT RAISE <class 'FileNotFoundError'>` on both missing-path tests, with the old `Warning: Config file /nonexistent/... not found. Using defaults.` print captured — a genuine behavioral red.

### Post-fix results

- Test 1: **3/3 PASSED** (bit-exact `jnp.array_equal` on every Param leaf; optimizer-state leaves equal with non-zero Adam moments asserted non-trivial; counters/h_state/key round-trip).
- Test 2: **PASSED** (`[RESUME] Stage 1:` in stdout; no back-transition / spurious stage-0 line after the marker; rc=0).
- Test 3: **3/3 PASSED**.
- **Pinned mismatch exception type** (Checkpoint 3): orbax 0.11.33 raises `builtins.ValueError` — `"Requested shape: (32, 4) is not compatible with the stored shape: (16, 4). Truncating/padding is disabled…"`; pinned as `pytest.raises(ValueError, match="not compatible with the stored shape")`.

### Suite results

Commands (interpreter `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest`):
- `tests/env/ tests/training/`: **204 passed, 492 skipped, 7 failed** (14:57 min).
- Full `tests/`: **388 passed, 494 skipped, 8 failed** (29:32 min).

All 8 failures are NOT this change's breakage:

1. `tests/env/test_unified_parity.py observability_gates_S1–S4` (4) — the documented known-red baseline (finding A1).
2. `tests/env/test_truncation_not_death.py` (2) + `tests/env/test_inactive_animal_offgrid.py::test_allactive_config_no_offgrid_parking` (1) — **pre-existing reds surfaced with a clearer error**. These tests reference `basic/06-sensory_noise_10x10.yaml` and `basic/04-far_sight_predator_10x10.yaml`, which commit b093023 (basic-ladder re-leveling, 2026-07-02) renamed/retired. **Empirically verified pre-fix red**: with the old soft-fail `load_yaml` restored (probe `tmp/20260704_213000_h3_prefix_status_probe.py`), the same load path dies with `ValueError: Strict Config: Configuration key 'environment.resources' is required but missing` — i.e., they were already failing on HEAD before this change; H3 only changed the failure from a confusing ValueError to an explicit `FileNotFoundError` naming the missing file. **Follow-up (new independent issue, per scope fence: flagged, not fixed):** the two test files need their config paths updated to the re-leveled ladder (`05-sensory_noise_10x10.yaml`; a replacement for the retired far-sight config). Hand to `senior-developer` / `bug-curator`.
3. `tests/scripts/test_dreamer_srl_offline_wm_test.py::test_offline_wm_smoke` (1) — `RuntimeError: Only 36 valid starting states (< 50)`. **Empirically verified independent of this change**: fails identically (same error, deterministic) with the H3 fix temporarily reverted; its config paths all exist so neither H1/H2 (train.py-only) nor H3 can reach it. Pre-existing red — flagged as follow-up, not fixed.

### Checkpoint-6 note (old-checkpoint compatibility)

A real April rPPO checkpoint (`results/JAX_RecurrentPPO/20260420-152414_rppo_MC_basic-01/models`, step 100069) was probed read-only: even the raw `PyTreeRestore` read fails on CPU (`ValueError: sharding passed to deserialization should be specified, concrete…`) — the checkpoint predates the current payload/format. This falls under the plan's accepted limitation (no rPPO checkpoint has ever successfully resumed anyway); relying on regression test 1, which saves and restores through the exact production payload shape.

### Speed check

**Skipped, per the plan's own Test plan section**: all three fixes are one-time startup/restore code plus a file-existence check at config-load time — provably outside the training hot loop. No throughput impact is possible.

### Deviations from the plan

1. Test-2 stage YAMLs additionally set `training.video_during_training: false` / `stats_during_training: false` (test-speed only; documented above).
2. None otherwise — helper module, train.py blocks, and config.py change are verbatim from the plan (plain-int counter variant confirmed workable).

### Blockers / follow-ups

- No blockers. Tree left dirty (uncommitted) for verification; parallel WP-B files (`src/models/recurrent_ppo_trainer.py`, `tests/models/test_mc_window_bootstrap.py`) untouched.
- Follow-ups flagged above: stale test config paths after b093023 (3 tests), dreamer_srl offline-WM smoke red (1 test). After verification, ask `bug-curator` to update/record the H1/H2/H3 registry rows (per Context §Registry status) and optionally record the two follow-ups.

> Implemented by: developer

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-05

Verified independently against the plan: full `git diff HEAD` read, scope-fence spot checks, caller enumeration re-run from scratch, and the three regression tests re-executed by the verifier.

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/utils/checkpoint_restore.py` (NEW) | Restore-to-target helper (H1) | ✅ | Matches File Change 1 verbatim (plain-int counter variant, as the report records). Fatal `FileNotFoundError` on no-step; no exception swallowing anywhere in the module. |
| `train.py` — restore block | Blanket `except` deleted; DreamerV3 no-step now fatal; rPPO delegates to helper; explicit else-raise for PPO/DQN/DRQN | ✅ | Diff has exactly 2 hunks (import + lines 1113–1185). `grep except` in lines 1050–1250 finds only the comment mentioning the old blanket except. Remaining `except Exception` at 289/2532 are pre-existing, outside the restore region, out of scope. |
| `train.py` — H2 rebuild block | Unconditional post-restore rebuild; stage derived from `schedule.stage_for_episode`; env rebuilt from `stage_configs[current_stage]`; rPPO h-state reset; `[RESUME] Stage k:` marker | ✅ | Verbatim per File Change 3, inserted before `start_time = datetime.now()`. Runtime-exercised end-to-end by the subprocess regression test (real `train.py` resume). |
| `src/utils/config.py` | `load_yaml` raises `FileNotFoundError` on missing path (H3) | ✅ | Verbatim per File Change 4. |
| `docs/environment/CONFIG_GUIDE.md` | Hard-fail paragraph (Maintenance Contract) | ✅ | Added in §1; `grep -rn "Using defaults" docs/environment/` clean. `02_config_schema.md` untouched — correct, no stale mention existed (conditional change). |
| `tests/training/test_checkpoint_restore_roundtrip.py` (NEW) | Regression test 1 (H1) | ✅ | Asserts bit-exact `jnp.array_equal` on every Param leaf, optimizer-leaf equality **with an explicit non-zero-Adam-moment sanity assert**, counter/h_state/key round-trip, pinned `ValueError` on hidden-size mismatch, `FileNotFoundError` on empty dir. Includes an anti-vacuity assert (seed-0 vs seed-99 weights differ pre-restore). Save side replicates train.py's exact `StandardSave` payload. |
| `tests/training/test_continual_resume_rebuild.py` (NEW) | Regression test 2 (H2) | ✅ | Subprocess harness per File Change 7 (checkpoint-doctoring to force a mid-stage-1 resume; asserts `[RESUME] Stage 1:`, no back-transition, rc=0, no traceback). Deviation (stage YAMLs disable video/stats eval during training) is test-speed-only and documented — accepted. |
| `tests/env/test_config_strict_load.py` (NEW) | Regression test 3 (H3) | ✅ | Missing-path raise on both the direct and the `load_env_config` (`train.py --config`) path; existing-path non-empty guard against over-tightening. |

**Scope fence (out-of-scope Findings 3, 4, 6, 7, 8, 9)**: ✅ honored. The train.py diff contains only the two planned hunks; grep of the diff for `total-timesteps`, `--lr`/`hidden-size`, `job_type`, `SIGINT`/`signal`, and noise-mode strings returns nothing. No other `src/` file modified (WP-B files committed separately in 5488c98, not in this tree).

**H3 caller safety (re-derived independently)**: ✅. Re-ran the `load_yaml`/`load_env_config` call-site enumeration; it matches the plan's table plus four sites the table did not name explicitly — `scripts/verification/verify_noise.py:16` (tracked `default.yaml`), `scripts/media/record_env_demo.py:23` (guarded by `os.path.exists`), `scripts/eval/eval_rollout.py:535` (run-artifact/user path — hard-fail desired), `tests/environment/test_per_entity_info.py:23` (tracked archive config, existence verified). None relies on the old soft-return-empty behavior. Conclusion of the plan's audit stands.

**Regression-test red→green contract**: ✅ satisfied.
- Test 3's red is genuinely behavioral (`DID NOT RAISE FileNotFoundError` + captured old warning print).
- Test 2's red is genuinely behavioral **and doubles as a live behavioral red for H1**: the captured pre-fix subprocess output shows the resumed run training 8/8 episodes from scratch — the silent restore no-op, observed on the real `train.py` path.
- Test 1's `ModuleNotFoundError` red is exactly what the plan pre-registered (File Change 6), with the behavioral proof carried by the diagnosis scripts `tmp/20260704_diag_ckpt_restore_test.py` / `..._test2.py` (existence verified; their assertions are replicated in the production-path test). Combined with Test 2's behavioral H1 red, no stash-based re-derivation was needed.

**Test re-run by verifier**: ✅ all 7 tests green in 59s (`7 passed, 1 warning`). The one warning is `PytestUnknownMarkWarning` for `@pytest.mark.slow` — pre-existing pattern: the sibling `test_continual_bm_transition.py:166` uses the same unregistered mark, and `pyproject.toml` registers only `integration`. Cosmetic; consistent with the plan's "follow the sibling precedent" instruction. Optional future cleanup: register `slow` in `pyproject.toml` markers.

**Known-red baseline sanity check**: ✅ reasoning confirmed. `basic/06-sensory_noise_10x10.yaml` and `basic/04-far_sight_predator_10x10.yaml` (referenced by `test_truncation_not_death.py` and `test_inactive_animal_offgrid.py`) were retired by commit b093023 (2026-07-02, ladder now 00–05) — two days **before** this fix. Pre-fix, the soft-fail returned an empty config which then died on `get_mandatory('environment.resources')`, so those tests were already red on HEAD; H3 only renamed the error. The 4 × `test_unified_parity` reds (finding A1) and the dreamer_srl offline-WM smoke red are the documented unrelated baseline. Not counted against this package.

**Speed check**: ✅ no regression — skip justified. All three fixes are one-time startup/restore code plus a file-existence check at config-load time; no training-hot-loop code is touched (verified by the diff hunk locations). A before/after speed run would measure nothing.

**Out-of-plan files in the tree** (not verification failures): `train_command-agent.sh` (+67 lines — an appended launch-audit block for the re-leveled 6-level basic-ladder runs of 2026-07-04, run 00 active + runs 01–05 commented; a `training-runner` artifact unrelated to this package), `docs/develop/INDEX.md` (auto-regen), `docs/develop/active/issues/diag_fable5_20260704/01_train_entry_config.md` (+3-line cross-link to this plan), diary files.

**Follow-ups (inherited from the Implementation Report, endorsed)**: (1) ask `bug-curator` to update/record the H1/H2/H3 registry rows; (2) new plan to repoint the two stale-config test files to the re-leveled ladder; (3) dreamer_srl offline-WM smoke red (36-valid-states) — independent pre-existing issue.

**Conclusion**: PASS — implementation matches the plan exactly (one documented, accepted test-speed deviation), the scope fence held, all three regression tests are meaningful and green, and no unexplained changes exist in the tree. Ready to commit.

> Verified by: senior-developer
