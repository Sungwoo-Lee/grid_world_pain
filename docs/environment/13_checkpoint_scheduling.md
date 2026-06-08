# Checkpoint Scheduling

> **Status**: Description only — current behavior is **accepted as-is** per user direction (no change prescribed).
> **Primary sources**: `train.py:2387–2430` (RecurrentPPO / DreamerV3 path) | `src/algorithms/dreamer_srl/dreamer_srl_main.py:816–843` (dreamer-srl path) | `src/algorithms/dreamer_srl/checkpoint.py` (dreamer-srl helper module)
> **Last updated**: 2026-06-08

This document describes how checkpoint saving is scheduled across the two active training entry points — `train.py` (for RecurrentPPO and DreamerV3) and `dreamer_srl_main.py` (for the dreamer-srl variant) — and how the two paths differ in gate logic, state payload, and resume behaviour. It exists to make the observed drift predictable and to serve as a reference for the continual-learning plan ([`docs/develop/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)), which inherits the same drift for its stage-transition gate. **No scheduler change is recommended or planned.** The "Alternative Schemes" appendix is retained as a reference for future reviewers only.

---

## TL;DR — Plain-language entry point

Checkpoints are not written at clean, round-number milestones. Instead, the training loop checks whether enough episodes have completed **at the end of each iteration**, where one iteration covers all `num_envs` parallel environments for one rollout (RecurrentPPO) or one env-step batch (dreamer-srl). Because many episodes can finish in a single iteration, the actual save point *overshoots* the intended milestone — you set `checkpoint_frequency: 100` and see save directories named `156/`, `224/`, `312/` instead of `100/`, `200/`, `300/`.

This is called **gate drift**: the "gate" that triggers a save is evaluated coarsely (once per training iteration), not finely (once per completed episode). Under `N` parallel environments, a single iteration can complete up to `N` episodes, so the gate can miss the exact milestone by up to `N − 1` episodes in the best case.

A second effect is **milestone collapse**: when the number of episodes per iteration equals or exceeds the checkpoint interval (possible when `num_envs ≥ checkpoint_frequency`), one gate-crossing can represent *multiple* nominal milestones. Only one checkpoint is written, and the intervening milestones simply never appear on disk.

A third effect is **silent rotation**: Orbax keeps only the most recent `max_checkpoints_to_keep` checkpoints. Once that limit is reached, the oldest save is deleted. Early checkpoints that were correctly written disappear without warning.

None of these three phenomena are JAX or Orbax quirks. They are mechanical consequences of the per-iteration gate design, which exists because halting the JIT-compiled training loop at every individual `done` flag would break GPU throughput. Both training paths accept this behaviour; the continual-learning stage-transition gate uses the same granularity.

**Why this is called "gate drift under vmap":** the RecurrentPPO path runs the entire rollout through `jit_train` (an NNX-jitted function, `train.py:778`), which internally vmaps the environment step over all `num_envs` environments simultaneously. The checkpoint gate is evaluated *after* `jit_train` returns — there is no opportunity to stop mid-iteration when a milestone is crossed. The dreamer-srl path does not use `jit_train`; it steps one env at a time in Python. But it too only checks the gate inside the `if dones_idxes:` branch (`dreamer_srl_main.py:713`), meaning the gate fires only after at least one episode ends, and then only once per single-env done-processing block, not per episode. The per-iteration-granularity label applies to both paths.

---

## How Checkpointing Is Implemented Today

### Path A — `train.py` (RecurrentPPO and DreamerV3)

#### Setup

`train.py:550–559` creates the Orbax `CheckpointManager`:

```python
# train.py:550–559
max_checkpoints = config.get('training.max_checkpoints_to_keep', _missing)
if max_checkpoints is _missing:
    raise ValueError("Strict Config: ... 'training.max_checkpoints_to_keep' is required ...")
checkpointer = ocp.CheckpointManager(
    os.path.abspath(models_dir),
    checkpointers=ocp.StandardCheckpointer(),
    options=ocp.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
)
```

`models_dir` is `<results_dir>/models/`. Checkpoint subdirectories are written directly under `models_dir` as `<models_dir>/<episode>/`.

#### Gate

```python
# train.py:2387–2401
if schedule is not None:
    checkpoint_freq = schedule.checkpoint_frequencies[current_stage]
else:
    checkpoint_freq = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')

if not hasattr(main, 'last_checkpoint_save'):
    main.last_checkpoint_save = 0

should_checkpoint = (total_episodes_completed >= main.last_checkpoint_save + checkpoint_freq)

if should_checkpoint:
    main.last_checkpoint_save = (total_episodes_completed // checkpoint_freq) * checkpoint_freq
```

Key facts:
- **`last_checkpoint_save`** is stored as a function attribute on the `main` object. It is initialised to `0` on the first call within a Python process and is **not reset between training runs in the same process** (see Concern A below).
- **Gate fires** when `total_episodes_completed ≥ last_checkpoint_save + checkpoint_freq`.
- **Cursor update** snaps `last_checkpoint_save` **down** to the nearest multiple of `checkpoint_freq` below the current count: `floor(total / freq) × freq`. This means the gate can fire again immediately in the next iteration if that iteration also crosses a multiple — but in practice only one multiple is crossed per gate-crossing iteration (Regime 1 below), unless `num_envs ≥ checkpoint_freq` (Regime 2).

#### Save payload — RecurrentPPO

```python
# train.py:2405–2414
ckpt_data = {
    'model':     nnx.state(model, nnx.Param),
    'optimizer': nnx.state(optimizer),
    'h_state':   h_state,
    'key':       key,
    'iteration': iteration,
    'step':      global_step,
    'episode':   total_episodes_completed,
    'stage':     current_stage,
}
```

#### Save payload — DreamerV3 (via train.py)

```python
# train.py:2416–2425
ckpt_data = {
    'wm':        nnx.state(trainer.agent.wm, nnx.Param),
    'actor':     nnx.state(trainer.agent.ac.actor, nnx.Param),
    'critic':    nnx.state(trainer.agent.ac.critic, nnx.Param),
    'key':       key,
    'iteration': iteration,
    'step':      global_step,
    'episode':   total_episodes_completed,
    'stage':     current_stage,
}
```

Note: DreamerV3 (via `train.py`) does **not** save `target_critic`, `moments`, `policy_step`, or `cumulative_grad_steps`. These fields exist only in the dreamer-srl checkpoint (see Path B below).

#### Step key

```python
# train.py:2429
checkpointer.save(total_episodes_completed, args=ocp.args.StandardSave(ckpt_data))
checkpointer.wait_until_finished()
```

The Orbax step key equals `total_episodes_completed` — the *actual* cumulative episode count at the end of that iteration, not the nominal milestone.

---

### Path B — `dreamer_srl_main.py` (dreamer-srl variant)

#### Setup

`dreamer_srl_main.py:295–296` reads both checkpoint config keys as mandatory:

```python
checkpoint_frequency = env_cfg.get_mandatory('training.checkpoint_frequency', int)
max_checkpoints_keep = env_cfg.get_mandatory('training.max_checkpoints_to_keep', int)
```

`dreamer_srl_main.py:498–500` constructs the manager via the helper module:

```python
from src.algorithms.dreamer_srl.checkpoint import make_checkpoint_manager, save_checkpoint as _save_checkpoint
_ckpt_manager = make_checkpoint_manager(results_dir, max_to_keep=max_checkpoints_keep)
```

`checkpoint.py:34` writes checkpoints under `<results_dir>/checkpoints/` (not `models/`) — the directory convention differs from Path A.

#### Gate

```python
# dreamer_srl_main.py:822–825
_just_saved_ckpt = False
if (total_episodes_completed > 0 and
        total_episodes_completed // checkpoint_frequency >
        last_ckpt_episode // checkpoint_frequency):
```

This is a **different gate formula** from Path A. Rather than `total >= last + freq`, it uses integer division: fire if `floor(total / freq) > floor(last / freq)`. Behaviourally these are equivalent in Regime 1 (one milestone per iteration), but the dreamer-srl gate fires **at most once per episode completion block** (it lives inside the `if dones_idxes:` block at `dreamer_srl_main.py:713`) rather than at the end of a full multi-step rollout.

The cursor:

```python
# dreamer_srl_main.py:841
last_ckpt_episode = total_episodes_completed
```

Unlike Path A, the dreamer-srl cursor stores the *actual* episode count (not floored to a multiple of freq). This means the next gate condition `floor(actual / freq) > floor(actual / freq)` is `False` until the counter reaches the next multiple — equivalent behaviour, different storage.

`last_ckpt_episode` is a plain local variable initialised to `0` at `dreamer_srl_main.py:494`, so it does **not** leak across Python process calls (no function-attribute singleton — Concern A from Path A does not apply here).

#### Save payload — dreamer-srl

`checkpoint.py:77–101`:

```python
ckpt_data = {
    'world_model':              nnx.state(world_model, nnx.Param),
    'actor':                    nnx.state(actor, nnx.Param),
    'critic':                   nnx.state(critic, nnx.Param),
    'target_critic':            nnx.state(target_critic, nnx.Param),
    'key':                      key,
    'iter_num':                 jnp.array(iter_num, dtype=jnp.int32),
    'policy_step':              jnp.array(policy_step, dtype=jnp.int32),
    'total_episodes_completed': jnp.array(total_episodes_completed, dtype=jnp.int32),
    'cumulative_grad_steps':    jnp.array(cumulative_grad_steps, dtype=jnp.int32),
    'moments':                  <dict of JAX arrays>,
}
```

Extra fields versus Path A: `target_critic`, `policy_step`, `cumulative_grad_steps`, `moments`. Missing versus Path A: `stage` (dreamer-srl does not currently support continual-learning stage tracking), `optimizer` state (optimiser state is **not** saved — see Suspected Bug 1 below).

The step key equals `total_episodes_completed` (same convention as Path A):

```python
# checkpoint.py:101
manager.save(episode, args=ocp.args.StandardSave(ckpt_data))
manager.wait_until_finished()
```

---

## Episode-Counter Advancement

### RecurrentPPO (train.py)

```python
# train.py:1327–1353  (inner loop over rollout steps)
for t in range(num_steps):
    ...
    dones_t = done_np[t].astype(bool)
    if np.any(dones_t):
        completed_indices = np.where(dones_t)[0]
        for i in completed_indices:
            total_episodes_completed += 1
```

One iteration covers `num_steps × num_envs` environment transitions. The counter advances for every `done` across all timesteps in the rollout. In typical usage (episode length much greater than `num_steps`), most iterations yield 0 dones; occasional iterations yield up to `num_envs` dones. The checkpoint gate is checked **after** this inner loop at `train.py:2395`.

### DreamerV3 (train.py)

```python
# train.py:1625–1657
done_indices = np.where(done_steps)   # done_steps shape = [T, B]
for i in np.unique(done_indices[1]):
    for d_idx in d_idxs:
        total_episodes_completed += 1
```

DreamerV3 can count multiple dones per env per iteration (if an env completes more than one episode within the `num_steps` window). The gate fires once per iteration; milestone collapse risk is the same.

### dreamer-srl (dreamer_srl_main.py)

```python
# dreamer_srl_main.py:719–722
for i in dones_idxes:
    ep_len = int(episode_lengths[i]) + 1
    total_episodes_completed += 1
```

One "iteration" = one env-step (all `num_envs` environments step once). At most `num_envs` dones per iteration — the same bound as Path A's RecurrentPPO, but with much finer temporal resolution. The gate fires inside `if dones_idxes:`, which means it only executes when at least one environment finishes an episode. This is slightly finer granularity than Path A (where the gate fires once per multi-step rollout regardless of dones), but still subject to the same milestone-collapse regime when `num_envs ≥ checkpoint_frequency`.

---

## Directory Layout

| Path | Checkpoint root |
|---|---|
| `train.py` (RPPO / DreamerV3) | `<results_dir>/models/<episode>/` |
| `dreamer_srl_main.py` | `<results_dir>/checkpoints/<episode>/` |

Both use `total_episodes_completed` as the Orbax step key (= subdirectory name). The subdirectory names are actual episode counts and will not match clean multiples of `checkpoint_frequency`.

---

## Configuration Keys

Both paths read from the YAML config hierarchy. The canonical defaults are in `configs/train/default.yaml`:

```yaml
# configs/train/default.yaml:5–6
training:
  checkpoint_frequency: 10000
  max_checkpoints_to_keep: 20
```

Some dreamer-srl experiment configs override `checkpoint_frequency` at a higher value (e.g. `200000` in `configs/models/dreamer_srl/01_food_only_buf256k.yaml`) to avoid excessive eval overhead.

For continual-learning runs via `train.py`, `checkpoint_frequency` is replaced by a per-stage list `schedule.checkpoint_frequencies` (`train.py:2389`). This list is parallel to the stage list in the schedule YAML and indexed by `current_stage`.

| Config key | Where read | Mandatory? | Default |
|---|---|---|---|
| `training.checkpoint_frequency` | `train.py:2391`, `dreamer_srl_main.py:295` | Yes (mandatory) | `10000` (from default.yaml) |
| `training.max_checkpoints_to_keep` | `train.py:552–554`, `dreamer_srl_main.py:296` | Yes (mandatory, hard `ValueError` if missing) | `20` (from default.yaml) |
| `--checkpoint-frequency` (CLI) | `train.py:230, 2391` | No | `None` (falls back to config) |

---

## Drift Analysis

Let `f = checkpoint_frequency`, `N = num_envs`, and let `Δₖ` be the number of episodes completed in iteration `k`.

### Regime 1 — normal drift ( `max(Δₖ) < f` )

One checkpoint per gate-crossing iteration. The saved step key overshoots the nominal milestone by up to `Δₖ − 1` episodes.

**Example**: `f=100`, `num_envs=128`, average episode length ≈ 200 steps, `num_steps=64`.
Typical iteration yields `Δₖ ≈ 40–80` dones.

```
iter    episodes   gate(last+f)      action          step_saved    last_save
 1      0 → 73     73 ≥ 100 ? no     —               —             0
 2      73 → 156   156 ≥ 100? yes    save            156           100
 3      156 → 224  224 ≥ 200? yes    save            224           200
 4      224 → 312  312 ≥ 300? yes    save            312           300
 5      312 → 395  395 ≥ 400? no     —               —             300
 6      395 → 467  467 ≥ 400? yes    save            467           400
 ...
```

Expected milestones `{100, 200, 300, 400}` became `{156, 224, 312, 467}`.

### Regime 2 — milestone collapse ( `Δₖ ≥ f` )

When one iteration completes more episodes than the entire checkpoint interval, the gate fires **once** for many milestones at once.

**Example**: `f=100`, `num_envs=512`, short episodes. Iter 1 produces Δ₁=450 dones.

```
iter    episodes   gate            action    step    last_save
 1      0 → 450    yes (≥100)      save      450     400   ← milestones 100/200/300/400 all missed
 2      450 → 890  yes (≥500)      save      890     800   ← 500/600/700/800 all missed
```

On disk you see `{450, 890}` — the 100, 200, 300, 500, 600, 700 checkpoints simply never existed.

### Regime 3 — silent rotation by `max_checkpoints_to_keep`

Even when drift is small, early saves are deleted once the limit is reached. A 20k-episode run with `f=1000` and the default `max_checkpoints_to_keep: 20` leaves only the last 20 saves (roughly episodes 1k–20k), but all saves before the first retained checkpoint are gone. Set `training.max_checkpoints_to_keep: null` in your YAML to retain all checkpoints (Orbax `max_to_keep: None` keeps everything).

---

## Additional Concerns

### (A) Module-level singleton for `last_checkpoint_save` (Path A only)

```python
# train.py:2395–2396
if not hasattr(main, 'last_checkpoint_save'):
    main.last_checkpoint_save = 0
```

`main` is the function object; its attribute persists across calls within the same Python process. In notebook or test-runner contexts that call `main()` more than once, the cursor leaks into the second call, suppressing early checkpoints of the second run. **This concern applies only to `train.py`.** The dreamer-srl path uses a local variable (`last_ckpt_episode: int = 0` at `dreamer_srl_main.py:494`) and does not have this problem.

### (B) Step key vs. payload episode-count agreement

Both paths set the Orbax step key to `total_episodes_completed` and also store `total_episodes_completed` (or `episode`) inside `ckpt_data`. These agree today. Any future fix that makes the step key a clean milestone (option F1 in the appendix) must still store the actual count in `ckpt_data['episode']` so resume logic reads the correct value.

### (C) Checkpoint triggers post-save evaluation

Both paths chain video/stats evaluation onto every successful save:
- `train.py:2432–2481`: `evaluate_jax_checkpoint` for video and stats.
- `dreamer_srl_main.py:845–913`: `dreamer_srl_eval_rollout` for video and stats.

If drift causes checkpoints to bunch up (Regime 2), evaluation is also skipped for those missed milestones.

### (D) dreamer-srl path — `last_ckpt_episode` initialised to `0`

Because the gate condition is `total // freq > last // freq`, and `last_ckpt_episode` starts at `0`, the **very first save** fires as soon as `total_episodes_completed >= 1` **and** `total_episodes_completed // checkpoint_frequency >= 1` — i.e., at the first iteration that crosses the `checkpoint_frequency` threshold. There is no off-by-one; the initialisation is correct.

---

## Resume Semantics

### Path A — `train.py` (RecurrentPPO and DreamerV3)

`train.py:1069–1154` handles checkpoint restoration via `--load-checkpoint <path>`:

```python
restore_mngr = ocp.CheckpointManager(os.path.abspath(args.load_checkpoint))
step = restore_mngr.latest_step()      # returns largest step key on disk
restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())
```

**`latest_step()`** picks the subdirectory with the largest integer name — i.e., the most recent actual episode count, not a nominal milestone. With drift, this will be a non-round number.

State restored per algorithm:

| Field | RecurrentPPO | DreamerV3 (train.py) |
|---|---|---|
| Model parameters | `nnx.update(model, restored['model'])` | `nnx.update(wm, restored['wm'])` + actor + critic |
| Optimizer state | `nnx.update(optimizer, restored['optimizer'])` | Not saved / not restored |
| RNN hidden state | `h_state = restored['h_state']` | Not applicable |
| PRNG key | `key = restored['key']` | `key = restored['key']` |
| Global step | `global_step = restored.get('step', ...)` | same |
| Iteration counter | `iteration = restored.get('iteration', ...)` | same |
| Episode count | `total_episodes_completed = restored.get('episode', ...)` | same |
| Continual stage | `current_stage = restored.get('stage', 0)` | same |

After restoration, `total_episodes_completed` resumes from the saved value. The training loop's budget check (`while total < episodes`) therefore continues the cumulative count seamlessly — WandB step keys and checkpoint directory names above the resume point will be above the restored value.

**`last_checkpoint_save` is not restored.** After loading a checkpoint, `main.last_checkpoint_save` starts at `0` again (unless the process has run a previous training call). This means the gate will fire at `total_episodes_completed >= 0 + checkpoint_freq`, which would trigger immediately if the resumed counter is already above `checkpoint_freq`. The cursor then snaps to `floor(total / freq) × freq`, so the very first post-resume save may duplicate the loaded checkpoint's episode count by a few episodes. This is a low-severity known quirk — it produces an extra checkpoint, not a corrupt or missing one.

### Path B — `dreamer_srl_main.py` (dreamer-srl)

As of the current codebase (2026-06-08), **dreamer-srl has no `--load-checkpoint` CLI argument and no resume path.** The `add_argument` entries end at `dreamer_srl_main.py:211`; none of them provide a checkpoint-restore flag. To restart a dreamer-srl run from a saved checkpoint, you would need to load the checkpoint manually after `make_checkpoint_manager` is called and before the training loop starts — this is currently unsupported in the driver.

---

## `max_checkpoints_to_keep` Rotation Detail

Orbax keeps the most recent `max_to_keep` saved steps. On each `manager.save(step, ...)` call, Orbax records the new step, then deletes the oldest entry if the count exceeds the limit. This is a **last-N retention** policy, not an LRU cache — the `N` most recent saves (by step key, which equals episode count) are always kept; everything older is deleted.

Practical consequences:
- Once the limit is hit, increasing it in the config only affects **future saves** — previously deleted checkpoints cannot be recovered.
- Orbax re-attaches to an existing `models_dir` (Path A) or `checkpoints/` (Path B) directory on resume and re-discovers existing saves. If fewer than `max_to_keep` saves exist, no deletion occurs. If more exist (e.g., after manually copying old checkpoints back in), Orbax will prune to the limit on the next save.
- Setting `training.max_checkpoints_to_keep: null` passes `max_to_keep=None` to `CheckpointManagerOptions`, which retains all checkpoints indefinitely.

---

## Relevance to Continual Learning

See [`docs/develop/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md). In continual-learning mode, `train.py:2389` selects `checkpoint_freq = schedule.checkpoint_frequencies[current_stage]` — a per-stage frequency drawn from the schedule YAML. Everything else in the checkpoint block is unchanged:

- A stage scheduled to end at episode `3000` will actually transition at the first iteration whose `total_episodes_completed ≥ 3000` — typically a few dozen episodes past the boundary when `num_envs = 128`. This drift is **accepted**.
- The last in-stage checkpoint won't align exactly with the stage boundary. No forced save is triggered at a stage transition.
- The stage index is stored in `ckpt_data['stage']` so resume can restart from the correct stage.

Dreamer-srl does not currently support continual-learning schedules.

---

## Summary of Observed Behavior

| Behaviour | Cause | Applies to |
|---|---|---|
| Orbax step keys are actual episode counts (e.g. `156`), not nominal milestones (e.g. `100`) | Step key = `total_episodes_completed` | Both paths |
| Milestones may collapse when `num_envs ≥ checkpoint_freq` | Per-iteration gate; per-iteration Δ can exceed freq | Both paths |
| `main.last_checkpoint_save` persists across `main()` calls in the same process | Function-attribute singleton (train.py only) | Path A only |
| Post-resume gate fires early (extra near-duplicate save) | `last_checkpoint_save` not restored | Path A only |
| `max_checkpoints_to_keep` prunes oldest saves | Orbax last-N retention | Both paths |
| Dreamer-srl checkpoints land in `checkpoints/` not `models/` | Different directory convention in helper module | Path B only |
| Dreamer-srl has no resume path | No `--load-checkpoint` CLI flag | Path B only |
| Dreamer-srl does not save optimiser state | Not included in `ckpt_data` in checkpoint.py | Path B only |
| Dreamer-srl saves `target_critic`, `moments`, `policy_step`, `cumulative_grad_steps` | Richer payload vs DreamerV3 in train.py | Path B only |

---

## Clarifications / FAQ

**Q: What is `total_episodes_completed` counting — cumulative over the whole run, or per-iteration?**
A: **Cumulative across the entire training run.** It is initialised to 0 at start (or to the resumed value when loading a checkpoint on Path A), and only ever increments. Checkpoint step keys reflect this cumulative total, so a resumed run's checkpoint directories have step keys above the loaded value.

**Q: Is the drift deterministic given the same config and seed?**
A: Yes. Under a fixed seed, env stepping is deterministic, so the per-iteration `Δₖ` sequence is fixed and the drift pattern reproduces exactly. Between runs with different seeds, drift values jitter but always overshoot `last_save + f`.

**Q: How do I force a checkpoint at a specific episode milestone?**
A: Not supported by the current gate. Workaround: set `checkpoint_frequency` low (e.g. 1) and post-process. Or implement option F2 (catch-up saves) — the plan explicitly does not.

**Q: Does `max_to_keep` apply per-run or per-resume?**
A: Per-`CheckpointManager` instance. A fresh run starts with an empty tracking list; resuming into the same directory re-attaches and Orbax re-discovers the existing saves up to the limit. Previously pruned checkpoints stay gone.

**Q: Can I raise `max_to_keep` without re-training?**
A: Yes — change `training.max_checkpoints_to_keep` in your YAML. The new limit only affects future saves. Previously-pruned checkpoints are not recovered.

**Q: Do WandB logs track nominal or actual episode counts?**
A: WandB step = `total_episodes_completed` (actual count). Checkpoint filenames and WandB `Episode/Number` use the same number — neither corresponds to clean milestones.

**Q: Why per-iteration gating and not per-episode gating?**
A: For RecurrentPPO and DreamerV3, the rollout runs inside `jit_train` — a JIT-compiled function. Halting at every `done` would break JIT tracing. The dreamer-srl path steps one env at a time in Python so per-episode gating would be technically feasible, but the current gate (inside `if dones_idxes:`) already fires once per episode-completion block, which is nearly equivalent for most `num_envs` settings.

**Q: What happens if `checkpoint_frequency=0` or negative?**
A: Untested. `floor(total / 0)` is a `ZeroDivisionError`; the dreamer-srl gate would crash. Path A's additive gate `total >= last + 0` would fire every iteration. Keep `checkpoint_frequency >= 1`.

**Q: Does the resume path reconstruct the correct `total_episodes_completed`?**
A: For Path A yes — `ckpt_data['episode']` stores the actual count, and `train.py:1146` seeds `total_episodes_completed` from it. For Path B (dreamer-srl) resume is not implemented.

**Q: Is Orbax's step key required to be monotonically increasing?**
A: Yes. Both gates ensure `total_episodes_completed` strictly increases on each call, so step keys are monotone. Orbax uses step keys for sorting and pruning.

**Q: Does a training crash mid-iteration lose partial episode counts?**
A: Yes — there is no intra-iteration checkpoint. The last saved `total_episodes_completed` is the recovery point. Episodes counted during the crashed iteration are not reflected in the restored counter.

**Q: Is this gate the same code path for RecurrentPPO and DreamerV3 in train.py?**
A: Yes. Both increment `total_episodes_completed` differently (RPPO: inner loop over timesteps; DreamerV3: done_indices scan), but both hit the same gate block at `train.py:2387–2430`. Any gate change applies to both.

**Q: Does dreamer-srl restore optimiser state on resume?**
A: No. The `save_checkpoint` helper at `checkpoint.py:77` does not include the optimiser in `ckpt_data`. On restart (if resume were implemented), the optimiser would reinitialise from scratch. This is a known gap.

---

## Appendix: Alternative Schemes (reference only — not planned)

Retained for future reviewers. **None of these are being implemented.**

Two axes: **WHEN** the save fires, and **WHAT step key** is written.

| Option | WHEN | Step key | Cursor update | Notes |
|---|---|---|---|---|
| **F0** *(current, kept)* | First iter where `total ≥ last+f` | `total` (actual) | `floor(total / f) * f` | Drift + overshoot; round-number directories never occur. |
| **F1** | Same as F0 | `last+f` (nominal) | `last + f` | Directories become `100/, 200/, …`; requires `ckpt_data['episode']` to store actual count for resume accuracy. |
| **F2** | F1 **+ catch-up loop** for missed milestones | nominal | advance through all crossed milestones | Saves same model state under multiple step keys — guaranteed milestone coverage, wasteful on disk. Pair with higher `max_to_keep`. |
| **F3** | Require `f > max_expected_Δ` (config validation) | nominal | nominal | Guideline instead of fix. Rejects configs where `num_envs ≥ f`. |
