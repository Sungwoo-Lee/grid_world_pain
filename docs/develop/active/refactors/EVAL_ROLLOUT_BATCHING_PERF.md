---
title: "Behavior-probe eval-results pipeline: batched single-process rollout (Tier 2) + decoupled video render (Tier 3)"
topic: refactors
status: active
created: 2026-07-06
last_updated: 2026-07-06
---

# Behavior-probe eval pipeline performance refactor — batched rollout (Tier 2) + video decoupling (Tier 3)

> **Status**: PLANNED
> **Opened**: 2026-07-06
> **Related**: [[behavior_measure_toolkit_v1_plan]] (the toolkit these recordings feed), `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` (eval-script callers)

---

## Context

Regenerating the results tables for a behavior-probe "avoidance sweep" — the study that asks *does the agent avoid predators, hide in the bush, and flee when threatened?* across 6 trained models × 12 animal/injury conditions (72 config runs, 30 episodes each) — currently takes about **56 minutes on a 20-core CPU box**, and it is dominated by avoidable overhead, not real computation. Three costs stack up:

1. **~17 min wasted on JAX startup.** The runner launches **72 separate Python processes**, and each one re-imports JAX and re-compiles the environment from cold (~14 seconds × 72 processes).
2. **~16 min of slow rollout.** The eval script `scripts/eval/eval_rollout.py` plays each episode through a plain Python `while` loop, one environment and one step at a time, and iterates the 30 episodes with another Python `for` loop — so almost none of JAX's batch parallelism is used.
3. **~24 min rendering videos** with matplotlib, one frame at a time, sequentially — even though the results *tables* do not need any video at all.

This plan fixes costs 2 and 3. (Cost 1 — pure runner parallelization with no change to eval code — is "Tier 1", already done and out of scope here.)

- **Tier 2 (primary):** rewrite the rollout so one process starts JAX once and plays **all 30 episodes of a config at the same time** as a single batched environment (`num_envs = 30`), the exact same way `train.py` already collects experience. Target: the whole 72-config sweep drops from ~56 min to **~1–2 min**. Because this is shared, correctness-critical measurement tooling, Tier 2 is **gated by a bit-for-bit parity harness**: the new batched path must reproduce the old path's 11 avoidance measures per episode, exactly, before it is allowed to become the default.
- **Tier 3 (companion, lower stakes):** stop the results-table path from depending on videos at all, then make video rendering faster (parallel workers, only representative conditions). Tier 3 must **not** land until Tier 2 has passed its parity gate.

The two tiers share exactly one thing: the on-disk **recording format** (`.rec.gz` files written by the rollout and read by the renderer and the stats script). This plan pins that format as an explicit contract so the two tiers stay decoupled.

### Plain-language glossary (for a fresh reader)

- **behavior probe / avoidance sweep** — a fixed battery of evaluation episodes run against a frozen trained model to measure *what the agent does* (does it hide in the bush, keep its distance from the predator, flee), summarized as a heatmap table.
- **the 11 avoidance measures** — the per-episode numbers the results table is built from: bush-use rate, steps-to-bush, time-in-bush, flight-initiation distance, time-near-animal, closest approach, time-moving, spatial spread, longest-chase duration, injury change, and survival steps. Computed by `scripts/behavior_measures/avoidance_stats_heatmap.py::episode_measures`.
- **`.rec.gz` recording** — one gzip-pickle file per episode holding the full per-step state trajectory (agent position, injury, animal positions, bush position, plus the observation vectors). Written by `eval_rollout.py --record`; read by both the renderer and the stats script via `src/utils/eval_recording.load_episode`.
- **batched / `ParallelEnv` rollout** — instead of stepping one environment 30 times in a Python loop, step 30 environments once per tick using `jax.vmap`; the whole 30-episode batch advances together on the accelerator.
- **parity harness** — a throwaway-proof script that runs the old rollout and the new rollout on the same checkpoint, config, and seeds and checks the 11 measures match exactly. It is the acceptance test for Tier 2.

---

## Analysis

### Why the batched path can be bit-for-bit identical

The single-environment path (`_run_episode` / `_run_episode_with_recording` in `eval_rollout.py`, ~lines 68–244) does, per episode:

```
state = jax_reset(params, jax.random.PRNGKey(seeds[i]))   # eval_rollout.py:782, 79/176
while step < max_steps and not done:
    action, carry = policy_fn(state, carry, rng_key, deterministic=True)  # argmax, key unused in eval
    next_state, reward, done, info = jax_step(state, action, params)
    ... record ...
    state = next_state
```

Two facts make an exact batched reproduction possible:

1. **All per-episode randomness is carried in `state.key`.** `jax_reset(params, key)` seeds the state (core.py:944–965), and every `jax_step` re-splits `state.key` (core.py:505: `jax.random.split(state.key, 6)`) for animal movement, damage, respawn, etc. So an episode is a *pure function of its reset key*. The reset key is `jax.random.PRNGKey(seeds[i])`.
2. **The policy is deterministic in eval mode** (`eval_mode=True` → `jnp.argmax(logits)`, network is pure per-sample), and the model has **no cross-batch coupling** (the only normalizers are LayerNorm variants, which normalize over the feature axis per sample — never across the batch axis). So env *i* in a batch of 30 evolves identically to env *i* run alone, provided its reset key is identical.

**The single most important parity requirement** falls out of fact 1: the batched reset must build the per-env keys as `jnp.stack([jax.random.PRNGKey(s) for s in seeds])` and `jax.vmap(jax_reset, in_axes=(None, 0))` over them. It must **NOT** call `ParallelEnv.reset(key, num_envs)`, because that helper derives its keys with `jax.random.split(key, num_envs)` (wrapper.py:18) — a *different* set of keys than `PRNGKey(seeds[i])`, which would silently change every trajectory and fail parity. `ParallelEnv`'s vmapped `step`/`obs` are fine to reuse; only its `reset` keying is wrong for our purpose.

### The four correctness concerns the reviewers must sign off (call-outs)

These are flagged here so `code-reviewer` (JAX/vmap/PRNG correctness) and `math-reviewer` (measure equivalence) know exactly where to look.

**(a) PRNG threading / per-episode seed semantics.** As above: per-env reset key must equal `jax.random.PRNGKey(seeds[i])`, stacked and vmapped. Reuse of `ParallelEnv.reset` is a trap. There is no other RNG in the eval path (eval policy is argmax; key argument unused). Reviewer check: assert `state.key[i]` after batched reset equals `jax_reset(params, PRNGKey(seeds[i])).key` for every i.

**(b) vmap axis correctness.** The batched forward pass feeds obs of shape `(num_envs, input_dim)` and hidden state built via `model.initial_state(num_envs)` (batched, recurrent_ppo_network.py:367–381 returns `(num_envs, hidden_size)` leaves; modulation hidden is also batched). Action selection MUST argmax over the **last** axis: `jnp.argmax(logits, axis=-1)` — the existing helper `get_action_and_value_nnx` uses bare `jnp.argmax(logits)` (network.py:393) which flattens and is only correct for the unbatched case, so the batched path calls `model(obs, h)` directly and argmaxes with `axis=-1`. Reviewer check: LayerNorm normalizes over features not batch (no leakage); `value.squeeze()` is not used in a way that collapses the batch axis.

**(c) Done-masking / no post-death pollution.** The Python loop stops at death; a `lax.scan` over `max_steps` cannot early-stop and will keep stepping dead envs. The batched path must, per env, find the first tick where `done` fired (`first_done = argmax(done_seq)` with an all-false guard → run to `max_steps`), and **slice every recorded array to that episode's true length** before writing the `.rec.gz`. This makes post-death steps invisible to the measures. Concretely, episode length `T` must equal the single-env `T` (number of `jax_step` calls until `done`, inclusive of the terminal step), and `snapshots` must have length `T + 1` (initial state + one per step) — matching `_run_episode_with_recording` exactly. Measures like `survival_steps` (= `len(snapshots)`), `bush_dwell`, and `injury_change` (uses `snapshots[-1]`) are the ones that corrupt if the slice is off by even one step. Reviewer check: for a seed that dies at step k, batched `len(snapshots) == 1 + (k+1)` and the last snapshot is the terminal state, identical to the legacy path.

**(d) `.rec.gz` format compatibility with `load_episode`.** The batched writer must produce byte-structurally the same payload the legacy `EpisodeRecorder.write` produces (eval_recording.py:62–76): keys `version, episode_index, train_episode, seed, snapshots, obs, true_obs, actions, rewards`, with `snapshots` a list of dicts holding exactly the fields `_snapshot_state` emits (eval_recording.py:24–39: `agent_pos, satiation, nutrition, injury_level, rest_streak, res_pos, res_active, animal_pos, obs_pos`), `obs`/`true_obs` as stacked arrays, `actions` int32, `rewards` float32. Simplest safe implementation: **reuse `EpisodeRecorder`** — after the batched scan, loop over envs and per env replay the sliced arrays into an `EpisodeRecorder` via `.append(...)`, then `.write(...)`. This guarantees format identity by construction rather than re-deriving the serialization. Note that the 11 avoidance measures read **only** `snapshots` fields, so the stats parity gate depends on the snapshot trajectory; `obs`/`true_obs` (the noised/true observation vectors) feed only the Tier-3 renderer. The plan still requires structural `.rec.gz` compatibility, but flags `obs`/`true_obs` numerical reproduction as a **secondary** (Tier-3-relevant) parity concern, separately verified, so a noise-key subtlety in the observation vector cannot silently block the Tier-2 stats gate.

### Recording-format contract (the one shared interface)

Pin this so Tier 2 and Tier 3 stay independent. The `.rec.gz` payload schema is **owned by `src/utils/eval_recording.py`** (`RECORDING_FORMAT_VERSION = 1`, `_snapshot_state` field set, `EpisodeRecorder.write` payload dict). Both the batched writer (Tier 2) and the renderer (Tier 3) go through this module — neither re-implements pickle layout. Any field addition bumps `RECORDING_FORMAT_VERSION` and updates all three consumers (`load_episode`, `render_recordings.py`, `avoidance_stats_heatmap.py`). Tier 3 changes must not require a format change; if one is discovered necessary, it comes back here first.

---

## Implementation Plan

### Design

**Tier 2 — additive `--batched` flag, legacy path preserved as the parity reference.**

Rather than replace the per-episode loop, add the batched rollout as a **new code path behind a flag**, leaving `_run_episode` / `_run_episode_with_recording` untouched. This is deliberate: the parity harness must compare "current path" vs "new path", and requirement 1 forbids routing both sides through the new (suspected-changed) code. Keeping the legacy loop verbatim in-file means the reference side of the diff runs code that this change never touched.

Rollout order of operations in the new `_run_episodes_batched(...)`:

1. `keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])` → `states = jax.vmap(jax_reset, in_axes=(None, 0))(params, keys)`. (NOT `ParallelEnv.reset`.)
2. `h = model.initial_state(num_envs)`; `obs = jax.vmap(get_observation, in_axes=(0, None))(states, params)`.
3. `lax.scan` (or a jitted Python-free step fn) over `max_steps`: each tick computes `logits, value, h = model(obs, h)`, `action = jnp.argmax(logits, axis=-1)`, `next_states, rewards, dones, infos = jax.vmap(jax_step, in_axes=(0,0,None))(states, action, params)`, `next_obs = vmap(get_observation)`. Carry all per-step arrays needed for measures + recording (agent_pos, injury via snapshot fields, animal_pos, obs_pos, actions, rewards, obs, true_obs, done). Accumulate `done` cumulatively so already-dead envs stay dead (mask actions/records after first death — safe either way since we slice later, but freeze to avoid wasted divergence).
4. Bring arrays to host once (`np.asarray`). Per env: compute `first_done` → `T`; slice `[0:T]` (steps) / `[0:T+1]` (snapshots incl. initial); replay into an `EpisodeRecorder` and `.write(...)`; also assemble the `.npz` per-episode dict in the same schema `_run_episode` returns (agent_pos, action, ate_food, agent_in_bush, dist_per_predator, dist_per_neutral, hit_predator, hit_neutral, nociception, termination_reason, length, seed). The threat-onset index and `online_replay.json` (eval_rollout.py:822–843) are computed from these dicts exactly as today — unchanged downstream.

The batched step needs the same per-step `info` fields the legacy loop reads (`ate_food, agent_in_bush, dist_per_predator, dist_per_neutral, hit_predator, hit_neutral, nociception, termination_reason`). `jax_step` already returns these in `info`; under `vmap` they come back batched — collect them in the scan carry. (These feed the `.npz` schema and the threat-onset/online-replay cross-checks, not the 11 avoidance measures, but must stay identical so `motif_cluster.py` and the online-replay sanity file are unaffected.)

**Tier 3 — decouple video from stats, then parallelize.**

The results-table path (`avoidance_stats_heatmap.py`) already reads `.rec.gz` and needs no video — so "decoupling" is mostly *organizational*: make `eval_rollout.py --record` (stats + recordings) and video rendering two independent invocations, and ensure the sweep runner's stats step never triggers a render. Then speed rendering by (i) running `render_recordings.py` workers in parallel (it already has a `ProcessPoolExecutor` — confirm the runner uses `--workers`), and (ii) rendering only a representative subset of conditions/episodes rather than all 72×30. Tier 3 lands only after Tier 2's parity gate is green.

### File Changes

#### `scripts/eval/eval_rollout.py` — add batched rollout behind `--batched` (Tier 2)

- Add CLI flag `--batched` (`action="store_true"`, default `False`) and, initially, keep the default path = legacy loop. After parity is proven and signed off, a follow-up one-line change flips the default (or the sweep runner passes `--batched`); do **not** delete the legacy functions in this change — they remain the parity reference and regression guard.
- Add `_run_episodes_batched(params, model, seeds, max_steps, record=..., record_n=..., quiet=...)` implementing the Design steps above. Reuse `jax.vmap(jax_reset, in_axes=(None,0))`, `jax.vmap(jax_step, in_axes=(0,0,None))`, `jax.vmap(get_observation, in_axes=(0,None))` (the same vmaps `ParallelEnv` builds — you may instantiate `ParallelEnv(params)` for `step`/`obs` but must do reset with the per-seed stacked keys, not `ParallelEnv.reset`).
- In `main()`, branch on `args.batched`: batched branch produces the same `episodes` list of per-episode dicts and writes the same `episodes/*.npz`, `recordings/`, `windows/threat_onsets.parquet`, `online_replay.json`, `metadata.json` outputs. Add `"rollout_mode": "batched"|"legacy"` to `metadata.json`.
- No config-key changes required (all inputs already come from `bm_cfg` / CLI). No `get_mandatory` additions.
- Speed note the `developer` must record in the Implementation Report: wall-clock for one 30-episode config, legacy vs batched, same checkpoint/config/seeds/CPU.

#### `scripts/eval/parity_check_eval_rollout.py` — NEW parity harness (Tier 2 deliverable)

- New standalone script (not imported by anything). Signature: `--config --agent_config --checkpoint --seeds S... [--n-episodes N] [--tolerance 0.0] [--golden-dir DIR]`.
- Behavior: (1) run `eval_rollout.py --legacy` (unbatched) and `eval_rollout.py --batched` on the **same** checkpoint/config/seed set into two temp output roots; (2) for each seed, load both `.rec.gz` files and compute the **11 avoidance measures via the shared `avoidance_stats_heatmap.episode_measures`** (import it — do NOT re-implement the measures, and do NOT route both sides through the new batched code); (3) diff per-episode per-measure. **Default tolerance = exact equality** (`==` for the integer/position-derived measures; the measures are deterministic functions of integer positions and a float injury level, so bit-for-bit is the expected and required outcome). Any non-zero `--tolerance` must be justified in the harness output and defaults off.
- Also diff the `.npz` per-episode arrays (agent_pos, action, length, termination_reason, …) for a stronger check than the 11 measures alone.
- Emit a compact PASS/FAIL table (seed × measure) and exit non-zero on any mismatch, so it can gate CI / the runner.
- **Golden-baseline mode:** with `--golden-dir`, compare the batched output against a *frozen* set of legacy recordings captured from the pristine pre-change file (see Implementation Order step 1), guarding against accidental edits to the legacy functions.

#### `scripts/behavior_measures/avoidance_stats_heatmap.py` — export `episode_measures` (Tier 2, minor)

- No logic change. Confirm `episode_measures` is importable (it is a module-level function). If the harness needs it and the file's `sys.path` insert (lines 39–42) or `_heatmap_style` import interferes with import-as-module, guard the heatmap-render imports under `if __name__ == "__main__"` so the harness can `from avoidance_stats_heatmap import episode_measures` without pulling matplotlib. Keep changes surgical.

#### `src/utils/eval_recording.py` — format contract owner (Tier 2, likely no change)

- No change expected. This file is named in scope as the **owner of the `.rec.gz` contract**: the batched writer must go through `EpisodeRecorder` / `write_run_meta` here rather than re-implementing serialization. If (and only if) a field must be added, bump `RECORDING_FORMAT_VERSION` and update `load_episode` + both consumers in the same change.

#### `scripts/eval/render_recordings.py` — decouple + parallelize (Tier 3, separate landing)

- Ensure the stats path never invokes rendering; make rendering an explicit separate step. Confirm `--workers` parallelism is actually used by the sweep runner; add a `--conditions`/`--episodes` subset selector so only representative recordings render. No `.rec.gz` format change.

#### `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` — REQUIRED update (Maintenance Contract)

- This change **adds a new file under `scripts/`** (`parity_check_eval_rollout.py`) and adds a caller relationship (harness → `eval_rollout.py`, harness → `avoidance_stats_heatmap.episode_measures`). Per the map's Maintenance Contract, `developer` must add the new script and its callees to the dependency map **in the same change**. Also refresh the existing rows for `eval_rollout.py` (new `--batched`/`--legacy` flags) and, when Tier 3 lands, `render_recordings.py`.

### Checkpoints (during implementation)

- [x] After batched reset: assert `states.key[i]` equals `jax_reset(params, PRNGKey(seeds[i])).key` for all i (PRNG parity, concern a). — Implemented as a permanent runtime guard in `_run_episodes_batched` (not a one-off manual check); passed on every run.
- [x] Print `logits.shape == (num_envs, action_dim)` and confirm `argmax(axis=-1)` shape `(num_envs,)` (vmap axis, concern b). — Implemented as `assert scan_out["action"].shape == (max_steps, num_envs)` + an out-of-range guard against `model.action_dim` (avoids an extra eager forward probe — see the eager/jit pitfall below).
- [x] For one seed known to die early, print legacy `T` and batched `T` and `len(snapshots)`; they must match (done-masking, concern c). — Verified manually on `avoid_pred_inj70` seed index 20 (legacy length 11, termination_reason 4/injury-death) and confirmed via the parity harness (exact match on `survival_steps` and all other measures for that seed).
- [x] Load one batched `.rec.gz` with `load_episode` and assert key set + `snapshots[0]` field set match a legacy file (format, concern d). — Verified directly (see Implementation Report); also implicit in every harness PASS since `episode_measures` reads `snapshots` fields exclusively.
- [x] Run the parity harness on a single checkpoint / 5 seeds → all 11 measures exact-match before scaling to the full sweep. — 5/5 exact on `avoid_pred_inj00`, b03_randinit@8500010 (golden-baseline mode).
- [x] Record legacy-vs-batched wall-clock for one 30-episode config in the Implementation Report. — See Implementation Report Speed Check.

**Unplanned checkpoint discovered during implementation:** an `nnx.jit`-vs-eager numerical-parity trap (not anticipated by the plan's concerns a–d). See Implementation Report "Deviations" for the full root-cause and fix.

### Implementation Order (mandatory sequence)

1. **Build the parity harness AND capture the golden baseline first**, from the pristine current `eval_rollout.py` (git HEAD, before any batched code exists). Run the legacy path on a representative checkpoint + the canonical seed set (e.g. seeds 0–29) and freeze its `.rec.gz` + `.npz` + computed 11-measure table to a golden dir under `tmp/`. This is the immovable reference.
2. **Implement `_run_episodes_batched` behind `--batched`.** Leave legacy default. Iterate against the harness (single checkpoint, 5 seeds) until the 11 measures + `.npz` arrays match the golden baseline **exactly**.
3. **Scale the harness** to the full seed set and 2–3 checkpoints (ideally spanning a zero-animal config and a predator config, and one that dies early vs. survives) → all exact.
4. **Only after the parity gate is green:** flip the sweep runner (or the default) to `--batched`; measure the end-to-end sweep wall-clock.
5. **Tier 3 is a separate landing**, planned in this doc but implemented + verified independently *after* Tier 2 clears its gate. Tier 3 must not touch the `.rec.gz` format.

### Verification steps (who owns the parity gate)

- **`developer`** implements, runs the harness, and pastes the PASS/FAIL table + wall-clock deltas into the Implementation Report. The harness exiting non-zero on any mismatch is a hard blocker — `developer` may not report success with any non-exact measure unless a bounded, justified `--tolerance` was explicitly approved here (default: none).
- **`senior-developer`** (this agent) runs the Verification Protocol: diff-stat + `git diff` against this File Changes list, confirm the legacy functions are untouched (they are the reference), confirm `SCRIPTS_DEPENDENCY_MAP.md` was updated, re-run the parity harness independently, and review the speed delta (Tier 2 is expected to be a large *speedup*, so the >5%/>15% slowdown thresholds apply only as a regression tripwire — a slowdown here would itself be a red flag). Sign off only when the parity table is exact and the golden-baseline comparison passes.
- **Recommended reviewers before merge:** `code-reviewer` for concerns (a)–(c) (PRNG stacking vs. split, vmap axis, done-mask slicing) and `math-reviewer` / the `episode_measures` equivalence is covered by the exact-parity harness itself.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-06

### What this section is (plain-language entry point)

Tier 2 only (per the task scope) is implemented and **passes its parity gate exactly**: the new batched rollout produces bit-for-bit identical per-episode behavior measures to the untouched legacy rollout, verified on 3 real trained checkpoints × 4 different environment configs (predator present/absent, zero-animal, early-death and full-survival episodes). Tier 3 (video decoupling) was explicitly NOT touched, per the task instructions.

### Summary of what was built (file-by-file)

- **`scripts/eval/eval_rollout.py`**
  - Added `_rollout_scan_jit(model, params, states0, h0, max_steps)` — the `nnx.jit`-wrapped `jax.lax.scan` body that steps all episodes in parallel for `max_steps` ticks, collecting per-tick action/reward/done/info + the 9 snapshot fields (`agent_pos, satiation, nutrition, injury_level, rest_streak, res_pos, res_active, animal_pos, obs_pos`) + noised/true observations.
  - Added `_run_episodes_batched(params, model, seeds, max_steps, record, record_n_episodes)` — builds the per-seed stacked reset keys (`jnp.stack([PRNGKey(s) for s in seeds])` + `jax.vmap(jax_reset, in_axes=(None,0))`, **not** `ParallelEnv.reset`), runs the PRNG reset-key parity guard (concern a, permanent runtime check), calls `_rollout_scan_jit` via `nnx.jit`, brings the whole trajectory to host once, computes each env's true episode length via `argmax(done_seq, axis=0) + 1`, slices every array to `[0:T]` (or `[0:T+1]` for snapshots), and replays the sliced snapshot arrays directly into `EpisodeRecorder`'s internal lists (reusing `_snapshot_state` from `src/utils/eval_recording.py` on a lightweight `SimpleNamespace` per (env, t) so the write path is byte-identical to the legacy recorder by construction).
  - Added `--batched` CLI flag (default `False` — legacy stays default), a guard that raises `NotImplementedError` if `--batched` is combined with a non-`"deterministic"` `eval_policy_mode` (batched stochastic sampling is out of scope for this change), a `main()` branch that calls the batched path and reuses the same `.npz`/`.rec.gz` writer loop, and `"rollout_mode": "batched"|"legacy"` added to `metadata.json`.
  - **The legacy `_run_episode` / `_run_episode_with_recording` functions are untouched** (verified: default-path output is byte-identical to the pre-change golden baseline — see Parity Results below).

- **`scripts/eval/parity_check_eval_rollout.py`** (new) — the Tier 2 acceptance harness. Runs `eval_rollout.py` twice (legacy, then `--batched`) as subprocesses on the same config/checkpoint/seeds (or, in `--golden-dir` mode, skips the legacy re-run and compares against a frozen capture), loads each episode's `.rec.gz` via `src/utils/eval_recording.load_episode`, computes the 11 avoidance measures through the **shared, imported** `avoidance_stats_heatmap.episode_measures` (never re-derived), additionally diffs the raw `.npz` per-episode arrays, prints a PASS/FAIL table (seed × measure), and exits non-zero on any mismatch. Default tolerance is `0.0` (exact); no tolerance was needed or used.

- **`scripts/behavior_measures/avoidance_stats_heatmap.py`** — **no change needed.** Contrary to the plan's contingency ("guard heatmap-render imports under `if __name__` if `_heatmap_style`'s matplotlib import interferes with import-as-module"), `matplotlib`/`seaborn` are already installed in the project's conda env, so `from avoidance_stats_heatmap import episode_measures` succeeds cleanly with no guard needed. Verified directly before writing the harness.

- **`src/utils/eval_recording.py`** — no change, as expected. The batched writer imports and reuses `EpisodeRecorder` and the private `_snapshot_state` helper directly (not re-implemented).

- **`docs/environment/SCRIPTS_DEPENDENCY_MAP.md`** — updated per its Maintenance Contract: added the new `parity_check_eval_rollout.py` script with its two caller relationships (bare import of `avoidance_stats_heatmap.episode_measures`/`KEYS` in §1a; subprocess calls to `eval_rollout.py` in §1b), refreshed the `eval_rollout.py` and `avoidance_stats_heatmap.py` roll-up rows in §3, and extended Cluster B in §4.

- **`scripts/eval/render_recordings.py`** — untouched (Tier 3, explicitly out of scope for this pass, per the task instructions).

### An unplanned but load-bearing discovery: an `nnx.jit`-vs-eager parity trap

While iterating the first batched run against the golden baseline, the 11 measures mismatched badly (survival lengths of 21–95 instead of 100, injury pinned at the death cap) even at `num_envs=1`. Root-cause, isolated by direct experiment (see below): after `nnx.update(model, restored_tree)` restores a checkpoint's weights, calling `model(x, h)` **eagerly** — with no `nnx.jit` anywhere in the call stack — reads a subtly different (and wrong) view of the restored parameters than calling the exact same `model(x, h)` through an `nnx.jit`-wrapped function, even though `nnx.state(model)` checksums identically either way. This is invisible in the legacy path because its only forward-pass entry point, `get_action_and_value_nnx`, is itself `@nnx.jit`-decorated — the first-ever forward pass of the whole process (and every one after) already goes through the split/merge that correctly materializes the restored state. A bare `jax.lax.scan` around an eager `model(...)` call does **not** fix this (`lax.scan` only compiles the outer XLA loop; it doesn't perform nnx's graphdef/state split) — confirmed by reproducing the exact same wrong numbers with `jax.lax.scan` alone, and then reproducing exact parity by wrapping the identical scan body in `nnx.jit`. **Fix**: `_run_episodes_batched` now calls `_rollout_scan_jit` via `nnx.jit(...)`, making the function self-contained and correct with no reliance on a caller-side warm-up call. This is flagged explicitly in the function's docstring for `code-reviewer`.

### Parity results (exact — tolerance 0.0)

| Checkpoint | Config | n_episodes | Mode | Result |
|---|---|---|---|---|
| b03_randinit (`.../20260704-230444_rppo_basic03_randinit_n108/models/8500010`) | `avoid_pred_inj00.yaml` (predator, all die at max_steps=100 via starvation) | 5 | golden-baseline | 5/5 PASS |
| b03_randinit @ 8500010 | `avoid_pred_inj70.yaml` (predator, varied lengths incl. one early injury-death at step 11) | 30 | golden-baseline | 30/30 PASS |
| b04_jump (`.../20260704-230521_rppo_basic04_jump_n109/models/9800003`) | `avoid_pred_inj70.yaml` | 30 | fresh legacy + batched | 30/30 PASS |
| b05_noise (`.../20260704-230555_rppo_basic05_noise_n109/models/10000004`) | `avoid_pred_inj70.yaml` | 30 | fresh legacy + batched | 30/30 PASS |
| b03_randinit @ 8500010 | `avoid_none_inj00.yaml` (zero-animal edge case, `dist_per_predator` shape `[T,0]`) | 5 | fresh legacy + batched | 5/5 PASS |
| b03_randinit @ 8500010 | `avoid_rabbit_inj00.yaml` (neutral-animal / olfaction path) | 5 | fresh legacy + batched | 5/5 PASS |

All 6 runs: every one of the 11 `episode_measures` values matched exactly (including `NaN==NaN` for the no-animal config's `fid`/`closest_approach`/`time_near_animal`), and the raw `.npz` per-episode arrays (`agent_pos, action, ate_food, agent_in_bush, dist_per_predator, dist_per_neutral, hit_predator, hit_neutral, nociception, termination_reason, length`) matched exactly too. The default (non-`--batched`) path was independently re-verified byte-identical to a golden capture taken from the pristine pre-change file (all snapshot fields, obs, true_obs, actions, rewards — full `.rec.gz` diff, not just the 11 measures).

**Harness invocation** (re-runnable by the verifier):
```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/eval/parity_check_eval_rollout.py \
  --config configs/environment/experiment/behavior_probes/core/avoidance/avoid_pred_inj70.yaml \
  --checkpoint results/JAX_RecurrentPPO/20260704-230521_rppo_basic04_jump_n109/models/9800003 \
  --n-episodes 30 --device cpu \
  --work-dir tmp/parity_run_b04_jump_pred70_30
```
Golden-baseline mode (frozen legacy capture, faster re-run):
```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/eval/parity_check_eval_rollout.py \
  --config configs/environment/experiment/behavior_probes/core/avoidance/avoid_pred_inj70.yaml \
  --checkpoint results/JAX_RecurrentPPO/20260704-230444_rppo_basic03_randinit_n108/models/8500010 \
  --golden-dir tmp/golden_b03_pred70_30 --n-episodes 30 --device cpu
```

### Speed check (before/after)

Same hardware (local CPU box), same config/checkpoint/seeds, `--device cpu`, measured via `eval_rollout.py`'s own internal timer (excludes config/model-load, includes JIT compile) on `avoid_pred_inj70.yaml` × b04_jump@9800003, 30 episodes, max_steps=100:

| Mode | With `--record --record-n-episodes 30` (real sweep usage) | Without recording |
|---|---|---|
| Legacy | 17.3s | 10.9s |
| Batched | 9.5s | 7.3s |
| Speedup | **1.8x** | **1.5x** |

Full-subprocess wall-clock (includes JAX cold-start + model restore, not just the internal timer) for 3 different checkpoints on `avoid_pred_inj70.yaml`, 30 episodes with recording (via the harness): b03_randinit 30/30 exact (golden mode, no fresh-legacy timing); b04_jump legacy 30.6s / batched 32.7s (0.93x — dominated by per-process JAX startup+compile, which Tier 1, not Tier 2, addresses); b05_noise legacy 36.6s / batched 27.2s (1.35x).

**Honest read**: the rollout-only speedup (1.5–1.8x) is real but well short of the plan's headline "~56 min → ~1–2 min" for the full 72-config sweep — that number bundles in Tier 1 (fewer JAX cold starts, already done, out of scope here) and likely assumes longer episodes than this test's `max_steps=100`. At this scale, per-process JAX startup/compile (~14–20s) and the still-Python-side `EpisodeRecorder` replay loop (unavoidable — the study's stats pipeline needs `.rec.gz` for every recorded episode) both eat into the win. No regression was observed in any run (batched was never slower than legacy on the *internal-timer* rollout-only metric); the one 0.93x "slowdown" data point was full-subprocess wall-clock dominated by cold-start noise, not the rollout itself. Flagging this honestly per the Speed Check Protocol rather than only reporting the best number.

### Deviations from the plan (and why)

1. **`scripts/behavior_measures/avoidance_stats_heatmap.py` needed no edit.** The plan's contingency (guard the heatmap-render imports under `if __name__` if `_heatmap_style`'s matplotlib import blocks import-as-module) turned out to be unnecessary — matplotlib/seaborn are already installed in the project conda env, so the plain `from avoidance_stats_heatmap import episode_measures` import used by the harness works with zero changes to that file. Verified directly before writing the harness, not assumed.
2. **`nnx.jit` wrapping was added, not in the plan's Design/File-Changes text.** The plan's Design section says the batched path should "call `model(obs, h)` directly" — this is correct as written, but the plan's authors (and my own first implementation) assumed an eager `model(...)` call inside `jax.lax.scan` would be numerically identical to `get_action_and_value_nnx`'s `@nnx.jit`-wrapped call, since it's "the same function." It is not, for the reason above. The fix is additive and surgical: the scan body now runs inside `nnx.jit(_rollout_scan_jit, static_argnames=("max_steps",))` instead of a bare `jax.lax.scan`. This does not change the plan's architecture (still one vmapped batch + one scan over max_steps) — it changes only how the model is invoked. Flagging for `code-reviewer` per instructions (also documented in the function's own docstring) since this is exactly the kind of JAX/NNX correctness subtlety in scope for that review.
3. **The parity harness's CLI omits a literal `--legacy` flag.** `eval_rollout.py` was not given a `--legacy` flag (only `--batched`, per the plan's File Changes list for that file). The harness's "legacy" subprocess invocation is simply the eval_rollout.py CLI **without** `--batched`, which is the pre-existing default behavior — functionally identical to what a `--legacy` flag would do, without adding an unrequested CLI surface to `eval_rollout.py`.
4. **`--record-n-episodes` is always set equal to `--n-episodes` in the harness** (record every episode, not just a subset). This matches how the real production sweep is actually run today (verified: the existing `results/eval/avoidance/sweeps/.../recordings/<ckpt>/` directories hold all 30 `.rec.gz` files, not a subset) — the 11 avoidance measures are computed from `.rec.gz` snapshots, so every scored episode must be recorded regardless of path.
5. **Tier 3 not touched**, per the task's explicit instruction — `render_recordings.py` is unmodified.

### Test suite

`tests/scripts/test_eval_rollout_stage_config.py` (5 tests, exercises `eval_rollout.py`'s continual-stage-config resolution, module-level import surface unaffected by the additive changes): **5 passed**. No `src/` files were touched by this change, so the broader environment/agent test suite was not re-run (the plan's File Changes list touches only `scripts/` files + docs).

### Blockers / follow-ups for `senior-developer`

- None blocking. The `nnx.jit`-vs-eager parity trap (Deviation #2) is worth a `code-reviewer` pass specifically on `_rollout_scan_jit`/`_run_episodes_batched` given how easy it would be to reintroduce (e.g. a future refactor that calls `model(...)` eagerly anywhere in the batched path, even outside the scan, would silently reintroduce wrong numbers that still "look like" plausible policy behavior rather than crashing).
- Tier 3 (video decoupling) is unstarted, as instructed — separate landing after this Tier 2 sign-off.

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-06

### Plain-language verdict (for a fresh reader)

I own the parity gate: a new behavior-results table is trusted only after I independently confirm the fast "play all 30 episodes at once" rollout (the *batched* path, behind the new `--batched` flag) produces **exactly** the same per-episode numbers as the old "one episode at a time" rollout (the *legacy* path, still the default). I did not accept the developer's self-report — I re-ran the acceptance harness myself, in its default mode (fresh legacy run vs fresh batched run, same checkpoint/config/seeds), on all three trained checkpoints named in the task, at 30 episodes each on CPU.

**Result: every one of the 90 episodes (3 checkpoints × 30) matched bit-for-bit** — all 11 avoidance measures AND the raw per-episode `.npz` arrays, tolerance `0.0`, in three independent harness runs I launched (logs: `tmp/verif_b03.log`, `tmp/verif_b04.log`, `tmp/verif_b05.log`).

Two separate verdicts, which must not be conflated:

- **(a) Correctness + parity: ✅ PASS.** The batched path is bit-for-bit identical to the untouched legacy path. Safe to trust its results tables.
- **(b) Safe to make `--batched` available (kept opt-in): ✅ PASS.** The flag is purely additive, legacy stays the default, and stochastic eval is fenced off with a `NotImplementedError`. Nothing forces anyone onto the new path.
- **Speed (separate from correctness): CORRECT but NOT yet FAST at the sweep level.** The rollout-only inner loop is ~1.5–1.8× faster (developer's internal-timer measurement, which I did not re-instrument), but full-process wall-clock in my three runs was 1.50× / 0.86× / 1.12× — dominated by per-process JAX cold-start, not the rollout. The plan's headline "~56 min → ~1–2 min" for the 72-config sweep is **not delivered by this change alone**: it needs a follow-on in-process driver (Tier 2b, one Python process launching JAX once for all 72 configs) plus the already-done runner parallelization (Tier 1). This is a correct, mergeable building block — not the finished speedup. The developer reported this honestly.

### Independent parity re-run (I ran this, not the developer)

Harness: `scripts/eval/parity_check_eval_rollout.py`, default mode (re-runs legacy fresh as a subprocess, then batched, then diffs), `--device cpu`, `--n-episodes 30`, config `avoid_pred_inj70.yaml` (predator present, mixed early-death/full-survival episodes):

| Checkpoint | Result | Full-process wall-clock (legacy → batched) |
|---|---|---|
| b03_randinit @ 8500010 | **30/30 exact-match (tol 0.0)** | 27.3s → 18.3s (1.50×) |
| b04_jump @ 9800003 | **30/30 exact-match (tol 0.0)** | 15.5s → 18.1s (0.86×) |
| b05_noise @ 10000004 | **30/30 exact-match (tol 0.0)** | 20.2s → 18.0s (1.12×) |

Batched wall-clock is near-constant (~18s) across all three because it is fixed JAX cold-start + compile dominated; legacy varies with episode-length totals. The 0.86× on b04 is cold-start noise on a fast legacy run, **not** a rollout regression — the internal-timer rollout-only metric never regressed (developer's Speed Check). No blocker.

### Requirement-by-requirement sign-off

1. **Re-ran the harness myself on all three ladder6 checkpoints** — 90/90 episodes exact, bit-for-bit, including raw `.npz` arrays. ✅
2. **Comparison is meaningful, not tautological** — the harness's legacy side calls `eval_rollout.py` with `batched=False` (no `--batched` flag), routing through the untouched per-episode loop, never the new code; measures come from the imported shared `avoidance_stats_heatmap.episode_measures`, not a re-derivation. I verified via `git diff` (ground truth, not a shared code path) that the legacy loop change is **whitespace-only** — it was re-indented into an `else:` branch and every removed line reappears verbatim modulo indentation; `_run_episode`/`_run_episode_with_recording` bodies are untouched. So the working-tree legacy path is behaviorally identical to HEAD, making the developer's golden-capture claim sound. ✅
3. **PRNG design correct** — `_run_episodes_batched` builds keys as `jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])` + `jax.vmap(jax_reset, in_axes=(None,0))`, explicitly NOT `ParallelEnv.reset` (which appears only in docstring/error text, never as a `.reset()` call), with a permanent runtime guard asserting `states0.key[i] == jax_reset(params, PRNGKey(s)).key` for every env. ✅
4. **`SCRIPTS_DEPENDENCY_MAP.md` updated per Maintenance Contract** — new `parity_check_eval_rollout.py` script recorded with both caller relationships (bare import of `episode_measures`/`KEYS` in §1a; subprocess ×2 of `eval_rollout.py` in §1b/§3), roll-up rows and Cluster B refreshed. Already committed (working tree clean; HEAD contains the entries). ✅
5. **Only Tier-2 files touched** — the other dirty files (`src/algorithms/dreamer_srl/dreamer_srl_main.py`, `src/behavior/accumulators.py`, `src/models/dreamer_v3_trainer.py`, `train.py`, `train_command-agent.sh`, diag_fable5 docs) belong to a separate concurrent dreamer-diagnosis workstream, are not imported by `eval_rollout.py` or the harness (the two grep hits are a code comment and a `"dreamer"` string match, not imports), and were not disturbed by this change. ✅
6. **Speed assessed honestly** — CORRECT ≠ FAST, stated above. The delivered code is correct and mergeable; the headline sweep speedup requires the Tier 2b in-process driver. ✅

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `scripts/eval/eval_rollout.py` | `--batched` path added, legacy preserved | ✅ | Legacy loop change is whitespace-only re-indent into `else:`; new funcs inserted after `_run_episode_with_recording`; PRNG + vmap + done-slice guards all present |
| `scripts/eval/parity_check_eval_rollout.py` | new harness | ✅ | Legacy side does not route through batched code; measures imported (not re-derived); also diffs raw `.npz`; exits non-zero on any mismatch — re-run by me, 90/90 exact |
| `scripts/behavior_measures/avoidance_stats_heatmap.py` | `episode_measures` importable | ✅ | No change needed (matplotlib present in env); imports cleanly |
| `src/utils/eval_recording.py` | format-contract owner (no change expected) | ✅ | Unchanged; batched writer reuses `EpisodeRecorder` + `_snapshot_state` — format identity by construction |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | new script + callers recorded | ✅ | Updated + committed; §1a/§1b/§3/§4 all reflect the harness |
| `scripts/eval/render_recordings.py` | Tier 3 (separate landing) | ✅ | Correctly untouched, per task scope |

**Conclusion**: ✅ PASS on both (a) correctness+parity (90/90 episodes bit-for-bit exact across 3 checkpoints, independently re-run) and (b) safe to ship `--batched` opt-in (legacy stays default, additive, stochastic-eval fenced off). Speed verdict: ✅ no rollout regression; the sweep-level headline speedup is deferred to Tier 2b (in-process driver) and is not claimed by this change. Signed off for merge. — Verified by: senior-developer

---

## Implementation Report — Tier 3 (video decoupling + fast rendering)

> **Implemented by**: developer
> **Date**: 2026-07-06

### What this section is (plain-language entry point)

Tier 3 makes video rendering — turning a saved eval recording into an MP4 you can actually watch — fast and crash-proof, without changing anything about how the behavior-probe results tables are computed. It touches only `scripts/eval/render_recordings.py`. The headline fix: rendering several episodes' videos at once used to crash with `OSError: [Errno 24] Too many open files` whenever the script was launched from a background/non-interactive context (a `nohup`'d job, some SSH non-interactive commands, job schedulers) — those contexts commonly cap a process at ~1024 open files even though an interactive shell here allows over a million. The script now raises its own limit at startup, so it survives regardless of how it was launched.

### Summary of what was built (file-by-file)

- **`scripts/eval/render_recordings.py`**
  - **FD-limit fix (`_raise_fd_limit`, L48–71; called at `main()` L158 before the worker pool is created, and again defensively inside `_worker_init` L82).** Reads the process's current soft/hard `RLIMIT_NOFILE` and raises the soft limit toward `min(8192, hard)` (or straight to 8192 if the hard limit is unlimited), wrapped in `try/except (ValueError, OSError): pass` so it can never crash the render if the limit can't be changed. Called in the main process *before* the `ProcessPoolExecutor` is created so forked workers inherit the raised limit (Linux's default multiprocessing start method is `fork`); also called defensively inside `_worker_init` in case a different start method is ever used.
  - **`--max-episodes N`** (L150–152) and **`--stride S`** (L153–155) flags: `episode_files` (sorted `episode_*.rec.gz` glob) is sliced `[::stride]` then `[:max_episodes]` (L174–179) before building the render task list. Both default to "no filtering" (`stride=1`, `max_episodes=None`), so **default behavior renders every episode, unchanged** from before this change.
  - **Decoupling note added to the module docstring** (L15–29): documents explicitly that the stats path (`eval_rollout.py --record` → `avoidance_stats_heatmap.py`) never invokes this script (confirmed by grep — no `render_recordings`/`subprocess` references in `eval_rollout.py` besides a docstring mention and a `--record` help string), and that the two `src/` auto-render callers (`evaluation_core.py`, `dreamer_srl/eval.py`) are separate, config-gated invocations, not the stats path.
  - No `--workers` default change: the existing `max(1, cpu_count-1)` default is safe now that the FD-limit fix is in place; the docstring documents the FD-per-render budget (~250 fds/render is the plan's stated worst case; measured ~70–105 fds/render in this environment — see Verify section) so a future maintainer understands why the fix exists.

### Deviations from the plan (and why)

- None in scope. The plan's contingency wording ("if the current worker default is unsafe, pick a sensible default") turned out unnecessary once the FD-limit fix is applied — no `--workers` default change was needed, only documentation of the interaction.
- `docs/environment/12_renderer.md` cites `render_recordings.py`'s two renderer imports at "lines 31 and 45" (that doc's own line-number citation, not something this plan's File Changes list named). Adding the FD-limit function and the two new CLI flags shifted those imports to **L78/L93**. Per task scope (only `render_recordings.py` and `SCRIPTS_DEPENDENCY_MAP.md` were authorized for this Tier), I did **not** edit `12_renderer.md` — instead I flagged the now-stale citation in `SCRIPTS_DEPENDENCY_MAP.md`'s `render_recordings.py` row (§3) so `senior-developer` can decide whether to route a follow-up edit there. This is a flag, not a silent scope expansion.

### Verify — reproducing the real failure, then confirming the fix

**Reproduction is real, not simulated by assertion.** A per-process `RLIMIT_NOFILE` is per-process, not a sum across parallel worker processes, so I first measured actual per-render fd usage directly (`/proc/<pid>/fd` sampling): a single worker process peaks at **~70–105 concurrently-open fds** during a render in this environment (lower than the plan's "~250" worst-case estimate, likely matplotlib/fontconfig-version-dependent — noted, not disputed, since the plan's number is a stated upper bound not a promise). To get a clean, deterministic repro of the exact failure mode, I set the soft limit below that measured usage (`ulimit -Sn 50`) — this is the same failure class the plan describes (a background context imposing a low soft limit below what a render needs), just made deterministic instead of environment-dependent.

- **Pre-fix (git HEAD copy of the script, `JAX_PLATFORMS=cpu`, `ulimit -Sn 50`, `--workers 6`, 16 staged episodes): crashes with exactly `OSError: [Errno 24] Too many open files`**, raised from `imageio_ffmpeg`'s `subprocess.run(...)` → `os.pipe()` inside a render worker, propagated through the `ProcessPoolExecutor` as the top-level exception (full traceback captured at `tmp/20260706_render_test/prefix_run_50_6w_cpu.log`).
- **Post-fix, identical setup, launched via `nohup` (a genuine backgrounded/non-interactive context, not just a simulated one) with the same `ulimit -Sn 50`: exit code 0, all 16 episodes rendered to valid MP4s**, confirmed via `file episode_000000.mp4` → `ISO Media, MP4 Base Media v1` (log at `tmp/20260706_render_test/postfix_run_50_6w_cpu.log`).
- **`--max-episodes 3`**: rendered exactly episodes 0,1,2 (verified file listing). **`--stride 4`**: rendered exactly episodes 0,4,8,12 (verified). **Default (no flags), `--concat --cleanup-per-episode`**: all 16 episodes rendered, consolidated into `eval_8500010.mp4`, per-episode files cleaned up — confirms backward-compatible default and that existing `--concat`/`--cleanup-per-episode` flows are unaffected.
- **Existing test suite**: `tests/algorithms/dreamer_srl/test_eval_recording.py` (4 tests) + `tests/algorithms/dreamer_srl/test_render_upload.py` (3 tests, including one that runs `render_recordings.py` as a real subprocess and asserts an MP4 is produced) — **7/7 passed**.

Staged test data: 16 real episodes copied from `results/eval/avoidance/sweeps/b03_randinit/avoid_rabbitwander_inj00/models/8500010/recordings/8500010/` into `tmp/20260706_render_smoke/recordings/8500010/` (gitignored scratch, not committed).

### Speed check (before/after)

The FD-limit fix is one cheap `getrlimit`/`setrlimit` syscall pair at startup — expected to have no measurable throughput cost, confirmed directly:

| Config | Pre-fix (git HEAD) | Post-fix | Command |
|---|---|---|---|
| 16 episodes, `--workers 16`, normal ulimit | 41.3s wall | 39.0s wall (no regression — within run-to-run noise) | `time (JAX_PLATFORMS=cpu python .../render_recordings.py <dir> --workers 16 --fps 5)` |

Parallelism itself (not new to this change, but confirmed working end-to-end with the fix in place):

| Config | Wall clock | Speedup |
|---|---|---|
| 4 episodes, `--workers 1` (sequential) | 87.7s | 1.0x (baseline) |
| 4 episodes, `--workers 4` (parallel) | 25.3s | **3.5x** |

**Fast full-sweep invocation** (one process, internal `--workers` parallelism, FD-safe regardless of launch context):
```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/eval/render_recordings.py \
  <results_dir>/recordings/<checkpoint_pct> --workers <cpu_count-1> --fps 5 \
  --concat --cleanup-per-episode
```
Representative-subset fast pass (e.g. spot-checking a sweep without rendering all 30 episodes per config):
```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/eval/render_recordings.py \
  <results_dir>/recordings/<checkpoint_pct> --workers <cpu_count-1> --fps 5 --max-episodes 5
```

### Blockers / follow-ups for `senior-developer`

- `docs/environment/12_renderer.md`'s "lines 31 and 45" citation for the two renderer imports in `render_recordings.py` is now stale (they moved to L78/L93). Flagged in `SCRIPTS_DEPENDENCY_MAP.md`'s `render_recordings.py` row; out of this Tier's authorized scope (not `render_recordings.py` or the dependency map itself) so not edited directly.
- None blocking merge. `.rec.gz` format untouched, `eval_rollout.py` untouched, stats/heatmap path untouched, no new config keys.

---

<!-- NEW ISSUES discovered during implementation: append as "## Issue #2" (related) or new doc (independent), cross-referenced both ways. -->
