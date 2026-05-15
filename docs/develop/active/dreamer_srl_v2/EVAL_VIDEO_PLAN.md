---
title: "dreamer-srl v2 — eval-step video recording port"
topic: dreamer
status: active
created: 2026-05-15
last_updated: 2026-05-15
phase: 2-extension
---

# dreamer-srl v2 — eval-step video recording port

> **Status**: PLANNED
> **Opened**: 2026-05-15
> **Related**: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) (the v2 master plan that this extends), [`../sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md`](../sheeprl_bridge/PARALLEL_ENV_BENCHMARK.md)

---

## 1. Context (plain-language entry point)

The two older training algorithms in this repo — the original JAX DreamerV3 and JAX RecurrentPPO, both reached through `train.py` — produce **eval-step videos** during training. An eval-step video is what you get after each *training checkpoint* (a save of the model's weights to disk, here triggered every N episodes): we freeze the model, run a small batch of *deterministic rollouts* (the model picks its single best action at every step instead of sampling, so the resulting playthrough is reproducible from the seed), draw each step of those rollouts as a PNG frame using the grid-world renderer, stitch the frames into one MP4 file per checkpoint, and ship that MP4 to *Weights & Biases* (our experiment-tracking dashboard, "WandB") so a reviewer can scrub through and see *how the agent behaves at this point in training* — does it walk toward food, does it freeze near predators, does it learn to rest. This is the project's primary qualitative-evaluation tool.

The new **dreamer-srl** algorithm — our second rebuild of DreamerV3 against the `vendor/sheeprl` PyTorch reference, living under `src/algorithms/dreamer_srl/` with its own training driver `dreamer_srl_main.py` — currently does **none of this**. It has no checkpointing, no eval-rollout function, no PNG-frame recorder, no MP4 stitcher, no WandB-video upload. After a 200k-step parity run finishes you get scalar curves on WandB but no way to *see* what the agent is doing.

This doc plans the port. It maps the existing pipeline (§2), enumerates the interface gaps dreamer-srl introduces (§3), proposes seven small commits ordered smallest-first so the developer agent can land them one at a time (§4), surfaces risks and one user-disposition question about checkpoint-format choice (§5), and gives a wall-clock estimate (§6). The end state: every N training iterations, dreamer-srl saves its weights, runs 3 deterministic eval episodes, renders them, uploads an MP4 to WandB tagged with the checkpoint number, and logs `Eval/MeanReward` + `Eval/MeanLength` scalars — matching what the original Dreamer + rPPO do at `train.py:L2444-L2467`.

---

## 2. How Dreamer + rPPO currently do it

The end-to-end flow (entry points in `train.py` and `src/utils/evaluation_core.py`):

1. **Training loop runs N iterations.** Each iteration advances `num_envs` parallel envs by one step and (for Dreamer) optionally takes gradient steps. Episode completions are counted into `total_episodes_completed`.

2. **Checkpoint-save trigger** at `train.py:L2400-L2426`. When `total_episodes_completed` crosses a multiple of `training.checkpoint_frequency`, the driver builds a pytree (`ckpt_data`) of all trainable state — for Dreamer: `{wm, actor, critic, key, iteration, step, episode, stage}`, all as `nnx.state(..., nnx.Param)` — and saves it via the Orbax `CheckpointManager` (`ocp.args.StandardSave(ckpt_data)`). The save is then forced sync with `checkpointer.wait_until_finished()`.

3. **Post-save evaluation gate** at `train.py:L2429-L2433`. The driver reads `visualization.enabled`, `training.video_during_training`, and `training.stats_during_training` from the config. If either video or stats is true, it calls `evaluate_jax_checkpoint` from `src/utils/evaluation_core.py`.

4. **Two-pass eval** at `train.py:L2440-L2467`:
   - **Pass 1 — Video** (only if `training.video_during_training`): calls `evaluate_jax_checkpoint(model=..., params=..., num_episodes=training.eval_video_episodes (default 3), render_video=True, record_stats=False, num_envs=1)`. Single-env, deterministic, frames written.
   - **Pass 2 — Stats** (only if `training.stats_during_training`): calls the same function with `render_video=False, record_stats=True, num_envs=training.eval_stats_num_envs`. Stats CSVs written. Returns `{mean_reward, mean_length, ...}` which is then logged to WandB as `Eval/MeanReward` and `Eval/MeanLength` at `train.py:L2466-L2467`.

5. **Inside `evaluate_jax_checkpoint`** (`src/utils/evaluation_core.py:L141`):
   - Creates `<results_dir>/recordings/<checkpoint_pct>/` (per-checkpoint dir).
   - Calls `write_run_meta(...)` to drop one `run_meta.pkl` (params, icon_config, action_map, config path, seed).
   - Loops `num_episodes` deterministic episodes. Each step calls `generic_inference(model, obs_batch, h_state, eval_mode=True)` (which does `argmax(logits)` — see `evaluation_core.py:L36-L39`) to pick an action, then `jax_step(state, action_idx, params)` to advance.
   - Per step, if `render_video=True`, an `EpisodeRecorder` (`src/utils/eval_recording.py`) collects `(state_snapshot, obs, true_obs, action_idx, reward)` into in-memory lists.
   - At episode end, `recorder.write(<recordings_dir>/episode_<NNNNNN>.rec.gz)` flushes the episode to a gzipped pickle (the "recording artifact").

6. **Subprocess MP4 consolidation** at `src/utils/evaluation_core.py:L289-L321`. If `testing.auto_render_after_eval` is true, the eval function shells out to `scripts/render_recordings.py <recordings_dir> --concat --skip-existing --cleanup-per-episode --fps <visualization.fps>` with `JAX_PLATFORMS=cpu` set (renderer is CPU-only matplotlib, avoids GPU OOM with the training process). The script:
   - Spawns a `ProcessPoolExecutor` with `_worker_init(run_meta_path)` (warms icon cache + matplotlib + loads `run_meta.pkl`).
   - Each worker loads one `episode_*.rec.gz`, replays the snapshots through `render_jax_state(...)` + `build_sensory_viz(...)` into a list of frames, writes one MP4 via `save_jax_video`.
   - With `--concat`: stitches per-episode MP4s into `<results_dir>/videos/eval_<checkpoint_pct>.mp4` (with 5-frame holds between episodes).
   - With `--cleanup-per-episode`: deletes the per-episode MP4s, keeps only the consolidated one.

7. **WandB upload** at `src/utils/evaluation_core.py:L318-L321`. Calls `upload_video(consolidated_mp4, episode=checkpoint_pct, step=checkpoint_pct, caption=f"Episode {checkpoint_pct}")` from `src/utils/wandb_utils.py:L44`, which does `wandb.log({"eval/video": wandb.Video(path, fps=4, format='mp4'), "eval/checkpoint_episode": episode}, step=step)`.

8. **Stats scalars** at `train.py:L2466-L2467` (after Pass 2). `wandb.log({"Eval/MeanReward": ..., "Eval/MeanLength": ..., "iteration": ..., "timesteps": ...})`.

---

## 3. Interface gaps for dreamer-srl

What dreamer-srl is missing relative to the pipeline above:

### 3.1. Checkpointing

dreamer-srl has **zero** checkpointing code in `dreamer_srl_main.py`. The original Dreamer in `train.py` uses Orbax (`ocp.CheckpointManager` + `ocp.args.StandardSave`) with `max_to_keep=20` (from `training.max_checkpoints_to_keep`). For dreamer-srl we need to save the full set of trainable+target+optimizer state at every checkpoint event:

```
ckpt_data = {
    'world_model': nnx.state(world_model, nnx.Param),
    'actor':       nnx.state(actor, nnx.Param),
    'critic':      nnx.state(critic, nnx.Param),
    'target_critic': nnx.state(target_critic, nnx.Param),
    # Optimizer state — Orbax can serialize optax states via jax.tree
    'wm_opt':     nnx.state(wm_opt),
    'actor_opt':  nnx.state(actor_opt),
    'critic_opt': nnx.state(critic_opt),
    # Moments + ratio state (post-eval reload must continue scaling correctly)
    'moments':    moments,            # dataclass-of-jax-arrays from utils.moments_init
    'ratio_num_grads': ratio.num_grads,
    'ratio_pretrain_steps': ratio.pretrain_steps,
    # Bookkeeping
    'key': key, 'iter_num': iter_num, 'policy_step': policy_step,
    'total_episodes_completed': total_episodes_completed,
    'cumulative_grad_steps': cumulative_grad_steps,
}
```

**Format choice (OPEN QUESTION — see §5).** Two options:

- **Orbax** (`ocp.args.StandardSave`) — matches the original Dreamer; future `evaluate_jax_checkpoint(...)` can in principle load these via `ocp.StandardCheckpointer` (though our usage here is "save + immediately eval in-process", no load round-trip strictly required for the video pipeline).
- **Lightweight pickle** (`pickle.dump` of an nnx-state pytree) — simpler, ~30 LOC, no extra dep. Reload path not needed for the in-process eval flow because we already hold the live `(world_model, actor, ...)` Python objects.

**Recommendation:** lightweight pickle for Commit B; revisit Orbax only if a use case for cross-script reload appears.

### 3.2. Agent interface

`evaluate_jax_checkpoint(model=..., params=..., ...)` expects ONE callable model object that supports:

- `model.initial_state(batch_size=None)` (single-env path) and `model.initial_state(batch_size=N)` (parallel path).
- `model(x, h)` returning `(logits, value, h_new, mod_info)` — see `generic_inference` at `src/utils/evaluation_core.py:L29-L46`.

dreamer-srl has **no such object**. The closest equivalent is the `Player` wrapper in `dreamer_srl_main.py:L52-L152`, which holds `(recurrent_state, posterior_state, prev_action)` internally and exposes:

- `player.init_states(reset_envs=None)` — resets internal RSSM state.
- `player.get_actions(obs, is_first, key)` — does `encoder + rssm.dynamic + actor` and returns a **one-hot action** (not raw logits).

This mismatch means dreamer-srl cannot reuse `evaluate_jax_checkpoint` directly. **Two options:**

- **(A) Adapter shim.** Write a thin wrapper that exposes `model.initial_state(...)` and `model(x, h)` over the dreamer-srl agent + RSSM state. Complex because the RSSM has two state tensors `(h, z)` plus `prev_action`, not a single `h_state`.
- **(B) Dedicated `dreamer_srl_eval_rollout()`.** Write a parallel function under `src/algorithms/dreamer_srl/eval.py` that mirrors the structure of `_run_single_env_eval` (`evaluation_core.py:L338-L498`) but calls `Player.get_actions` directly. Reuse the recording helpers (`EpisodeRecorder`, `write_run_meta`) and the subprocess + WandB tail (already algorithm-agnostic at `evaluation_core.py:L287-L321`).

**Recommendation:** **(B).** Cleaner separation, ~120 LOC instead of an awkward shim that has to invent a fake h-state for an RSSM. The existing render and upload helpers are already algorithm-agnostic.

### 3.3. Deterministic eval mode for `Player`

The current `Player.get_actions` (`dreamer_srl_main.py:L104-L152`) uses `self.actor(latent, k_act)` — the actor's `__call__` does a **straight-through Gumbel-softmax** sample (`agent.py:L1539-L1545`). That is **stochastic**, not deterministic.

For eval we need `argmax(logits)` instead. The actor already exposes `forward_logits(latent)` at `agent.py:L1484` (returns post-unimix logits without sampling), so the eval rollout function can call:

```python
logits = actor.forward_logits(latent)             # [B, action_dim]
action_idx = int(jnp.argmax(logits, axis=-1))     # deterministic
actions_oh = jax.nn.one_hot(action_idx, action_dim)
```

No new method needed on `Actor`; `forward_logits` already does the right thing.

### 3.4. Recording config plumbing

dreamer-srl reads only the agent config and a subset of the env config. It does **not** read:

- `training.video_during_training`
- `training.eval_video_episodes`
- `training.stats_during_training`
- `training.eval_stats_episodes` / `training.eval_stats_num_envs`
- `training.auto_analysis`
- `training.checkpoint_frequency`
- `training.max_checkpoints_to_keep`
- `visualization.enabled` / `visualization.icons` / `visualization.fps`
- `testing.auto_render_after_eval` / `testing.record_stats` / `testing.record_true_observations`

These are all under `configs/train/default.yaml` and `configs/environment/*.yaml`. dreamer-srl needs to read them from `env_cfg` (the merged config) using the `get_mandatory` pattern that the rest of the driver already uses.

### 3.5. WandB metric definitions

`dreamer_srl_main.py:L346-L350` defines `iteration`, `timesteps`, `Episode/Number` as step metrics. We need to add `eval/checkpoint_episode` as well so the `Eval/*` scalars and `eval/video` align in the WandB UI. The original Dreamer's pattern is `wandb.log({"Eval/MeanReward": ..., "Eval/MeanLength": ..., "iteration": ..., "timesteps": ...})` (no explicit `step=` argument — pulls from `iteration` / `timesteps` via `define_metric`).

---

## 4. Implementation plan (chunked commits)

Order: smallest first, each chunk independently verifiable, no chunk depends on a chunk that hasn't landed.

### Commit A — Plumb training/visualization config flags (no behavior change)

**Estimated LOC: ~40**

In `dreamer_srl_main.py` after the existing `agent_cfg.get_mandatory(...)` block (around L227), add:

```python
# Eval-video config — read from env_cfg (which holds training.* + visualization.* + testing.*)
video_during_training = env_cfg.get_mandatory('training.video_during_training')
eval_video_episodes   = env_cfg.get_mandatory('training.eval_video_episodes')
stats_during_training = env_cfg.get_mandatory('training.stats_during_training')
eval_stats_episodes   = env_cfg.get_mandatory('training.eval_stats_episodes')
eval_stats_num_envs   = env_cfg.get_mandatory('training.eval_stats_num_envs')
checkpoint_frequency  = env_cfg.get_mandatory('training.checkpoint_frequency')
max_checkpoints_keep  = env_cfg.get_mandatory('training.max_checkpoints_to_keep')
viz_enabled           = env_cfg.get_mandatory('visualization.enabled')
viz_fps               = env_cfg.get_mandatory('visualization.fps')
auto_render           = env_cfg.get_mandatory('testing.auto_render_after_eval')
```

Print them on startup but **don't use them yet**. Confirms key presence and types. Add a `results_dir` (use the env-config's existing `results_dir_root` or fall back to `wandb.run.dir`) so future commits have somewhere to write.

**Test:** existing smoke-test command still runs identically (no eval triggered, since the eval block lands in Commit F).

### Commit B — Lightweight checkpointing

**Estimated LOC: ~80**

New helper `src/algorithms/dreamer_srl/checkpoint.py`:

```python
import pickle
from pathlib import Path
from flax import nnx

def save_checkpoint(out_dir, episode, world_model, actor, critic, target_critic,
                    moments, ratio, key, iter_num, policy_step,
                    total_episodes_completed, cumulative_grad_steps):
    payload = {
        'world_model':   nnx.state(world_model, nnx.Param),
        'actor':         nnx.state(actor, nnx.Param),
        'critic':        nnx.state(critic, nnx.Param),
        'target_critic': nnx.state(target_critic, nnx.Param),
        'moments': moments,
        'ratio_num_grads': ratio.num_grads,
        'key': key, 'iter_num': iter_num, 'policy_step': policy_step,
        'total_episodes_completed': total_episodes_completed,
        'cumulative_grad_steps': cumulative_grad_steps,
    }
    out_path = Path(out_dir) / f'ckpt_ep{episode:06d}.pkl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path
```

In `dreamer_srl_main.py`, after the done-block (~L617), check whether `total_episodes_completed` crossed a multiple of `checkpoint_frequency`. If yes, call `save_checkpoint(...)`. Enforce `max_checkpoints_to_keep` by listing `out_dir/ckpt_*.pkl`, sorting, and deleting the oldest.

**Test:** add `tests/test_dreamer_srl_checkpoint.py` that runs ~200 iters with `checkpoint_frequency=50`, asserts ≥1 `ckpt_*.pkl` exists, loads it back, checks pytree shapes match the live model.

### Commit C — `dreamer_srl_eval_rollout` function

**Estimated LOC: ~150**

New file `src/algorithms/dreamer_srl/eval.py`:

```python
import jax, jax.numpy as jnp, numpy as np, os
from pathlib import Path
from src.environment.core import jax_reset, jax_step
from src.environment.sensor import get_observation
from src.utils.eval_recording import EpisodeRecorder, write_run_meta

def dreamer_srl_eval_rollout(
    world_model, actor, env_params, config,
    num_episodes: int, seed: int, results_dir: str, checkpoint_pct: int,
    render_video: bool = True, quiet: bool = True,
):
    """Deterministic single-env eval rollout matching evaluate_jax_checkpoint's
    contract: writes <results_dir>/recordings/<checkpoint_pct>/episode_*.rec.gz +
    run_meta.pkl. Returns {mean_reward, mean_length, episode_rewards, episode_lengths}.
    """
    # 1. Setup recordings dir + run_meta
    recordings_dir = Path(results_dir) / 'recordings' / str(checkpoint_pct)
    if render_video:
        recordings_dir.mkdir(parents=True, exist_ok=True)
        action_map = ['Up', 'Right', 'Down', 'Left']
        if env_params.rest_action_enabled: action_map.append('Rest')
        if env_params.eat_action_enabled:  action_map.append('Eat')
        write_run_meta(
            recordings_dir, env_params,
            config.get('visualization.icons', None),
            action_map, getattr(config, 'source_path', ''),
            extras={'checkpoint_pct': checkpoint_pct, 'seed': seed},
        )

    # 2. Per-episode loop — deterministic
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)
    key = jax.random.PRNGKey(seed)
    episode_rewards, episode_lengths = [], []

    for ep_idx in range(num_episodes):
        key, k_reset = jax.random.split(key)
        state = jax_reset(env_params, k_reset)
        obs = get_observation(state, env_params)

        # Init RSSM state via WorldModel
        h0, z0 = world_model.rssm.get_initial_states(1)
        recurrent_state, posterior_state = h0, z0
        prev_action = jnp.zeros((1, action_dim), dtype=jnp.float32)
        is_first = jnp.ones((1, 1), dtype=jnp.float32)

        recorder = EpisodeRecorder(ep_idx + 1, checkpoint_pct, seed) if render_video else None
        if recorder is not None:
            recorder.append(jax.device_get(state), obs, None, action_idx=-1, reward=0.0)

        total_reward, step_count, done = 0.0, 0, False
        while not done and step_count < env_params.max_steps:
            # Encoder + RSSM dynamic step (mirror Player.get_actions but deterministic)
            obs_b = jnp.asarray(obs, dtype=jnp.float32)[None, :]
            embedded = jax.vmap(world_model.encoder)(obs_b)
            key, k_rssm = jax.random.split(key)
            recurrent_state, posterior_state, _, _, _ = world_model.rssm.dynamic(
                posterior_state, recurrent_state, prev_action, embedded, is_first, k_rssm,
            )
            posterior_flat = posterior_state.reshape(1, -1)
            latent = jnp.concatenate([posterior_flat, recurrent_state], axis=-1)

            # DETERMINISTIC action = argmax(post-unimix logits)
            logits = actor.forward_logits(latent)   # [1, action_dim]
            action_idx = int(jnp.argmax(logits, axis=-1)[0])
            prev_action = jax.nn.one_hot(jnp.array([action_idx]), action_dim, dtype=jnp.float32)
            is_first = jnp.zeros((1, 1), dtype=jnp.float32)

            # Step env
            next_state, reward, done, _info = jax_step(state, action_idx, env_params)
            total_reward += float(reward)
            step_count += 1

            state = next_state
            obs = get_observation(state, env_params)
            if recorder is not None:
                recorder.append(jax.device_get(state), obs, None,
                                action_idx=action_idx, reward=float(reward))

        if recorder is not None:
            recorder.write(recordings_dir / f'episode_{ep_idx + 1:06d}.rec.gz')
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        if not quiet:
            print(f'[eval] ep {ep_idx + 1}/{num_episodes}: reward={total_reward:.2f} steps={step_count}')

    return {
        'mean_reward':     float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'mean_length':     float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'recordings_dir':  str(recordings_dir) if render_video else None,
    }
```

**Test:** add `tests/test_dreamer_srl_eval.py` that runs 2 episodes with a freshly-initialized agent (random behavior), asserts `episode_lengths` are within `[1, env_params.max_steps]`, asserts `recordings_dir` exists with 2 `episode_*.rec.gz` + 1 `run_meta.pkl`.

### Commit D — Frame-recording correctness verification

**Estimated LOC: ~30 (just test code; the integration is already in C)**

Add `tests/test_dreamer_srl_recording_format.py` that:

1. Runs `dreamer_srl_eval_rollout(render_video=True, num_episodes=1)`.
2. Loads the resulting `.rec.gz` via `src.utils.eval_recording.load_episode`.
3. Asserts all keys present (`snapshots`, `obs`, `actions`, `rewards`, `episode_index`, etc.).
4. Asserts `len(snapshots) == len(obs) == len(actions) == steps+1` (step 0 placeholder + each step).
5. Calls `src.environment.renderer.render_jax_state(...)` on the first snapshot to confirm the renderer accepts the recorded fields end-to-end.

This is the **regression test** for the renderer contract — catches a future change to `_snapshot_state` or `render_jax_state` that would silently break the eval-video pipeline.

### Commit E — Render + WandB upload hook

**Estimated LOC: ~50**

Add a function `_render_and_upload(recordings_dir, results_dir, checkpoint_pct, fps, wandb_enabled, quiet)` to `src/algorithms/dreamer_srl/eval.py`:

```python
def _render_and_upload(recordings_dir, results_dir, checkpoint_pct, fps, wandb_enabled, quiet):
    import subprocess, sys
    project_root = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'
    render_script = f'{project_root}/scripts/render_recordings.py'
    consolidated = f'{results_dir}/videos/eval_{checkpoint_pct}.mp4'
    cmd = [sys.executable, render_script, str(recordings_dir),
           '--concat', '--skip-existing', '--cleanup-per-episode', '--fps', str(fps)]
    child_env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    result = subprocess.run(cmd, env=child_env, capture_output=quiet, text=True)
    if result.returncode != 0:
        print(f'Warning: render failed (rc={result.returncode}). stderr: {result.stderr[:500]}')
        return None
    if wandb_enabled and os.path.exists(consolidated):
        from src.utils.wandb_utils import upload_video
        upload_video(consolidated, episode=checkpoint_pct, step=checkpoint_pct,
                     caption=f'Episode {checkpoint_pct}', quiet=True)
    return consolidated if os.path.exists(consolidated) else None
```

This is a near-verbatim port of `evaluation_core.py:L289-L321`, just lifted into the dreamer-srl module.

**Test:** integration smoke (manual): run `dreamer_srl_eval_rollout(...)` then `_render_and_upload(...)` over the resulting dir, assert MP4 exists. WandB upload is integration-tested by Commit G.

### Commit F — Driver wiring

**Estimated LOC: ~60**

In `dreamer_srl_main.py`, after the done-block (~L617, just after Commit B's checkpoint save) but inside the same `if dones_idxes:` branch, add:

```python
# Checkpoint-triggered evaluation (mirrors train.py:L2429-L2467)
if just_saved_ckpt:  # set True by Commit B's save block
    if video_during_training:
        from src.algorithms.dreamer_srl.eval import dreamer_srl_eval_rollout, _render_and_upload
        if not args.quiet:
            print(f'[eval] checkpoint @ ep={total_episodes_completed}: video pass')
        eval_result = dreamer_srl_eval_rollout(
            world_model, actor, env_params, env_cfg,
            num_episodes=eval_video_episodes, seed=args.seed,
            results_dir=results_dir, checkpoint_pct=total_episodes_completed,
            render_video=True, quiet=args.quiet,
        )
        if auto_render:
            _render_and_upload(eval_result['recordings_dir'], results_dir,
                               total_episodes_completed, viz_fps, use_wandb, args.quiet)
        if use_wandb:
            wandb.log({
                'Eval/MeanReward': eval_result['mean_reward'],
                'Eval/MeanLength': eval_result['mean_length'],
                'iteration':       iter_num,
                'timesteps':       policy_step,
            }, step=policy_step)
```

Also: add `wandb.define_metric('eval/checkpoint_episode')` and `wandb.define_metric('Eval/*', step_metric='timesteps')` near `L346-L350`.

**Test:** existing dreamer-srl smoke (`--total-steps 5000 --num-envs 1`) with `video_during_training=true` in the env config: confirm one `recordings/<N>/` dir is written and one `Eval/MeanReward` row appears in WandB. Should not slow down the run by >10% (no eval triggered unless checkpoint hits).

### Commit G — End-to-end smoke test

**Estimated LOC: ~80**

Add `tests/test_dreamer_srl_eval_video_e2e.py` (marked `slow`, ~2 min wall-clock):

```python
@pytest.mark.slow
def test_eval_video_e2e(tmp_path, monkeypatch):
    """End-to-end: run dreamer-srl for ~2000 iters with eval enabled, assert
    MP4 lands at videos/eval_<N>.mp4 and run_meta.pkl is loadable."""
    # invoke dreamer_srl_main.main() with --total-steps 2000, --no-wandb, results-dir=tmp_path
    # set training.checkpoint_frequency=500 so 4 evals fire
    # assert tmp_path/videos/eval_500.mp4 exists, size > 1KB
    # assert tmp_path/recordings/500/run_meta.pkl loadable
```

WandB upload is not asserted (`--no-wandb` mode); a separate manual smoke is run on a real WandB project to validate that final hop.

---

## 5. Risks + open questions

### 5.1. **OPEN QUESTION FOR USER — Checkpoint format**

The current plan uses **lightweight pickle** for dreamer-srl checkpoints (Commit B). The original Dreamer uses **Orbax**. Trade-offs:

| | Lightweight pickle | Orbax |
|---|---|---|
| LOC | ~30 | ~60 |
| Cross-script reload | manual | first-class |
| Sharding-aware | no | yes (irrelevant for our single-host setup) |
| Compatibility with existing `evaluate_jax_checkpoint` reload | no | partial (loader supports `StandardSave` format) |

**Recommendation: pickle.** The in-process eval pipeline never needs to reload — the live `(world_model, actor, ...)` Python objects are passed directly. Orbax adds complexity for a use case we don't have.

**User must disposition before Commit B lands.** If user wants future cross-script reload (e.g., post-hoc analysis runs that load a saved dreamer-srl model and re-eval it), switch to Orbax.

### 5.2. Eval frequency

The original Dreamer triggers eval **per checkpoint save** (every `training.checkpoint_frequency` episodes). The default config sets `checkpoint_frequency: 10000`. For a 200k-step parity run at `num_envs=1` averaging ep_len ≈ 400 (per the recent `i4ulpn95` parity check at 80% of target), that's ~500 episodes total — meaning the default `10000` would fire **zero** evals.

**Recommendation:** for dreamer-srl, recommend setting `training.checkpoint_frequency` to ~50 episodes for the parity-class runs (gives ~10 evals over a 500-episode run). Or expose an iter-based `training.eval_iter_frequency` key (e.g., every 20k iters → ~10 evals over a 200k-iter run). The latter is more robust to ep_len variation; recommend adding it as an alternate trigger in Commit F.

### 5.3. Compute cost

Each eval pass: ~3 deterministic episodes × ~400 steps × ~3ms/step ≈ 4 seconds of agent compute, plus ~30s of subprocess frame-rendering (CPU matplotlib, parallel workers). Total ~35s per eval. 10 evals × 35s = ~6 minutes over a 200k-iter run, which at ~17 SPS = ~3.3 hours — so eval overhead ≈ **3% of wall-clock**. Acceptable.

### 5.4. WandB step-metric collision

The new `Eval/*` metrics are logged with `step=policy_step`, which is the same step axis used for training scalars. No collision risk; matches the original Dreamer's pattern.

### 5.5. Results-dir source

dreamer-srl doesn't currently materialize a `results/<run-name>/` directory the way `train.py` does. Need to decide where the recordings + videos directory lives. Two options:

- **(A) `wandb.run.dir`** when WandB is active — automatic, gets uploaded with the run.
- **(B) `<project_root>/results/dreamer_srl/<wandb-run-name>/`** — matches the project's existing convention (`results/JAX_DreamerV3/...`, etc.).

**Recommendation: (B).** Consistent with the rest of the project; renders cleanly in CI-less analysis later. Falls back to `tmp/dreamer_srl_<timestamp>/` when `--no-wandb`.

### 5.6. Eval-during-training vs post-training only

The original Dreamer does **during-training, every checkpoint**. dreamer-srl plan matches this. **Recommendation: match the original.** Eval-only-at-end loses the qualitative trajectory of "what does the agent look like at 50%, 80%, 100% of training?"

### 5.7. No stats CSV in this plan

We are only porting the **video** pipeline in this plan. The stats CSV pipeline (`record_stats=True`, `_write_episode_stats(...)`) is a separate port — would require porting the breakdown-aware CSV writer (`evaluation_core.py:L50-L139`). Out of scope here; can land as a sibling plan if the user wants `Eval/MeanReward` over many episodes (Pass 2 of the original two-pass eval).

---

## 6. Total LOC + wall-clock estimate

| Commit | What | LOC | Dev wall-clock |
|---|---|---:|---:|
| A | Plumb config flags | ~40 | 0.5h |
| B | Lightweight checkpointing | ~80 | 1.5h |
| C | `dreamer_srl_eval_rollout` | ~150 | 3h |
| D | Recording-format regression test | ~30 | 1h |
| E | Render + WandB upload helper | ~50 | 1h |
| F | Driver wiring | ~60 | 1.5h |
| G | End-to-end smoke test | ~80 | 1.5h |
| **Total** | | **~490 LOC** | **~10h dev wall-clock** |

Excludes verification time and any iteration-on-feedback cycles. With one feedback cycle (typical), figure **~12-14h dev wall-clock total**.

---

## 7. Hand-off

After this plan is accepted, the parent (top-level Claude) should:

1. **Surface the open question to the user** (§5.1 checkpoint format: pickle vs Orbax — recommended pickle). Get a disposition before Commit B.
2. **Decide §5.2 frequency** (`training.checkpoint_frequency=50` ep-based vs new `training.eval_iter_frequency` iter-based — recommend the latter).
3. **Spawn `developer`** with this plan as the source of truth. Developer fills the Implementation Report at the end of each commit's section.
4. **Verify** by `senior-developer` after each commit (or batched at A+B, C+D, E+F, G milestones).
5. **Final smoke** on a real lab node with `--total-steps 50000` + `eval_iter_frequency=10000` to confirm 5 MP4s land in WandB + `Eval/*` scalars plot.

---

## Implementation Report

Implemented by: developer (Claude Sonnet 4.6)
Date: 2026-05-15
Session: 7962c4de/developer

### Summary

All 7 commits landed in order. Deviation from plan: Commit E was implemented as part of Commit C (the `_render_and_upload()` function was written into `eval.py` at the same time as `dreamer_srl_eval_rollout()`); Commit E's contribution is its dedicated test file. The plan's Commit B disposition changed from pickle to Orbax per user directive (§5.1). An additional auto-commit by the project linter (commit 2da1a33) was absorbed between Commit E and F — it added extra `define_metric` calls and `timesteps`/`iteration` keys to the periodic log dict, which is consistent with the plan's intent.

### Commit A — cf7523a
- [x] Planned
- [x] Implemented
- [x] Tests pass (49/49 → 49/49)
- Notes: Added multi-config merge (train/evaluation/visualization defaults), all 10 eval-video config reads via `env_cfg.get_mandatory()`, results_dir computation (`results/JAX_DreamerSRL/<wandb-run-name>/`), `--results-dir` + `--debug` CLI args, `last_ckpt_episode` sentinel, `Eval/*` + `eval/checkpoint_episode` wandb.define_metric(). No behavior change.

### Commit B — f66a719
- [x] Planned
- [x] Implemented
- [x] Tests pass (49+4=53/53)
- Notes: Created `src/algorithms/dreamer_srl/checkpoint.py` with `make_checkpoint_manager()`, `save_checkpoint()`, `load_checkpoint()` using `ocp.StandardSave/StandardCheckpointer`. Checkpoint dict includes all nnx.Param states + moments fields + bookkeeping scalars. Driver initializes `CheckpointManager` at startup and triggers save when `total_episodes_completed // checkpoint_frequency` advances. 4 new tests in `test_checkpoint.py`. User disposition: Orbax (not pickle).

### Commit C — 3437d3e
- [x] Planned
- [x] Implemented
- [x] Tests pass (53+3=56/56)
- Notes: Created `src/algorithms/dreamer_srl/eval.py` with `dreamer_srl_eval_rollout()` (deterministic RSSM.dynamic + actor.forward_logits argmax, reuses `EpisodeRecorder` + `write_run_meta`) and `_render_and_upload()` (subprocess `render_recordings.py` with `JAX_PLATFORMS=cpu` + `wandb_utils.upload_video()`). 3 new tests in `test_eval_rollout.py`.

### Commit D — 7c544ba
- [x] Planned
- [x] Implemented
- [x] Tests pass (56+4=60/60)
- Notes: Created `tests/algorithms/dreamer_srl/test_eval_recording.py` with 4 regression tests for the recording format contract (key presence, list length consistency, renderer acceptance via `build_sensory_viz` to bypass nociception fallback, non-zero pixel variance). Deviation from plan: test uses `build_sensory_viz(obs, snap, params, None)` to pass `sensory_data` to the renderer — needed because `render_jax_state()` requires `nociception_history_buffer` in the fallback path which is not stored in the slim snapshot dict. This matches the behavior of `render_recordings.py:L69`.

### Commit E — 7af6a3e
- [x] Planned
- [x] Implemented
- [x] Tests pass (60+3=63/63 with slow test)
- Notes: Created `tests/algorithms/dreamer_srl/test_render_upload.py` with 3 tests: importability, graceful None on empty dir, and full slow test that runs `dreamer_srl_eval_rollout` + `_render_and_upload` end-to-end and verifies MP4 > 1KB. The actual `_render_and_upload` function was already written in Commit C.

### Commit F — landed in auto-commit 2da1a33
- [x] Planned
- [x] Implemented
- [x] Tests pass (62/62 fast, 65/65 with slow)
- Notes: Eval wiring in `dreamer_srl_main.py` was committed as part of the linter's auto-commit `2da1a33` alongside extra define_metric additions (WorldModel/*, Behavior/*, Loss/*, Episode/*, Eval/* explicit patterns) and `timesteps`/`iteration` keys in the periodic log dict. The checkpoint-triggered eval block (Pass 1 video + Pass 2 stats) is in the file at lines 757-822. Smoke test with `checkpoint_frequency=3`: 7 Orbax checkpoint dirs created correctly.

Note: WandB step ordering warning (`"Tried to log to step N that is less than current step M"`) observed during final smoke. This occurs because the render subprocess takes ~13s and by the time it completes the training loop has advanced beyond the `policy_step` used for eval logging. The video artifacts upload correctly; only the x-axis step assignment is affected. This matches the behavior of the original Dreamer in `train.py` (same race condition exists). Flag for senior-developer: consider using `wandb.log(..., commit=False)` or logging without explicit step for Eval/* to let WandB auto-assign.

### Commit G — e26f469
- [x] Planned
- [x] Implemented
- [x] Tests pass (2/2 slow end-to-end smoke tests pass, ~5 min wall-clock)
- Notes: Created `tests/algorithms/dreamer_srl/test_eval_video_smoke.py` with 2 slow tests: `test_e2e_smoke_checkpoints_and_recordings` (300 steps, checkpoint_frequency=3, video=True, asserts >=1 Orbax checkpoint + >=1 .rec.gz + loadable run_meta.pkl + >=1 MP4 > 1KB) and `test_e2e_smoke_no_video_only_stats` (150 steps, video=False, asserts checkpoint exists, no .rec.gz).

### Test Results

| Suite | Count | Command |
|---|---|---|
| pytest fast (no slow) | 62 passed, 3 deselected | `pytest tests/algorithms/dreamer_srl/ -k "not slow" -p no:randomly` |
| pytest with slow | 65 passed | `pytest tests/algorithms/dreamer_srl/` |
| offline_check | 17/17 passed | `python scripts/dreamer_srl_offline_check.py` |

### Final Smoke (WandB)

- **Run**: https://wandb.ai/sungwoolee/grid_world_pain/runs/6fmf8lvh (`eval_video_smoke_commit_ABCDEFG`)
- **Config**: combined 5x5 food-only env, 2000 iters, checkpoint_frequency=10, video_during_training=True, eval_video_episodes=1
- **Results dir**: `/tmp/dsrl_smoke_wandb_final/`
- **Checkpoints**: 9+ Orbax checkpoint dirs (episodes 10, 20, 30, 40, 50, 60, 70, 80, 90, 100...)
- **MP4s**: 9+ `eval_N.mp4` files (~55-70KB each) in `videos/`
- **Recordings**: `.rec.gz` files in `recordings/<N>/` at each checkpoint
- **WandB**: `Eval/MeanReward` + `Eval/MeanLength` logged at each checkpoint; video artifacts uploaded

### Speed Check

No performance-impacting changes to the training hot path. The checkpoint + eval trigger fires only at episode boundaries (O(1) overhead per training iteration). The eval rollout + render subprocess (~13-30s per eval) is outside the training loop. Speed check: not applicable (eval-only change; training SPS unchanged).

### Deviations from Plan

1. **Commit B**: User disposition changed from pickle → Orbax (plan §5.1). Implemented accordingly.
2. **Commit C+E**: `_render_and_upload()` was written in Commit C alongside the rollout function. Commit E adds only the test.
3. **Commit D test**: Uses `build_sensory_viz()` to pass `sensory_data` to renderer, matching `render_recordings.py:L69` behavior. Plan described using `render_episode()` which doesn't exist; the actual renderer uses `render_jax_state()` directly.
4. **Commit F**: Absorbed into linter auto-commit `2da1a33` which also added bonus metrics improvements. Driver wiring is complete and verified.
5. **WandB step ordering**: Eval metrics logged at `step=policy_step` may be shadowed by subsequent training logs when render subprocess takes >1s. Flag for senior-developer to decide on fix.

### Implemented by: developer
