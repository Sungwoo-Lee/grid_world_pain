# ISSUE 05 — Decouple Video Rendering from Evaluation (Record-then-Render Offline)

> **Status**: PLANNED
> **Opened**: 2026-04-24
> **Related**: [docs/environment/12_renderer.md](../environment/12_renderer.md), [src/utils/evaluation_core.py](../../src/utils/evaluation_core.py), [src/environment/renderer.py](../../src/environment/renderer.py)

---

## Context

`render_jax_state` is a pure matplotlib + NumPy function called inline inside the evaluation loop for every step when `--render` is enabled. This has three downstream pains:

1. **Eval wall-clock time** blows up with render enabled — each frame takes ~100–300 ms on CPU (GridSpec + pod drawing + icon pasting), so a 1000-step episode adds 1.5–5 min of pure render time per episode on top of the JAX step cost.
2. **GPU sits idle** during rendering: the JAX env + policy are extremely fast, but eval blocks on matplotlib on the CPU before advancing.
3. **Videos are skipped** on many evaluation runs because the render cost is prohibitive, which hurts debugging and paper figures.

The renderer is already decoupled in principle — it operates on host-side pytrees via `jax.device_get` — so the fix is an architectural one, not a GPU/JAX issue.

## Analysis

### Current pipeline (`src/utils/evaluation_core.py`)

The single-env path at [evaluation_core.py:369–435](../../src/utils/evaluation_core.py#L369-L435) does per-step inline rendering:

```python
# Single-env render loop (simplified)
for ep in range(num_episodes):
    state = jax_reset(params, reset_key)
    obs = get_observation(state, params)
    all_frames.append(render_jax_state(jax.device_get(state), params, ...))   # ← blocks
    while not done and step_count < max_steps:
        action, _, _, h_state, _ = generic_inference(model, obs[None, :], h_state, eval_mode=True)
        state, reward, done, info = jax_step(state, action_idx, params)
        obs = get_observation(state, params)
        true_obs = get_observation(state, params, apply_noise=False)
        all_frames.append(render_jax_state(jax.device_get(state), params, ...,
                                           sensory_data=build_sensory_viz(obs, state, params, true_obs)))   # ← blocks
# Finally:
save_jax_video(all_frames, video_path, fps=fps)
```

The parallel-env path at [evaluation_core.py:479–644](../../src/utils/evaluation_core.py#L479-L644) **does not render** today — there is a TODO-shaped gap: `all_frames` is left empty on the parallel path (see [evaluation_core.py:273-275](../../src/utils/evaluation_core.py#L273-L275)). Post-hoc rendering fixes this too.

### What the renderer actually reads

From [renderer.py:353](../../src/environment/renderer.py#L353) (signature) and grep of `state.` / `params.` / `info` usage:

| Source | Fields consumed per frame |
|---|---|
| `state` | `agent_pos`, `res_pos`, `res_active`, `pred_pos`, `neutral_pos`, `obs_pos`, `satiation`, `nutrition`, `injury_level` |
| `params` | static per-episode — `height`, `width`, `local_view_size`, `max_satiation`, `max_nutrition`, `max_injury`, `sensor_range`, `visual_sensor_range`, `res_type`, `obs_blocking`, `obs_hides_agent`, `neutral_property`, plus boolean toggles |
| scalars | `episode`, `step`, `train_episode`, `action` |
| `sensory_data` | built from `(obs, state, params, true_obs)` by `build_sensory_viz` at [sensor.py:335](../../src/environment/sensor.py#L335) |
| `info` | **accepted but unused** in current renderer — can be omitted from recordings |
| `icon_config` | loaded from YAML config, static per run |

The stats recorder already snapshots exactly the state fields above (plus `rest_streak`) at [evaluation_core.py:345–356](../../src/utils/evaluation_core.py#L345-L356) / [evaluation_core.py:439–450](../../src/utils/evaluation_core.py#L439-L450). **The recording payload is a strict superset of the renderer input plus `rest_streak`** — we can reuse that collection pathway without collecting anything new except `action`, `obs`, and `true_obs` vectors (already collected).

### Decomposition

```
EVAL (fast, on GPU)                 RENDER (slow, on CPU, parallel)
┌──────────────────────┐            ┌──────────────────────────────┐
│ jax_step loop        │            │ ProcessPoolExecutor(N)       │
│   → collect:         │  disk      │   worker(episode_file):      │
│     • state snapshots├──────────▶ │     load recording           │
│     • obs, true_obs  │  .npz/     │     for step in episode:     │
│     • action, reward │  .pkl      │       sensory_data = build…  │
│   → dump episode.rec │            │       frame = render_jax_… │
│                      │            │     save_jax_video(frames)   │
└──────────────────────┘            └──────────────────────────────┘
```

- **Eval** no longer calls matplotlib at all — it only writes recordings.
- **Rendering script** fans out N workers, one episode per task. Workers load icons + matplotlib once (via pool `initializer`).
- **`params` is serialized once per run**, not per episode (identical across episodes in a single eval).
- Videos are idempotent: recordings can be re-rendered with different `fps` / `icon_scale` / future renderer versions without re-running the agent.

### Why we need the benchmark first

We don't know today:
- **Per-step render cost** under this checkpoint/config combo (could be 80 ms, could be 350 ms — depends on grid size, pod count, visual sensor range).
- **Per-step recording payload size** — a 10×10 grid with typical entity counts is tiny, but a 30×30 grid with 20 obstacles is not.
- **Serialization overhead**: raw pickle vs `np.savez_compressed` vs zstd-pickled.
- **Worker startup cost** vs per-episode render cost — determines whether pooling is even worth it for short episodes.

These numbers drive concrete decisions (format, compression, pool size, whether to chunk multiple episodes per worker task). Phase 0 below produces the numbers.

---

## Implementation Plan

### Design

Three phases, each independently verifiable:

- **Phase 0** — Benchmark script. Measures per-step render time, recording payload size under multiple serialization formats, and worst-case episode memory footprint. Output: a markdown table with decisions.
- **Phase 1** — Recording layer. A `RecordingWriter` that dumps per-step data during eval, and a `RecordingReader` that reconstructs what the renderer needs.
- **Phase 2** — Parallel rendering CLI. `scripts/render_recordings.py` with a `ProcessPoolExecutor` that renders all episodes in a run directory.

Out of scope for this issue: replacing matplotlib with a lighter backend. That is a follow-up once Phase 2 lands (expected 5–10× further speedup, but much larger engineering cost).

### Phase 0 — Benchmark Script

Deliverable: `scripts/benchmark_render.py` — a single-file script that reports concrete numbers before any behavioral change lands. Writes results to `tmp/benchmark_render_<timestamp>.md`.

**Full script content** (Gemini: drop this into `scripts/benchmark_render.py` verbatim, then run it):

```python
"""Benchmark render_jax_state cost and recording payload size.

Purpose: produce concrete numbers for ISSUE_05 (video pipeline decoupling)
before committing to a serialization format or parallelism strategy.

Run from project root:
    python scripts/benchmark_render.py \
        --config configs/<your-default-config>.yaml \
        --steps 300

Writes a markdown report to tmp/benchmark_render_<timestamp>.md.
"""
import argparse
import datetime as _dt
import gzip
import io
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config  # noqa: E402
from src.environment.config_loader import load_env_params  # noqa: E402
from src.environment.core import jax_reset, jax_step  # noqa: E402
from src.environment.sensor import (  # noqa: E402
    get_observation, build_sensory_viz,
)
from src.environment.renderer import render_jax_state  # noqa: E402


def _snapshot_state(state):
    """Minimal per-step snapshot the renderer needs (same fields as stats recorder)."""
    return {
        'agent_pos': np.asarray(state.agent_pos),
        'satiation': float(state.satiation),
        'nutrition': float(state.nutrition),
        'injury_level': float(state.injury_level),
        'rest_streak': int(state.rest_streak),
        'res_pos': np.asarray(state.res_pos),
        'res_active': np.asarray(state.res_active),
        'pred_pos': np.asarray(state.pred_pos),
        'neutral_pos': np.asarray(state.neutral_pos),
        'obs_pos': np.asarray(state.obs_pos),
    }


def _size_raw_pickle(snapshot_list, obs_list, true_obs_list, action_list):
    buf = io.BytesIO()
    pickle.dump({
        'snapshots': snapshot_list,
        'obs': np.stack(obs_list),
        'true_obs': np.stack(true_obs_list),
        'actions': np.asarray(action_list, dtype=np.int32),
    }, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return len(buf.getvalue())


def _size_gzip_pickle(snapshot_list, obs_list, true_obs_list, action_list):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=5) as gz:
        pickle.dump({
            'snapshots': snapshot_list,
            'obs': np.stack(obs_list),
            'true_obs': np.stack(true_obs_list),
            'actions': np.asarray(action_list, dtype=np.int32),
        }, gz, protocol=pickle.HIGHEST_PROTOCOL)
    return len(buf.getvalue())


def _size_npz(snapshot_list, obs_list, true_obs_list, action_list):
    """Stack snapshot fields into arrays, compress with np.savez_compressed."""
    stacked = {}
    for k in snapshot_list[0].keys():
        stacked[f'snap_{k}'] = np.array([s[k] for s in snapshot_list])
    stacked['obs'] = np.stack(obs_list)
    stacked['true_obs'] = np.stack(true_obs_list)
    stacked['actions'] = np.asarray(action_list, dtype=np.int32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **stacked)
    return len(buf.getvalue())


def run_benchmark(config_path: str, max_steps: int = 300):
    with open(config_path) as f:
        config = Config(yaml.safe_load(f))
    params = load_env_params(config)
    icon_config = config.get('visualization.icons', None)

    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)
    obs = get_observation(state, params)

    # Action space size (random-policy rollout is fine for benchmark)
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

    snapshots, obs_list, true_obs_list, action_list = [], [], [], []
    render_times_ms = []

    # Warm-up: exclude first render from timing (matplotlib + icon cache lazy-init)
    true_obs = get_observation(state, params, apply_noise=False)
    sensory_data = build_sensory_viz(np.asarray(obs), state, params, np.asarray(true_obs))
    _ = render_jax_state(jax.device_get(state), params, episode=1, step=0,
                        sensory_data=sensory_data, icon_config=icon_config)

    for step in range(max_steps):
        key, ak = jax.random.split(key)
        action_idx = int(jax.random.randint(ak, (), 0, action_dim))
        state, reward, done, info = jax_step(state, action_idx, params)
        obs = get_observation(state, params)
        true_obs = get_observation(state, params, apply_noise=False)

        # Record payload (host-side copies)
        state_host = jax.device_get(state)
        snapshots.append(_snapshot_state(state_host))
        obs_list.append(np.asarray(obs))
        true_obs_list.append(np.asarray(true_obs))
        action_list.append(action_idx)

        # Time render
        sensory_data = build_sensory_viz(np.asarray(obs), state, params, np.asarray(true_obs))
        t0 = time.perf_counter()
        _ = render_jax_state(state_host, params, episode=1, step=step + 1, action=action_idx,
                             sensory_data=sensory_data, icon_config=icon_config)
        render_times_ms.append((time.perf_counter() - t0) * 1000.0)

        if bool(done):
            break

    # Sizes
    n = len(snapshots)
    raw = _size_raw_pickle(snapshots, obs_list, true_obs_list, action_list)
    gzp = _size_gzip_pickle(snapshots, obs_list, true_obs_list, action_list)
    npz = _size_npz(snapshots, obs_list, true_obs_list, action_list)

    rt = np.asarray(render_times_ms)
    report = [
        f"# Render benchmark — {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Config: `{config_path}`",
        f"- Steps measured: **{n}**",
        f"- Grid: {int(params.height)} × {int(params.width)}, "
        f"local_view={int(params.local_view_size)}",
        "",
        "## Render time per frame (ms)",
        "",
        "| mean | p50 | p90 | p99 | max |",
        "|------|-----|-----|-----|-----|",
        f"| {rt.mean():.1f} | {np.percentile(rt, 50):.1f} | "
        f"{np.percentile(rt, 90):.1f} | {np.percentile(rt, 99):.1f} | {rt.max():.1f} |",
        "",
        f"- **1000-step episode estimated render cost**: {rt.mean() * 1000 / 1000:.1f} s "
        f"(mean × 1000)",
        "",
        "## Recording payload size (bytes for whole episode)",
        "",
        "| format | total | per step |",
        "|--------|-------|----------|",
        f"| raw pickle        | {raw:,} | {raw / n:.0f} |",
        f"| gzip pickle (5)   | {gzp:,} | {gzp / n:.0f} |",
        f"| np.savez_compressed | {npz:,} | {npz / n:.0f} |",
        "",
        "## Decisions this should drive",
        "",
        "- Storage format: pick smallest that stays under 10 MB per 1000-step episode.",
        "- Pool size: if `p90 < 150 ms`, 4 workers already saturate a typical 8-core box; "
        "if `p90 > 300 ms`, go to 8+ workers.",
        "- If `raw pickle > 50 MB`: mandatory compression; otherwise raw pickle is fine "
        "for Phase 1 and compression can wait.",
        "",
    ]

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    out = tmp_dir / f"benchmark_render_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text("\n".join(report))
    print(f"Wrote {out}")
    print("\n".join(report))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a YAML config file")
    ap.add_argument("--steps", type=int, default=300, help="Max steps to benchmark")
    args = ap.parse_args()
    run_benchmark(args.config, args.steps)


if __name__ == "__main__":
    main()
```

**Usage**:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/benchmark_render.py \
    --config configs/<your-default-config>.yaml --steps 300
```

Run against the reference checkpoint's config (whichever config matches `results/JAX_recurrentPPO/20260422-011429_rppo_MC_basic-03-prop75_std4-noise/`). Paste the resulting `tmp/benchmark_render_*.md` table into the **Implementation Report** below before proceeding to Phase 1. The numbers dictate:

- Format choice for Phase 1 (`RECORDING_FORMAT` constant).
- Whether to ship compression in Phase 1 or defer to a later pass.
- Default `--workers` for Phase 2 CLI.

### Phase 1 — Recording Layer

New module: `src/utils/eval_recording.py`.

```python
"""Episode recording format for offline (post-hoc) video rendering.

Each episode is serialized to ONE file per episode:

    <results_dir>/recordings/<checkpoint_pct>/episode_<NNNNNN>.rec

Plus ONE shared metadata file per eval run:

    <results_dir>/recordings/<checkpoint_pct>/run_meta.pkl
    (contains: params pytree, icon_config, action_map, config path, git sha)

Format: pickle-gzip (level 5) chosen based on Phase 0 benchmark.
[GEMINI: replace default format here if benchmark says otherwise.]
"""
import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

RECORDING_FORMAT_VERSION = 1
_DEFAULT_COMPRESSLEVEL = 5


def _snapshot_state(state) -> Dict[str, Any]:
    """Exactly the fields render_jax_state reads. Keep in lockstep with renderer.py."""
    return {
        'agent_pos': np.asarray(state.agent_pos),
        'satiation': float(state.satiation),
        'nutrition': float(state.nutrition),
        'injury_level': float(state.injury_level),
        'rest_streak': int(state.rest_streak),
        'res_pos': np.asarray(state.res_pos),
        'res_active': np.asarray(state.res_active),
        'pred_pos': np.asarray(state.pred_pos),
        'neutral_pos': np.asarray(state.neutral_pos),
        'obs_pos': np.asarray(state.obs_pos),
    }


class EpisodeRecorder:
    """Accumulates per-step data for one episode, then writes a single file."""

    def __init__(self, episode_index: int, train_episode: int, seed: int):
        self.episode_index = int(episode_index)
        self.train_episode = int(train_episode)
        self.seed = int(seed)
        self.snapshots: List[Dict[str, Any]] = []
        self.obs: List[np.ndarray] = []
        self.true_obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def append(self, state, obs, true_obs, action_idx: int, reward: float):
        self.snapshots.append(_snapshot_state(state))
        self.obs.append(np.asarray(obs))
        self.true_obs.append(np.asarray(true_obs) if true_obs is not None else None)
        self.actions.append(int(action_idx))
        self.rewards.append(float(reward))

    def write(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': RECORDING_FORMAT_VERSION,
            'episode_index': self.episode_index,
            'train_episode': self.train_episode,
            'seed': self.seed,
            'snapshots': self.snapshots,
            'obs': np.stack(self.obs),
            'true_obs': (np.stack([t for t in self.true_obs]) if self.true_obs[0] is not None else None),
            'actions': np.asarray(self.actions, dtype=np.int32),
            'rewards': np.asarray(self.rewards, dtype=np.float32),
        }
        with gzip.open(out_path, 'wb', compresslevel=_DEFAULT_COMPRESSLEVEL) as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def write_run_meta(out_dir: Path, params, icon_config, action_map, config_path: str, extras: Dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': RECORDING_FORMAT_VERSION,
        'params': params,           # EnvParams is a Flax struct.dataclass — picklable
        'icon_config': icon_config,
        'action_map': list(action_map),
        'config_path': str(config_path),
        'extras': dict(extras or {}),
    }
    with open(out_dir / 'run_meta.pkl', 'wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_run_meta(recording_dir: Path) -> Dict[str, Any]:
    with open(recording_dir / 'run_meta.pkl', 'rb') as fh:
        return pickle.load(fh)


def load_episode(path: Path) -> Dict[str, Any]:
    with gzip.open(path, 'rb') as fh:
        return pickle.load(fh)
```

### Phase 1 — Wire recorder into eval

Two small edits to `src/utils/evaluation_core.py`. The default behaviour changes: `render_video=True` now writes **recordings**, not frames, and video generation is deferred to `scripts/render_recordings.py`. A new flag `--render-inline` (config key `testing.render_inline`) preserves today's old behavior for local debugging.

#### `src/utils/evaluation_core.py` (signature, around line 135-150)

```python
# BEFORE (line 135-150):
def evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, checkpoint_pct,
                            render_video=False, record_stats=None, wandb_enabled=False, debug=False, quiet=True, num_envs=1, device=None):
    ...
    # Video output setup
    video_dir = os.path.join(results_dir, "videos")
    if render_video:
        os.makedirs(video_dir, exist_ok=True)
        from src.environment.renderer import render_jax_state, save_jax_video

# AFTER:
def evaluate_jax_checkpoint(model, params, config, num_episodes, seed, results_dir, checkpoint_pct,
                            render_video=False, record_stats=None, wandb_enabled=False, debug=False,
                            quiet=True, num_envs=1, device=None):
    ...
    # Video output setup
    video_dir = os.path.join(results_dir, "videos")
    recordings_dir = os.path.join(results_dir, "recordings", str(checkpoint_pct))
    # New default: record-then-render offline. render_inline preserves old behaviour.
    render_inline = bool(config.get('testing.render_inline', False))
    if render_video:
        if render_inline:
            os.makedirs(video_dir, exist_ok=True)
            from src.environment.renderer import render_jax_state, save_jax_video
        else:
            os.makedirs(recordings_dir, exist_ok=True)
            from src.utils.eval_recording import EpisodeRecorder, write_run_meta
            write_run_meta(
                Path(recordings_dir), params, icon_config,
                action_map, getattr(config, 'source_path', ''),
                extras={'checkpoint_pct': checkpoint_pct, 'seed': seed},
            )
```

#### `src/utils/evaluation_core.py::_run_single_env_eval` (lines 369-435)

Replace the two `all_frames.append(render_jax_state(...))` calls with recorder appends when `render_inline=False`:

```python
# BEFORE (line 369-380):
if render_video:
    if debug: print(f"    [Render] Initial frame...", end="", flush=True)
    state_for_render = jax.device_get(state)
    all_frames.append(render_jax_state(
        state_for_render, params_ref, episode=ep+1, step=0,
        train_episode=checkpoint_pct,
        sensory_data=get_sensory_viz(obs, true_obs),
        info=None,
        icon_config=icon_config
    ))

# AFTER:
recorder = None
if render_video:
    if render_inline:
        state_for_render = jax.device_get(state)
        all_frames.append(render_jax_state(
            state_for_render, params_ref, episode=ep+1, step=0,
            train_episode=checkpoint_pct,
            sensory_data=get_sensory_viz(obs, true_obs),
            info=None, icon_config=icon_config,
        ))
    else:
        from src.utils.eval_recording import EpisodeRecorder
        recorder = EpisodeRecorder(episode_index=ep+1, train_episode=checkpoint_pct, seed=seed)
        recorder.append(jax.device_get(state), obs, true_obs, action_idx=-1, reward=0.0)
```

Symmetric change for the per-step render call at line 425-434:

```python
# BEFORE:
if render_video:
    state_for_render = jax.device_get(state)
    all_frames.append(render_jax_state(
        state_for_render, params_ref, ..., sensory_data=get_sensory_viz(next_obs, true_obs), ...))

# AFTER:
if render_video:
    if render_inline:
        state_for_render = jax.device_get(state)
        all_frames.append(render_jax_state(
            state_for_render, params_ref, ..., sensory_data=get_sensory_viz(next_obs, true_obs), ...))
    elif recorder is not None:
        recorder.append(jax.device_get(state), next_obs, true_obs, action_idx=action_idx, reward=float(reward))
```

After the episode loop, write the recording:

```python
# NEW (after `for _ in range(5): all_frames.append(all_frames[-1])` block, around line 477):
if render_video and recorder is not None and not render_inline:
    from pathlib import Path
    out_path = Path(recordings_dir) / f"episode_{ep+1:06d}.rec.gz"
    recorder.write(out_path)
    if debug:
        print(f"    [Recording] Wrote {out_path}", flush=True)
```

And skip the old `save_jax_video(all_frames, ...)` when not rendering inline — gated by the same flag around line 274-289:

```python
# BEFORE:
if render_video and all_frames:
    video_path = os.path.join(video_dir, f"eval_{checkpoint_pct}.mp4")
    ...
    save_jax_video(all_frames, video_path, fps=fps, quiet=quiet)

# AFTER:
if render_video and render_inline and all_frames:
    video_path = os.path.join(video_dir, f"eval_{checkpoint_pct}.mp4")
    ...
    save_jax_video(all_frames, video_path, fps=fps, quiet=quiet)
elif render_video and not render_inline:
    if not quiet:
        print(f"  --- Recordings written to {recordings_dir}. "
              f"Render with: python scripts/render_recordings.py {recordings_dir} ---", flush=True)
    last_video_path = None
```

#### `src/utils/evaluation_core.py::_run_parallel_env_eval`

The parallel path currently does **not** render (see TODO-shape gap at line 273). Wire recordings into the per-slot completion branch at line 573-588:

```python
# AFTER (new block inside `if dones[i] and slot_active[i]:`, after existing _write_episode_stats):
if render_video and not render_inline:
    from src.utils.eval_recording import EpisodeRecorder
    rec = EpisodeRecorder(episode_index=completed_episodes,
                          train_episode=checkpoint_pct, seed=seed)
    # slot_states[i] contains dicts of per-step state arrays — reconstruct appends
    for t in range(len(slot_states[i])):
        fake_state = _DictState(slot_states[i][t])  # see helper below
        rec.append(fake_state, slot_obs[i][t],
                   (slot_true_obs[i][t] if slot_true_obs is not None else None),
                   action_idx=slot_actions[i][t],
                   reward=slot_rewards[i][t])
    rec.write(Path(recordings_dir) / f"episode_{completed_episodes:06d}.rec.gz")
```

A tiny helper at module scope:

```python
class _DictState:
    """Adapter that lets EpisodeRecorder._snapshot_state read from a dict slot buffer."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
```

### Phase 2 — Parallel Rendering CLI

New script: `scripts/render_recordings.py`.

```python
"""Render saved eval recordings to MP4 in parallel.

Usage:
    python scripts/render_recordings.py <recordings_dir> [--workers N] [--fps 5]

<recordings_dir> is the directory produced by evaluate_jax_checkpoint, e.g.:
    results/<run>/recordings/<checkpoint_pct>/

Writes one MP4 per episode to:
    results/<run>/videos/<checkpoint_pct>/episode_<NNNNNN>.mp4

And a single consolidated:
    results/<run>/videos/eval_<checkpoint_pct>.mp4
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_WORKER_STATE = {}


def _worker_init(run_meta_path: str):
    """Runs once per worker: loads icons + matplotlib + run metadata into globals."""
    import pickle, matplotlib
    matplotlib.use('Agg')
    from src.environment.renderer import _load_icons  # warm icon cache
    from src.utils.eval_recording import load_run_meta
    from pathlib import Path as _P

    meta = load_run_meta(_P(run_meta_path).parent)
    _WORKER_STATE['params'] = meta['params']
    _WORKER_STATE['icon_config'] = meta['icon_config']
    _WORKER_STATE['action_map'] = meta['action_map']
    _WORKER_STATE['checkpoint_pct'] = meta['extras'].get('checkpoint_pct')
    _load_icons(meta['icon_config'])  # side-effect: populates _ICON_CACHE


def _render_episode(episode_path_str: str, out_video_path_str: str, fps: int) -> dict:
    """Worker body: load one episode, render frames, write MP4."""
    from src.environment.renderer import render_jax_state, save_jax_video
    from src.environment.sensor import build_sensory_viz
    from src.utils.eval_recording import load_episode
    import time

    ep = load_episode(Path(episode_path_str))
    params = _WORKER_STATE['params']
    icon_config = _WORKER_STATE['icon_config']
    checkpoint_pct = _WORKER_STATE['checkpoint_pct']

    frames = []
    num_steps = len(ep['snapshots'])
    t0 = time.perf_counter()
    for t in range(num_steps):
        snap = ep['snapshots'][t]

        class _S:
            pass
        s = _S()
        for k, v in snap.items():
            setattr(s, k, v)

        obs_t = ep['obs'][t]
        true_obs_t = ep['true_obs'][t] if ep['true_obs'] is not None else None
        sensory_data = build_sensory_viz(obs_t, s, params, true_obs_t)
        action_t = int(ep['actions'][t]) if ep['actions'][t] >= 0 else None

        frames.append(render_jax_state(
            s, params,
            episode=ep['episode_index'], step=t,
            train_episode=checkpoint_pct,
            action=action_t, sensory_data=sensory_data,
            info=None, icon_config=icon_config,
        ))

    out = Path(out_video_path_str)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jax_video(frames, str(out), fps=fps, quiet=True)
    return {
        'episode_index': ep['episode_index'],
        'steps': num_steps,
        'render_seconds': time.perf_counter() - t0,
        'out': str(out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings_dir", help="Directory containing run_meta.pkl + episode_*.rec.gz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--concat", action="store_true",
                    help="Also write a consolidated MP4 concatenating every episode in order.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip episodes whose MP4 already exists.")
    args = ap.parse_args()

    rec_dir = Path(args.recordings_dir)
    run_meta_path = rec_dir / "run_meta.pkl"
    if not run_meta_path.exists():
        raise SystemExit(f"run_meta.pkl not found in {rec_dir}")

    # videos/<checkpoint_pct>/episode_*.mp4 — sibling of recordings/<checkpoint_pct>
    run_root = rec_dir.parent.parent     # .../results/<run>/
    video_dir = run_root / "videos" / rec_dir.name
    video_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(rec_dir.glob("episode_*.rec.gz"))
    if not episode_files:
        raise SystemExit(f"No episode_*.rec.gz files in {rec_dir}")

    tasks = []
    for ep_file in episode_files:
        out_mp4 = video_dir / (ep_file.stem.replace(".rec", "") + ".mp4")
        if args.skip_existing and out_mp4.exists():
            continue
        tasks.append((str(ep_file), str(out_mp4)))

    if not tasks:
        print("All episodes already rendered.")
        return

    print(f"Rendering {len(tasks)} episodes with {args.workers} workers "
          f"(fps={args.fps}) → {video_dir}")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(run_meta_path),),
    ) as pool:
        futures = [pool.submit(_render_episode, ep, out, args.fps) for ep, out in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            print(f"  ep {r['episode_index']:>4}: {r['steps']} steps, "
                  f"{r['render_seconds']:.1f}s → {r['out']}")

    if args.concat:
        from src.environment.renderer import save_jax_video
        import imageio
        # Simple re-encode concat: decode each MP4, stack frames
        frames = []
        for t_in, _ in tasks:
            mp4 = video_dir / (Path(t_in).stem.replace(".rec", "") + ".mp4")
            rdr = imageio.get_reader(str(mp4))
            frames.extend(list(rdr))
            rdr.close()
        consolidated = run_root / "videos" / f"eval_{rec_dir.name}.mp4"
        save_jax_video(frames, str(consolidated), fps=args.fps, quiet=True)
        print(f"Consolidated → {consolidated}")


if __name__ == "__main__":
    main()
```

### File Changes Summary

| File | Change | Approx lines |
|---|---|---|
| `scripts/benchmark_render.py` | **NEW** — Phase 0 benchmark | +220 |
| `src/utils/eval_recording.py` | **NEW** — `EpisodeRecorder`, `load_episode`, `write_run_meta` | +100 |
| `src/utils/evaluation_core.py` | Wire recorder into both eval paths; add `render_inline` flag | ~50 edits |
| `scripts/render_recordings.py` | **NEW** — parallel MP4 generator | +150 |
| `configs/<eval configs>` | Add `testing.render_inline: false` (default) | +1 per file |

No changes to `src/environment/renderer.py` or `src/environment/renderer_v2.py`.

## Checkpoints

The implementing agent should verify each before moving to the next:

- [ ] **Phase 0 benchmark runs end-to-end** — produces a `tmp/benchmark_render_*.md` with non-zero render times and all three size columns populated. Report the three size numbers in the Implementation Report; if gzip-pickle is >3× smaller than raw pickle, keep the default; otherwise switch `_DEFAULT_COMPRESSLEVEL = 0` and document why.
- [ ] **Phase 1 single-env recording round-trip** — run eval with `render_video=True`, `render_inline=False`, and 1 episode. Confirm `results/<run>/recordings/<ckpt>/episode_000001.rec.gz` and `run_meta.pkl` exist and re-load via `load_episode` / `load_run_meta` without error.
- [ ] **Phase 1 `--render-inline` still works** — rerun the same eval with `testing.render_inline: true`. Confirm `results/<run>/videos/eval_<ckpt>.mp4` is produced exactly as before.
- [ ] **Phase 1 parallel-env recording** — run eval with `num_envs=4`, `num_episodes=8`, `render_video=True`. Confirm 8 `episode_*.rec.gz` files, numbered in completion order.
- [ ] **Phase 2 single-worker render** — `python scripts/render_recordings.py <recordings_dir> --workers 1 --fps 5`. Confirm one MP4 per recording, visually identical to an inline-rendered MP4 from the same seed.
- [ ] **Phase 2 pool render** — same command with `--workers 4`. Confirm total wall-time ≤ `(single_worker_time / 3.5)` (≥3.5× speedup on 4 workers — leaves headroom for pool overhead).
- [ ] **Idempotency**: rerun `render_recordings.py --skip-existing`. Confirm no re-renders and exit code 0.

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

Paste the Phase 0 `tmp/benchmark_render_*.md` table here before Phase 1 starts. Document any format/compression deviation from the plan default and the reason.

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `scripts/benchmark_render.py` | NEW | | |
| `src/utils/eval_recording.py` | NEW | | |
| `src/utils/evaluation_core.py` | Recorder integration + `render_inline` flag | | |
| `scripts/render_recordings.py` | NEW | | |

**Conclusion**: [one-line summary]
