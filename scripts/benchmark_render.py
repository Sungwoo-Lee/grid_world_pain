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
