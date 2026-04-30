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
        def frame_generator():
            for t_in, _ in tasks:
                mp4 = video_dir / (Path(t_in).stem.replace(".rec", "") + ".mp4")
                rdr = imageio.get_reader(str(mp4))
                last_frame = None
                for frame in rdr:
                    yield frame
                    last_frame = frame
                rdr.close()
                # Padding: hold last frame for 5 steps (legacy behavior)
                if last_frame is not None:
                    for _ in range(5):
                        yield last_frame
        consolidated = run_root / "videos" / f"eval_{rec_dir.name}.mp4"
        save_jax_video(frame_generator(), str(consolidated), fps=args.fps, quiet=True)
        print(f"Consolidated → {consolidated}")


if __name__ == "__main__":
    main()
