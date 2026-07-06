"""Render saved eval recordings to MP4 in parallel.

Usage:
    python scripts/render_recordings.py <recordings_dir> [--workers N] [--fps 5]

<recordings_dir> is the directory produced by evaluate_jax_checkpoint, e.g.:
    results/<run>/recordings/<checkpoint_pct>/

Writes one MP4 per episode to:
    results/<run>/videos/<checkpoint_pct>/episode_<NNNNNN>.mp4

And a single consolidated:
    results/<run>/videos/eval_<checkpoint_pct>.mp4

Fully decoupled from the stats/results-table path: the behavior-probe pipeline
(`eval_rollout.py --record` → `avoidance_stats_heatmap.py`) reads `.rec.gz`
recordings directly and never invokes this script. Rendering is always a
separate, optional, later invocation over the same recordings directory —
run it only when you actually want videos.

FD-limit note: each rendered episode opens ~250 matplotlib font/icon file
descriptors, so `--workers` parallel renders can exhaust a low soft
`RLIMIT_NOFILE` (some background/non-interactive launch contexts, e.g. nohup'd
jobs or job schedulers, default the soft limit to ~1024) and crash with
`OSError: [Errno 24] Too many open files`. This script raises its own soft
limit toward the hard limit at startup (see `_raise_fd_limit`) so a single
`render_recordings.py --workers N` invocation is robust regardless of launch
context — prefer this over launching multiple separate render processes,
which would only multiply FD pressure rather than fixing it.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_WORKER_STATE = {}

# Target soft RLIMIT_NOFILE. Each render worker opens ~250 font/icon fds via
# matplotlib; this comfortably covers a couple dozen parallel workers plus
# normal process overhead (stdio, pipes, sockets).
_TARGET_SOFT_NOFILE = 8192


def _raise_fd_limit(target: int = _TARGET_SOFT_NOFILE) -> None:
    """Raise the soft RLIMIT_NOFILE toward the hard limit, best-effort.

    Background/non-interactive launch contexts (nohup, some SSH non-interactive
    commands, job schedulers) commonly default the soft fd limit to ~1024 even
    when the hard limit is much higher (or unlimited). Rendering several
    episodes in parallel with matplotlib (~250 fds per render) can exhaust a
    1024 soft limit with only a handful of concurrent workers, crashing with
    `OSError: [Errno 24] Too many open files`. Call this once, in the main
    process, before spawning the worker pool (fork inherits the raised limit),
    so the fix applies regardless of how the script was launched. Never
    raises: if the limit can't be changed (e.g. sandboxed further, or already
    at the hard cap), it silently leaves the limit as-is.
    """
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= target:
            return
        new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError):
        pass  # never crash the render because we couldn't raise the fd limit


def _worker_init(run_meta_path: str):
    """Runs once per worker: loads icons + matplotlib + run metadata into globals."""
    import pickle, matplotlib
    matplotlib.use('Agg')
    from src.environment.renderer import _load_icons  # warm icon cache
    from src.utils.eval_recording import load_run_meta
    from pathlib import Path as _P

    _raise_fd_limit()  # best-effort; also applied in the parent, but cheap and safe to repeat
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
    ap.add_argument("--cleanup-per-episode", action="store_true",
                    help="After --concat succeeds, delete per-episode MP4s. Keeps only eval_<pct>.mp4.")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Render only the first N episodes (after --stride selection, if given). "
                         "Default: render all episodes (backward-compatible).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Render only every Nth episode (1 = every episode, the default). "
                         "Useful for a fast representative-subset pass instead of a full sweep.")
    args = ap.parse_args()

    _raise_fd_limit()  # do this before creating the worker pool so forked workers inherit it

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

    if args.stride > 1:
        episode_files = episode_files[::args.stride]
    if args.max_episodes is not None:
        episode_files = episode_files[:args.max_episodes]
    if not episode_files:
        raise SystemExit(f"--stride/--max-episodes selected 0 episodes from {rec_dir}")

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

        if args.cleanup_per_episode:
            removed = 0
            for ep_file in episode_files:
                out_mp4 = video_dir / (ep_file.stem.replace(".rec", "") + ".mp4")
                if out_mp4.exists():
                    out_mp4.unlink()
                    removed += 1
            
            # Remove the now-empty per-episode directory if it has no other content
            try:
                video_dir.rmdir()  # only succeeds if empty
            except OSError:
                pass  # not empty (e.g., manual files); leave it
            print(f"Cleaned up {removed} per-episode MP4(s); kept consolidated only.")


if __name__ == "__main__":
    main()
