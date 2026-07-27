"""Async checkpoint-video render: dispatch / poll / drain.

Port of rPPO's during-training experiment-eval idiom (train.py:256-385 —
`_maybe_dispatch_experiment_eval` / `_poll_and_log_experiment_results` /
`_drain_experiment_results`) applied to the checkpoint-video MP4 render, shared
by BOTH trainers (train.py for RecurrentPPO, dreamer_srl_main.py for Dreamer).
Design doc: docs/develop/active/refactors/ASYNC_CHECKPOINT_VIDEO_RENDER.md.

Why: the in-training render used to run via a blocking `subprocess.run`
(evaluation_core.py / dreamer_srl/eval.py), freezing the training loop (and the
GPU) for the full 15-195 s render. Here the render is `subprocess.Popen`-
dispatched instead; the training loop polls once per iteration (a cheap
`proc.poll()`), and the PARENT uploads the finished MP4 to WandB — the child
never touches WandB (single-writer invariant).

Invariants / policies (all inherited from the source idiom):
  * At most ONE concurrent render child; if the previous render is still
    running when the next checkpoint fires, that checkpoint's video is SKIPPED
    (never queued). The `.rec.gz` recordings are already on disk, so a skipped
    video is offline-recoverable via `scripts/eval/render_recordings.py
    <recordings_dir>` — never a data-loss event. Skips are counted and surfaced
    on WandB as `Eval/video/render_skipped_total` (plan-reviewer finding 4).
  * `dispatch_render` POLLS BEFORE touching pending state (plan-reviewer
    finding 2): a render that finished between the last per-iteration poll and
    this dispatch gets its upload first, instead of being silently overwritten.
  * `drain_render` = bounded `proc.wait()` at exit (RENDER_DRAIN_TIMEOUT_S
    normal / RENDER_DRAIN_TIMEOUT_S_INTERRUPTED on Ctrl-C) followed by a FINAL
    POLL (plan-reviewer finding 1) so the last video of the run is still
    uploaded when the child finishes within the drain window. On timeout the
    child is deliberately LEFT RUNNING (orphaned): the render is finite,
    CPU-only work, and `--skip-existing` makes any later re-render harmless.
    Same orphan class as the known SIGINT behavior (wiki:
    docs/llm_wiki/entries/cluster_ops/20260513_0018_train_py_orphan_render_workers_on_sigint.md).
  * Every step is failure-isolated: a render/upload problem must never crash
    or stall training.

Per-algorithm WandB step-stamping is preserved as-is (upload_step_mode):
  * "policy_step"    — Dreamer: stamp the CURRENT env-step clock passed by the
                       caller at poll time (forward-monotone; commit 39f851b).
  * "checkpoint_pct" — rPPO: stamp the dispatching checkpoint's episode count,
                       exactly as evaluation_core.py's blocking path does.
"""

import os
import subprocess
import sys

# Bounded-drain timeouts (seconds). Same normal-exit value as the experiment-eval
# drain (train.py EXPERIMENT_EVAL_DRAIN_TIMEOUT_S = 300); the interrupted value
# is 10 s per the plan's drain spec.
RENDER_DRAIN_TIMEOUT_S = 300
RENDER_DRAIN_TIMEOUT_S_INTERRUPTED = 10

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Module-level so tests can monkeypatch it with a stub script.
_RENDER_SCRIPT = os.path.join(_PROJECT_ROOT, "scripts", "eval", "render_recordings.py")


def new_render_state():
    """Fresh per-run dispatch/poll/drain state dict."""
    return {
        "proc": None,        # live subprocess.Popen, or None
        "log_f": None,       # open log-file handle for the live child
        "pending": None,     # {"mp4_path", "checkpoint_pct", "recordings_dir", "log_path"}
        "ckpt_index": 0,     # monotonic count of render-triggering checkpoints (every_n gate)
        "skips": 0,          # total skip-if-busy events
        "skips_logged": 0,   # skips already surfaced to WandB
    }


def _render_env():
    """Child env: CPU-only render, belt-and-braces GPU isolation (adopted from
    train.py's _experiment_eval_env) so the render can never touch the training
    GPU now that it runs CONCURRENTLY with training."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def dispatch_render(state, recordings_dir, results_dir, checkpoint_pct, fps,
                    workers=None, every_n=1, quiet=True,
                    wandb_enabled=False, step=None, upload_step_mode="policy_step"):
    """Non-blocking render dispatch for one checkpoint's recordings.

    Returns True if a render child was spawned, False otherwise (every_n gate
    or skip-if-busy). Callers should wrap in try/except (failure isolation),
    like the experiment-eval dispatch call site.
    """
    # Plan-reviewer finding 2: poll BEFORE touching pending state, so a render
    # that finished between the last per-iteration poll and this dispatch gets
    # its upload instead of being silently overwritten.
    try:
        poll_render(state, wandb_enabled=wandb_enabled, step=step,
                    upload_step_mode=upload_step_mode, quiet=quiet)
    except Exception as e:
        print(f"[render] WARNING: pre-dispatch poll failed (non-fatal): {e}")

    # every_n gate (Option 3 folded in; default 1 = render every checkpoint).
    # Recordings/checkpoints are written by the caller regardless — this gates
    # the MP4 only.
    state["ckpt_index"] = state.get("ckpt_index", 0) + 1
    if every_n > 1 and state["ckpt_index"] % every_n != 0:
        return False

    # Skip-if-busy: at most one concurrent render child.
    proc = state.get("proc")
    if proc is not None and proc.poll() is None:
        state["skips"] = state.get("skips", 0) + 1
        if not quiet:
            print(f"[render] checkpoint {checkpoint_pct}: previous render "
                  f"(pid={proc.pid}) still running -- skipping this checkpoint's "
                  f"video (recordings preserved at {recordings_dir}; backfill "
                  f"offline with `python scripts/eval/render_recordings.py "
                  f"{recordings_dir}` -- delete any partial episode_*.mp4 there "
                  f"first, or --skip-existing would keep a truncated file).")
        return False
    state["proc"] = None

    videos_dir = os.path.join(results_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    consolidated_mp4 = os.path.join(videos_dir, f"eval_{checkpoint_pct}.mp4")
    # Exact command the blocking path builds (evaluation_core.py /
    # dreamer_srl/eval.py::_render_and_upload), plus optional --workers.
    cmd = [
        sys.executable, _RENDER_SCRIPT,
        str(recordings_dir),
        "--concat",
        "--skip-existing",
        "--cleanup-per-episode",
        "--fps", str(fps),
    ]
    if workers is not None:
        # NOTE: render_recordings.py parallelises per EPISODE (one task per
        # .rec.gz), so with eval_video_episodes=3 at most 3 workers ever run —
        # this is a cap, not a contention lever (plan-reviewer finding 5).
        cmd += ["--workers", str(workers)]

    log_path = os.path.join(videos_dir, f"render_{checkpoint_pct}.log")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=_render_env(), stdout=log_f, stderr=subprocess.STDOUT)
    state["proc"] = proc
    state["log_f"] = log_f
    state["pending"] = {
        "mp4_path": consolidated_mp4,
        "checkpoint_pct": checkpoint_pct,
        "recordings_dir": str(recordings_dir),
        "log_path": log_path,
    }
    if not quiet:
        print(f"[render] dispatched checkpoint {checkpoint_pct} render "
              f"(pid={proc.pid}); log: {log_path}")
    return True


def _flush_skip_counter(state, wandb_enabled, step, upload_step_mode):
    """Surface skip-if-busy drops on the dashboard (plan-reviewer finding 4).
    Parent-only WandB write; logs only when the counter advanced."""
    if not wandb_enabled:
        return
    skips = state.get("skips", 0)
    if skips <= state.get("skips_logged", 0):
        return
    try:
        import wandb
        if wandb.run is None:
            return
        if upload_step_mode == "policy_step" and step is not None:
            wandb.log({"Eval/video/render_skipped_total": skips}, step=int(step))
        else:
            # rPPO convention: no explicit step (matches its other wandb.log calls).
            wandb.log({"Eval/video/render_skipped_total": skips})
        state["skips_logged"] = skips
    except Exception as e:
        print(f"[render] WARNING: could not log skip counter (non-fatal): {e}")


def poll_render(state, wandb_enabled, step, upload_step_mode="policy_step", quiet=True):
    """Cheap once-per-iteration poll. On child completion: close the log file,
    clear state, and (rc==0, MP4 present, wandb enabled) upload the MP4 from
    the PARENT — the sole WandB writer. Every branch failure-isolated."""
    try:
        _flush_skip_counter(state, wandb_enabled, step, upload_step_mode)
    except Exception:
        pass

    proc = state.get("proc")
    if proc is None:
        return
    try:
        rc = proc.poll()
    except Exception as e:
        print(f"[render] WARNING: poll failed (non-fatal): {e}")
        return
    if rc is None:
        return  # still running

    # Child finished — consume pending state.
    pending = state.get("pending") or {}
    log_f = state.get("log_f")
    state["proc"] = None
    state["log_f"] = None
    state["pending"] = None
    try:
        if log_f is not None:
            log_f.close()
    except Exception:
        pass

    ckpt = pending.get("checkpoint_pct")
    mp4_path = pending.get("mp4_path")
    if rc != 0:
        print(f"[render] Warning: render for checkpoint {ckpt} failed (rc={rc}). "
              f"Recordings preserved at {pending.get('recordings_dir')}; "
              f"log: {pending.get('log_path')}")
        return

    if not (mp4_path and os.path.exists(mp4_path)):
        print(f"[render] Warning: render for checkpoint {ckpt} exited 0 but no MP4 "
              f"at {mp4_path}. Log: {pending.get('log_path')}")
        return

    if not quiet:
        print(f"[render] checkpoint {ckpt} render finished: {mp4_path}")
    if wandb_enabled:
        try:
            from src.utils.wandb_utils import upload_video
            if upload_step_mode == "policy_step":
                upload_step = int(step) if step is not None else None
            else:  # "checkpoint_pct" — rPPO's existing stamping, preserved as-is
                upload_step = ckpt
            upload_video(mp4_path, episode=ckpt, step=upload_step,
                         caption=f"Episode {ckpt}", quiet=True)
        except Exception as e:
            print(f"[render] Warning: WandB video upload failed (non-fatal): {e}")


def drain_render(state, wandb_enabled=False, step=None, upload_step_mode="policy_step",
                 interrupted=False, quiet=True, timeout_s=None):
    """Bounded drain before wandb.finish(). Waits up to RENDER_DRAIN_TIMEOUT_S
    (RENDER_DRAIN_TIMEOUT_S_INTERRUPTED on Ctrl-C) for an in-flight render, then
    runs a FINAL POLL so a render that completed during (or before) the wait
    still gets uploaded (plan-reviewer finding 1 — without this the last video
    of every run would silently never reach WandB; mirrors the source idiom's
    drain-ends-with-poll, train.py:381-383). On timeout the child is
    deliberately LEFT RUNNING — finite CPU-only work; only its upload is missed
    (the MP4 itself still lands on disk)."""
    if timeout_s is None:
        timeout_s = RENDER_DRAIN_TIMEOUT_S_INTERRUPTED if interrupted else RENDER_DRAIN_TIMEOUT_S
    proc = state.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if not quiet:
                print(f"[render] in-flight render (pid={proc.pid}) did not finish "
                      f"within {timeout_s}s at shutdown -- leaving it running in "
                      f"the background (CPU-only, finite; its MP4 will still be "
                      f"written; only its WandB upload is missed).")
        except Exception as e:
            print(f"[render] WARNING: drain wait failed (non-fatal): {e}")

    # Final poll (finding 1): uploads the MP4 if the child finished in time.
    try:
        poll_render(state, wandb_enabled=wandb_enabled, step=step,
                    upload_step_mode=upload_step_mode, quiet=quiet)
    except Exception as e:
        print(f"[render] WARNING: final drain poll failed (non-fatal): {e}")

    # Timeout path: close our log handle (the child keeps its own fd).
    log_f = state.get("log_f")
    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
        state["log_f"] = None
