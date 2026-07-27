"""Tests for the async checkpoint-video render (dispatch / poll / drain).

Plan: docs/develop/active/refactors/ASYNC_CHECKPOINT_VIDEO_RENDER.md
(Test Plan items 1-4, plus the two plan-reviewer amendments:
finding 1 — drain must end with a poll so the LAST video of a run is uploaded;
finding 2 — dispatch must poll before overwriting pending state so a render
finishing between the last poll and the next dispatch is not silently lost).

The render script is monkeypatched to a stub (writes the same
<results>/videos/eval_<ckpt>.mp4 the real renderer writes, after an optional
env-controlled sleep), so no matplotlib/ffmpeg runs. WandB upload is mocked at
src.utils.wandb_utils.upload_video (the exact attribute poll_render resolves
lazily at call time).
"""
import os
import sys
import time
import inspect
import subprocess
from types import SimpleNamespace

import pytest

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

from src.utils import async_render
from src.utils.async_render import (
    new_render_state, dispatch_render, poll_render, drain_render, _render_env,
)


STUB = """\
import os, sys, time
recordings_dir = sys.argv[1]
time.sleep(float(os.environ.get("RENDER_STUB_SLEEP", "0")))
rc = int(os.environ.get("RENDER_STUB_RC", "0"))
if rc == 0:
    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(recordings_dir)))
    ckpt = os.path.basename(os.path.normpath(recordings_dir))
    videos = os.path.join(results_dir, "videos")
    os.makedirs(videos, exist_ok=True)
    with open(os.path.join(videos, "eval_%s.mp4" % ckpt), "w") as f:
        f.write("stub-mp4")
sys.exit(rc)
"""


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Stub render script + results/recordings layout + upload_video recorder."""
    stub = tmp_path / "stub_render.py"
    stub.write_text(STUB)
    monkeypatch.setattr(async_render, "_RENDER_SCRIPT", str(stub))
    monkeypatch.delenv("RENDER_STUB_SLEEP", raising=False)
    monkeypatch.delenv("RENDER_STUB_RC", raising=False)

    results_dir = tmp_path / "results"

    def rec_dir(ckpt):
        d = results_dir / "recordings" / str(ckpt)
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    uploads = []
    import src.utils.wandb_utils as wu
    monkeypatch.setattr(wu, "upload_video",
                        lambda path, **kw: uploads.append((path, kw)))

    return SimpleNamespace(results_dir=str(results_dir), rec_dir=rec_dir,
                           uploads=uploads, monkeypatch=monkeypatch)


def _dispatch(rig, state, ckpt, **kw):
    defaults = dict(
        recordings_dir=rig.rec_dir(ckpt),
        results_dir=rig.results_dir,
        checkpoint_pct=ckpt,
        fps=5,
        quiet=True,
        wandb_enabled=True,
        step=ckpt * 10,
        upload_step_mode="policy_step",
    )
    defaults.update(kw)
    return dispatch_render(state, **defaults)


def _kill(state):
    proc = state.get("proc")
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Test Plan 1 — non-blocking dispatch; poll no-op while running; upload once.
# ---------------------------------------------------------------------------

def test_dispatch_nonblocking_poll_detects_completion_uploads_once(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "2.0")
    state = new_render_state()
    try:
        t0 = time.monotonic()
        assert _dispatch(rig, state, 100) is True
        assert time.monotonic() - t0 < 1.0, "dispatch must not block on the render"
        assert state["proc"] is not None
        assert state["pending"]["checkpoint_pct"] == 100

        # Poll while running: no-op, no upload.
        poll_render(state, wandb_enabled=True, step=1000)
        assert state["proc"] is not None
        assert rig.uploads == []

        # Wait for the child, then poll: exactly one upload, state cleared.
        state["proc"].wait(timeout=20)
        poll_render(state, wandb_enabled=True, step=1234)
        assert len(rig.uploads) == 1
        path, kw = rig.uploads[0]
        assert path.endswith(os.path.join("videos", "eval_100.mp4"))
        assert os.path.exists(path)
        assert kw["episode"] == 100
        assert kw["step"] == 1234  # Dreamer mode: CURRENT policy_step at poll time
        assert state["proc"] is None and state["pending"] is None

        # Further polls: nothing more.
        poll_render(state, wandb_enabled=True, step=1300)
        assert len(rig.uploads) == 1
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# Test Plan 2 — skip-if-busy: one concurrent child max; skip counter surfaced.
# ---------------------------------------------------------------------------

def test_skip_if_busy_single_child_and_skip_counter(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "10.0")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 100) is True
        first_pid = state["proc"].pid

        assert _dispatch(rig, state, 200) is False
        assert state["proc"].pid == first_pid, "no second child may be spawned"
        assert state["pending"]["checkpoint_pct"] == 100, "pending must not be overwritten"
        assert state["skips"] == 1

        # Amendment 4 (finding 4): skip counter reaches WandB at poll time.
        import wandb
        logged = []
        rig.monkeypatch.setattr(wandb, "run", SimpleNamespace(), raising=False)
        rig.monkeypatch.setattr(wandb, "log",
                                lambda d, **kw: logged.append((d, kw)), raising=False)
        poll_render(state, wandb_enabled=True, step=2000)
        assert logged and logged[0][0] == {"Eval/video/render_skipped_total": 1}
        assert state["skips_logged"] == 1
        # Counter unchanged -> not re-logged.
        poll_render(state, wandb_enabled=True, step=2100)
        assert len(logged) == 1
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# Test Plan 3 — bounded drain: returns on timeout, child LEFT RUNNING, log closed.
# ---------------------------------------------------------------------------

def test_drain_bounded_leaves_orphan_and_closes_log(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "30.0")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 100) is True
        log_f = state["log_f"]
        t0 = time.monotonic()
        drain_render(state, wandb_enabled=True, step=999, timeout_s=2)
        elapsed = time.monotonic() - t0
        assert 1.5 <= elapsed < 10, f"drain must return ~timeout_s, took {elapsed:.1f}s"
        assert state["proc"] is not None and state["proc"].poll() is None, \
            "leave-orphan policy: child must NOT be killed on drain timeout"
        assert log_f.closed
        assert state["log_f"] is None
        assert rig.uploads == []  # nothing finished -> nothing uploaded
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# Amendment 1 (finding 1) — drain ends with a poll: the LAST video of a run
# in flight at exit is still uploaded when the child finishes within the window.
# ---------------------------------------------------------------------------

def test_drain_uploads_render_that_finishes_within_window(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "1.0")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 300) is True
        # No per-iteration poll happens again (loop has ended) — straight to drain.
        drain_render(state, wandb_enabled=True, step=5555, timeout_s=30)
        assert len(rig.uploads) == 1, \
            "drain must poll after the wait — otherwise the last video is never uploaded"
        assert rig.uploads[0][1]["episode"] == 300
        assert state["proc"] is None
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# Amendment 2 (finding 2) — dispatch polls BEFORE overwriting pending state:
# a render that finished after the last poll is uploaded, not silently lost.
# ---------------------------------------------------------------------------

def test_dispatch_does_not_overwrite_unpolled_completed_render(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "0")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 100) is True
        # Child finishes, but NO poll_render runs (models the race window
        # between the last per-iteration poll and the next checkpoint).
        state["proc"].wait(timeout=20)
        assert rig.uploads == []

        # Next checkpoint dispatches: the pre-dispatch poll must upload ckpt 100
        # first, then dispatch ckpt 200.
        assert _dispatch(rig, state, 200) is True
        assert len(rig.uploads) == 1 and rig.uploads[0][1]["episode"] == 100
        assert state["pending"]["checkpoint_pct"] == 200

        state["proc"].wait(timeout=20)
        poll_render(state, wandb_enabled=True, step=9999)
        assert len(rig.uploads) == 2 and rig.uploads[1][1]["episode"] == 200
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# every_n gate (Option 3 folded in) + rPPO step-stamping mode.
# ---------------------------------------------------------------------------

def test_every_n_checkpoints_gate(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "0")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 100, every_n=2) is False   # index 1: gated
        assert state["proc"] is None
        assert _dispatch(rig, state, 200, every_n=2) is True    # index 2: fires
        state["proc"].wait(timeout=20)
        poll_render(state, wandb_enabled=True, step=1)
        assert len(rig.uploads) == 1 and rig.uploads[0][1]["episode"] == 200
    finally:
        _kill(state)


def test_rppo_checkpoint_pct_step_mode(rig):
    rig.monkeypatch.setenv("RENDER_STUB_SLEEP", "0")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 400, upload_step_mode="checkpoint_pct") is True
        state["proc"].wait(timeout=20)
        # rPPO mode: upload stamped with the checkpoint episode count, NOT the
        # (here deliberately different) current step.
        poll_render(state, wandb_enabled=True, step=123456,
                    upload_step_mode="checkpoint_pct")
        assert len(rig.uploads) == 1
        assert rig.uploads[0][1]["step"] == 400
        assert rig.uploads[0][1]["episode"] == 400
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# Failure path: rc != 0 -> warning, no upload, state cleared.
# ---------------------------------------------------------------------------

def test_render_failure_no_upload_state_cleared(rig):
    rig.monkeypatch.setenv("RENDER_STUB_RC", "3")
    state = new_render_state()
    try:
        assert _dispatch(rig, state, 100) is True
        state["proc"].wait(timeout=20)
        poll_render(state, wandb_enabled=True, step=1)
        assert rig.uploads == []
        assert state["proc"] is None and state["pending"] is None
    finally:
        _kill(state)


# ---------------------------------------------------------------------------
# Test Plan 4 — blocking fallback is intact (feature off == today's behavior).
# ---------------------------------------------------------------------------

def test_evaluate_jax_checkpoint_defaults_to_blocking():
    """Every existing caller (standalone eval, feature-off training) hits the
    legacy blocking path: the new parameter defaults to None."""
    from src.utils.evaluation_core import evaluate_jax_checkpoint
    sig = inspect.signature(evaluate_jax_checkpoint)
    assert "async_render_state" in sig.parameters
    assert sig.parameters["async_render_state"].default is None


def test_dreamer_blocking_fallback_uses_subprocess_run(monkeypatch, tmp_path):
    """The kill-switch path (_render_and_upload) still runs a BLOCKING
    subprocess.run with the unchanged render command."""
    from src.algorithms.dreamer_srl.eval import _render_and_upload

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rdir = tmp_path / "recordings" / "0"
    rdir.mkdir(parents=True)
    result = _render_and_upload(
        recordings_dir=str(rdir), results_dir=str(tmp_path), checkpoint_pct=0,
        fps=5, wandb_enabled=False, policy_step=0, quiet=True,
    )
    assert result is None  # rc=1 -> None (legacy behavior)
    assert len(calls) == 1
    assert calls[0][1].endswith(os.path.join("scripts", "eval", "render_recordings.py"))
    assert "--concat" in calls[0] and "--skip-existing" in calls[0]


# ---------------------------------------------------------------------------
# Child env: GPU isolation (belt-and-braces, from _experiment_eval_env).
# ---------------------------------------------------------------------------

def test_render_child_env_is_gpu_isolated():
    env = _render_env()
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
