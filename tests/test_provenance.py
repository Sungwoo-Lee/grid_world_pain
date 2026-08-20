"""Tests for the training-run provenance stamp (src/utils/provenance.py).

WHAT THIS IS ABOUT, in plain English: a training run used to record nothing about *which
version of the code* produced it, so when a bug was fixed we could not tell which runs
were affected. `train.py` now writes `<run>/models/provenance.json` the moment it creates
the run directory. These tests pin the two properties that make that stamp trustworthy:

  1. it contains every field, as valid JSON;
  2. it NEVER crashes training — if `git` is unavailable the fields read `"unknown"` and
     the run continues. This is the one place the project's no-fallback-defaults rule is
     deliberately inverted, so it is tested as such: a future edit that "fixes" the
     best-effort behaviour into a hard failure must break a test.

The collector-side half (reading the stamp back into a trajectory-store manifest, and the
`null` sentinel for runs that predate it) is tested here at the helper level and again
end-to-end in tests/test_trajectory_collection.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "eval" / "traj_collect"))

from src.utils import provenance as pv                               # noqa: E402

EXPECTED_KEYS = {"git_sha", "git_short", "branch", "git_dirty",
                 "started_utc", "python", "argv"}


# ── The stamp is written, complete, and valid JSON ────────────────────────────

def test_write_provenance_writes_complete_valid_json(tmp_path):
    path = pv.write_provenance(tmp_path, argv=["train.py", "--config", "x.yaml"])
    assert path is not None
    p = Path(path)
    assert p.name == "provenance.json"
    assert p.parent == tmp_path

    rec = json.loads(p.read_text())          # raises if not valid JSON
    assert set(rec) == EXPECTED_KEYS, f"key set drifted: {set(rec) ^ EXPECTED_KEYS}"

    # This repo IS a git repo, so these must be real values, not the failure sentinel.
    assert rec["git_sha"] != "unknown" and len(rec["git_sha"]) == 40
    assert rec["git_short"] != "unknown"
    assert rec["git_sha"].startswith(rec["git_short"])
    assert rec["branch"] != "unknown"
    assert isinstance(rec["git_dirty"], bool)
    assert rec["python"] == sys.version.split()[0]
    assert rec["argv"] == ["train.py", "--config", "x.yaml"]

    # ISO 8601 WITH an offset — a bare timestamp is ambiguous across nodes.
    from datetime import datetime
    assert datetime.fromisoformat(rec["started_utc"]).utcoffset() is not None


def test_write_provenance_creates_missing_directory(tmp_path):
    """train.py calls this right after makedirs, but the helper must not assume that."""
    target = tmp_path / "run" / "models"
    assert pv.write_provenance(target, argv=["train.py"]) is not None
    assert (target / "provenance.json").exists()


# ── A broken git NEVER crashes training ───────────────────────────────────────

@pytest.mark.parametrize("failure", ["missing_binary", "nonzero_exit", "timeout"])
def test_git_failure_yields_unknown_and_does_not_raise(tmp_path, monkeypatch, failure):
    """Simulate the three ways git can fail and assert the stamp degrades, not dies.

    `missing_binary` = FileNotFoundError from subprocess (no git on the node);
    `nonzero_exit`   = git ran but refused (not a repo / detached weirdness);
    `timeout`        = the NAS stalled and the call timed out.
    """
    import subprocess as _sp

    class _Fake:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    def fake_run(*a, **kw):
        if failure == "missing_binary":
            raise FileNotFoundError("git")
        if failure == "timeout":
            raise _sp.TimeoutExpired(cmd="git", timeout=10)
        return _Fake()

    monkeypatch.setattr(pv.subprocess, "run", fake_run)

    rec = pv.collect_provenance(argv=["train.py"])          # must not raise
    assert rec["git_sha"] == "unknown"
    assert rec["git_short"] == "unknown"
    assert rec["branch"] == "unknown"
    # "unknown" is a THIRD value, distinct from False: "we could not tell" is not "clean".
    assert rec["git_dirty"] == "unknown"
    # Non-git fields still carry real information.
    assert rec["python"] == sys.version.split()[0]
    assert rec["argv"] == ["train.py"]

    path = pv.write_provenance(tmp_path, argv=["train.py"])  # must not raise either
    assert set(json.loads(Path(path).read_text())) == EXPECTED_KEYS


def test_write_provenance_returns_none_when_file_unwritable(tmp_path, monkeypatch):
    """Even an unwritable destination must not propagate an exception into train.py."""
    def boom(*a, **kw):
        raise PermissionError("read-only filesystem")
    monkeypatch.setattr("builtins.open", boom)
    assert pv.write_provenance(tmp_path, argv=["train.py"]) is None


# ── git_dirty counts TRACKED modifications only ───────────────────────────────

def test_git_dirty_ignores_untracked_files(tmp_path):
    """An untracked scratch file must NOT make a run read as dirty.

    This project writes into `tmp/` and `results/` constantly; if untracked files counted,
    every run would be dirty and the flag would carry no signal. Built as a throwaway repo
    so the assertion holds regardless of the working tree this suite runs in.
    """
    import subprocess
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": __import__("os").environ.get("PATH", ""), "HOME": str(tmp_path)}

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, env=env,
                              capture_output=True, text=True, check=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("one\n")
    git("add", "a.txt")
    git("commit", "-qm", "init")
    assert pv.git_dirty(cwd=tmp_path) is False

    (tmp_path / "scratch.log").write_text("untracked noise\n")
    assert pv.git_dirty(cwd=tmp_path) is False, "untracked file must not count as dirty"

    (tmp_path / "a.txt").write_text("two\n")
    assert pv.git_dirty(cwd=tmp_path) is True, "modified TRACKED file must count as dirty"

    git("add", "a.txt")
    assert pv.git_dirty(cwd=tmp_path) is True, "staged-not-committed must count as dirty"


# ── Collector side: absent → null sentinel, present → real values ─────────────

def _read_training_provenance():
    import collect_trajectories as ct
    return ct.read_training_provenance


def test_collector_records_null_sentinel_when_stamp_absent(tmp_path):
    """Every run trained before 2026-08-20 lacks the file. That must be recorded as
    `null`, distinguishably from a stamped run whose git read failed — and must not
    raise."""
    run_dir = tmp_path / "run"
    (run_dir / "models").mkdir(parents=True)
    got = _read_training_provenance()(run_dir)
    assert got == {"training_git_sha": None, "training_git_dirty": None,
                   "training_started_utc": None}


def test_collector_records_real_values_when_stamp_present(tmp_path):
    run_dir = tmp_path / "run"
    models = run_dir / "models"
    models.mkdir(parents=True)
    pv.write_provenance(models, argv=["train.py", "--config", "x.yaml"])
    truth = json.loads((models / "provenance.json").read_text())

    got = _read_training_provenance()(run_dir)
    assert got["training_git_sha"] == truth["git_sha"] != "unknown"
    assert got["training_git_dirty"] == truth["git_dirty"]
    assert got["training_started_utc"] == truth["started_utc"]


def test_collector_distinguishes_unknown_from_absent(tmp_path):
    """A stamped run whose git could not be read reads `"unknown"`, NOT `null`. The two
    are different claims: 'pre-stamp run' vs 'stamped, git unreadable'."""
    run_dir = tmp_path / "run"
    models = run_dir / "models"
    models.mkdir(parents=True)
    (models / "provenance.json").write_text(json.dumps({
        "git_sha": "unknown", "git_short": "unknown", "branch": "unknown",
        "git_dirty": "unknown", "started_utc": "2026-08-20T00:00:00+00:00",
        "python": "3.11.0", "argv": ["train.py"]}))

    got = _read_training_provenance()(run_dir)
    assert got["training_git_sha"] == "unknown"
    assert got["training_git_dirty"] == "unknown"
    assert got["training_git_sha"] is not None


def test_collector_survives_corrupt_stamp(tmp_path):
    run_dir = tmp_path / "run"
    models = run_dir / "models"
    models.mkdir(parents=True)
    (models / "provenance.json").write_text("{not json")
    with pytest.warns(UserWarning):
        got = _read_training_provenance()(run_dir)
    assert got["training_git_sha"] is None


# ── The readers share ONE implementation ──────────────────────────────────────

def test_eval_rollout_and_collector_delegate_to_shared_helper():
    """No third copy of the git-sha logic: both readers must call src/utils/provenance."""
    sys.path.insert(0, str(_ROOT / "scripts" / "eval"))
    import eval_rollout                                             # noqa: E402
    import collect_trajectories as ct                               # noqa: E402

    # eval_rollout's documented output format is the SHORT sha; the collector's is full.
    assert eval_rollout._get_git_commit() == pv.git_commit(short=True)
    assert ct._git_sha() == pv.git_commit(short=False)
