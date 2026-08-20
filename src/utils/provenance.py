"""provenance.py — record WHICH CODE produced a run.

Why this module exists
----------------------
A training run used to record no version of the code that produced it. When a
config-resolution bug was fixed on 2026-07-23 (`828b77e`) and we needed to know which
runs were affected, there was no recorded fact to consult — 334 saved configs under
`results/` had to be scanned for a structural fingerprint. That worked only because that
particular bug happened to leave one; the next one might not.

Note the asymmetry this closes: the tools that READ runs
(`scripts/eval/eval_rollout.py`, `scripts/eval/traj_collect/collect_trajectories.py`)
already stamped their own git commit into their output. The tool that CREATES runs
(`train.py`) did not.

THE ONE PLACE THE NO-FALLBACK-DEFAULTS RULE DOES NOT APPLY
----------------------------------------------------------
Every function here is best-effort and returns the string ``"unknown"`` rather than
raising. That is deliberate and is NOT an oversight to be "fixed" into a hard failure:
provenance is a diagnostic note, not a critical config. A training run that dies at
startup because `git` was missing, the repo was in a detached/odd state, or a subprocess
timed out on a slow NAS would be a strictly worse outcome than a run carrying an
incomplete note. The project's `config.get_mandatory()` rule governs values training
*depends on*; nothing here is one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN = "unknown"

# Short enough that a wedged/NFS-stalled git can never hold up training startup, long
# enough that a cold git on this NAS still answers.
_GIT_TIMEOUT_S = 10


def _git(args: Sequence[str], cwd: Optional[Union[str, Path]] = None):
    """Run a git command; return CompletedProcess, or None if git could not run at all."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True,
            cwd=str(cwd or PROJECT_ROOT), timeout=_GIT_TIMEOUT_S,
        )
    except Exception:
        return None


def git_commit(short: bool = False, cwd=None) -> str:
    """HEAD sha — full 40 chars by default, abbreviated with ``short=True``.

    Returns ``"unknown"`` on any failure (no git, not a repo, timeout).
    """
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    r = _git(args, cwd)
    if r is None or r.returncode != 0:
        return UNKNOWN
    return r.stdout.strip() or UNKNOWN


def git_branch(cwd=None) -> str:
    """Current branch name, or ``"unknown"`` on failure.

    A detached HEAD reports ``"HEAD"`` from git itself — that is a real answer, not a
    failure, and is passed through as-is so a reader can tell it apart from ``"unknown"``.
    """
    r = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if r is None or r.returncode != 0:
        return UNKNOWN
    return r.stdout.strip() or UNKNOWN


def git_dirty(cwd=None) -> Union[bool, str]:
    """Were there uncommitted edits to TRACKED files at this moment?

    This matters as much as the sha: if it is true, the sha does not fully describe what
    ran and any later reader should distrust it.

    TRACKED FILES ONLY (`git status --porcelain --untracked-files=no`, i.e.
    `git diff --quiet HEAD` semantics). This project writes scratch artifacts into `tmp/`
    and training outputs into `results/` as a matter of routine, so counting untracked
    files would mark essentially every run dirty and the flag would carry no signal. The
    question worth recording is "did the code differ from the commit", and only tracked
    modifications can answer that. Staged-but-uncommitted changes DO count (porcelain
    reports both index and worktree columns).

    Returns ``True``/``False``, or the string ``"unknown"`` on any failure — a distinct
    third value, because "we could not tell" is not "it was clean".
    """
    r = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    if r is None or r.returncode != 0:
        return UNKNOWN
    return bool(r.stdout.strip())


def collect_provenance(argv: Optional[Sequence[str]] = None, cwd=None) -> dict:
    """Build the provenance record. Never raises."""
    return {
        "git_sha": git_commit(short=False, cwd=cwd),
        "git_short": git_commit(short=True, cwd=cwd),
        "branch": git_branch(cwd=cwd),
        "git_dirty": git_dirty(cwd=cwd),
        # tz-aware, so `.isoformat()` carries an explicit "+00:00" offset — an ISO 8601
        # timestamp with no offset is ambiguous across the nodes this project runs on.
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "argv": list(argv if argv is not None else sys.argv),
    }


def write_provenance(out_dir, argv: Optional[Sequence[str]] = None,
                     filename: str = "provenance.json", cwd=None) -> Optional[str]:
    """Write ``<out_dir>/provenance.json`` and return its path (None if it could not be
    written).

    MUST NEVER RAISE — see the module docstring. Called at training startup, right after
    `models_dir` is created, so that a run killed ten minutes in still carries its
    provenance. Writing it at the end would mean the runs most worth investigating (the
    ones that died) are exactly the ones without it.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(str(out_dir), filename)
        with open(path, "w") as f:
            json.dump(collect_provenance(argv=argv, cwd=cwd), f, indent=2)
            f.write("\n")
        return path
    except Exception as e:                      # unwritable dir, full disk, NAS hiccup …
        try:
            print(f"[provenance] could not write provenance file under {out_dir}: {e} "
                  "(continuing — provenance is diagnostic, never fatal)",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
        return None
