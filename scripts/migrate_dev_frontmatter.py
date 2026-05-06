#!/usr/bin/env python3
"""ONE-SHOT migration: add YAML frontmatter to every docs/develop/*.md.

Run once during the develop-docs reorganization
(docs/develop/DEVELOP_DOCS_REORGANIZATION_PLAN.md). After this script runs and
the migration is verified, this file may be removed (`git rm`); it has no
ongoing role in the project.

For each file under docs/develop/ (recursive), this script:
1. Skips files that already have YAML frontmatter (idempotent).
2. Looks up topic/status/supersedes/superseded_by from a hard-coded triage
   table below.
3. Pulls created/last_updated dates from `git log`.
4. Pulls the title from the first `# H1` line in the file.
5. Inserts a frontmatter block at the top of the file.

Usage:
    python scripts/migrate_dev_frontmatter.py            # dry run
    python scripts/migrate_dev_frontmatter.py --apply    # write changes
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEVELOP = ROOT / "docs" / "develop"

# Triage table: filename -> dict of frontmatter fields.
# Topics must match VALID_TOPICS in regen_dev_index.py.
TRIAGE: dict[str, dict] = {
    # ---- active (flat develop/) ----
    "FILM_MODULATION_PLAN.md": {"topic": "filim", "status": "active"},
    "FiLM_PAPERS_REVIEW.md": {"topic": "filim", "status": "active"},
    "FiLM_ENSEMBLE_SENSORY_PRECISION.md": {"topic": "filim", "status": "active"},
    "PRECISION_MODULATION.md": {"topic": "precision", "status": "active"},
    "PRECISION_MODULATION_ARCHITECTURE.md": {"topic": "precision", "status": "active"},
    "SOLVING_GATE_COLLAPSE.md": {"topic": "precision", "status": "active"},
    "NEUROMODULATION_ALGORITHM.md": {"topic": "neuromodulation", "status": "active"},
    "NMN_ARCHITECTURE_REVIEW.md": {"topic": "neuromodulation", "status": "active"},
    "NMN_METRICS_REFERENCE.md": {"topic": "neuromodulation", "status": "active"},
    "NMN_PERFORMANCE_DIAGNOSIS_v8.md": {
        "topic": "diagnosis",
        "status": "active",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v7.md",
    },
    "DREAMER_DIAGNOSTICS_PLAN_v2.md": {
        "topic": "dreamer",
        "status": "active",
        "supersedes": "DREAMER_DIAGNOSTICS_PLAN_v1.md",
    },
    "DREAMER_REVIEW.md": {"topic": "dreamer", "status": "active"},
    "DREAMER_IMPLEMENTATION_AUDIT.md": {"topic": "dreamer", "status": "active"},
    "CONTINUAL_LEARNING_CONFIG_SCHEDULE.md": {
        "topic": "continual_learning",
        "status": "active",
    },
    "CONTINUAL_LEARNING_REVIEW.md": {
        "topic": "continual_learning",
        "status": "active",
    },
    "BEHAVIOR_ANALYSIS.md": {"topic": "behavior", "status": "active"},
    "TRAINING_METRICS_ANALYSIS.md": {"topic": "behavior", "status": "active"},
    "WANDB_METRICS_REFERENCE.md": {"topic": "behavior", "status": "active"},
    "OLFACTORY_PROPERTY_VARIANCE.md": {"topic": "sensors", "status": "active"},
    "NOISE_RENDERING_RENDERER_FIX.md": {"topic": "sensors", "status": "active"},
    "UNIFY_UNIMODAL_GROUPING.md": {"topic": "sensors", "status": "active"},
    "LAYERNORM_ORTHOGONAL_REFACTOR_PLAN.md": {
        "topic": "refactors",
        "status": "active",
    },
    "MIXTURE_SAMPLING_PLAN.md": {"topic": "refactors", "status": "active"},
    "FIX_PPO_MODULATION_LOGGING.md": {"topic": "refactors", "status": "active"},
    "UI_REDESIGN_PROPOSAL.md": {"topic": "refactors", "status": "active"},
    "IMPORTANT_ISSUES.md": {"topic": "meta", "status": "active"},
    "DEVELOP_DOCS_REORGANIZATION_PLAN.md": {"topic": "meta", "status": "active"},
    # ---- superseded chain (flat develop/) ----
    "NMN_PERFORMANCE_DIAGNOSIS_v1.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v2.md",
    },
    "NMN_PERFORMANCE_DIAGNOSIS_v2.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v1.md",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v3.md",
    },
    "NMN_PERFORMANCE_DIAGNOSIS_v3.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v2.md",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v4.md",
    },
    "NMN_PERFORMANCE_DIAGNOSIS_v4.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v3.md",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v5.md",
    },
    "NMN_PERFORMANCE_DIAGNOSIS_v5.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v4.md",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v6.md",
    },
    "NMN_PERFORMANCE_DIAGNOSIS_v6.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v5.md",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v7.md",
    },
    "NMN_PERFORMANCE_DIAGNOSIS_v7.md": {
        "topic": "diagnosis",
        "status": "superseded",
        "supersedes": "NMN_PERFORMANCE_DIAGNOSIS_v6.md",
        "superseded_by": "NMN_PERFORMANCE_DIAGNOSIS_v8.md",
    },
    # ---- archive (one-off, completed) ----
    "TASKS_2026-04-24.md": {"topic": "meta", "status": "archive"},
    "ISSUE_01_PREDATOR_COUNT.md": {"topic": "issues", "status": "archive"},
    "ISSUE_02_PROPERTY_KEY_UNIFY.md": {"topic": "issues", "status": "archive"},
    "ISSUE_03_DANGER_TO_HIDING_PREDATOR.md": {"topic": "issues", "status": "archive"},
    "ISSUE_04_CHECKPOINT_RETENTION.md": {"topic": "issues", "status": "archive"},
    "ISSUE_05_PERFORMANCE_REPORT.md": {"topic": "issues", "status": "archive"},
    "ISSUE_05_VIDEO_PIPELINE_DECOUPLING.md": {"topic": "issues", "status": "archive"},
    "ISSUE_06_HIDDEN_STATES_INTEROCEPTIVE_NOCICEPTION.md": {
        "topic": "issues",
        "status": "archive",
    },
    "ISSUE_07_AUTO_RENDER_AFTER_EVAL.md": {"topic": "issues", "status": "archive"},
    "ISSUE_08_VIDEO_OUTPUT_AND_INTERO_PANEL.md": {
        "topic": "issues",
        "status": "archive",
    },
    "ISSUE_09_OLFACTION_PARITY_AND_OBSERVABILITY_GATES.md": {
        "topic": "issues",
        "status": "archive",
    },
    # ---- existing archive/ files ----
    "ABLATION_REVIEW.md": {"topic": "behavior", "status": "archive"},
    "BEHAVIORAL_METRICS_PLAN.md": {"topic": "behavior", "status": "archive"},
    "BRANCH_COMPARISON.md": {"topic": "meta", "status": "archive"},
    "DREAMER_DIAGNOSTICS_PLAN_v1.md": {
        "topic": "dreamer",
        "status": "superseded",
        "superseded_by": "DREAMER_DIAGNOSTICS_PLAN_v2.md",
    },
    "EVALUATION_RECORDING_STATS.md": {"topic": "refactors", "status": "archive"},
    "EVALUATION_VIDEO_COMPARISON.md": {"topic": "refactors", "status": "archive"},
    "FIX_EVAL_TRUE_OBS_UNBOUND.md": {"topic": "refactors", "status": "archive"},
    "GAE_VS_MC_RETURNS.md": {"topic": "diagnosis", "status": "archive"},
    "MC_ZMEMORY_CLAMP_BUG.md": {"topic": "diagnosis", "status": "archive"},
    "NETWORK_ENCODING_REVIEW.md": {"topic": "refactors", "status": "archive"},
    "NOISE_DEBUGGING_PLAN.md": {
        "topic": "noise",
        "status": "superseded",
        "superseded_by": "NOISE_DEBUGGING_PLAN_V2.md",
    },
    "NOISE_DEBUGGING_PLAN_V2.md": {
        "topic": "noise",
        "status": "archive",
        "supersedes": "NOISE_DEBUGGING_PLAN.md",
    },
    "OBJECT_GENERATION.md": {"topic": "refactors", "status": "archive"},
    "OBJECT_GENERATION_AND_REST.md": {"topic": "refactors", "status": "archive"},
    "OBSERVATION_SCALE_PLAN.md": {"topic": "sensors", "status": "archive"},
    "OBSTACLE_SYSTEM_REVIEW.md": {"topic": "sensors", "status": "archive"},
    "OLFACTORY_SYSTEM_REVIEW.md": {"topic": "sensors", "status": "archive"},
    "RESTORE_EVALUATION_CORE.md": {"topic": "refactors", "status": "archive"},
    "RPPO_DIAGNOSTICS_PLAN.md": {"topic": "diagnosis", "status": "archive"},
    "VERIFY_NOISE_DISABLED.md": {"topic": "noise", "status": "archive"},
    "VIDEO_TRUE_OBS_DIAGNOSIS.md": {"topic": "diagnosis", "status": "archive"},
    "WANDB_LOG_INTERVAL.md": {"topic": "behavior", "status": "archive"},
    "WANDB_SKILL_GENERALIZATION_PLAN.md": {"topic": "meta", "status": "archive"},
    "temp_wandb_extraction.md": {"topic": "meta", "status": "archive"},
    "NOISE_RENDERING_DEBUG.md": {"topic": "sensors", "status": "archive"},
    # The frontmatter contract doc itself already has frontmatter — skipped automatically.
}


def git_date_created(path: Path) -> str:
    """Earliest commit date for a file (`git log --diff-filter=A`)."""
    rel = path.relative_to(ROOT)
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--follow", "--format=%cs", "--", str(rel)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    lines = [ln for ln in out.stdout.strip().splitlines() if ln]
    if lines:
        return lines[-1]  # earliest
    # New file not yet committed — use today.
    from datetime import date

    return date.today().isoformat()


def git_date_updated(path: Path) -> str:
    """Most recent commit date for a file."""
    rel = path.relative_to(ROOT)
    out = subprocess.run(
        ["git", "log", "-1", "--follow", "--format=%cs", "--", str(rel)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    line = out.stdout.strip()
    if line:
        return line
    from datetime import date

    return date.today().isoformat()


H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def extract_title(text: str, fallback: str) -> str:
    m = H1_RE.search(text)
    if m:
        return m.group(1).strip()
    return fallback


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n")


def build_frontmatter(fields: dict) -> str:
    """Build a YAML frontmatter block. Use stable key order for diffability."""
    order = [
        "title",
        "topic",
        "status",
        "created",
        "last_updated",
        "supersedes",
        "superseded_by",
        "phase",
    ]
    lines = ["---"]
    for k in order:
        if k not in fields:
            continue
        v = fields[k]
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, str):
            # Quote if contains special chars; otherwise leave bare for readability.
            if any(ch in v for ch in ":#{}[],&*!|>'\"%@`"):
                v_str = '"' + v.replace('"', '\\"') + '"'
            else:
                v_str = v
            lines.append(f"{k}: {v_str}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def migrate_file(path: Path, apply: bool) -> str:
    name = path.name
    text = path.read_text()

    if has_frontmatter(text):
        return f"SKIP (has frontmatter): {path.relative_to(ROOT)}"

    if name not in TRIAGE:
        return f"SKIP (not in triage table): {path.relative_to(ROOT)}"

    triage = TRIAGE[name]
    fields = {
        "title": extract_title(text, fallback=name.removesuffix(".md")),
        "topic": triage["topic"],
        "status": triage["status"],
        "created": git_date_created(path),
        "last_updated": git_date_updated(path),
    }
    if "supersedes" in triage:
        fields["supersedes"] = triage["supersedes"]
    if "superseded_by" in triage:
        fields["superseded_by"] = triage["superseded_by"]

    new_text = build_frontmatter(fields) + "\n" + text

    if apply:
        path.write_text(new_text)
        return f"WROTE: {path.relative_to(ROOT)} ({fields['topic']}/{fields['status']})"
    return f"WOULD WRITE: {path.relative_to(ROOT)} ({fields['topic']}/{fields['status']})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually modify files")
    args = ap.parse_args()

    if not DEVELOP.exists():
        print(f"docs/develop/ not found at {DEVELOP}", file=sys.stderr)
        return 1

    files = sorted(DEVELOP.rglob("*.md"))
    untriaged = []
    for f in files:
        result = migrate_file(f, apply=args.apply)
        print(result)
        if "not in triage table" in result and f.name not in {
            "INDEX.md",
            "FRONTMATTER_CONTRACT.md",
        }:
            untriaged.append(f.name)

    if untriaged:
        print("\nWARNING — untriaged files:", file=sys.stderr)
        for n in untriaged:
            print(f"  - {n}", file=sys.stderr)

    if not args.apply:
        print("\n(dry run — pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
