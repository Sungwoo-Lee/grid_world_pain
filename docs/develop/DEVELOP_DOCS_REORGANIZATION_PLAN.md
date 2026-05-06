# Develop Docs Reorganization Plan

> **Status**: PLANNED
> **Opened**: 2026-05-06
> **Related**: [project_plan.md](../project/project_plan.md), [CLAUDE.md](../../CLAUDE.md)

---

## Context

`docs/develop/` currently contains 42 active markdown files in a flat directory plus 22 in `archive/`. The directory mixes design plans, diagnosis series, paper reviews, refactor proposals, and one-off issue write-ups, with no systematic way to tell what's load-bearing right now versus what has been superseded. The pain points:

1. **Discovery** — an agent or new contributor cannot tell which doc to read first for a given topic without scanning all 42 files.
2. **Staleness** — `NMN_PERFORMANCE_DIAGNOSIS_v1` through `v8` all sit at the same level. A reader landing on `v3` has no signal it's been superseded twice over.
3. **No supersession trail** — when v8 replaces v7, there is currently no link forward from v7 → v8 or backward from v8 → v7. Reasoning history is lost.
4. **Ad-hoc archiving** — `archive/` exists but the move criterion is implicit; some superseded docs are still flat.

This plan introduces a lightweight, mostly-automated system for managing the lifecycle of develop docs without imposing manual maintenance overhead.

## Analysis

### Why the Obvious Solutions Don't Work

- **Timestamp-prefixed filenames** (`YYYYMMDD_HHMMSS_*.md`): redundant with `git log --diff-filter=A`; doesn't track currency, only birth date; pollutes filenames.
- **Hand-maintained `INDEX.md`**: same staleness problem as the docs themselves — one more thing to forget. Drifts the moment a contributor adds a doc and skips the index update.
- **Pure topic subfolders** (`FiLM/`, `precision/`): solves discovery but not staleness — a superseded `NMN_PERFORMANCE_DIAGNOSIS_v3` would still sit next to `v8` inside the same topic folder.

### What Actually Works (Industry Pattern)

The fix combines three independent mechanisms, each addressing one of the three pain points:

| Pain point | Mechanism |
|---|---|
| Discovery | Auto-generated `INDEX.md` from frontmatter |
| Staleness | Frontmatter `status` field + lifecycle folder split (`active/` vs `archive/`) |
| Supersession trail | `supersedes:` / `superseded_by:` frontmatter fields forming a linked chain |

Topic clustering (`FiLM/`, `precision/`, ...) is added inside `active/` for human navigation but is not the primary lifecycle signal.

## Implementation Plan

### Design

**1. Frontmatter Contract**

Every doc under `docs/develop/{active,archive}/` must carry this YAML frontmatter:

```yaml
---
title: "Human-readable title"
topic: filim | precision | noise | hypervigilance | neuromodulation | diagnosis | dreamer | continual_learning | behavior | sensors | refactors | issues | meta
status: active | superseded | archive
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
supersedes: <prev-doc-filename>.md      # optional
superseded_by: <next-doc-filename>.md   # optional, fills in later
phase: 1 | 2 | 3 | 4 | null              # which project_plan phase this serves
---
```

Required fields: `title`, `topic`, `status`, `created`, `last_updated`. Others optional.

**2. Folder Layout (Target State)**

```
docs/develop/
├── INDEX.md                   ← AUTO-GENERATED; do not hand-edit
├── DEVELOP_DOCS_REORGANIZATION_PLAN.md  ← this plan, lives here until complete
├── active/                     ← every doc here is load-bearing right now
│   ├── filim/
│   ├── precision/
│   ├── neuromodulation/
│   ├── diagnosis/              ← only the latest (v8) stays here
│   ├── dreamer/
│   ├── continual_learning/
│   ├── behavior/
│   ├── sensors/
│   ├── refactors/
│   ├── issues/                 ← in-progress only; closed issues archive
│   └── meta/                   ← IMPORTANT_ISSUES, TASKS_*
└── archive/                    ← superseded; kept for traceability
    └── (existing 22 files plus migrated v1–v7 etc.)
```

`status: superseded` and `status: archive` both live under `archive/`. The distinction is semantic: `superseded` carries a `superseded_by:` link forward; `archive` is for one-offs (ad-hoc bug fixes, completed one-time tasks) without a successor.

**3. Index Generation Script**

`scripts/regen_dev_index.py` walks `docs/develop/`, parses frontmatter, and rewrites `docs/develop/INDEX.md`. It runs deterministically — same input always produces the same output — so it can be safely re-run and diffed.

**4. Cross-Reference Discipline**

- [project_plan.md](../project/project_plan.md) links only to docs under `docs/develop/active/`. When a doc is superseded, its incoming links from the project plan must be updated to the successor at the same time as the move.
- Active docs may link to archived predecessors via `supersedes:` frontmatter, but body-text references should point to the current active doc.
- The auto-generated `INDEX.md` validates `supersedes:` / `superseded_by:` chains and fails if they dangle.

### Migration Strategy: Three Phases

**Phase A — System setup (no doc moves yet)**

A.1. Create empty folder structure: `docs/develop/active/{filim,precision,neuromodulation,diagnosis,dreamer,continual_learning,behavior,sensors,refactors,issues,meta}` and `docs/develop/archive/` (already exists).

A.2. Implement `scripts/regen_dev_index.py` (see File Changes below).

A.3. Document the frontmatter contract in `docs/develop/active/meta/FRONTMATTER_CONTRACT.md`.

A.4. Verify the script runs on the empty structure and emits a valid (mostly empty) `INDEX.md`.

**Phase B — Frontmatter mechanical pass**

For every existing doc under `docs/develop/` (flat) and `docs/develop/archive/`:

B.1. Add the frontmatter block. Required fields filled from:
   - `title`: derived from `# H1` of the doc.
   - `topic`: assigned per the categorization in the table below.
   - `status`: `active` for flat-folder docs by default; `archive` for already-archived; `superseded` for `NMN_PERFORMANCE_DIAGNOSIS_v1–v7` and `DREAMER_DIAGNOSTICS_PLAN_v1`.
   - `created`: `git log --diff-filter=A --format=%cs -- <file> | tail -1`.
   - `last_updated`: `git log -1 --format=%cs -- <file>`.

B.2. For superseded series, fill `supersedes:` and `superseded_by:` chains.

B.3. Re-run the index script; confirm it validates clean.

**Phase C — Lifecycle moves**

For every doc:

C.1. **Use `git mv`, never `mv`** — preserves git history. This is mandatory; a plain `mv` followed by `git add` breaks `git log --follow`.

C.2. Move based on `status:` in frontmatter:
   - `status: active` → `docs/develop/active/<topic>/`
   - `status: superseded` or `status: archive` → `docs/develop/archive/`

C.3. Update incoming links in [project_plan.md](../project/project_plan.md), [CLAUDE.md](../../CLAUDE.md), and [docs/environment/ENVIRONMENT_SUMMARY.md](../environment/ENVIRONMENT_SUMMARY.md) to the new paths. Use `grep -rn "docs/develop/<filename>"` to find references.

C.4. Re-run the index script; confirm it validates clean.

C.5. Commit each phase separately so the diff is reviewable.

### Initial Topic Triage Table

This table is the seed assignment. Implementing agent should adjust if a doc obviously belongs in a different topic (file owners know best).

| Doc | Topic | Status | Notes |
|---|---|---|---|
| `FILM_MODULATION_PLAN.md` | filim | active | |
| `FiLM_PAPERS_REVIEW.md` | filim | active | |
| `FiLM_ENSEMBLE_SENSORY_PRECISION.md` | filim | active | |
| `PRECISION_MODULATION.md` | precision | active | |
| `PRECISION_MODULATION_ARCHITECTURE.md` | precision | active | |
| `SOLVING_GATE_COLLAPSE.md` | precision | active | gate-collapse remediation |
| `NEUROMODULATION_ALGORITHM.md` | neuromodulation | active | canonical algorithm doc |
| `NMN_ARCHITECTURE_REVIEW.md` | neuromodulation | active | |
| `NMN_METRICS_REFERENCE.md` | neuromodulation | active | |
| `NMN_PERFORMANCE_DIAGNOSIS_v8.md` | diagnosis | active | latest in chain |
| `NMN_PERFORMANCE_DIAGNOSIS_v1.md` ... `v7.md` | diagnosis | superseded | `superseded_by: <next>` chain forward to v8 |
| `DREAMER_DIAGNOSTICS_PLAN_v2.md` | dreamer | active | v1 already in archive/ |
| `DREAMER_REVIEW.md` | dreamer | active | |
| `DREAMER_IMPLEMENTATION_AUDIT.md` | dreamer | active | |
| `CONTINUAL_LEARNING_CONFIG_SCHEDULE.md` | continual_learning | active | |
| `CONTINUAL_LEARNING_REVIEW.md` | continual_learning | active | |
| `BEHAVIOR_ANALYSIS.md` | behavior | active | |
| `TRAINING_METRICS_ANALYSIS.md` | behavior | active | |
| `WANDB_METRICS_REFERENCE.md` | behavior | active | |
| `OLFACTORY_PROPERTY_VARIANCE.md` | sensors | active | |
| `NOISE_RENDERING_RENDERER_FIX.md` | sensors | active | |
| `UNIFY_UNIMODAL_GROUPING.md` | sensors | active | |
| `LAYERNORM_ORTHOGONAL_REFACTOR_PLAN.md` | refactors | active | |
| `MIXTURE_SAMPLING_PLAN.md` | refactors | active | |
| `FIX_PPO_MODULATION_LOGGING.md` | refactors | active | |
| `UI_REDESIGN_PROPOSAL.md` | refactors | active | |
| `ISSUE_01_PREDATOR_COUNT.md` ... `ISSUE_09_*.md` | issues | active or archive — triage per file (closed → archive) |
| `IMPORTANT_ISSUES.md` | meta | active | running task tracker |
| `TASKS_2026-04-24.md` | meta | archive | dated task list |
| Existing `archive/*` (22 files) | (per content) | archive | only need frontmatter; no moves |

Triage of `ISSUE_*` files: implementing agent should `grep -l "Status.*COMPLETED" ISSUE_*.md` to find closed issues; those move to `archive/` with `status: archive`. Open issues stay in `active/issues/` with `status: active`.

### File Changes

#### `scripts/regen_dev_index.py` (new file)

```python
#!/usr/bin/env python3
"""Regenerate docs/develop/INDEX.md from frontmatter of all develop docs.

Walks docs/develop/{active,archive}/, parses YAML frontmatter, and writes a
deterministic INDEX.md grouped by topic and lifecycle status. Validates that
supersedes/superseded_by chains resolve to existing files; exits nonzero on
validation errors so pre-commit hooks fail loudly.

Usage: python scripts/regen_dev_index.py
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DEVELOP = ROOT / "docs" / "develop"
INDEX = DEVELOP / "INDEX.md"

REQUIRED_FIELDS = {"title", "topic", "status", "created", "last_updated"}
VALID_STATUS = {"active", "superseded", "archive"}
VALID_TOPICS = {
    "filim", "precision", "noise", "hypervigilance", "neuromodulation",
    "diagnosis", "dreamer", "continual_learning", "behavior", "sensors",
    "refactors", "issues", "meta",
}


def parse_frontmatter(path: Path) -> dict | None:
    """Read a markdown file's YAML frontmatter; return None if absent."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    return yaml.safe_load(text[4:end]) or {}


def collect_docs() -> list[tuple[Path, dict]]:
    docs = []
    for sub in ("active", "archive"):
        base = DEVELOP / sub
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            fm = parse_frontmatter(md)
            if fm is None:
                continue  # skip non-frontmatter files (e.g., this plan)
            docs.append((md, fm))
    return docs


def validate(docs: list[tuple[Path, dict]]) -> list[str]:
    errors = []
    by_name = {p.name: (p, fm) for p, fm in docs}
    for path, fm in docs:
        rel = path.relative_to(DEVELOP)
        missing = REQUIRED_FIELDS - set(fm)
        if missing:
            errors.append(f"{rel}: missing required fields: {sorted(missing)}")
        if fm.get("status") not in VALID_STATUS:
            errors.append(f"{rel}: invalid status: {fm.get('status')}")
        if fm.get("topic") not in VALID_TOPICS:
            errors.append(f"{rel}: invalid topic: {fm.get('topic')}")
        for chain_field in ("supersedes", "superseded_by"):
            target = fm.get(chain_field)
            if target and target not in by_name:
                errors.append(f"{rel}: {chain_field} -> {target} (not found)")
    return errors


def render_index(docs: list[tuple[Path, dict]]) -> str:
    active = [(p, fm) for p, fm in docs if fm.get("status") == "active"]
    archived = [(p, fm) for p, fm in docs if fm.get("status") in ("superseded", "archive")]

    by_topic: dict[str, list] = {}
    for p, fm in active:
        by_topic.setdefault(fm["topic"], []).append((p, fm))

    lines = [
        "# Develop Docs Index",
        "",
        f"> AUTO-GENERATED by `scripts/regen_dev_index.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        "> Do not edit by hand. Re-run the script after changing any frontmatter.",
        "",
        "## Active",
        "",
    ]
    for topic in sorted(by_topic):
        lines.append(f"### {topic}")
        lines.append("")
        for p, fm in sorted(by_topic[topic], key=lambda x: x[1].get("last_updated", "")):
            rel = p.relative_to(DEVELOP)
            phase = f" — phase {fm['phase']}" if fm.get("phase") else ""
            lines.append(f"- [{fm['title']}]({rel}) — updated {fm['last_updated']}{phase}")
        lines.append("")

    lines.append("## Archive (superseded or one-off)")
    lines.append("")
    for p, fm in sorted(archived, key=lambda x: x[1].get("last_updated", ""), reverse=True):
        rel = p.relative_to(DEVELOP)
        chain = ""
        if fm.get("superseded_by"):
            chain = f" → superseded by `{fm['superseded_by']}`"
        lines.append(f"- [{fm['title']}]({rel}){chain}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    docs = collect_docs()
    errors = validate(docs)
    if errors:
        print("Validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    INDEX.write_text(render_index(docs))
    print(f"Wrote {INDEX} ({len(docs)} docs indexed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Dependencies: `pyyaml`. Add to project dependencies if not already present.

#### `docs/develop/active/meta/FRONTMATTER_CONTRACT.md` (new file)

Short doc (under 50 lines) describing the frontmatter schema, the valid topic enum, and the workflow for superseding a doc (set old `status: superseded`, new doc carries `supersedes: <old>`, both files re-run script). Owned by `senior-developer` for future amendments.

#### `CLAUDE.md` (small update — one bullet)

Add to the Project-Wide Rules section under **Templates**:

> - **Develop docs.** Files under `docs/develop/` carry frontmatter (`title, topic, status, created, last_updated`). The `INDEX.md` there is auto-generated by `scripts/regen_dev_index.py` — do not hand-edit. When superseding a doc, set `status: superseded` and link with `supersedes:`/`superseded_by:`. Move with `git mv` to preserve history.

#### `docs/project/project_plan.md` (link rewrites in Phase C only)

After Phase C moves complete, all `[link](../develop/<file>.md)` references in `project_plan.md` must be rewritten to the new paths. The implementing agent should produce a diff showing all rewrites for review.

## Checkpoints

Phase A:
- [ ] Empty subfolder tree exists under `docs/develop/active/`.
- [ ] `scripts/regen_dev_index.py` runs without error on the empty tree.
- [ ] `FRONTMATTER_CONTRACT.md` exists and is referenced from `CLAUDE.md`.

Phase B:
- [ ] Every `*.md` under `docs/develop/` (recursively) has a valid frontmatter block.
- [ ] `python scripts/regen_dev_index.py` exits 0 — no validation errors.
- [ ] `NMN_PERFORMANCE_DIAGNOSIS_v1` through `v7` carry `superseded_by:` pointing to the next version; `v8` is `status: active`.
- [ ] `DREAMER_DIAGNOSTICS_PLAN_v2.md` carries `supersedes: DREAMER_DIAGNOSTICS_PLAN_v1.md` (which lives in `archive/`).

Phase C:
- [ ] Every move was performed with `git mv`. Verify with `git log --follow <new-path>` showing pre-move history.
- [ ] No `*.md` files remain directly in `docs/develop/` (flat) except `INDEX.md` and this plan doc.
- [ ] `grep -rn "docs/develop/<old-path>"` returns no hits in `docs/`, `CLAUDE.md`, or `.claude/agents/` after rewrites.
- [ ] Final index re-run is clean.

## Implementation Report

> **Implemented by**: [pending]
> **Date**: [pending]

## Verification Report

> **Verified by**: [pending]
> **Date**: [pending]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [pending]
