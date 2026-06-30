---
title: "Develop Docs Frontmatter Contract"
topic: meta
status: active
created: 2026-05-06
last_updated: 2026-05-06
phase: null
aliases: [frontmatter_contract]
---

# Develop Docs Frontmatter Contract

Every markdown file under `docs/develop/{active,archive}/` carries a YAML frontmatter block that drives the auto-generated `INDEX.md`. Hand-editing `INDEX.md` is forbidden — re-run `scripts/claude/regen_dev_index.py` instead.

## Required fields

| Field | Type | Description |
|---|---|---|
| `title` | string | Human-readable title shown in the index. Usually matches the doc's H1. |
| `topic` | enum | Topic cluster (see valid values below). Determines subfolder under `active/`. |
| `status` | enum | Lifecycle: `active`, `superseded`, or `archive`. |
| `created` | `YYYY-MM-DD` | Date of doc creation. Derive from `git log --diff-filter=A --format=%cs -- <file> \| tail -1`. |
| `last_updated` | `YYYY-MM-DD` | Date of last meaningful update. Derive from `git log -1 --format=%cs -- <file>`. |

## Optional fields

| Field | Type | Description |
|---|---|---|
| `supersedes` | filename | The previous doc this one replaces (e.g. `NMN_PERFORMANCE_DIAGNOSIS_v7.md`). |
| `superseded_by` | filename | The successor doc that replaced this one. Set when this doc is moved to `archive/`. |
| `phase` | 1 \| 2 \| 3 \| 4 | Project-plan phase this doc serves (per [project_plan.md](../../../project/project_plan.md)). |

## Valid `topic` values

`filim`, `precision`, `noise`, `hypervigilance`, `neuromodulation`, `diagnosis`, `dreamer`, `continual_learning`, `behavior`, `sensors`, `refactors`, `issues`, `meta`.

To add a new topic: extend `VALID_TOPICS` in `scripts/claude/regen_dev_index.py` and update this contract in the same commit.

## Valid `status` values

- **`active`** — load-bearing right now. Lives under `docs/develop/active/<topic>/`. Linked from `project_plan.md` and other active docs.
- **`superseded`** — replaced by a newer doc. Lives under `docs/develop/archive/`. Carries `superseded_by:` pointing forward.
- **`archive`** — completed one-off, no successor. Lives under `docs/develop/archive/`.

## Superseding a doc

When a new version replaces an older one:

1. Write the new doc with `status: active` and `supersedes: <old-filename>.md`.
2. Edit the old doc's frontmatter: set `status: superseded` and add `superseded_by: <new-filename>.md`.
3. **`git mv`** the old doc into `docs/develop/archive/` — never plain `mv`. Preserving `git log --follow` is mandatory.
4. Run `python scripts/claude/regen_dev_index.py` and confirm exit 0.
5. Update incoming links in `docs/project/project_plan.md` (and elsewhere if found via `grep -rn`) to point at the new doc.

## Validation

`scripts/claude/regen_dev_index.py` validates:

- All required fields are present.
- `status` and `topic` are in the valid enums.
- `supersedes` and `superseded_by` resolve to files that actually exist somewhere under `docs/develop/`.

Validation failures block index regeneration and exit nonzero, so the script can serve as a pre-commit hook.
