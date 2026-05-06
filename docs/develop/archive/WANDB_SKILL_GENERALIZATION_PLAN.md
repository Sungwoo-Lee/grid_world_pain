---
title: "Plan: Generalize WandB Analysis Skill"
topic: meta
status: archive
created: 2026-03-03
last_updated: 2026-04-12
---

# Plan: Generalize WandB Analysis Skill

## Context

The current wandb-analysis skill has DreamerV3-specific metric lists, categories, healthy ranges, and debugging playbooks hardcoded into both the Python scripts and SKILL.md. The goal is to make the skill work with **any algorithm** — discover what metrics exist in a WandB run and extract them for analysis, without needing to know the algorithm upfront.

## Approach: Auto-Discovery + Optional Presets

The key idea: **use `run.summary` from the WandB API to discover all available metrics in a run**, auto-group them by prefix, and present them. Optionally support named presets (like `dreamer_v3`) for curated views with healthy ranges.

## Changes

### 1. New shared utility: `scripts/wandb_utils.py`

Extract duplicated code from both scripts into a shared module:
- `parse_timestamp_from_name()` — timestamp parsing
- `match_wandb_run()` — run matching (exact + fuzzy timestamp)
- `fetch_wandb_runs()` — fetch runs from entity/project with fallback
- `compute_stats()` — summary statistics (mean, std, last 20%, first/last, min/max)
- `fmt()` — smart float formatter
- Constants: `WANDB_ENTITY`, `WANDB_PROJECT`, `TIMESTAMP_TOLERANCE_SEC`
- Fix timezone: make it configurable (default to local system timezone instead of hardcoded KST)

### 2. New general-purpose script: `scripts/wandb_metrics.py`

Replace the hardcoded comparison script with a general-purpose metrics tool.

**New CLI interface:**

```bash
# Discover all metrics in a run (list what's available)
python scripts/wandb_metrics.py discover RUN_NAME

# Compare runs with auto-discovery (groups metrics by prefix)
python scripts/wandb_metrics.py compare RUN1 RUN2 --labels "A,B"

# Compare with specific metric filter (glob patterns on metric names)
python scripts/wandb_metrics.py compare RUN1 RUN2 --metrics "Episode/*,*/loss_*"

# Compare with a named preset (backward compat for DreamerV3)
python scripts/wandb_metrics.py compare RUN1 RUN2 --preset dreamer_v3

# Extract single run metrics
python scripts/wandb_metrics.py extract RUN_NAME
python scripts/wandb_metrics.py extract RUN_NAME --metrics "Episode/*"
```

**Auto-discovery logic:**
1. Call `run.summary.keys()` to get all logged metric names
2. Filter out WandB internal keys (starting with `_`)
3. Group by prefix (text before first `/`), metrics without `/` go in "General" group
4. For each group, pull time series via `run.history(keys=[...])` and compute stats
5. Output markdown tables grouped by discovered categories

**Preset system:**
- Presets are simple Python dicts mapping category name → list of (metric_key, display_name, criterion)
- `PRESETS = {"dreamer_v3": {...}, "recurrent_ppo": {...}}`
- Defined in the script (not external files) — keeps it simple
- `--preset` flag selects one; without it, auto-discovery is used

**Metric filtering (`--metrics`):**
- Comma-separated patterns supporting `*` wildcards
- e.g., `"Episode/*,WorldModel/loss_*,value_mae"`
- Applied on top of discovered metrics

### 3. Update `scripts/benchmark_wandb_speed.py` (minimal)

- Import shared utilities from `wandb_utils.py` (remove duplicate functions)
- No functional changes — it's already general-purpose

### 4. Update `scripts/compare_wandb_runs.py` (backward compat wrapper)

Keep the old script but make it a thin wrapper that calls `wandb_metrics.py compare` with `--preset dreamer_v3`. Existing usage doesn't break.

### 5. Update `.claude/skills/wandb-analysis/SKILL.md`

Make the skill doc algorithm-agnostic:

- **Remove**: hardcoded healthy metric ranges table, DreamerV3 debugging playbook, DreamerV3-specific metric naming conventions
- **Add**: documentation for `discover` subcommand as the first step in any analysis
- **Add**: documentation for `--metrics` and `--preset` flags
- **Update workflow**: Step 1 becomes "discover metrics in the run", Step 2 becomes "extract/compare based on what was found"
- **Keep**: project context (entity, project), speed benchmark docs, general analysis structure
- **Move** DreamerV3-specific healthy ranges to the preset definition in the Python code (or to `docs/DREAMER_DIAGNOSTICS_PLAN_v2.md` where they already exist)

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/wandb_utils.py` | **CREATE** — shared utilities |
| `scripts/wandb_metrics.py` | **CREATE** — new general-purpose tool |
| `scripts/compare_wandb_runs.py` | **EDIT** — thin wrapper for backward compat |
| `scripts/benchmark_wandb_speed.py` | **EDIT** — import from wandb_utils |
| `.claude/skills/wandb-analysis/SKILL.md` | **EDIT** — generalize |

## Verification

1. `python scripts/wandb_metrics.py discover <any_recent_run>` — should list all metrics grouped by prefix
2. `python scripts/wandb_metrics.py compare <run1> <run2>` — auto-discovered comparison
3. `python scripts/wandb_metrics.py compare <run1> <run2> --preset dreamer_v3` — same output as old script
4. `python scripts/compare_wandb_runs.py <run1> <run2>` — backward compat still works
5. `python scripts/benchmark_wandb_speed.py <run1>` — still works unchanged
