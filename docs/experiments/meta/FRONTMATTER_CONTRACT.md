---
title: "Experiments Docs Frontmatter Contract"
topic: meta
status: active
created: 2026-05-07
last_updated: 2026-05-07
---

# Experiments Docs Frontmatter Contract

Every markdown file under `docs/experiments/{active,archive}/` carries a YAML frontmatter block. Unlike `docs/develop/`, there is **no auto-generated INDEX** for this tree (yet) — validation is by convention. If the tree grows beyond ~10 docs we will revisit auto-generation.

This contract is **separate from** [`docs/develop/active/meta/FRONTMATTER_CONTRACT.md`](../../../develop/active/meta/FRONTMATTER_CONTRACT.md). The two trees serve different purposes:

- `docs/develop/` — building the platform (architecture plans, refactor specs, sensor design, infrastructure, bug-fix plans). Owned by `senior-developer` and `developer`.
- `docs/experiments/` — using the platform to do science (experimental designs, ablations, post-hoc training analyses, run comparisons). Owned by `experiment-designer` and `senior-developer` (for analysis-phase fill-in).

## Required fields

| Field | Type | Description |
|---|---|---|
| `title` | string | Human-readable title shown in any future index. Usually matches the doc's H1. |
| `topic` | slug | Topic cluster (free-form slug). Determines subfolder under `active/`. Suggested values below. |
| `status` | enum | Lifecycle: `active`, `superseded`, or `archive`. |
| `created` | `YYYY-MM-DD` | Date of doc creation. Derive from `git log --diff-filter=A --format=%cs -- <file> \| tail -1`. |
| `last_updated` | `YYYY-MM-DD` | Date of last meaningful update. Derive from `git log -1 --format=%cs -- <file>`. |

## Optional fields

| Field | Type | Description |
|---|---|---|
| `supersedes` | filename | Previous doc this one replaces (e.g. `HYPERVIGILANCE_NOISE_SWEEP_v1.md`). |
| `superseded_by` | filename | Successor doc that replaced this one. Set when this doc is moved to `archive/`. |
| `phase` | 1 \| 2 \| 3 \| 4 | Project-plan phase this doc serves (per [project_plan.md](../../../project/project_plan.md)). |
| `wandb_tag` | string | The WandB tag pattern used by the runs this doc analyzes/plans. Lets future analyses grep for related work. |
| `develop_link` | path | If the experiment depends on a develop-side spec (e.g., an algorithm doc), point at that doc here. |

## Suggested `topic` values

Free-form slugs. Suggested starter set; extend on demand:

- `hypervigilance` — emergent cross-domain post-injury responses and probes.
- `noise` — perceptual-noise sweeps, σ schedules, modality-specific noise studies.
- `ablation` — generic ablation studies (factor-on/factor-off comparisons).
- `sweep` — parameter sweeps (lambda, learning rate, seed counts).
- `baseline` — baseline characterizations of unmodulated agents.
- `comparison` — multi-architecture or multi-config comparisons.
- `diagnosis` — post-hoc analyses of training failures or surprising results.
- `meta` — tree-level docs (this contract, conventions, etc.).

If a topic doesn't fit, pick a slug that names the experimental theme cleanly (e.g., `dreamer_tuning`). Match an existing `docs/develop/active/<topic>/` slug when the experiment is closely tied to a development line.

## Valid `status` values

- **`active`** — load-bearing right now. Lives under `docs/experiments/active/<topic>/`.
- **`superseded`** — replaced by a newer doc. Lives under `docs/experiments/archive/`. Carries `superseded_by:` pointing forward.
- **`archive`** — completed one-off, no successor. Lives under `docs/experiments/archive/`.

## Naming convention

Match the existing develop-side convention: SCREAMING_SNAKE_CASE filenames (e.g., `HYPERVIGILANCE_NOISE_SWEEP.md`, `DREAMER_LAMBDA_PRECISION_ABLATION.md`). Versioned designs use a `_v2` suffix and supersede the prior file.

## Superseding a doc

When a new version replaces an older one:

1. Write the new doc with `status: active` and `supersedes: <old-filename>.md`.
2. Edit the old doc's frontmatter: set `status: superseded` and add `superseded_by: <new-filename>.md`.
3. **`git mv`** the old doc into `docs/experiments/archive/` — never plain `mv`. Preserving `git log --follow` is mandatory.
4. Update incoming links in `docs/project/project_plan.md` and any other referencing docs (use `grep -rn` to find them).

## Launch Manifest (in-doc convention)

Experiment design docs that drive training carry a `## 3. Launch Manifest` table — the system-of-record binding each experimental cell to its WandB folder. Three agents share it with strict column ownership:

| Agent | Writes |
|---|---|
| `experiment-designer` | Planned columns: `Run`, `Cell`, `Tag (= wandb-name)`, `wandb-group`, `wandb-job-type`, `Seed`. Plus §3.1 (Configs to Produce). |
| `training-runner` | Actual columns: `Status`, `Node`, `GPU`, `Launched at`, `WandB run ID`, `Log path`. One row per launch, in place. |
| `experiment-analyzer` | Reads only. The whole table is its run-discovery surface. |

The schema is defined in [docs/TEMPLATES/training_analysis.md](../../TEMPLATES/training_analysis.md) §3. Use Tag = wandb-name (identical strings) so `train.py:592`'s fallback keeps logs and WandB correlated.

## Validation

There is no `regen_*_index.py` for this tree at present. The agents writing here (`experiment-designer`, `senior-developer`) are responsible for following the contract by inspection. If multiple frontmatter blocks drift out of sync, it's a soft warning, not a hard failure.

## Soft-split note (2026-05-07)

This tree was introduced when experimental work moved out of `docs/develop/active/`. Pre-existing experiment-shaped docs under `docs/develop/active/{hypervigilance,noise,diagnosis}/` are **not** retroactively migrated — they remain in develop until convenient to move. New experiment docs (designs, analyses) go here.
