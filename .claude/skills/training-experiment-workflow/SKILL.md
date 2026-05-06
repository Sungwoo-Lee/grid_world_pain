---
name: training-experiment-workflow
description: Workflow for designing, running, and analyzing training experiments and ablations in this RL project. Use this skill whenever the user wants to "run a training experiment", "compare these runs", "analyze WandB results", "test this hypothesis", "do an ablation on X", or attaches a screenshot of WandB run IDs and asks for analysis. Orchestrates the senior-developer (experiment design and analysis with the wandb-analysis skill) and optionally the developer (config changes only). Performance is always evaluated by survival steps, not cumulative reward — this is a project-wide convention. Use this skill even when the user does not say "experiment" or "workflow" — any training-results question or experiment-design request belongs here.
---

# Training Experiment Workflow

This skill orchestrates the workflow for empirical training work — designing experiments, running them (or analyzing already-run results), and producing publication-quality analysis docs.

It has two entry points depending on what the user is asking:

- **(A) Design a new experiment** — the user wants to test a hypothesis or run an ablation.
- **(B) Analyze existing results** — the user attaches a WandB screenshot or names datetime IDs and wants the data interpreted.

Both paths route through `senior-developer` as the primary agent. `developer` enters only if the experiment requires code or config changes.

## Path A — Design a New Experiment

```
[A1]   senior-developer    → writes training_analysis-shaped design doc
[A2]   USER APPROVAL       → confirms hypothesis and experiment design
[A3]   developer           → applies config changes (only if needed)
[A3.5] env-config-auditor  → pre-flight audit of the experiment configs
[A4]   USER                → runs training (manually or via existing scripts)
[A5]   senior-developer    → analyzes results, fills in the design doc
```

### A1 — Experiment design (senior-developer)

Delegate to the **`senior-developer`** agent. Its job:

1. Read [docs/TEMPLATES/training_analysis.md](../../../docs/TEMPLATES/training_analysis.md) and the [Frontmatter Contract](../../../docs/develop/active/meta/FRONTMATTER_CONTRACT.md). The template is hypothesis-driven (research question → design → expected outcomes → results → analysis → conclusions).
2. Write a doc to **`docs/develop/active/<topic>/<EXP_NAME>.md`** (typically `diagnosis/`, `hypervigilance/`, or `noise/` — pick the topic the experiment serves). The doc must start with YAML frontmatter (`title`, `topic`, `status: active`, `created`, `last_updated`; optional `phase`). Fill **only** the pre-results sections:
   - Research question and hypothesis.
   - Experimental design (configs, ablations, seeds, expected sample size).
   - Predicted outcomes (what would confirm vs. refute the hypothesis).
   - Empty Results / Analysis / Conclusions sections (filled in A5).
3. List any required config changes. If new YAML keys are needed, list them with exact paths and values per the Configuration Protocol.
4. Run `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/regen_dev_index.py` so the new doc appears in `docs/develop/INDEX.md`.

### A2 — User approval

The user approves the hypothesis and design before any compute is spent.

### A3 — Config changes (developer, only if needed)

If new configs are needed, hand off to **`developer`** (mini-version of `feature-workflow` Phase 3): apply config changes, no fallback defaults, write Implementation Report.

If no config changes are needed, **skip this phase** and go straight to A3.5.

### A3.5 — Config audit (env-config-auditor, pre-flight)

Before any training run, delegate to **`env-config-auditor`** to validate the experiment's configs. Mandatory whenever A3 produced config changes; recommended even when reusing existing configs (catches drift between the design doc and the actual YAML on disk). The auditor walks observation ↔ noise modality consistency, mandatory-key discipline, static-field recompile risk, latent-bug recurrences, schema padding, and (for sweeps) cross-config coherence.

The audit produces `docs/reviews/config_<exp-name>.md` with `🔴 / 🟡 / 🟢` findings. **Resolve all `🔴` blockers before A4.** Cross-link the audit from the experiment doc.

### A4 — User runs training

The user runs training. Not Claude's job. The skill pauses here.

### A5 — Result analysis (senior-developer)

When the user returns with run IDs (typically a WandB screenshot), delegate to **`senior-developer`** with the `wandb-analysis` skill to fill in the Results / Analysis / Conclusions sections of the same design doc.

## Path B — Analyze Existing Results

The user has a screenshot of WandB or a list of `YYYYMMDD_HHMMSS` run IDs and wants analysis. There is no design doc yet; you may be analyzing runs that were not pre-registered.

```
[B1] senior-developer  → extracts run IDs, locates local logs, runs wandb-analysis
[B2] senior-developer  → produces training_analysis doc
[B3] (optional) → if analysis reveals a bug, branch into bug-fix-workflow
```

### B1 — Run identification and extraction (senior-developer)

Delegate to **`senior-developer`** with these mandatory steps:

1. **Extract run datetime IDs** from the user's screenshot/message (`YYYYMMDD_HHMMSS` format).
2. **Locate local WandB logs** in `wandb/run-YYYYMMDD_HHMMSS-<wandb_id>/`. **Do NOT query the WandB web API** — use local files only.
3. Invoke the **`wandb-analysis` skill** to parse logs.
4. Save intermediate extractions to a `tmp/YYYYMMDD_HHMMSS_<topic>.md` working file as you go (Working File Convention).
5. **Temporal evolution analysis is mandatory** — show how key metrics change over training steps/episodes, including trends, inflection points, convergence behavior. Static end-of-training snapshots are not sufficient.

### B2 — Analysis doc (senior-developer)

Write the final doc to **`docs/develop/active/<topic>/<NAME>.md`** (typically `diagnosis/` for ablation/comparison analyses; `behavior/` for behavioral readouts) using `docs/TEMPLATES/training_analysis.md` and the [Frontmatter Contract](../../../docs/develop/active/meta/FRONTMATTER_CONTRACT.md). Use the hypothesis-driven structure: even for unplanned analyses, retroactively frame the comparison as a question being answered. Run `scripts/regen_dev_index.py` after writing.

**Performance evaluation MUST use survival steps**, not cumulative reward — project-wide convention.

### B3 — Optional bug-fix branch

If the analysis surfaces a bug or needed code change, the analysis doc must include a `[Related](link)` reference to a separate `bug-fix-workflow` plan. Cross-reference both directions.

## Parallel Analysis (When Useful)

When comparing many independent runs (e.g., 8 seeds × 4 ablations = 32 runs), parallelism is sometimes worth it.

- **Mechanical extraction** (the same metric across all runs): use a **single shell loop**, not parallel agents. Far cheaper.
- **Independent run-level reasoning** (each run has its own qualitative pattern to interpret): consider parallel `senior-developer` instances, one per run group. Merge the results into a single analysis doc afterward.

Default: sequential. Switch to parallel only when the per-run reasoning genuinely benefits from independent attention.

## Hand-off Artifact Summary

| Phase | Produces | Read by |
|---|---|---|
| A1 / B2 | `docs/develop/active/<topic>/<NAME>.md` (with frontmatter) | User; possibly bug-fix-workflow |
| A3 | Config changes (uncommitted) + Implementation Report | senior-developer (verify if substantive); env-config-auditor (A3.5) |
| A3.5 | `docs/reviews/config_<exp>.md` audit report | User (must clear 🔴 before A4) |
| A5 / B1–B2 | Filled-in analysis doc with temporal evolution and survival-based evaluation | User |

## Why This Workflow Has Two Paths

Pre-registered experiments (Path A) and post-hoc analyses (Path B) have different epistemological standing. A pre-registered hypothesis tested cleanly is strong evidence; a post-hoc explanation of what already happened is weaker, but still valuable. The training_analysis template's hypothesis-driven framing keeps both honest about which mode the user is in.

## Common Mistakes to Avoid

- Querying the WandB web API instead of reading local files in `wandb/run-*/`.
- Reporting only end-of-training metrics — temporal evolution is mandatory.
- Using cumulative reward as the headline performance metric — this project uses survival steps.
- Writing the Results section before approving the experimental design (Path A only) — that is the kind of motivated reasoning the hypothesis-driven template exists to prevent.
- Skipping the `tmp/` working file and trying to hold all 32 runs' metrics in conversation context.
