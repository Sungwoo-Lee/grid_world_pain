---
name: training-experiment-workflow
description: Workflow for designing, running, and analyzing training experiments and ablations in this RL project. Use this skill whenever the user wants to "run a training experiment", "compare these runs", "analyze WandB results", "test this hypothesis", "do an ablation on X", or attaches a screenshot of WandB run IDs and asks for analysis. Orchestrates experiment-designer (design + config generation), env-config-auditor (pre-flight), training-runner (launch), and experiment-analyzer (post-hoc result analysis). Performance is always evaluated by survival steps, not cumulative reward — this is a project-wide convention. Use this skill even when the user does not say "experiment" or "workflow" — any training-results question or experiment-design request belongs here.
---

# Training Experiment Workflow

This skill orchestrates the workflow for empirical training work — designing experiments, running them (or analyzing already-run results), and producing publication-quality analysis docs.

All experiment docs live under **`docs/experiments/active/<topic>/`** per the [experiments Frontmatter Contract](../../../docs/experiments/meta/FRONTMATTER_CONTRACT.md). This tree is separate from `docs/develop/`, which holds platform-development plans. Pre-existing experiment-shaped docs under `docs/develop/active/{hypervigilance,noise,diagnosis}/` are NOT migrated retroactively — read them as reference, write new ones to `docs/experiments/`.

It has two entry points depending on what the user is asking:

- **(A) Design a new experiment** — the user wants to test a hypothesis or run an ablation.
- **(B) Analyze existing results** — the user attaches a WandB screenshot or names datetime IDs and wants the data interpreted.

## Path A — Design a New Experiment

```
[A1]   experiment-designer → design doc + configs
[A1.5] env-config-auditor  → pre-flight audit of the configs
[A2]   USER APPROVAL       → confirms hypothesis, design, and configs
[A3]   training-runner     → launches via run_command.py on a chosen lab node
[A4]   USER                → waits for training to complete (or monitors)
[A5]   experiment-analyzer → analyzes results, fills in the design doc
```

`developer` enters Path A only when the experiment requires **schema changes** in the codebase (new mandatory YAML keys not yet read by `src/utils/config.py`). Pure parameter changes are written directly by `experiment-designer`.

### A1 — Experiment design + config generation (experiment-designer)

Delegate to the **`experiment-designer`** agent. Its job:

1. Read [docs/TEMPLATES/training_analysis.md](../../../docs/TEMPLATES/training_analysis.md) and the [experiments Frontmatter Contract](../../../docs/experiments/meta/FRONTMATTER_CONTRACT.md). The template is hypothesis-driven (research question → design → expected outcomes → results → analysis → conclusions).
2. Write a design doc to **`docs/experiments/active/<topic>/<EXP_NAME>.md`** with frontmatter (`title`, `topic`, `status: active`, `created`, `last_updated`; optional `phase`, `wandb_tag`, `develop_link`). Fill **only** the pre-results sections:
   - Research question and hypothesis.
   - Experimental design (configs, ablations, seeds, expected sample size).
   - Predicted outcomes (what would confirm vs. refute the hypothesis).
   - Configs to Produce table.
   - Analysis plan (pre-specified).
   - Failure-mode catalog.
   - Empty Results / Analysis / Conclusions sections (filled in A5).
3. **Generate the configs** under `configs/experiment/<topic>/` per the design's Configs to Produce table. Schema-affecting changes route through `developer` first; pure parameter configs are written directly.
4. Do NOT run `regen_dev_index.py` — there is no INDEX for `docs/experiments/`.

### A1.5 — Config audit (env-config-auditor, pre-flight)

Delegate to **`env-config-auditor`** to validate the new configs against `docs/environment/`. Walks observation ↔ noise modality consistency, mandatory-key discipline, static-field recompile risk, latent-bug recurrences, schema padding, and (for sweeps) cross-config coherence.

The audit produces `docs/reviews/config_<exp-name>.md` with `🔴 / 🟡 / 🟢` findings. **Resolve all `🔴` blockers before A2.** Cross-link the audit from the experiment doc.

### A2 — User approval

The user approves the hypothesis, design, and configs before any compute is spent. The user may iterate with `experiment-designer` (revise design or configs) or with `env-config-auditor` (re-audit) before approving.

### A3 — Launch training (training-runner)

Delegate to the **`training-runner`** agent. It:

- Re-reads the configs (read-only) for a final sanity check.
- `WebFetch`es http://192.168.0.101:1810 to propose a node + free GPU.
- Edits `train_command-new.sh` to point at the experiment's config + agent_config + chosen `--device cuda:N` and a unique `--tag`.
- Launches via `python3 run_command.py <node> grid_world_pain "bash train_command-new.sh"` (SSH key auth — no password).
- Confirms the run started; reports node, GPU, log path, and WandB tag back to the user.

If `training-runner` finds a config issue at this stage, it halts and routes back to `experiment-designer` to fix.

### A4 — User waits for training

Not Claude's job. The skill pauses here. The user (or `senior-developer` ad-hoc) may tail logs.

### A5 — Result analysis (experiment-analyzer)

When the user returns with run IDs (typically a WandB screenshot), delegate to **`experiment-analyzer`** with the `wandb-analysis` skill to fill in the Results / Analysis / Conclusions sections of the same design doc at `docs/experiments/active/<topic>/<EXP_NAME>.md`.

If the analysis surfaces a metric that should have been logged but wasn't, the analyzer adds a `## Metrics Requested` section to the same doc. The user reviews; if accepted, escalate to `feature-workflow` (`senior-developer` plans → `developer` implements) to add the logger. The original experiment may need to be re-run with the new metric.

## Path B — Analyze Existing Results

The user has a screenshot of WandB or a list of `YYYYMMDD_HHMMSS` run IDs and wants analysis. There is no design doc yet; you may be analyzing runs that were not pre-registered.

```
[B1] experiment-analyzer → extracts run IDs, locates local logs, runs wandb-analysis
[B2] experiment-analyzer → produces training_analysis doc (Mode B in the analyzer's profile)
[B3] (optional) → if analysis reveals a bug, branch into bug-fix-workflow
                 → if analysis reveals a missing metric, branch into feature-workflow
```

### B1 — Run identification and extraction (experiment-analyzer)

Delegate to **`experiment-analyzer`** with these mandatory steps:

1. **Extract run datetime IDs** from the user's screenshot/message (`YYYYMMDD_HHMMSS` format).
2. **Locate local WandB logs** in `wandb/run-YYYYMMDD_HHMMSS-<wandb_id>/`. **Do NOT query the WandB web API** — use local files only.
3. Invoke the **`wandb-analysis` skill** to parse logs.
4. Save intermediate extractions to a `tmp/YYYYMMDD_HHMMSS_<topic>.md` working file as you go (Working File Convention).
5. **Temporal evolution analysis is mandatory** — show how key metrics change over training steps/episodes, including trends, inflection points, convergence behavior. Static end-of-training snapshots are not sufficient.

### B2 — Analysis doc (experiment-analyzer)

Write the final doc to **`docs/experiments/active/<topic>/<NAME>.md`** (typically `diagnosis/` for ablation/comparison analyses; `comparison/` for multi-run comparisons; `hypervigilance/` for hypervigilance-related readouts) using `docs/TEMPLATES/training_analysis.md` and the [experiments Frontmatter Contract](../../../docs/experiments/meta/FRONTMATTER_CONTRACT.md). Use the hypothesis-driven structure: even for unplanned analyses, retroactively frame the comparison as a question being answered. Do NOT run `regen_dev_index.py` — there is no INDEX for this tree.

**Performance evaluation MUST use survival steps**, not cumulative reward — project-wide convention.

### B3 — Optional bug-fix branch

If the analysis surfaces a bug or needed code change, the analysis doc must include a `[Related](link)` reference to a separate `bug-fix-workflow` plan. Cross-reference both directions.

## Parallel Analysis (When Useful)

When comparing many independent runs (e.g., 8 seeds × 4 ablations = 32 runs), parallelism is sometimes worth it.

- **Mechanical extraction** (the same metric across all runs): use a **single shell loop**, not parallel agents. Far cheaper.
- **Independent run-level reasoning** (each run has its own qualitative pattern to interpret): consider parallel `experiment-analyzer` instances, one per run group. Merge the results into a single analysis doc afterward.

Default: sequential. Switch to parallel only when the per-run reasoning genuinely benefits from independent attention.

## Hand-off Artifact Summary

| Phase | Produces | Read by |
|---|---|---|
| A1 | `docs/experiments/active/<topic>/<EXP_NAME>.md` (design, with frontmatter) + `configs/experiment/<topic>/*.yaml` | env-config-auditor (A1.5); user (A2); training-runner (A3) |
| A1.5 | `docs/reviews/config_<exp>.md` audit report | User (must clear 🔴 before A2) |
| A3 | Edited `train_command-new.sh` + remote training process; log path + WandB tag reported back | User (A4); senior-developer (A5) |
| A5 / B2 | Filled-in analysis doc at `docs/experiments/active/<topic>/<NAME>.md` | User; possibly bug-fix-workflow |

## Why This Workflow Has Two Paths

Pre-registered experiments (Path A) and post-hoc analyses (Path B) have different epistemological standing. A pre-registered hypothesis tested cleanly is strong evidence; a post-hoc explanation of what already happened is weaker, but still valuable. The training_analysis template's hypothesis-driven framing keeps both honest about which mode the user is in.

## Common Mistakes to Avoid

- Querying the WandB web API instead of reading local files in `wandb/run-*/`.
- Reporting only end-of-training metrics — temporal evolution is mandatory.
- Using cumulative reward as the headline performance metric — this project uses survival steps.
- Writing the Results section before approving the experimental design (Path A only) — that is the kind of motivated reasoning the hypothesis-driven template exists to prevent.
- Skipping the `tmp/` working file and trying to hold all 32 runs' metrics in conversation context.
