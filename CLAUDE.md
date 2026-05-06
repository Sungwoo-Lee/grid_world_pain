## Agent Team

This project uses a multi-agent workflow. Each agent has a dedicated profile in `.claude/agents/` with its full responsibilities and tool scope. **Delegate to the matching agent rather than doing the work yourself.**

### Roster

| Agent | Profile | Role | File scope |
|---|---|---|---|
| **senior-developer** | [.claude/agents/senior-developer.md](.claude/agents/senior-developer.md) | Planning, code analysis, WandB / training analysis, verification | `docs/`, `CLAUDE.md`, `.claude/agents/` only |
| **developer** | [.claude/agents/developer.md](.claude/agents/developer.md) | Implementing approved plans, running tests, reporting results | Full code access (`src/`, `configs/`, `scripts/`, tests) |
| **literature-reviewer** | [.claude/agents/literature-reviewer.md](.claude/agents/literature-reviewer.md) | Academic literature review of PDFs and NotebookLM notebooks | `docs/` only |

### Delegation Guide

| User intent | Delegate to |
|---|---|
| Plan a feature / investigate a bug / draft an implementation strategy | `senior-developer` |
| Verify another agent's implementation against the plan | `senior-developer` |
| Analyze WandB training results / ablation runs | `senior-developer` (uses `wandb-analysis` skill) |
| Implement an approved plan / write code / run the test suite | `developer` |
| Review papers from a directory of PDFs or a NotebookLM link | `literature-reviewer` |

### Standard Hand-off Flow (Code Changes)

1. **`senior-developer`** writes a plan to `docs/` using the appropriate template.
2. User approves the plan.
3. **`developer`** implements the plan, runs tests, appends an Implementation Report and updates Checkpoints in the plan doc, and leaves the working tree dirty (uncommitted).
4. **`senior-developer`** runs the Verification Protocol on the diff and fills the Verification Report in the plan doc.

Each agent's profile contains the full procedure for its role — read the profile before delegating.

### Workflow Skills

Common multi-agent workflows are codified as skills under `.claude/skills/`. Use the matching skill rather than re-deriving the orchestration each time.

| Skill | Pattern | Use when |
|---|---|---|
| **`feature-workflow`** | Sequential: plan → approve → implement → verify | Adding new features, sensors, models, environment components |
| **`bug-fix-workflow`** | Sequential: root-cause → approve → fix + regression test → verify | Fixing bugs, regressions, crashes, unexpected behavior |
| **`training-experiment-workflow`** | Branching: design new experiment OR analyze existing runs | Running training experiments, ablations, WandB result analysis |
| **`parallel-literature-review`** | Parallel: shard corpus → K reviewer instances → merge | Large literature corpora (~10+ papers) where parallelism is worth the overhead |

For small literature reviews (under ~10 papers), invoke the `literature-reviewer` agent directly without the parallel skill.

---

## Project-Wide Rules

These rules apply to **every** agent. Agent profiles may extend them but must not relax them.

### Configuration Protocol

- **No fallback defaults** for critical config params. Use `config.get_mandatory('key')` — a missing YAML key must raise `ValueError`.
- New config keys introduced by a plan must be listed in the plan's File Changes section with the exact YAML path and value.

### Token Efficiency & Agent Policy

- **Parallel subagents are allowed.** Use them when tasks are genuinely independent (e.g., reviewing disjoint paper subsets, running ablation analyses on separate runs). Do not parallelize tasks with hand-off dependencies (plan → implement → verify) — keep those sequential.
- For mechanical batch work (e.g., extracting metrics from 16 WandB runs with the same script), prefer a single shell loop over spawning 16 agents — it is faster and cheaper. Reserve parallel agents for work that benefits from independent reasoning, not for mechanical iteration.
- Save intermediate results to `tmp/` files **after each extraction step** — never accumulate results only in context. This prevents data loss if the conversation is compressed and lets parallel agents share progress.
- Avoid redundant work: do not extract the same data through multiple paths.

### Working File Conventions

- All temporary working notes go in `tmp/` with a timestamped filename: `tmp/YYYYMMDD_HHMMSS_<topic>.md` (e.g., `tmp/20260310_143052_wandb_dreamer_comparison.md`).
- Multiple analyses may run in parallel — each gets its own timestamped file with a unique HHMMSS.
- Write to the working file **after each step**, not at the end.
- These files are temporary and may be cleaned up or overwritten as needed.

### Documentation Templates

Plans and analysis go in `docs/`. Two templates — read the relevant template file before writing:

- **[docs/TEMPLATES/issue_plan.md](docs/TEMPLATES/issue_plan.md)** — for development issues, bug fixes, and implementation plans (*"what to build/fix"*).
- **[docs/TEMPLATES/training_analysis.md](docs/TEMPLATES/training_analysis.md)** — for training experiment analysis (WandB results, ablation studies, run comparisons; *"what happened during training and why"*). Hypothesis-driven academic structure: research question → experimental design → results → analysis → conclusions.

**Cross-referencing between documents:**
- If a `training_analysis` reveals a bug or needed code change, create a separate `issue_plan` doc and link it from the analysis with `[Related](link)`.
- If an `issue_plan` investigation uncovers a closely related issue, append it to the same doc. If independent, create a separate doc.
- Always cross-reference related docs in **both directions**.
- Include enough detail (file paths, line numbers, code snippets) for the implementing agent to execute without ambiguity.

### Performance Evaluation Convention

- Agent training performance is evaluated based on **survival steps**, not cumulative reward. Any training analysis or report must respect this convention.

---

## Where the Detailed Workflows Live

The full step-by-step procedures previously listed in this file have moved into the agent profiles. Pointers:

- **Verification Protocol** (post-implementation diff review and Verification Report) → [senior-developer.md](.claude/agents/senior-developer.md)
- **WandB Analysis Protocol** (`tmp/` checkpointing, temporal evolution analysis) → [senior-developer.md](.claude/agents/senior-developer.md)
- **Training Results Analysis Workflow** (datetime ID extraction, local log files, survival-based evaluation) → [senior-developer.md](.claude/agents/senior-developer.md)
- **Literature Review Workflow** (source-type skill mapping, 4-step backbone, Phase 1/2 synthesis with LaTeX) → [literature-reviewer.md](.claude/agents/literature-reviewer.md)
- **Implementation Report & Checkpoint conventions** → [developer.md](.claude/agents/developer.md)

When in doubt, read the profile of the agent you intend to use.
