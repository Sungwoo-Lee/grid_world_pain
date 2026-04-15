## Workflow Rules

### Strict No-Implementation Policy

- **NEVER modify source code, configs, or scripts.** This includes creating, editing, or deleting any files under `src/`, `configs/`, `scripts/`, or any other code directories.
- The only files Claude may create or edit are documentation files in `docs/` and project management files like this one (`CLAUDE.md`).
- "Plan approved" means the plan document is accepted for review — it does NOT authorize code implementation.
- Even when explicitly asked to implement, confirm with the user first, as implementation is handled by other LLM agents.

### Planning & Analysis

- Claude is used for **planning and analysis only**. Implementation is done by other LLM agents.
- Plans and analysis should be written to `docs/` files.
- **Two templates** — read the relevant template file before writing:
  - **`docs/TEMPLATES/issue_plan.md`** — for development issues, bug fixes, and implementation plans.
  - **`docs/TEMPLATES/training_analysis.md`** — for training experiment analysis (WandB results, ablation studies, run comparisons). This template follows a hypothesis-driven academic structure: research question → experimental design → results → analysis → conclusions.
- **Choose the right template**: If the work is about *what to build/fix*, use issue_plan. If the work is about *what happened during training and why*, use training_analysis.
- **Cross-referencing between documents**:
  - If a training analysis reveals a bug or needed code change, create a separate issue_plan doc and link it from the analysis with `[Related](link)`.
  - If an issue_plan investigation uncovers a new closely related issue, append it to the same doc. If independent, create a separate doc.
  - Always cross-reference related docs in both directions.
- Include enough detail (file paths, line numbers, code snippets) for the implementing agent to execute without ambiguity.

## Iterative Technical Literature Review Protocol

When tasked with reviewing a collection of references or papers (e.g., from a specific folder or a literature review list), follow this systematic, tiered approach.

* **Source Mapping:** Identify all references from the specified source document or directory.
* **Contextual Alignment:** Ensure the technical depth matches the project’s existing documentation and specific requirements.
* **Skills:** Use pdf skill for this job.

For every individual paper or reference, generate a review consisting of two distinct sections:

Phase 1: Foundational Overview (Undergraduate-Level)
* **Introduction:** A basic-level summary of the paper’s core problem and concept.
* **Key Findings:** Summarize the main results and the primary algorithm or methodology used.
* **Initial Takeaway:** Explain the high-level significance of the work in simple terms.

Phase 2: Graduate-Level Deep Dive
* **Technical Analysis:** Provide an advanced technical breakdown of the methodology suitable for a graduate student or researcher.
* **Mathematical Rigor:** Include **all** critical equations from the paper.
* **Derivations:** Do not simply state formulas; provide step-by-step mathematical derivations to show how results are reached.
* **Formatting:** Use **LaTeX** for all mathematical variables, expressions, and standalone equations.

To maintain high accuracy and prevent information loss, you must process references **one-by-one** using the following loop:

1.  **Analyze:** Process a single reference according to the Phase 1 and Phase 2 requirements.
2.  **Update:** Append this analysis to the master "Reference Review" document.
3.  **Checkpoint:** Use your **"Ask User Question"** skill to provide a brief summary of what was just added and ask for permission to proceed to the next reference.
4.  **Repeat:** Do not attempt to batch multiple papers in a single turn.

* Use clear `##` and `###` headers for each paper title and sub-section.
* Maintain an auto-updating Table of Contents at the top of the file as new reviews are added.
* Ensure all LaTeX syntax is correctly formatted for Markdown rendering.

### Token Efficiency & Agent Policy

- **Do NOT use subagents or multiagent parallelization** unless the user explicitly requests it. If you believe subagents would help, **ask the user first** before spawning any.
- Prefer single batch shell scripts (loops) over launching many parallel agents/commands. One script that processes 16 runs sequentially is far cheaper than 16 separate agent calls.
- Always save intermediate results to `tmp/` files **after each extraction step** — never accumulate results only in context. This prevents data loss if the conversation is compressed.
- Avoid redundant work: do not extract the same data through multiple paths.

### Configuration Protocol

- **No fallback defaults** for critical config params. Implementation agents must use `config.get_mandatory('key')` — missing YAML key → `ValueError`.
- New config keys added by plans must be listed in the plan's File Changes section with the exact YAML path and value.

### WandB Analysis Protocol

- When analyzing WandB training results using the WandB skill, **always create a temporary document** in the `tmp/` folder to record all extracted data and findings as you go.
- Use a descriptive filename prefixed with a datetime stamp in `YYYYMMDD_HHMMSS` format (e.g., `tmp/20260310_143052_wandb_dreamer_comparison.md`, `tmp/20260310_150817_wandb_reward_analysis.md`). Multiple analyses may run in parallel, so each should have its own file with a unique timestamp.
- Write results to this file **after each extraction step** — do not wait until the end. This prevents loss of earlier results if the conversation context is compressed.
- The document should include: run IDs, metric values, tables, comparisons, and any intermediate observations.
- These files are temporary working notes — they can be cleaned up or overwritten as needed.

### Training Results Analysis Workflow

When the user asks for training results analysis (typically with an attached screenshot/image):

1. **Extract run datetime IDs** — read the attached image and extract the datetime identifiers for the runs to analyze (format: `YYYYMMDD_HHMMSS`).
2. **Locate local WandB log files** — find matching runs in the local `wandb/` folder (e.g., `wandb/run-YYYYMMDD_HHMMSS-<wandb_id>/`). Do NOT query the WandB web API for log data — use local files only.
3. **Use the WandB skill** — invoke the `wandb-analysis` skill to parse and analyze the local log files.
4. **Temporal evolution analysis** — the analysis **must** include temporal evolution of logged metrics (how key metrics change over training steps/episodes). Show trends, inflection points, and convergence behavior.
5. **Produce analysis report** — write the final analysis to a `docs/` file using the `docs/TEMPLATES/training_analysis.md` template. Follow the WandB Analysis Protocol above for intermediate results in `tmp/`.
6. **P**erformance Evaluation (Survival Logic)** - Evaluate the agent’s performance based on survival steps rather than cumulative reward.

### Verification Protocol

After Gemini completes implementation, verify using this workflow:

1. **Read the plan doc** — check Implementation Report and Checkpoints for what was done, any deviations, or blockers.
2. **Diff stats check** — run `git diff --stat HEAD` first. Check the insertion/deletion counts per file against the plan's expected scope. A plan that calls for a 2-line fix should not show hundreds of deletions. **Flag any file where the net line change is disproportionate to the planned change** — this catches accidental deletions, file truncations, or scope creep before detailed review.
3. **Git diff** — run `git diff HEAD` to see Gemini's uncommitted changes vs the last commit. Cross-reference against the plan's File Changes section.
4. **Flag unexpected changes** — any files modified that were not in the plan should be noted as out-of-scope in the Verification Report.
5. **Targeted reads** — only read specific lines if the diff is unclear or logic needs closer inspection.
6. **Fill the Verification Report** — complete the table with `✅`/`⚠️`/`❌` per file, then write a one-line conclusion. Use `Verified by: Claude`.

