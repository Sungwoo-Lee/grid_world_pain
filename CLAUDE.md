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

### Configuration Protocol

- **No fallback defaults** for critical config params. Implementation agents must use `config.get_mandatory('key')` — missing YAML key → `ValueError`.
- New config keys added by plans must be listed in the plan's File Changes section with the exact YAML path and value.

### WandB Analysis Protocol

- When analyzing WandB training results using the WandB skill, **always create a temporary document** in the `tmp/` folder to record all extracted data and findings as you go.
- Use a descriptive filename prefixed with a datetime stamp in `YYYYMMDD_HHMMSS` format (e.g., `tmp/20260310_143052_wandb_dreamer_comparison.md`, `tmp/20260310_150817_wandb_reward_analysis.md`). Multiple analyses may run in parallel, so each should have its own file with a unique timestamp.
- Write results to this file **after each extraction step** — do not wait until the end. This prevents loss of earlier results if the conversation context is compressed.
- The document should include: run IDs, metric values, tables, comparisons, and any intermediate observations.
- These files are temporary working notes — they can be cleaned up or overwritten as needed.

### Verification Protocol

After Gemini completes implementation, verify using this workflow:

1. **Read the plan doc** — check Implementation Report and Checkpoints for what was done, any deviations, or blockers.
2. **Diff stats check** — run `git diff --stat HEAD` first. Check the insertion/deletion counts per file against the plan's expected scope. A plan that calls for a 2-line fix should not show hundreds of deletions. **Flag any file where the net line change is disproportionate to the planned change** — this catches accidental deletions, file truncations, or scope creep before detailed review.
3. **Git diff** — run `git diff HEAD` to see Gemini's uncommitted changes vs the last commit. Cross-reference against the plan's File Changes section.
4. **Flag unexpected changes** — any files modified that were not in the plan should be noted as out-of-scope in the Verification Report.
5. **Targeted reads** — only read specific lines if the diff is unclear or logic needs closer inspection.
6. **Fill the Verification Report** — complete the table with `✅`/`⚠️`/`❌` per file, then write a one-line conclusion. Use `Verified by: Claude`.

