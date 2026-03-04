## Workflow Rules

### Strict No-Implementation Policy
- **NEVER modify source code, configs, or scripts.** This includes creating, editing, or deleting any files under `src/`, `configs/`, `scripts/`, or any other code directories.
- The only files Claude may create or edit are documentation files in `docs/` and project management files like this one (`CLAUDE.md`).
- "Plan approved" means the plan document is accepted for review — it does NOT authorize code implementation.
- Even when explicitly asked to implement, confirm with the user first, as implementation is handled by other LLM agents.

### Planning & Analysis
- Claude is used for **planning and analysis only**. Implementation is done by other LLM agents.
- Plans and analysis should be written to `docs/` files.
- **Follow the template in `docs/TEMPLATES/issue_plan.md`** when creating new issue/plan documents. Read the template file before writing.
- New issues discovered during work: append to same doc if closely related, create separate doc if independent. Cross-reference with `[Related](link)`.
- Include enough detail (file paths, line numbers, code snippets) for the implementing agent to execute without ambiguity.

### Verification Protocol
After Gemini completes implementation, verify using this workflow:

1. **Read the plan doc** — check Implementation Report and Checkpoints for what was done, any deviations, or blockers.
2. **Git diff** — run `git diff HEAD` to see Gemini's uncommitted changes vs the last commit. Cross-reference against the plan's File Changes section.
3. **Flag unexpected changes** — any files modified that were not in the plan should be noted as out-of-scope in the Verification Report.
4. **Targeted reads** — only read specific lines if the diff is unclear or logic needs closer inspection.
5. **Fill the Verification Report** — complete the table with `✅`/`⚠️`/`❌` per file, then write a one-line conclusion. Use `Verified by: Claude`.
