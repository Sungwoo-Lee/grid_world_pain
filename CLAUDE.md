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
