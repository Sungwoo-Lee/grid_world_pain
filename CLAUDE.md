## Workflow Rules

### Strict No-Implementation Policy
- **NEVER modify source code, configs, or scripts.** This includes creating, editing, or deleting any files under `src/`, `configs/`, `scripts/`, or any other code directories.
- The only files Claude may create or edit are documentation files in `docs/` and project management files like this one (`CLAUDE.md`).
- "Plan approved" means the plan document is accepted for review — it does NOT authorize code implementation.
- Even when explicitly asked to implement, confirm with the user first, as implementation is handled by other LLM agents.

### Planning & Analysis
- Claude is used for **planning and analysis only**. Implementation is done by other LLM agents.
- Plans and analysis should be written to `docs/` files (e.g., `docs/NOISE_DEBUGGING_PLAN.md`).
- Include enough detail in plans (file paths, line numbers, code snippets) for the implementing agent to execute without ambiguity.

### Verification Reports
- Start with a **summary table** showing all items and their status at a glance (file, change, status, notes).
- Use ✅ / ❌ / ⚠️ status icons in the table for quick scanning.
- Add a one-line conclusion after the table.
- For ✅ items: table row is sufficient, no extra detail needed.
- For ❌ / ⚠️ items: add a **detailed section below the table** explaining the issue, root cause, affected lines, and recommended fix — with enough context for the implementing LLM agent to resolve it without re-investigation.
