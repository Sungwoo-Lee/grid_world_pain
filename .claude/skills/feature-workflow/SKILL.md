---
name: feature-workflow
description: Sequential plan→implement→verify workflow for adding new features or capabilities to the project. Use this skill whenever the user asks to "implement", "build", "add", or "create" a new feature, model, environment component, sensor, or any other functional addition to the codebase. Orchestrates the senior-developer (planning), developer (implementation), and senior-developer (verification) agents in sequence with explicit user approval between phases. Trigger phrases include "let's add X", "implement Y", "build a new Z", "create a feature for W". Make sure to use this skill even when the user does not explicitly say "workflow" — any feature-addition request in this codebase should follow this pattern.
---

# Feature Workflow

This skill orchestrates the project's standard sequential workflow for adding new features. It enforces clean hand-offs between agents and ensures every feature is planned, implemented, tested, and verified before being declared complete.

## When to Use

- Adding new functionality to `src/` (new sensors, models, environments, training algorithms, etc.)
- Adding new config options or experiment configurations
- Extending existing components with new capabilities

**Do NOT use this skill for:**
- Bug fixes — use `bug-fix-workflow` instead (different planning emphasis: root cause first).
- Pure analysis of existing training results — use `training-experiment-workflow`.
- Documentation-only changes — single-agent task, no workflow needed.

## The Four Phases

```
[Phase 1] senior-developer  → writes plan to docs/develop/active/<topic>/  (no code touched)
[Phase 2] USER APPROVAL     → human checkpoint
[Phase 3] developer         → implements, tests, reports
[Phase 4] senior-developer  → verifies diff against plan
```

Each phase has clear hand-off artifacts. Do not skip phases or merge them.

### Phase 1 — Plan (senior-developer)

Delegate to the **`senior-developer`** agent. Its job:

1. Read [docs/TEMPLATES/issue_plan.md](../../../docs/TEMPLATES/issue_plan.md) and the [Frontmatter Contract](../../../docs/develop/active/meta/FRONTMATTER_CONTRACT.md) before writing.
2. Investigate the relevant code areas (read-only).
3. Write a plan doc to **`docs/develop/active/<topic>/<NAME>.md`** (NOT `docs/plans/`). Pick the right `<topic>` folder per [FRONTMATTER_CONTRACT.md](../../../docs/develop/active/meta/FRONTMATTER_CONTRACT.md) (`dreamer`, `behavior`, `sensors`, `refactors`, `neuromodulation`, `precision`, `filim`, `continual_learning`, etc.). Match each topic folder's existing naming convention (most use SCREAMING_SNAKE_CASE). The doc must start with YAML frontmatter:
   ```yaml
   ---
   title: "<plan title>"
   topic: <topic>           # must be in VALID_TOPICS
   status: active
   created: YYYY-MM-DD
   last_updated: YYYY-MM-DD
   ---
   ```
4. The plan must include:
   - Clear objectives and motivation.
   - File-by-file Change List with paths and line numbers.
   - Concrete code snippets where helpful.
   - Test plan (which tests to add or run).
   - Checkpoints (incremental milestones the developer can mark complete).
   - Empty Implementation Report and Verification Report sections (filled later).
5. Cross-reference any related docs.
6. After writing, run `python scripts/regen_dev_index.py` so the new file is registered in `docs/develop/INDEX.md` (auto-generated — never hand-edit).

**Output of Phase 1:** A complete plan doc at `docs/develop/active/<topic>/<NAME>.md` with valid frontmatter, registered in `INDEX.md`. Code is untouched.

### Phase 2 — User Approval (human checkpoint)

Pause and present the plan doc path to the user for review. Do not proceed without explicit approval. The user may request revisions, which loop back to Phase 1.

"Plan approved" by the user authorizes Phase 3 — and only Phase 3.

### Phase 3 — Implement (developer)

Delegate to the **`developer`** agent. Its job:

1. Read the approved plan end-to-end before any code changes.
2. Implement file-by-file in the order specified.
3. Honor the Configuration Protocol: `config.get_mandatory()` for critical params, no fallback defaults, new YAML keys exactly as listed.
4. Run the targeted tests from the plan's test plan after each meaningful change.
5. Append an **Implementation Report** to the bottom of the plan doc with:
   - File-by-file summary of changes.
   - Test results (commands run, pass/fail).
   - Any deviations from the plan and why.
   - Blockers or follow-up items.
6. Update the **Checkpoints** section.
7. **Leave the working tree dirty** (do not commit). This is critical — `senior-developer` needs the diff to verify.

**Output of Phase 3:** Code changes (uncommitted), Implementation Report appended, Checkpoints updated.

### Phase 4 — Verify (senior-developer)

Delegate back to the **`senior-developer`** agent for the Verification Protocol:

1. Read the Implementation Report and Checkpoints.
2. `git diff --stat HEAD` — check insertion/deletion counts against the plan's expected scope. Flag disproportionate changes.
3. `git diff HEAD` — cross-reference each modified file against the plan's File Changes section.
4. Flag any files modified that were not in the plan as out-of-scope.
5. Targeted reads only if the diff is unclear.
6. Fill the **Verification Report** in the plan doc — table with `✅`/`⚠️`/`❌` per file, one-line conclusion, signed `Verified by: senior-developer`.

**If the diff touches `configs/`, `src/environment/`, or any sensor / observation-breakdown / perceptual-noise code**, also delegate to **`env-config-auditor`** (in parallel with senior-developer's verification — they are orthogonal checks). Senior-developer verifies plan adherence; env-config-auditor verifies obs↔noise sync, mandatory-key discipline, static-field recompile risk, and latent-bug recurrences. The audit report goes to `docs/reviews/config_<plan-name>.md` and is cross-linked from the plan doc's Verification Report section.

**Output of Phase 4:** Verification Report in the plan doc; for env/config-touching changes, also a Config Audit report under `docs/reviews/`. The user then decides whether to commit, request fixes, or revert.

## Hand-off Artifact Summary

| Phase | Produces | Read by |
|---|---|---|
| 1 | `docs/develop/active/<topic>/<NAME>.md` (plan only, with frontmatter) | User (Phase 2), developer (Phase 3) |
| 2 | User approval | senior-developer flow controller |
| 3 | Code diff (uncommitted), Implementation Report, Checkpoints | senior-developer + env-config-auditor (Phase 4) |
| 4 | Verification Report; Config Audit report (if env/config touched) | User |

## Why This Sequence Matters

- **Plan-first** prevents code drift and surprise refactors. Reviewing a written plan is far cheaper than reviewing committed code.
- **User approval gate** ensures the human stays in control of *what* gets built before time is spent on *how*.
- **Uncommitted diff for verification** means the verifier sees exactly what the implementer did — no commits to wade through, no rebases, no mystery.
- **Separate verifier agent** prevents the implementer from grading their own work and catches scope creep that the implementer normalized to themselves.

## Common Mistakes to Avoid

- Skipping Phase 2 because the plan "looks obvious" — there is no such thing as an obviously correct plan.
- Letting the developer commit before verification — the verifier loses signal once changes are squashed into a commit.
- Running the verifier on the committed branch instead of the dirty working tree — same problem.
- Asking the senior-developer to also implement "since it has Read access" — the strict separation is the point of the workflow.
