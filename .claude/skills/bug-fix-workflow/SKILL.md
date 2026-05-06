---
name: bug-fix-workflow
description: Sequential root-cause-analysis→fix→verify workflow for bug investigation and resolution. Use this skill whenever the user reports a bug, error, regression, or unexpected behavior — phrases like "fix the bug in X", "this is broken", "X is throwing an error", "training crashes when Y", "investigate why Z", "debug this issue". The skill orchestrates the senior-developer (root cause analysis and fix plan), developer (implementation plus regression test), and senior-developer (verification) agents. Use even when the user does not say "workflow" or "bug" explicitly — any report of broken behavior in this codebase should follow this pattern.
---

# Bug Fix Workflow

This skill orchestrates the standard sequential workflow for diagnosing and fixing bugs. It differs from `feature-workflow` in two important ways: the planning phase emphasizes **root cause analysis** (not just "what to build"), and Phase 3 always includes a **regression test**.

## When to Use

- A test is failing or producing unexpected results.
- Training is crashing, NaN-ing, or diverging unexpectedly.
- A previously-working feature has stopped working.
- The user reports observed behavior that conflicts with intended behavior.

**Do NOT use this skill for:**
- Adding new functionality — use `feature-workflow`.
- Re-running training to confirm a fix worked — that's part of `training-experiment-workflow`.
- Bugs in the plan itself (logic errors in a `docs/` file) — single-agent edit, no workflow needed.

## The Four Phases

```
[Phase 1] senior-developer  → root cause analysis + fix plan to docs/
[Phase 2] USER APPROVAL     → human checkpoint on diagnosis and proposed fix
[Phase 3] developer         → applies fix + regression test, runs full suite
[Phase 4] senior-developer  → verifies fix against plan, confirms regression test catches the bug
```

### Phase 1 — Root Cause Analysis & Fix Plan (senior-developer)

Delegate to the **`senior-developer`** agent. Its job:

1. Read [docs/TEMPLATES/issue_plan.md](../../../docs/TEMPLATES/issue_plan.md).
2. **Reproduce the bug first.** Identify the minimum command, config, or test that triggers the broken behavior. If the user described the bug informally, ask for the exact reproducer.
3. Trace the bug to its root cause. Read the relevant source files. Use `grep` and `git log` / `git blame` to understand when and why the offending code was introduced. Distinguish *symptom* from *cause* — a NaN in the loss is a symptom; the cause might be a config default that should not exist.
4. Write a fix plan to `docs/plans/bugfix-<short-name>.md` with these sections:
   - **Symptom** — the observed broken behavior.
   - **Reproduction** — exact command or test that triggers it.
   - **Root cause** — file:line of the underlying issue, with explanation.
   - **Proposed fix** — file-by-file changes with code snippets.
   - **Regression test** — exact test to add (path + name) that would catch this bug if it returned.
   - **Test plan** — which existing tests to run to confirm no regressions elsewhere.
   - **Checkpoints** and empty Implementation/Verification Report sections.
5. If the root cause turns out to be in the project's design rather than a localized bug, **stop and flag it** — design changes belong in `feature-workflow`, not bug-fix-workflow.

**Output of Phase 1:** A diagnosis-and-fix plan. The bug is fully understood; no code is touched.

### Phase 2 — User Approval (human checkpoint)

Present the plan to the user. Two questions matter here:
- Is the **root cause analysis** correct?
- Is the **proposed fix** the right approach (vs. a workaround, a revert, etc.)?

The user's approval authorizes Phase 3.

### Phase 3 — Apply Fix + Regression Test (developer)

Delegate to the **`developer`** agent. Its job:

1. Read the approved plan.
2. **Add the regression test first** — confirm it fails on the current (pre-fix) code. This is the *test that proves the fix works*; if it passes before the fix, it's the wrong test.
3. Apply the fix as specified in the plan.
4. Confirm the regression test now passes.
5. Run the full test suite (or the targeted subset from the plan's test plan).
6. Append an Implementation Report including:
   - The pre-fix and post-fix state of the regression test (failed → passes).
   - Full suite results.
   - Any deviations from the plan and why.
7. Update Checkpoints. Leave the working tree dirty.

**Output of Phase 3:** Fix + regression test (uncommitted), Implementation Report, full suite green.

### Phase 4 — Verify (senior-developer)

Delegate back to the **`senior-developer`** agent. Verification Protocol with bug-specific additions:

1. Standard diff stats and `git diff HEAD` review.
2. **Confirm the regression test exists and is meaningful** — it should fail without the fix and pass with it. Reading the test alone should clarify what bug it guards against.
3. **Confirm the fix targets the root cause, not just the symptom.** If the plan said the cause was a missing config validation but the fix only catches the resulting NaN, that's a `⚠️` or `❌`.
4. Fill the Verification Report. Sign as `Verified by: senior-developer`.

## Why Bug Fixes Get Their Own Workflow

Bugs and features look similar from the outside but require different mental modes:

- **Features** start from "what should the system do?" — the plan describes the *destination*.
- **Bugs** start from "why did the system do this wrong thing?" — the plan must first describe the *current broken state* before the *correct state*.

A feature plan that omits motivation can still be implemented. A bug plan that omits the root cause produces a workaround that masks the underlying issue, leaving a time bomb in the codebase.

The mandatory regression test exists because bugs return. The cheapest insurance against a regression is a test that would have caught the original bug.

## Common Mistakes to Avoid

- Treating the symptom as the cause (e.g., catching a NaN instead of fixing the upstream divergence).
- Skipping the "reproduce first" step — without a reliable reproducer, you cannot tell if the fix worked.
- Adding a regression test that passes both before and after the fix — it is not testing what you think it is.
- Bundling unrelated cleanups into the bugfix PR — the verifier will flag these as out-of-scope, and rightly so.
