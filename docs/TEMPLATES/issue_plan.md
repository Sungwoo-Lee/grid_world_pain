# [Issue Title]

> **Status**: [INVESTIGATING | PLANNED | IN PROGRESS | COMPLETED]
> **Opened**: [date]
> **Related**: [links to related docs if any]

---

## Context

What is the problem? Why does it matter? How was it discovered?

<!-- Keep this concise — 2-5 sentences. Link to external evidence (WandB runs, logs) if applicable. -->

## Analysis

Investigation findings, root cause, supporting data.

<!-- Include tables, code traces, or data that justify the implementation direction.
     This section is the "proof" — it should convince a reader the fix is correct. -->

## Implementation Plan

### Design

High-level approach and rationale.

<!-- Data flow diagrams, architecture decisions, why this approach over alternatives. -->

### File Changes

For each file to modify:

#### `path/to/file.py` (lines X–Y)

```python
# BEFORE:
existing_code()

# AFTER:
new_code()
```

<!-- Include enough surrounding context for the implementing agent to locate the exact change. -->

## Checkpoints

What the implementing agent should verify **during** implementation:

- [ ] Checkpoint 1 — description
- [ ] Checkpoint 2 — description
- [ ] Checkpoint 3 — description

<!-- These are sanity checks to catch mistakes early, not final verification.
     Examples: "print obs shape after change", "assert no NaN in output", "run single episode". -->

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the implementing agent after code changes are made.
     Describe what was done, any deviations from the plan, and why. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]

<!-- For ⚠️/❌ items, add detailed sections below the table with:
     root cause, affected lines, and recommended fix. -->

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.
-->
