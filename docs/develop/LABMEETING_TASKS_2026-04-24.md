# Lab Meeting Task List — 2026-04-24

> **Status**: PLANS DRAFTED — awaiting per-issue implementation by Gemini (Claude is analysis-only per [CLAUDE.md](../../CLAUDE.md))
> **Opened**: 2026-04-24

Four discrete issues were identified from a lab-meeting task screenshot. Each has its own detailed plan document following [docs/TEMPLATES/issue_plan.md](../TEMPLATES/issue_plan.md). The recommended execution order is below — earlier fixes are prerequisite-free and low-risk; later fixes touch more of the codebase.

## Execution order

| # | Plan | One-line summary | Risk | Recommended gate before next |
|---|------|------------------|:----:|------------------------------|
| 1 | [ISSUE_01 — predator count ignored](ISSUE_01_PREDATOR_COUNT.md) | Respect `count:` field on predator YAML entries | Low | Render a frame with `count: 3`; confirm 3 predators visible |
| 2 | [ISSUE_02 — property / properties key unified](ISSUE_02_PROPERTY_KEY_UNIFY.md) | Accept both keys, canonicalize on `properties`, migrate bundled configs | Low | Byte-compare `EnvParams` arrays pre/post migration |
| 3 | [ISSUE_03 — danger → hiding_predator rename](ISSUE_03_DANGER_TO_HIDING_PREDATOR.md) | **User must pick Option A vs B first.** Rename strings and info-dict keys (A), or split into a new entity type (B) | A=Low, B=High | Render-identical golden-frame check (Option A); new plan required for Option B |
| 4 | [ISSUE_04 — checkpoint retention config](ISSUE_04_CHECKPOINT_RETENTION.md) | Expose `max_to_keep` as a mandatory training config key | Low | Run saves >5 checkpoints with `max_checkpoints_to_keep: 20`; confirm all 20 persist |

Each plan includes **Checkpoints** the implementing agent must pass during the change. Run the checkpoints **within** the issue's edit before moving to the next issue.

## Open decision

**ISSUE_03 requires the user to pick a scope (A = rename only vs B = architectural split)** before implementation begins. Option A is drafted in detail; Option B will be drafted in a separate plan if chosen.

## Notes for the implementing agent

- The four plans are **independent** and can be merged in separate commits or PRs.
- Claude (this assistant) wrote the plans but must not write code — implementation is handled by Gemini or another code-writing agent.
- After each plan is implemented, fill in its **Implementation Report** and **Verification Report** sections.
