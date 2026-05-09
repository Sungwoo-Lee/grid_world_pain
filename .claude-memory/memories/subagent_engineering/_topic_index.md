# _topic_index.md — `subagent_engineering` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `subagent_engineering` topic.

**Folder definition**: Subagent + worktree usage gotchas
**Insights**: 3
**Last updated**: 2026-05-09

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-09 | 15:37 | `20260509_1537_professor_analysis_resets_exotic_investigation` | Routing pattern: when an investigation has drifted toward bespoke instrumentation (per-action probes, fork-rollouts, custom metrics), pulling in a professor-level conventional-cause analysis BEFORE writing platform-development plans short-circuits cycles. Concrete arc: user pushback "this isn't too difficult" → professor-rl-bayesian-dl → top-3 ranked conventional causes → 2-cell battery → clean refutation in <12h wall-clock. |
| 2026-05-09 | 15:34 | `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs` | Developer agents asked for "smoke tests" will sometimes substitute synthetic Python scripts that bypass the entrypoint's dict-assembly step and miss missing-key wiring bugs. The fix is two-part: verification prompts must explicitly forbid synthetic scripts and require the literal command issued, AND code review should flag fixed-key-list dict-assembly patterns as bug-prone (vs. `dict.get()`-style direct extraction). |
| 2026-05-08 | 04:30 | `20260508_0430_worktree_isolation_path_safety` | Agent-tool worktree isolation is filesystem-only, not path-namespace; subagents using absolute paths escape the sandbox; subagent prompts must enforce repo-relative paths. |

---

## Change history

- 2026-05-09: Added insight `20260509_1537_professor_analysis_resets_exotic_investigation` from the dreamer conventional-fixes session — routing pattern: professor-rl analysis BEFORE platform-development plans when investigation has gone exotic. No new tags (all reused: subagent, learned_lesson, meta, decision).
- 2026-05-09: Added insight `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs` from the hypervigilance Round 2 partial-verdict + per-tag metrics ship session — generalizable lesson about developer-agent smoke tests. No new tags promoted (all reused: subagent, learned_lesson, meta, decision).
- 2026-05-08: Folder created. Added insight `20260508_0430_worktree_isolation_path_safety` (genesis insight for this topic).
