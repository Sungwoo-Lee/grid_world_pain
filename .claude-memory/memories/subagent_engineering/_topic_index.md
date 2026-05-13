# _topic_index.md — `subagent_engineering` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `subagent_engineering` topic.

**Folder definition**: Subagent + worktree usage gotchas
**Insights**: 6
**Last updated**: 2026-05-10

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-10 | 22:41 | `20260510_2241_residual_error_pattern_directs_next_fix` | Methodology pattern: when a partial fix lands (H2-band outcome), the residual-error pattern in the diagnostic output identifies the next-best candidate, NOT the original list-order or the plan's prescribed-next-step. Concretely: Z1's residual error was disproportionate on negative rewards → bin-range deviation (§6 item 2, mechanistic match) was selected over the plan §4 H2 prescription (candidate #1, GRU reset gate). |
| 2026-05-10 | 22:40 | `20260510_2240_reference_impl_compare_only_act_intersections` | User-stated rule for using a third-party reference implementation (sheeprl) as a comparator: default action is to document differences in the concept doc; act on a difference (i.e., plan code changes to match) only when it intersects the live failure investigation. External reference impls always surface N differences, most of which are deliberate or framework-only divergences; reflexively matching them is a category error. |
| 2026-05-09 | 16:21 | `20260509_1621_multi_agent_research_chain_v2_pattern` | Multi-agent research chains (postdoc triage → professor directions → postdoc synthesis) handle mid-chain user expansion via append-only sibling versioning, NOT in-place revision. Worked example: NMN meta/continual pivot — professor was re-spawned with a v2 task and wrote `nmn_meta_context_conditioning_v2.md` alongside the v1; postdoc then wrote `nmn_meta_continual_synthesis_v2.md` alongside the v1. v1 memos preserved as historical snapshots. |
| 2026-05-09 | 15:37 | `20260509_1537_professor_analysis_resets_exotic_investigation` | Routing pattern: when an investigation has drifted toward bespoke instrumentation (per-action probes, fork-rollouts, custom metrics), pulling in a professor-level conventional-cause analysis BEFORE writing platform-development plans short-circuits cycles. Concrete arc: user pushback "this isn't too difficult" → professor-rl-bayesian-dl → top-3 ranked conventional causes → 2-cell battery → clean refutation in <12h wall-clock. |
| 2026-05-09 | 15:34 | `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs` | Developer agents asked for "smoke tests" will sometimes substitute synthetic Python scripts that bypass the entrypoint's dict-assembly step and miss missing-key wiring bugs. The fix is two-part: verification prompts must explicitly forbid synthetic scripts and require the literal command issued, AND code review should flag fixed-key-list dict-assembly patterns as bug-prone (vs. `dict.get()`-style direct extraction). |
| 2026-05-08 | 04:30 | `20260508_0430_worktree_isolation_path_safety` | Agent-tool worktree isolation is filesystem-only, not path-namespace; subagents using absolute paths escape the sandbox; subagent prompts must enforce repo-relative paths. |

---

## Change history

- 2026-05-10: Added 2 insights from the dreamer sheeprl-comparison + zero-init session: `20260510_2240_reference_impl_compare_only_act_intersections` (rule for handling third-party reference impl comparisons — document by default, act only on intersections with live failure) and `20260510_2241_residual_error_pattern_directs_next_fix` (residual-error pattern dictates next-fix selection in iterative cascades, not list order). No new tags (all reused: subagent, learned_lesson, decision, meta).
- 2026-05-09: Added insight `20260509_1621_multi_agent_research_chain_v2_pattern` from the NMN meta/continual pivot session — append-only sibling versioning when the user expands the brief mid-chain in a multi-agent research chain. No new tags (all reused: subagent, design, decision, learned_lesson).
- 2026-05-09: Added insight `20260509_1537_professor_analysis_resets_exotic_investigation` from the dreamer conventional-fixes session — routing pattern: professor-rl analysis BEFORE platform-development plans when investigation has gone exotic. No new tags (all reused: subagent, learned_lesson, meta, decision).
- 2026-05-09: Added insight `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs` from the hypervigilance Round 2 partial-verdict + per-tag metrics ship session — generalizable lesson about developer-agent smoke tests. No new tags promoted (all reused: subagent, learned_lesson, meta, decision).
- 2026-05-08: Folder created. Added insight `20260508_0430_worktree_isolation_path_safety` (genesis insight for this topic).
