---
id: 20260509_1537_professor_analysis_resets_exotic_investigation
date: 2026-05-09
time: "15:37"
folder: subagent_engineering
tags: [subagent, learned_lesson, meta, decision]
summary: "Routing pattern: when an investigation has drifted toward bespoke instrumentation (per-action probes, fork-rollouts, custom metrics), pulling in a professor-level conventional-cause analysis BEFORE writing platform-development plans short-circuits cycles. Concrete arc this session: user pushback 'this isn't too difficult' → professor-rl-bayesian-dl analysis → top-3 ranked conventional causes → 2-cell battery → clean refutation localized the failure to a single component (the reward head) in <12h wall-clock."
related: ["20260508_0430_worktree_isolation_path_safety"]
session_origin: claude_code
session_label: "dreamer_conventional_fixes_battery_2026-05-09"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Routing pattern: professor analysis resets exotic investigation

## Key conclusion
When an investigation has gone deep into bespoke instrumentation territory (custom probes, novel metrics, multi-cell batteries grounded in project-specific intuitions), the cheapest forward move is to pull in a professor-level domain expert FIRST and ask them to walk the textbook failure-mode checklist for the underlying class of system, BEFORE writing more platform-development plans. The right user-side trigger for this routing is informal: phrases like "this isn't too difficult" or "what's the most conventional issue" signal that the user thinks we are over-engineering and want a sanity-check from the published literature. The right Claude-side response is to spawn the relevant `professor-*` agent with explicit instructions to ground in conventional causes, NOT to skip the professor and write the plan directly. Plan-authorship comes AFTER the analysis.

## Evidence, measurements, facts
- Concrete arc this session (DreamerV3 hypervigilance investigation):
  1. Initial direction (post-probe-battery): plan a per-action conditional `cont` probe + checkpoint-based imagination probe (bespoke, ~2 weeks of platform work).
  2. User pushback verbatim: "I don't think this problem is too difficult thing. For example, what if we don't use homeostatic reward and no pred?"
  3. Routing response: spawned `professor-rl-bayesian-dl` directly (single-agent, bypassed `agent-manager`) with explicit framing in the prompt — "Don't propose a bespoke probe; ground the analysis in the textbook DreamerV3 failure modes."
  4. Output: 2 memos under `docs/project/` (256 + 201 lines combined), top-3 ranked conventional causes, two specific one-line config fixes that hadn't been tried.
  5. Followed through: `experiment-designer` 2-cell battery → `env-config-auditor` pre-flight (4 PASS) → `training-runner` launch on n113 → `experiment-analyzer` verdict at 13:36.
  6. Battery verdict: top-2 conventional causes refuted on predator (sibling insight `20260509_1535`). Conventional checklist exhausted; a previously-deferred `senior-developer` plan for an OFFLINE WM-imagination diagnostic was then written and executed → reward-head failure localized (sibling insight `20260509_1534`).
  - Total wall-clock from user pushback to localized verdict: <12 hours.
- Pattern dependency: requires the professor-level agent to honor "ground in conventional, not bespoke" framing. The instruction needs to be explicit in the spawn-prompt, not assumed. The professor's own bias might be toward novel framings; the framing needs to be load-bearing in the prompt.
- The user's own informal heuristic ("this isn't too difficult") was the trigger — Claude did NOT propose this routing pattern unprompted. Capturing this insight is partly a directive to future-Claude: take that signal seriously and route through the professor before the plan, even when it feels like a delay.

## Decisions and actions
- Add to investigation playbook: when initial planning lands on bespoke instrumentation AND the user signals "we're overcomplicating," spawn `professor-*` first with explicit conventional-cause framing. Plan-authorship comes after.
- The right professor depends on the system class — `professor-rl-bayesian-dl` for RL/world-model issues, `professor-bayesian-brain` for inference-framing issues, `professor-pain-modeling` for construct-validity, `professor-neuromodulation` for biological-substrate. Pick by the class of the problem, not the urgency.
- Even when this routing seems wasteful (the professor "knows what we know"), the value is in the ranked conventional-checklist that the bespoke plan would skip. The 2-cell battery here only became cheap-and-decisive BECAUSE the professor pre-ranked the candidates.

## Open questions and follow-ups
- Does this pattern generalize beyond DreamerV3? Likely yes — the same logic applies to any complex ML system where bespoke probes can be deferred until conventional knobs are checked. Worth testing on the next NMN or RPPO investigation that drifts into bespoke instrumentation.
- Should this routing be encoded into `agent-manager`'s default flow (e.g., "if the conversation has produced 2+ bespoke-probe plans, suggest a professor-analysis step first")? Or kept as a user-driven trigger? Keeping it user-driven preserves agency; encoding it risks over-routing.
- Is there a symmetric anti-pattern to capture — e.g., when the user says "let's run a quick experiment" but the conventional checklist has NOT been walked, Claude should resist and route through the professor anyway? Possibly out of scope; depends on user preference.

## References
- Sibling insights from this session (the routing pattern's downstream artifacts): `20260509_1535_conventional_fixes_battery_verdict_predator_refute` (the experimental verdict the routing produced) and `20260509_1534_wm_reward_head_localized_failure_a1` (the localized component-level failure that the verdict-driven follow-up surfaced).
- Professor outputs: `docs/project/critiques/dreamer_conventional_failure_modes_for_our_setup.md`, `docs/project/directions/dreamer_minimum_viable_strip_down.md`.
- Existing subagent_engineering insight (different topic, same folder): `20260508_0430_worktree_isolation_path_safety`.
- Project routing manual: `CLAUDE.md` (Agent Team table; Researchers section).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
