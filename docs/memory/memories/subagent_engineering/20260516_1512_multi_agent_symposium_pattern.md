---
id: 20260516_1512_multi_agent_symposium_pattern
date: 2026-05-16
time: "15:12"
folder: subagent_engineering
tags: [subagent, design, decision, meta, learned_lesson]
summary: "Multi-agent symposium pattern (evolution of the v2 research chain): 4 professors run in parallel with explicit fresh-ideas framing; postdoc synthesises and applies user-refined constraints; PI frames the portfolio call; top-level Claude surfaces AskUserQuestion because PI cannot in subagent mode. Each prof writes to a sidecar to avoid parallel-write race."
related: ["20260509_1621_multi_agent_research_chain_v2_pattern", "20260516_1504_symposium_substrate_right_rhetoric_wrong", "20260516_1505_target_one_acknowledge_many_defer_full_coverage", "20260516_1511_math_reviewer_catches_silent_direction_errors"]
session_origin: claude_code
session_label: "pi-probe-prioritization / symposium round + v5"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# Multi-agent symposium pattern (4 profs + postdoc + PI)

## Pattern at a glance

```
            ┌─ professor-neuromodulation ─┐
trigger ───▶├─ professor-rl-bayesian-dl  ─┤───▶ postdoc synthesis ───▶ PI call doc
            ├─ professor-bayesian-brain  ─┤        (collector)             │
            └─ professor-pain-modeling   ─┘                                ▼
                                                              top-level AskUserQuestion
                                                              (PI cannot surface from
                                                               background subagent)
```

## Key conclusion

The 2026-05-16 impactful-vs-reasonable symposium evolved the prior [[20260509_1621_multi_agent_research_chain_v2_pattern]] into a **symposium-scale pattern**: 4 professors in parallel → postdoc synthesis → PI portfolio call → top-level AskUserQuestion. **The pattern has 3 non-obvious mechanics that distinguish it from a simple chain**: (1) each professor writes to its own sidecar file (avoids the parallel-write race four agents create on a shared file); (2) the postdoc compensates for any user-refined-mid-round constraints that the professors didn't have (constraint refinement happens via user messages between professor spawns and postdoc spawn); (3) the PI runs in subagent mode and *cannot directly call AskUserQuestion* — top-level Claude surfaces the user-facing choice.

## Evidence, measurements, facts

- **Pattern shape this session**: 4 professor symposium contributions (~1500-3700 words each, in `docs/project/symposium/20260516_impact_vs_reasonable/`) → postdoc synthesis (5300-word body) → PI call doc (~3300 words) at `docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md` → top-level Claude surfaces 4-path AskUserQuestion.
- **Sidecar pattern (mechanic 1)**: each professor wrote to a sidecar named `professor_<name>_contribution.md` in a session-specific subdirectory. Top-level Claude later committed all four + the postdoc synthesis as one logical commit (`89697aa` on v1.4). Avoids the prior pattern's parallel-write race (seen in v1 round when 2 professors had to write to v1 §8 placeholders).
- **Mid-round refinement (mechanic 2)**: between the professor spawn and the postdoc spawn, the user clarified the "no one-to-one" constraint (see [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]]). The professors had the original (over-strong) wording; the postdoc applied the refined version in the synthesis without requiring re-spawning the professors. Net result: one round-trip cost, correct final synthesis.
- **PI cannot AskUserQuestion in subagent mode (mechanic 3)**: the PI returned the call doc + the proposed user-facing options; top-level Claude then invoked `AskUserQuestion` from the main thread. The PI agent profile lists `AskUserQuestion` as a tool but in async background subagent mode the tool is not surfaced. Workaround: PI produces the choice payload; top-level surfaces it.
- **Compared to the v2 research chain**: the v2 pattern was 2-3 sequential agents producing iterative refinements. The symposium is 4 parallel agents producing complementary perspectives, synthesised once. Different problem shape (consolidation vs iteration).
- **Convergence vs disagreement handling**: all four professors converged on "substrate right, rhetoric wrong" but disagreed on which lens to lead with — see [[20260516_1504_symposium_substrate_right_rhetoric_wrong]]. The postdoc surfaces the disagreement as a path-choice rather than resolving it; the PI then frames the portfolio-level call to the user.

## Decisions and actions

- This pattern is reusable for future portfolio-level decisions: when the project hits a multi-domain framing question, spawn 4 (or appropriate-N) professors in parallel + synthesis + PI + user-question.
- Prompt engineering convention adopted: each professor's prompt includes (a) the user's verbatim constraint, (b) the 3 other professors' likely lenses for cross-reference, (c) "bring NEW perspective, not refinement", (d) sidecar output path.
- The postdoc prompt must include the user-refined constraints if any landed mid-round.

## Open questions and follow-ups

- Whether a 5- or 6-professor symposium is ever justified is open. The 4-prof set covered the project's domain space; bigger panels risk dilution.
- Whether to fold the sidecar-then-commit-once pattern into an `agent-manager` template is open.

## References

- Prior pattern (v2 research chain): [[20260509_1621_multi_agent_research_chain_v2_pattern]].
- This session's symposium artefacts: [`docs/project/symposium/20260516_impact_vs_reasonable/`](../../../docs/project/symposium/20260516_impact_vs_reasonable/).
- PI call doc: [`docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md`](../../../docs/pi/calls/2026-05-16_impact_vs_reasonable_call.md).
- Related insights: [[20260516_1504_symposium_substrate_right_rhetoric_wrong]] (the convergent finding), [[20260516_1505_target_one_acknowledge_many_defer_full_coverage]] (the mid-round refinement), [[20260516_1511_math_reviewer_catches_silent_direction_errors]] (the prior math-reviewer audit that informed the symposium prompts).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260516_1504_symposium_substrate_right_rhetoric_wrong]] (nmn_diagnosis, 2026-05-16) — Four professors (neuromod / rl-bayesian-dl / bayesian-brain / pain-modeling) ind
- [[20260516_1508_v5_last_rhetorical_round_before_experiments]] (nmn_diagnosis, 2026-05-16) — PI pace flag: project cycled through 4 versions of direction memo + concept memo
<!-- END BACKLINKS -->
