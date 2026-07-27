---
id: 20260516_1511_math_reviewer_catches_silent_direction_errors
date: 2026-05-16
time: "15:11"
folder: subagent_engineering
tags: [subagent, learned_lesson, meta, design]
summary: "Math-reviewer audit caught two silent direction-memo errors (Hessian-direction reversal, Bellman-γ vs FiLM-γ dimensional incoherence) AND validated the project's novel-architecture claim (FiLM-Ensemble + heteroscedastic-precision compound). Without the audit, both errors would have propagated into experimental design."
related: ["20260509_1621_multi_agent_research_chain_v2_pattern", "20260516_1506_film_hypernet_bhn_bnn_2x2_lineage", "20260516_1507_gamma_bellman_not_forward_pass_filmable"]
session_origin: claude_code
session_label: "pi-probe-prioritization / concept memo + math audit"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/72649f91-6f90-4cac-b01a-c01c59b9d302.jsonl
raw_completeness: full
---

# Math-reviewer catches silent direction-memo errors and validates novel claims

## Key conclusion

The math-reviewer agent audited the FiLM-neuromod concept memo (10,486-word body across 18 integration rows + 4 flagged equations) and produced an audit (§10, ~3,300 words) that caught **two load-bearing dimensional errors** that would have propagated silently into experimental design, AND **validated** the project's novel-architecture claim (Row EE-6 = FiLM-Ensemble + heteroscedastic-precision compound) as mathematically defensible — leading to a clean reframe as "first instantiation of an approximate Bayesian Hypernetwork with heteroscedastic likelihood for an RL policy". **Workflow lesson: math-heavy direction memos benefit from a math-reviewer audit before they reach experiment-designer.**

## Evidence, measurements, facts

- **Flag (c) — Hessian-eigenvalue probe direction REVERSED in §6.3**: original concept memo claimed gain → flattening at the raw-weight Hessian; audit derived that under Rodriguez-Garcia 2026's NGM-SGD, H_W is *amplified* by g², not flattened — flattening lives at H_{W_eff} = H_W/g². Without this fix, the experimental discriminator would have measured the wrong direction. Fixed in v4 §5.6.
- **Flag (d) — Bellman-γ vs FiLM-γ dimensional incoherence at GRU update-gate**: GRU update-gate is per-unit recurrent-state retention, NOT a value-function horizon. The claim "additive β at GRU update-gate ⇔ effective Bellman γ change" was incoherent. Audit corroborated `professor-rl-bayesian-dl`'s prior warning from v1 §8 Q3 — two independent reviews flagged the same error. Fixed in v4 (site B renamed plasticity gating); see [[20260516_1507_gamma_bellman_not_forward_pass_filmable]] for the full architectural consequence.
- **Flag (a) — Abdollahzadeh reinterpretation lemma**: audit verified the Tsuda ⊂ FiLM ⊂ KML nesting chain. HOLDS with one-sentence clarification (conv-kernel vs generic-weight-matrix operand). Anchored Row EE-6's lineage claim — see [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]].
- **Flag (b) — FiLM-Ensemble + heteroscedastic-precision compound (project's unique novel claim)**: audit validated sign convention (matches Kendall-Gal Eq. 6 exactly) + gradient-flow structure. HOLDS with three minor explicitness fixes (post-softmax U^epist, ρ=2 starting value, explicit f functional form). This validation is what enabled the v3-round professor to reframe Row EE-6's headline as "first instantiation of an approximate BHN with heteroscedastic likelihood for an RL policy" (cleaner novelty claim, lineage-grounded).
- Audit verdict: "the memo's headline integration claim is mathematically defensible. Two required surgical fixes (flags c + d), three minor revisions (flag b). No re-spawn needed."

## Decisions and actions

- Math-reviewer added to the post-concept-memo / pre-experimental-design workflow as a recommended step for math-heavy direction memos. Not yet codified into a standing flow, but the v4 → v5 cycle relied on the audit.
- Fixed errors carried into v4 (`028cdde` Hessian direction; site B rename) and v5 (target-as-basket framing absorbs the deeper lesson).
- Confirmed value: the audit was ~3300 words but caught errors that would have wasted weeks of experimental design.

## Open questions and follow-ups

- Whether math-reviewer should be a default agent in any `senior-developer` plan that touches equation-heavy code (FiLM γ/β, heteroscedastic precision, TD-target γ(c)) is open. Currently invoked ad-hoc.
- The "two independent reviews flag the same error" pattern (rl-bayesian-dl prof + math-reviewer both caught flag d) is a strong signal — when two independent agents converge on a problem, it is almost certainly real.

## References

- Concept memo §10 audit: [`docs/project/concepts/film_neuromod_integration.md`](../../../docs/project/concepts/film_neuromod_integration.md).
- v4 absorption commit: `028cdde` on v1.4.
- Related insights: [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]] (Claim 1 / Abdollahzadeh validated by audit), [[20260516_1507_gamma_bellman_not_forward_pass_filmable]] (flag d's architectural consequence), [[20260509_1621_multi_agent_research_chain_v2_pattern]] (prior multi-agent-flow learning).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `claude --resume 72649f91-6f90-4cac-b01a-c01c59b9d302`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260516_1506_film_hypernet_bhn_bnn_2x2_lineage]] (nmn_diagnosis, 2026-05-16) — Formal 2×2 lineage of FiLM / Hypernet / Bayesian Hypernet / Bayesian Neural Net 
- [[20260516_1512_multi_agent_symposium_pattern]] (subagent_engineering, 2026-05-16) — Multi-agent symposium pattern (evolution of the v2 research chain): 4 professors
<!-- END BACKLINKS -->
