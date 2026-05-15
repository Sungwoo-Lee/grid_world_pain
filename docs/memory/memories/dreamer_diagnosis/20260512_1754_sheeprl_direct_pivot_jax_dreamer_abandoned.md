---
id: 20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned
date: 2026-05-12
time: "17:54"
folder: dreamer_diagnosis
tags: [dreamer, decision, learned_lesson, refutation, meta]
summary: "After 3-reviewer ✅ PASS on a 1033-line JAX re-implementation plan, the user pivoted via PI call to using sheeprl PyTorch directly; static review of plan-against-paper does not predict integration-layer execution success, the cascade-debugging history was a stronger prior than the plan reviews."
related: []
session_origin: claude_code
session_label: "dreamer_srl plan + PI pivot to sheeprl-direct"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/268d07a3-2eac-4772-858a-c44fb812d10d.jsonl
raw_completeness: full
---

# Pivot from JAX dreamer-srl rebuild to sheeprl-direct: static review does not predict integration success

## Key conclusion

A 1033-line clean re-implementation plan for sheeprl's DreamerV3 in JAX/NNX (called `dreamer-srl`) reached unanimous ✅ PASS across three reviewers (`professor-rl-bayesian-dl`, `math-reviewer`, `code-reviewer`) over two audit rounds — 29 deviations resolved. The user then asked whether executing it was worth it, given that sheeprl's PyTorch implementation already learns the task 5× better out-of-the-box (~500 vs ~106 survival steps on food-only NoPred). PI was spawned, surfaced four candidate paths, and the user picked Option 1: sheeprl-direct with a minimal bridge. The deeper meta-lesson: the multi-month JAX Dreamer cascade-debugging (cascade items #2 / #27 / #28 / #29 / #30) reviewed each fix individually, signed off each, and still did not close the 5× gap. Static review of a plan against the cited paper or reference implementation does not detect integration-layer failures. The cascade-debugging history was therefore strong prior evidence that executing the dreamer-srl plan would replicate that failure mode, even with three clean ✅ PASS audits.

## Evidence, measurements, facts

- Sheeprl drop-in smoke run `jzgkcep4` (2026-05-11): survival ~500 steps with stock sheeprl on food-only NoPred.
- In-house JAX Dreamer baseline survival on the same task: ~106 steps.
- Gap: 5× — survived multi-month cascade fix attempts.
- Plan size: 1033 lines (`docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md` — now superseded).
- Three-reviewer audit rounds: v1 found 29 deviations (11 algorithm + 7 math + 11 JAX/NNX); v2 resolved all 29; v2 re-audit found 0 residual deviations.
- Estimated dreamer-srl implementation cost (per plan): 2–6 weeks before parity gate (3-seed survival ≥ ~500).
- PI flagged: "the same 3 reviewers who signed off ✅ PASS on the v2 plan are exactly the people who reviewed the cascade fixes one at a time and didn't catch the integration-layer issue either."
- PI call: `docs/pi/calls/2026-05-12_dreamer_backend.md` — four options offered, user picked Option 1 (sheeprl-direct minimal bridge).
- User's verbatim rationale: *"I have spent more time to debug and solve the implementation issue. So, now I more prefer to use sheeprl version of dreamer to my project."*

## Decisions and actions

- Dropped the `dreamer-srl` JAX re-implementation. Plan + 6 reviewer companion files moved to `docs/develop/archive/dreamer_srl/` with `status: superseded` and body-text `Superseded by:` pointer to the PI call.
- Adopted sheeprl's PyTorch DreamerV3 as the Dreamer backend for the NMN/pain-modeling paper. WandB project unified at `grid_world_pain` (same as existing rPPO + JAX Dreamer runs).
- NMN/FiLM/precision-modulation port to PyTorch is the next focus (senior-developer feasibility verdict: ✅ straightforward; ~1–2 weeks).
- User signaled long-term intent to migrate all agents (including rPPO) to PyTorch — this informs the new in-repo folder naming (`pytorch_agents/`, accommodating future rPPO without rename).
- The cascade-debugging history is now formally treated as evidence about a structural failure mode of the JAX Dreamer integration layer, not a sequence of correctable individual bugs.

## Open questions and follow-ups

- The 50k-step parity gate (CP-v2-4, WandB run `i4ulpn95`) is running on node 114 cuda:2 with `apply_noise=True`. If survival approaches ~500 by 30–50k steps, the bridge is paper-ready. If not, the integration-layer suspicion was wrong and we re-open the question.
- The "static review doesn't predict integration success" lesson generalizes — when the next big plan lands a clean ✅ PASS, weight the team's prior failure-modes history against the plan's apparent cleanliness.
- PI flagged `docs/pi/PORTFOLIO.md` active publication tracks are still TBD. A portfolio-ratification call before the next major strategic decision would anchor future calls.

## References

- Active plan: [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../../../docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md)
- Archived plan: [`docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md`](../../../docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md) — 1033 lines + 6 reviewer companion files
- PI call: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../docs/pi/calls/2026-05-12_dreamer_backend.md)
- Diagnosis (the 5× evidence): [`docs/develop/active/diagnosis/sheeprl_drop_in_test.md`](../../../docs/develop/active/diagnosis/sheeprl_drop_in_test.md)
- Related prior insight: [`20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry`](../dreamer_diagnosis/20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry.md) — last cascade fix before the pivot
- Related prior insight: [`20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt`](../dreamer_diagnosis/20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt.md) — paper-canonical bins partial-closure attempt
- WandB: sheeprl smoke [`jzgkcep4`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4) — the 5× evidence run
- WandB: v2 parity run [`i4ulpn95`](https://wandb.ai/sungwoolee/grid_world_pain/runs/i4ulpn95) — currently running, will confirm or refute the pivot's premise
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 268d07a3-2eac-4772-858a-c44fb812d10d` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_1417_jax_vmap_no_speedup_tiny_env]] (dreamer_diagnosis, 2026-05-13) — JAX-vmap parallel env over our 5×5 NoPred gridworld delivered no speedup vs shee
- [[20260513_2308_strong_strategy_validates_on_cp1]] (dreamer_diagnosis, 2026-05-13) — The Strong (A+B+C+D) deviation-prevention strategy paid off on the first checkpo
<!-- END BACKLINKS -->
