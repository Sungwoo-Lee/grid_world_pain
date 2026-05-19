---
id: 20260518_1511_dreamer_srl_v2_parity_pass_outperform
date: 2026-05-18
time: "15:11"
folder: dreamer_diagnosis
tags: [dreamer, decision, learned_lesson, meta]
summary: "dreamer-srl v2 PASS-OUTPERFORMs sheeprl on food-only parity (501 vs ~500) after the v2 audit chain caught 5 P-blockers that the v1 plan-only review had missed; the strong A+B+C+D discipline is now the default template for any JAX rebuild of a PyTorch reference."
related: ["20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned", "20260513_2308_strong_strategy_validates_on_cp1", "20260518_1512_reinforce_resampling_bug_imag_action_threading", "20260518_1513_production_recipe_xs_16_4m_hypervigilance", "20260518_1514_num_envs_vs_budget_interaction"]
session_origin: claude_code
session_label: "dreamer-srl v2 parity + 10x10 hyperparameter search"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# dreamer-srl v2 parity PASS-OUTPERFORMs after 5 P-blockers fixed

## Key conclusion
The dreamer-srl v2 rebuild reached parity with sheeprl on the food-only task (ep_len_avg = 501 vs sheeprl ~500), after the v1 attempt had failed at ep_len 103.8 (random floor). The five P-blockers that the v2 strong A+B+C+D audit chain identified — missing reset_data write, terminated/truncated conflation, D-014 burst, REINFORCE action threading (see [[20260518_1512_reinforce_resampling_bug_imag_action_threading]]), and target_critic→live critic — were the actual v1 root causes; the v1 plan-only review had passed them through because static review of a 1033-line plan does not predict integration-layer execution success. v2's bit-identity grad parity tests + 3-reviewer chain + vendored sheeprl diff tool would have flagged each of them at code-review time. Production recipe carryover validated by the 10×10 hypervigilance sweep ([[20260518_1513_production_recipe_xs_16_4m_hypervigilance]]).

## Evidence, measurements, facts
- v1 parity launch FAIL: mean ep_len 103.8 (random floor) on food-only; sheeprl reference ~500.
- v2 parity launch PASS-OUTPERFORM: ep_len_avg 501 at PARITY_LAUNCH_V2 commit `bfc79a8`.
- 5 P-blockers fixed across 6 commits `1d4c1f9` … `b813d48` in v2-CP9.
- v2 added gradient-side bit-identity tests (`tests/algorithms/dreamer_srl/test_grad_parity.py`, 11 new tests) covering `jax.grad` vs `torch.autograd` on each loss head.
- 20k-step CP10 policy-learning gate (ep_len_avg > 200) passed before the parity-launch was authorised.
- Closure docs: `docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md` (CP1–CP12 all closed) and `docs/develop/active/dreamer_srl_v2/PARITY_LAUNCH_V2.md` (verdict).
- v1 lineage preserved at `docs/develop/active/dreamer_srl_v1/` (renamed from `_v3/` at commit `e461628`, 8 files moved via `git mv`, 19 cross-references updated).
- Same Karpathy-style framing as [[20260513_2308_strong_strategy_validates_on_cp1]]: strong-discipline overhead is paid in code-review minutes, not in lost months of training.

## Decisions and actions
- v2 is the production codebase for JAX DreamerV3; v1 retained as historical reference only.
- Closes the chapter opened by [[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]]: after the user pivoted to sheeprl-direct, the v2 JAX rebuild via the Strong-discipline path succeeded and now outperforms it. The "static plan review does not predict integration success" lesson stands; it just means strong discipline is the right way to do a JAX rebuild, not that JAX rebuilds are not viable.
- v2 audit-chain template is the default for any future JAX rebuild of a PyTorch reference.

## Open questions and follow-ups
- None on the food-only parity itself. Open questions on extension are tracked under [[20260518_1513_production_recipe_xs_16_4m_hypervigilance]] and [[20260518_1514_num_envs_vs_budget_interaction]].

## References
- [[20260518_1512_reinforce_resampling_bug_imag_action_threading]] — mechanistic detail on P-blocker #4, the REINFORCE re-sampling bug.
- [[20260518_1513_production_recipe_xs_16_4m_hypervigilance]] — the downstream 10×10 hypervigilance recipe carryover.
- [[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]] — the prior chapter (PI-call pivot to sheeprl-direct after v1 failed).
- [[20260513_2308_strong_strategy_validates_on_cp1]] — Strong A+B+C+D discipline first-validated on CP1; this insight is the closure.
- `docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md`
- `docs/develop/active/dreamer_srl_v2/PARITY_LAUNCH_V2.md`
- Commits: `1d4c1f9` … `b813d48` (5-blocker fixes), `bfc79a8` (parity-launch).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260518_1511_dreamer_srl_v2_parity_pass_outperform.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260518_1512_reinforce_resampling_bug_imag_action_threading]] (dreamer_diagnosis, 2026-05-18) — v1 H1 root cause was REINFORCE re-sampling at loss-time — re-calling the actor o
- [[20260518_1513_production_recipe_xs_16_4m_hypervigilance]] (hypervigilance, 2026-05-18) — Production recipe for 10×10 hypervigilance with dreamer-srl v2: XS size preset /
- [[20260518_1514_num_envs_vs_budget_interaction]] (hypervigilance, 2026-05-18) — Size-vs-num_envs optimum is regime-dependent on training budget — at short budge
- [[20260518_1515_m_paradox_resolution_slow_learner]] (dreamer_diagnosis, 2026-05-18) — M (640/1024/mlp_layers=3) size cells appeared stuck at low ep_len despite the lo
- [[20260519_1507_dreamer_srl_v2_cpu_buffer_regression]] (dreamer_diagnosis, 2026-05-19) — dreamer-srl v2's replay buffer is 100% numpy/CPU (faithful to sheeprl's PyTorch 
<!-- END BACKLINKS -->
