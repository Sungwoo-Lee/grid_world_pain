---
id: 20260723_1910_sheeprl_parity_drift_lives_in_glue
date: 2026-07-23
time: "19:10"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, refutation, meta]
summary: "A from-scratch Fable-5 re-audit of both Dreamers vs sheeprl found the math cores bit-identical, but 11 undeclared training-impact deviations — all in the GLUE (optimizer wrapping, loss assembly, buffer write paths, config constants), not the equations. dreamer_srl had no gradient clipping (WM loss spiked ~1e30), 2x under-weighted recon loss, a gamma typo in 18 configs, and multi-env replay hole-rows."
related: ["20260510_2240_reference_impl_compare_only_act_intersections", "20260518_1511_dreamer_srl_v2_parity_pass_outperform", "20260723_1909_nnx_abandoned_archived_stack_confusion", "20260723_1912_fresh_empirical_reaudit_finds_new_bug_class"]
session_origin: claude_code
session_label: "Fable 5 re-diagnosis + parity fixes + NNX archival + live-path inspection"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/07436809-982c-4ed5-b1fa-e84c030f5fd6.jsonl
raw_completeness: full
---

# sheeprl parity drift lives in the glue, not the math

## Key conclusion

A from-scratch 5-reviewer parity re-audit of BOTH Dreamer stacks against vendored sheeprl (~170 items, paired torch/JAX bit-identity probes) found the **numerical cores at exact parity** — lambda-returns, the REINFORCE actor loss, the Moments return-normalizer, KL balance, and two-hot machinery were bit-identical or within float32 noise. Yet it surfaced **11 undeclared deviations with real training impact**, and every one lived in the *glue* around the math: optimizer wrapping, loss assembly, buffer write paths, and config constants — exactly the places line-by-line porting attention lapses. The practical rule: when auditing a port whose equations already passed review, spend the effort on assembly/plumbing/config, not on re-deriving the losses. A second corollary: `num_envs=1` masked two of the worst deviations (multi-env buffer corruption, iteration-counted prefill), so a "parity-passed" history run at single-env does NOT certify multi-env correctness.

## Evidence, measurements, facts

- **dreamer_srl (the live port) deviations fixed** (commits `8c0fcf9` + `867ec51`, one comparability epoch): (a) NO gradient clipping on any of the 3 optimizers vs sheeprl 1000/100/100 — WM loss empirically spiked to ~1e29-1e31 in live smokes; (b) reconstruction loss under-weighted EXACTLY 2x (extra 0.5 factor, probe measured 0.500000) plus decoder trained in real space (extra symlog on predictions) — and the faithful `reconstruction_loss` already existed in `loss.py`, imported but never called; (c) gamma typo `0.996840347` vs sheeprl `0.996996996996997` in all 18 gamma-carrying configs (~5% shorter horizon, shipped with a FALSE citation comment); (d) shared replay write-head punched garbage "hole rows" into non-done envs' columns at every reset (multi-env only, `[21,31,0,41]` probe); (e) episode survival-metric bleed (+1 step + predecessor's terminal reward); (f) learning_starts counted in iterations not env steps (16x prefill at 16 envs).
- Bit-identity probes that PASSED: lambda-returns diff 0.0, full REINFORCE actor loss diff 0.0, Moments chain 0.0, KL 1.4e-6, reward two-hot 7.2e-6, continue 1.2e-7.
- The DreamerV3-NNX half of the audit's fixes is now moot (that stack was archived — [[20260723_1909_nnx_abandoned_archived_stack_confusion]]).
- Process: the fresh Fable-5 model found these largely by (a) empirically executing suspect paths and (b) diffing the port against its upstream recipe line by line — read-only review of the same code had passed it.

## Decisions and actions

- Fixed all live-stack (dreamer_srl) deviations in one comparability epoch; declared kept deviations in DEVIATION_LOG (D-014 refresh, D-015, D-016).
- Standing caveat recorded: post-fix dreamer_srl / rPPO / (archived) NNX runs are NOT comparable to pre-fix runs — the old runs carried the removed biases. Relaunch from the fix commits as the new baseline.
- Registry cluster "Fixed — sheeprl-parity cluster (2026-07-08)" (`5e66df2`).

## Open questions and follow-ups

None blocking. The two DreamerV3-NNX open rows (decoder LayerNorm U6, sigmoid-vs-mode continues R1) are archived-stack / moot-unless-revived.

## References

- Master comparison: `docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/00_master_comparison.md` (+ 5 area reports).
- Related process lessons: [[20260510_2240_reference_impl_compare_only_act_intersections]], [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]], [[20260723_1912_fresh_empirical_reaudit_finds_new_bug_class]].
- Commits: `91be2bc` (audit), `8c0fcf9`+`867ec51` (srl fixes), `5e66df2` (registry).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 07436809-982c-4ed5-b1fa-e84c030f5fd6`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260723_1909_nnx_abandoned_archived_stack_confusion]] (dreamer_diagnosis, 2026-07-23) — DreamerV3-NNX is the abandoned stack (2026-05-12 pivot); dreamer_srl is live but
- [[20260723_1912_fresh_empirical_reaudit_finds_new_bug_class]] (dreamer_diagnosis, 2026-07-23) — A second independent audit pass with a fresh model generation (Fable 5) that EMP
<!-- END BACKLINKS -->
