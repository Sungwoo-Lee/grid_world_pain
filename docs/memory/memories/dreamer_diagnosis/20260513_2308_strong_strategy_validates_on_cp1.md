---
id: 20260513_2308_strong_strategy_validates_on_cp1
date: 2026-05-13
time: "23:08"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, meta, design]
summary: "The Strong (A+B+C+D) deviation-prevention strategy paid off on the first checkpoint (CP1, utils.py port) of the dreamer-srl v3 rebuild — caught a pre-CP0 gitignore blocker that would have broken Lever D + Lever A on a fresh clone, surfaced a latent wrong-reference bug in the D-002 distribution test when we tightened its threshold, and let the PI cleanly batch-approve all 3 deviations (D-001/D-002/D-003) under the user's 'nothing has to be changed in the meaning of functions' criterion. ~50% dev-time overhead vs naive impl, but the previous plan-only review process let the twohot encoding bug through — the strategy closes that loop."
related: ["20260510_2240_reference_impl_compare_only_act_intersections", "20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt", "20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned"]
session_origin: claude_code
session_label: "dreamer-srl v3 CP1 closure — Strong (A+B+C+D) strategy validation"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# Strong (A+B+C+D) deviation-prevention strategy paid off on CP1 of dreamer-srl v3

## Key conclusion

The four-lever Strong strategy the user chose for the dreamer-srl v3 rebuild — **A** per-function bit-identity tests + **B** source-citation discipline + **C** per-checkpoint 3-reviewer chain + **D** vendored sheeprl + diff tool, plus **E** PI on every deviation — caught three concrete classes of silent failure on CP1, none of which the previous plan-only review process would have surfaced. The historical cascade-debugging arc (twohot encoding bug, GRU reset, paper-canonical bins all slipping past static plan review) was the prior evidence; CP1 is now the positive proof that per-checkpoint reviewer gates against the code (not just the plan) closes that loop. ~50% dev-time overhead vs a naive port, paid in exchange for confidence that what merges has been verified at function granularity.

## Evidence, measurements, facts

- **CP1 scope**: `src/algorithms/dreamer_srl/utils.py`, 8 leaf functions (`symlog`, `symexp`, `init_weights`, `uniform_init_weights`, `compute_lambda_values`, `Moments` → pure-functional `moments_init` + `moments_update`, `Ratio`, `prepare_obs`). Vendored sheeprl source at `vendor/sheeprl/sheeprl/utils/distribution.py` + `vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py` (pinned to commit `33b6366`).
- **Three concrete catches during CP1**:
  1. **Pre-CP0 gitignore blocker (Lever-C code-reviewer at infrastructure layer)**: `.gitignore` line 15 had bare `sheeprl/` which silently shadowed the `!vendor/sheeprl/` negation on the next line. 232 files under `vendor/sheeprl/sheeprl/` (the inner Python package) existed on disk but were NOT tracked in git. A fresh clone would have broken Lever D (`scripts/sheeprl_jax_diff.py` couldn't import the reference) and Lever A (`tests/algorithms/dreamer_srl/*` couldn't import `vendor.sheeprl.sheeprl.<...>`). Fix: change `sheeprl/` → `/sheeprl/` (root-anchored). Commit `fc84a93`.
  2. **F2 latent test-bug surfaced via threshold tighten**: the D-002 `init_weights` distribution-property test was comparing `sampled_std` against `std_theoretical` (the *inflated input* passed to `nn.init.trunc_normal_`, equal to `sqrt(scale) / HAFNER_CONST`), when the correct reference is `std_theoretical * HAFNER_CONST = sqrt(scale)` (the actual *expected std after truncation* at ±2σ). The old test's persistent 12% gap was the truncation factor itself, NOT statistical error. The historical 0.8796 Hafner-truncation bug (the v2 plan's §Risks-5 case that motivated Lever B) would have produced a ~0.34% std shift — comfortably inside the loose 15% bound, so the test as written would NOT have caught the bug it was supposed to catch. Tightening forced the reference correction; D-002 now passes at 0.055% rel_err (well inside the new <1% bound). Commit `77382f2`.
  3. **PI clean batch-approval of 3 deviations**: code, math, and professor-rl-bayesian-dl all classified the 3 deviations cleanly — D-001 "JAX-mechanical: pure-functional refactor of stateful `Moments` + `fabric.all_gather` dropped on single-process" (max_abs_diff 8.2e-8); D-002 "cross-PRNG fundamental: same target distribution, different sampler streams"; D-003 "hardware float32 1-ULP rounding at exp(5)≈148.4". The user's "nothing has to be changed in the meaning of functions" framing + the technical reviewers' classifications let the PI batch-process all 3 in one call (`docs/pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md`, commit `f653260`). No silent rationalizations; deviation log persisted with PI verdicts.
- **Cost**: estimated ~50% dev-time overhead vs naive port at CP1. The senior-developer's v3 plan estimate was 1.5× (6 weeks vs 4 weeks naive). Drivers: per-function PyTorch-trace fixture generation (the largest new cost), and 8 × 3-reviewer gates × ~30 min each.
- **What the prior process would have missed**: the previous dreamer-srl-v2 plan was reviewed by the same three reviewers (code, math, professor-rl-bayesian-dl) and passed ✅ on all 29 deviations, but it was a *plan* review — there was no per-function bit-identity test, no diff tool, no per-checkpoint code-against-code audit. The twohot encoding bug + GRU reset gate fix + paper-canonical bins all slipped through because static plan review doesn't predict integration-layer execution. The Strong strategy specifically fires its reviewers *at the code* at each checkpoint.

## Decisions and actions

- **Decision (kept)**: continue the Strong (A+B+C+D+E) strategy for all 8 CPs of the dreamer-srl v3 rebuild. The CP1 evidence — three real catches at the infrastructure / test-design / deviation layers — justifies the ~50% overhead.
- **Decision (carried forward)**: PI-gate every deviation, no exceptions. The user's "nothing has to be changed in the meaning of functions" criterion is the binding constraint; the PI applies it at portfolio level after the technical reviewers classify each deviation's substrate-mechanical / cross-platform-fundamental / hardware-rounding nature.
- **Action (next)**: CP5 is the next checkpoint (slot #2 in user-reordered build order: CP1 → CP5 → CP2 → CP3 → CP4 → CP4b → CP6 → CP7 → CP8). CP5 is the historical-scar checkpoint (`loss.py` with `TwoHotEncoding` — the actual twohot encoding bug class). Expect heavier 3-reviewer scrutiny. Two things to over-document: (1) Hafner full-precision constants — never truncate/round/clean up cited values; (2) symlog-space bin discipline — `bins = linspace(-20, 20, 255)` lives in symlog space, `symexp` applied only at consumption via `transbwd`, `target` symlog-encoded before bin lookup. Triple-consistency check (cascade table + Checkpoint 5 spec + loss.py file-change row) per math-reviewer's spot-check protocol.

## Open questions and follow-ups

- **Lever-C reviewer-file persistence gap**: the CP1 code-reviewer agent's audit findings drove the F1+F3+F2 fixes, but their write to `docs/reviews/dreamer_srl_v3_cp1_code_review.md` did NOT persist to disk. Math + professor audits both persisted cleanly. Pattern: have the orchestrator (top-level Claude) check `git status` for the expected review file after each reviewer agent returns; if missing, route the agent again to re-write. Documented in the v3 plan's CP1 Verification subsection as a known gap; not blocking CP-PASS, but a Lever-C process improvement for CP5 onward.
- **The F4 + F5 deferred-nits from code-reviewer**: F4 (D-002 runner print-format opacity — synthetic `0.0 PASS` hides the actual rel-err) and F5 (`compute_lambda_values` test covers only one T,B shape) are still deferred. They are quality-of-life, not correctness gates. Pick up before CP-PASS scrutiny tightens at CP5 / CP6.

## References

- v3 plan (active, CP1 flipped to `CP-PASS`): [`docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md`](../../../docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
- v2 plan (archived; algorithmic backbone): [`docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md`](../../../docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md)
- PI call: [`docs/pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md`](../../../docs/pi/calls/2026-05-13_dreamer_srl_v3_cp1_deviations.md)
- Deviation log: [`docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md`](../../../docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md) (D-001/D-002/D-003 all `APPROVED`)
- Reviewer audits (persisted): `docs/reviews/dreamer_srl_v3_cp1_math_review.md`, `docs/reviews/dreamer_srl_v3_cp1_professor_rl_bayesian_dl_review.md`, `docs/reviews/dreamer_srl_v3_pre_cp0_code_review.md`
- Related insight (the pivot that opened the rebuild question): [[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]]
- Related insight (the cascade-debugging arc this closes the loop on): [[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]]
- Related insight (reference-impl-comparator discipline that informs Lever B): [[20260510_2240_reference_impl_compare_only_act_intersections]]
- Key CP1 commits: `fdcbfa5` (initial port), `46a18cb` (F1+F3 fixes), `f653260` (PI deviation call doc), `77382f2` (F2 tighten + wrong-reference fix), `ba362e3` (math+professor reviews), `82c3699` (CP-PASS plan-edit)
- Pre-CP0 commit chain: `292dd3a` (setup) → `fc84a93` (gitignore root-anchor + 232 files tracked) → `747e8c5` (diff tool nits) → `325c9da` (CP-id taxonomy)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]] (dreamer_diagnosis, 2026-05-18) — dreamer-srl v2 PASS-OUTPERFORMs sheeprl on food-only parity (501 vs ~500) after 
- [[20260518_1512_reinforce_resampling_bug_imag_action_threading]] (dreamer_diagnosis, 2026-05-18) — v1 H1 root cause was REINFORCE re-sampling at loss-time — re-calling the actor o
<!-- END BACKLINKS -->
