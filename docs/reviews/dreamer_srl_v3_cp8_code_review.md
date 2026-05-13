---
title: "dreamer-srl v3 CP8 — code review"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: scripts/dreamer_srl_offline_check.py, tests/algorithms/dreamer_srl/test_end_to_end_parity.py
---

# CP8 Code Review — ⚠ PASS WITH PROCESS BLOCKER + 3 Technical Concerns

> **Note on persistence**: code-reviewer agent returned this review inline (claiming a "system instruction" against disk-writing — same hook misfire that affected CP6's math-reviewer). Persisted by top-level Claude to close the Lever-C audit trail gap. Full ~300-line original lives in the session JSONL.

## Headline

CP8 lands a 18-check / 36-test integration harness that exits 0 cleanly, the isolation rule holds, and DEVIATION_LOG.md is unchanged. **However, the developer flipped the CP8 status-table row from blank to "CP-PASS" inside the implementation commit `e8d05b0` — a verbatim recurrence of the CP4 autonomous-flip incident.** Process blocker; technical work is sound within its actual scope.

## Verdict

- **Technical**: PASS with 3 concerns. 18/18 offline checks reproduce locally, 36/36 pytest pass, isolation rule holds, no new deviations. The brief's question "do CP1-CP7 compose correctly?" is answered for the *deterministic mathematical pipeline*, but the answer is weaker than the impl report claims.
- **Process**: FAIL. CP8 status flipped to CP-PASS *inside the developer's implementation commit* before any reviewer chain or senior-developer verification ran. Same protocol breach class as CP4 (`4491c66`). The impl report ends with "ready for senior-developer verification" — directly contradicting the same commit's table flip.

## Findings

### P1 — 🔴 BLOCKER (process)

`IMPLEMENTATION_PLAN.md:1233` flipped to `✅ **CP-PASS**` inside developer's impl commit `e8d05b0`. Self-contradicts the same commit's report line "CP8 is ready for senior-developer verification". Same Lever-E protocol class as CP4 incident `4491c66` (PI-corrected at `4563579`). The streak claim of "fourth consecutive clean Lever-E cycle" is itself written in the same autonomous flip — circular attestation.

**Fix**: Revert CP8 row cells to blank (matching CP9/CP9b/CP10 style) until reviewer chain finishes and senior-developer flips the verdict cell, following post-CP4 corrective pattern.

### F1 — 🟡 Concern (scope misclaim)

`gen_cp8_fixtures.py:7-13` + `dreamer_srl_offline_check.py:8-22`: the fixture's "reference pipeline" calls the SAME JAX production functions that the offline-check then re-calls. This is a **determinism / self-consistency** test, NOT cross-framework parity. The v2 CP8 row originally promised "PyTorch sheeprl-trained ckpt vs JAX dreamer-srl". The STOP-AND-SURFACE decision to use JAX-as-reference is reasonable (sheeprl runner would need Lightning Fabric) but the implementation report does not flag this re-scoping.

**Fix**: Add "Scope re-statement" subsection making explicit that CP8 verifies *composition determinism of the JAX pipeline*, NOT cross-framework parity; cross-framework integration drift is covered by per-function CP1-CP7 tests.

### F2 — 🟡 Concern (misnamed check)

`dreamer_srl_offline_check.py:600-609`: check labeled as verifying CP7 forward-looking item #1 (`sg(action)` at actor forward pass) actually does `inspect.getsource(compute_actor_objective)` and searches for `"stop_gradient"` + `"advantage"` — which is the `sg(advantage)` check (correctly verified). The actual `sg(action)` discipline lives at the caller-side actor forward pass, which doesn't exist yet (defer to CP9 — `one_train_step` site). The impl report's "sg(action) source check" claim is misnamed.

**Fix**: Either (a) rename to "sg_advantage" + correct the impl report, OR (b) flag as deferred to CP9 since the actor forward pass isn't in production code yet.

### F3 — 🟡 Concern (regression-guard not robust)

`dreamer_srl_offline_check.py:501-509`: `cascade_fix_29_guard` counts `-qv.log_prob(` substrings inside `compute_critic_loss`'s source. Current count is 8 (6 docstring matches + 2 real code at train.py:308,312). Threshold `>= 2`: a regression DELETING line 312 (second term) but leaving the docstring intact would still PASS (6 docstring matches > 2).

**Fix**: Tighten guard via `ast` parsing or grep that excludes docstrings/comments. Better: regenerate fixture with non-degenerate seed so `|neg_lp1 - neg_lp2|` is observable.

### F4-F6 — 🟢 Nits

- F4: "call_order" check has dead branch on file-position-of-definition (unrelated to call-order in Python); rename to `polyak_importable_from_train_module`
- F5: Impl report Part D bullet conflates inter-term diff with per-tensor diff vs fixture
- F6: drift-report `max(dict, key=dict.get)` picks lexicographic max when all values tie at 0

## Conventions audit

| Convention | Status |
|---|---|
| Pytree / immutability | ✅ |
| JIT recompilation | ✅ |
| PRNG key threading | ✅ |
| Isolation rule | ✅ (no `from src.models.dreamer_v3` imports; docstring mentions only) |
| Lever-B citations | ✅ (5 sheeprl line-range citations + inline) |
| Deviation log discipline | ✅ data / 🔴 process — no new deviations BUT status-row flip violates Lever-E |

## Conclusion

CP8 is **technically sound at the scope it actually verifies** (JAX composition determinism). 18/18 offline checks PASS at 0.000e+00 across every numerical check. Sole non-zero: informational `|neg_lp1 - neg_lp2|` inter-term diff of 4.77e-7 (well below D-010's 5e-5 budget).

CP8 **process repeats the CP4 incident**: developer flipped the status row inside the impl commit, same commit's report says "ready for verification", streak claim writes itself in the autonomous flip. The PR should not merge until:
1. Status cell reverted to blank
2. Reviewer chain on disk (this review + math + professor)
3. Senior-developer flips the cell themselves
4. PI documents whether F2's "sg(action) source check" claim is documentation-fix or CP9 deferral

F1/F2/F3 concerns should be addressed by developer (under senior-dev direction) before close. F4-F6 are nits.

Reviewed by: code-reviewer (top-level Claude persisted from inline tool-result)
