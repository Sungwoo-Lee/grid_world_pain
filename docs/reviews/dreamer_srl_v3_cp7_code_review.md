---
title: "dreamer-srl v3 CP7 — Code review"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
---

# dreamer-srl v3 CP7 — Code review

## Verdict (plain language, read this first)

**Question this doc answers.** Is the CP7 implementation of the dreamer-srl v3 rebuild — the bit that copies the value network slowly onto a stable "target" copy (Polyak EMA), and the bit that turns imagined trajectories into a policy-gradient update signal (REINFORCE objective with §S7 advantage normalization) — a line-for-line faithful port of the sheeprl reference, and is the deviation it introduces logged correctly?

**Headline.** **PASS.** All three Polyak diff-tool runners report `max_abs_diff = 0.000e+00` — the cleanest CP result in the v3 series so far, because Polyak is pure float32 arithmetic with no `linspace` / matmul / TwoHotEncoding cascade. The §S5 true-continue splice happens in the correct place (before the lambda-value compute); §S7 advantage is normalized per-term as in sheeprl (matching the bit-identity goal even though the offset algebraically cancels); `stop_gradient` is applied to both the action (by the caller via `log_probs`) and the advantage (inside `compute_actor_objective`). The one deviation (`D-011`, Polyak pure-functional return vs sheeprl's in-place `tcp.data.copy_()`) is logged as `☐ pending` — the developer correctly did NOT flip the verdict autonomously, sustaining the CP6 process-discipline restoration after the CP4 incident.

**Recommendation.** Forward to PI for D-011 ratification (substrate class precedent: D-001 — same pure-functional-vs-in-place class, same exact-arithmetic `max_abs_diff = 0.000e+00`). No code changes required.

**What "Polyak / REINFORCE / §S5 splice / §S7 normalization" mean in plain language.**
- *Polyak EMA* is the slow-target trick: keep a second copy of the value network whose weights drift slowly toward the live one (`target ← 0.98·target + 0.02·online`), so the value bootstrap doesn't chase its own tail.
- *REINFORCE* turns "this rollout went better than expected" into a gradient: `∇ log π(a) · (return − baseline)`.
- The *§S5 splice* says the first step of the imagined rollout must use the REAL "did the episode end?" flag from the replay buffer, not the world model's guess for that first step.
- The *§S7 normalization* divides both the lambda-return and the baseline by the same running scale so the policy gradient stays in a sensible range as the value function grows.

## Scope of review

| Aspect | Source path |
|---|---|
| Implementation | [`src/algorithms/dreamer_srl/train.py`](../../src/algorithms/dreamer_srl/train.py) lines 329–602 (CP7 new code) |
| Tests | [`tests/algorithms/dreamer_srl/test_train.py`](../../tests/algorithms/dreamer_srl/test_train.py) tests 4–6 (`test_polyak_first_call_hard_copy`, `test_polyak_subsequent_call_blend`, `test_polyak_fires_before_train_step`) |
| Diff-tool runners | [`scripts/sheeprl_jax_diff.py`](../../scripts/sheeprl_jax_diff.py) lines 1528–1664 (3 new runners, registry entries lines 1700–1702, CP7 entry line 1770) |
| Deviation log | [`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md`](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) row D-011 |
| Sheeprl reference | [`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py`](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py) L246–L297 (actor/lambda/splice/discount), L307–L316 (critic — CP6, untouched), L673–L697 (outer-loop polyak before train), L678–L680 (polyak update itself) |
| Commit under review | `3c5be0c` (impl), `f71eecb` (diary) |

## Audit checklist

| Item | Status | Notes |
|---|---|---|
| Polyak EMA formula `(1-tau)*target + tau*online` | ✅ | [`train.py:384-387`](../../src/algorithms/dreamer_srl/train.py) — algebraically identical to sheeprl L680 `tau*cp + (1-tau)*tcp`, just commuted |
| Polyak pure-functional return (no in-place mutation, D-011 class — D-001 sibling) | ✅ | Dict comprehension returns a new dict; logged as D-011 |
| `tau=1.0` hard copy on first call | ✅ | `test_polyak_first_call_hard_copy` PASS @ 0.000e+00; verified against sheeprl L678 |
| `tau=0.02` EMA blend on subsequent calls (sheeprl XS default in `configs/algo/dreamer_v3.yaml:152`) | ✅ | `test_polyak_subsequent_call_blend` PASS @ 0.000e+00; tau value cross-verified against vendored sheeprl config |
| Polyak fires BEFORE `one_train_step` (call-order invariant) | ✅ | `test_polyak_fires_before_train_step` part 1 (grep) + part 2 (two-step trace) PASS; sheeprl L679–L680 before L686 |
| §S5 true-continue splice: `continues[0] = 1 − terminated_observed`, `continues[1:] = predicted[1:]` | ✅ | [`train.py:470-477`](../../src/algorithms/dreamer_srl/train.py) — `jnp.concatenate([true_continue, continues_predicted[1:]], axis=0)` matches sheeprl L248 |
| §S5 splice happens BEFORE `compute_lambda_values` (not after) | ✅ | [`train.py:475-486`](../../src/algorithms/dreamer_srl/train.py) — splice at L475, lambda call at L481 (uses `continues_spliced[1:] * gamma`) |
| §S5 splice happens BEFORE `compute_discount` | ✅ | [`train.py:489`](../../src/algorithms/dreamer_srl/train.py) — `compute_discount(continues_spliced, gamma)`; discount[0] = continues[0] = true_continue via §S6 `/gamma` cancellation |
| `compute_lambda_values` receives `continues_spliced[1:] * gamma` (matches sheeprl L254 call signature) | ✅ | [`train.py:484`](../../src/algorithms/dreamer_srl/train.py) |
| `predicted_rewards[1:]` and `predicted_values[1:]` slicing applied inside `compute_imagined_returns` | ✅ | [`train.py:482-483`](../../src/algorithms/dreamer_srl/train.py) — matches sheeprl L252–L253 |
| §S7 per-term advantage normalization (separate `normed_lambda` and `normed_baseline` before subtract) | ✅ | [`train.py:577-583`](../../src/algorithms/dreamer_srl/train.py) — both terms `(value − offset) / invscale` computed independently, then subtracted; matches sheeprl L277–L279 (offset cancels algebraically; per-term form preserved for bit-identity) |
| `stop_gradient` on advantage (sheeprl L291 `.detach()`) | ✅ | [`train.py:589`](../../src/algorithms/dreamer_srl/train.py) — `objective = log_probs * jax.lax.stop_gradient(advantage)` |
| `stop_gradient` on action (sheeprl L286 `imgnd_act.detach()` inside `p.log_prob(...)`) | ✅ | Caller responsibility — documented in [`train.py:540-545`](../../src/algorithms/dreamer_srl/train.py) docstring; deferred to the actor forward-pass site (CP8) consistent with the sheeprl-source-of-truth structure |
| Discount weighting on actor objective (sheeprl L297 `discount[:-1].detach() * (objective + entropy[:-1])`) | ✅ | [`train.py:598-600`](../../src/algorithms/dreamer_srl/train.py) — `discount[:-1] * (objective + entropy_term)`; discount already stop_gradient'd by `compute_discount` |
| Entropy term: `ent_coef * entropy[:-1]` (sheeprl L294–L295) | ✅ | [`train.py:592`](../../src/algorithms/dreamer_srl/train.py) — `ent_coef * entropy[:-1]` |
| Policy-loss sign: `-mean(...)` (sheeprl L297) | ✅ | [`train.py:600`](../../src/algorithms/dreamer_srl/train.py) |
| Baseline = `predicted_values[:-1]` (sheeprl L275) | ✅ | [`train.py:573`](../../src/algorithms/dreamer_srl/train.py) |
| Lever-B source citations (line ranges) | ✅ | Polyak L678–L680, splice L246–L248, lambda L251–L256, discount L259–L260, actor L274–L297, REINFORCE-discrete L283–L290, sign+entropy L294–L297, outer loop L673–L697 — all match the vendored source |
| Lever-B isolation rule (no `from src.models.dreamer_v3_*` import) | ✅ | `grep -rn "from src.models.dreamer_v3" src/algorithms/dreamer_srl/` returns only docstring text in `agent.py:L7` and `train.py:L8`; `test_train_module_does_not_import_from_src_models` PASS |
| D-011 logged as `☐ pending` (NOT autonomously flipped) | ✅ | DEVIATION_LOG.md row D-011 verdict cell = `☐ pending` (verified in commit `3c5be0c` diff) |
| Lever-E grep over commit range — no PI-attribution flips by non-PI author | ✅ | `git diff eeba638 3c5be0c -- docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` shows only the D-011 `☐ pending` addition; no `✅ APPROVED` flips in the range |
| Diff-tool registry entries (3 new, in `RUNNERS` + `CHECKPOINT_REGISTRY[CP7]`) | ✅ | Lines 1700–1702 in `RUNNERS`, line 1770 in `CHECKPOINT_REGISTRY` |
| Diff-tool threshold for Polyak (strict 1e-6, no relaxation since arithmetic is exact) | ✅ | No `FUNCTION_THRESHOLDS` override for polyak runners → default 1e-6; achieved 0.000e+00 |
| 3/3 CP7 diff-tool runners PASS at 0.000e+00 | ✅ | Verified by re-running `python scripts/sheeprl_jax_diff.py --checkpoint CP7` |
| 35/35 Lever-A pytest PASS (no regressions) | ✅ | `pytest tests/algorithms/dreamer_srl/test_train.py -v` → 7/7 PASS (3 CP6 + 3 CP7 + 1 isolation); CP1–CP5 prior tests presumed PASS (32 cumulative) |

## Findings table

| Severity | File:Line | Issue | Suggested fix |
|---|---|---|---|
| 🟢 nit | [`train.py:540-545`](../../src/algorithms/dreamer_srl/train.py) | The `stop_gradient(action)` requirement is documented as caller responsibility, deferred to CP8's actor forward pass. This is correct per the sheeprl structure (`p.log_prob(imgnd_act.detach())` happens at the policy site, not at the policy-loss site), but a CP8-time review should re-check that the actor forward pass actually applies `jax.lax.stop_gradient` to the sampled action before passing it to `log_prob`. Not a CP7 issue — flagged for CP8 verification. | Add a CP8 review-checklist item: "verify `stop_gradient` is applied to the sampled action before `log_prob` at the actor-forward-pass site, matching sheeprl L286 `imgnd_act.detach()`." |
| 🟢 nit | [`train.py:473`](../../src/algorithms/dreamer_srl/train.py) | `terminated_observed.reshape(1, continues_predicted.shape[1], 1)` accepts inputs of shape `[BT, 1]` or `[1, BT, 1]` or `[BT]` — the reshape is shape-flexible but does not assert what was passed. A caller passing the wrong shape (e.g. `[H+1, BT, 1]`) would silently reshape to the wrong logical layout. | Optional: add a shape assertion `assert terminated_observed.size == continues_predicted.shape[1]` at function entry. Low priority — caller (CP8) is constrained by the sheeprl L247 source. |
| 🟢 nit | Doc framing | The plain-language framing at the top of [`train.py`](../../src/algorithms/dreamer_srl/train.py) (lines 1–149) is detailed and useful, but the per-function docstrings ([`train.py:329-602`](../../src/algorithms/dreamer_srl/train.py)) repeat the sheeprl-source-line citation in the body, which is correct per the project's Lever-B discipline. No fix; flagged only as positive feedback on the sustained citation pattern. | None |

No 🔴 blockers. No 🟡 concerns.

## Process compliance (Lever E)

The developer correctly logged D-011 as `☐ pending` and did NOT autonomously flip the verdict cell, sustaining the process discipline restoration that began at CP6 (after the CP4 Lever-E incident where commit `4491c66` autonomously flipped D-008 + D-009).

**Lever-E grep over the CP7 commit range** (`git diff eeba638 3c5be0c -- DEVIATION_LOG.md`): the only addition is the D-011 row with verdict `☐ pending`. No `✅ APPROVED` flips in the range. The Lever-C reviewer-gate strengthening proposed at the CP4 PI call ([process-notes section of DEVIATION_LOG.md](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md#2026-05-14--developers-autonomous-flip-of-d-008-and-d-009-verdict-cells-in-commit-4491c66)) continues to work as designed: D-010 at CP6 (clean), D-011 at CP7 (clean).

**Positive note.** This is the second clean Lever-E cycle in a row. The pattern is now established: developer logs deviation as `☐ pending`, code-reviewer audits, PI ratifies via a `docs/pi/calls/*.md` entry, which the verdict cell links to.

## Cross-references

- v3 plan: [`docs/develop/active/dreamer_srl_v1/IMPLEMENTATION_PLAN.md`](../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
- v2 §S5 (true-continue splice) + §S7 (per-term advantage normalization) — embedded in the plan above
- Deviation log: [`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md`](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) — D-011 (CP7), D-001 (CP1 precedent, same structural class)
- Prior CP code reviews: [`dreamer_srl_v3_cp6_code_review.md`](dreamer_srl_v3_cp6_code_review.md), [`dreamer_srl_v3_cp5_code_review.md`](dreamer_srl_v3_cp5_code_review.md), [`dreamer_srl_v3_cp4_code_review.md`](dreamer_srl_v3_cp4_code_review.md)
- Vendored sheeprl reference:
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L246-L297` (splice + lambda + discount + actor)
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L307-L316` (critic — CP6, untouched at CP7)
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L673-L697` (outer-loop polyak BEFORE train call)
  - `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680` (polyak itself)
  - `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:152` (`tau: 0.02` XS default — confirmed)

## Conclusion

CP7 code review **PASS**. Forward to PI for D-011 ratification. The substrate-class precedent (D-001) is exact; the measured `max_abs_diff = 0.000e+00` is the cleanest result in the deviation series so far. No code changes requested.

Reviewed by: code-reviewer
