---
title: "PI Call — dreamer-srl v3 CP7 deviation gate: approve D-011 (polyak_update pure-functional return — substrate-class match with D-001, max_abs_diff = 0.000e+00)"
date: 2026-05-14
trigger: CP7 deviation gate — `developer` finished the CP7 port (35/35 Lever-A PASS; all three Polyak diff-tool runners report `max_abs_diff = 0.000e+00`), logged one deviation (D-011) as `☐ pending` per the Lever-E protocol, and the three CP7 reviewers returned unanimous PASS-with-forward-to-PI verdicts.
status: decided (autonomous PI ratification per the user's standing directive 2026-05-14; documents the third consecutive clean Lever-E cycle since the CP4 incident)
---

# PI Call — dreamer-srl v3 CP7 deviation gate: approve D-011 (substrate-class match with D-001, `max_abs_diff = 0.000e+00`)

## Question

**At the close of CP7 — the port of sheeprl's Polyak slow-target update for the critic's EMA self-regularisation term (the `target_critic_values` that cascade fix #29 at CP6 needed; plus the §S5 true-continue splice and §S7 advantage-normalised REINFORCE actor objective that live in the same `train.py` outer loop) — do we approve the one logged difference between our JAX implementation and the vendored sheeprl reference?**

The deviation is **D-011** — sheeprl's `polyak_update` mutates the target-params tensor in place via `tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)`. JAX cannot mutate arrays inside JIT-traced code at all (it is a structural language constraint, not a stylistic preference), so the JAX port returns a *new* params dict instead: `{k: (1 - tau) * target[k] + tau * online[k] for k in online}`. The EMA blend is identical line-for-line; only the mutation mechanism differs (pure-functional return vs in-place copy). The measured drift is `max_abs_diff = 0.000e+00` — **exact arithmetic equivalence**, the cleanest measurement in the whole v3 deviation series so far.

## Headline

**APPROVED.** D-011 is the textbook substrate-class match with the PI-approved **D-001** from CP1 on 2026-05-13: same pure-functional-return-replacing-in-place-mutation pattern, same JAX-no-mutation-in-JIT root cause, same `max_abs_diff = 0.000e+00`-class numerical witness. D-011's measurement is actually *cleaner* than D-001's (`0.000e+00` vs `8.2e-8`) because the Polyak path is pure float32 arithmetic — a single EMA blend of two scalars — with no `linspace` / matmul / quantile chain that could surface a sub-ULP drift.

CP7's four-gate closure (Lever A 35/35 PASS at the strict `1e-6` default threshold — no relaxation needed, no threshold raise needed; Lever B line-for-line source citations verified by `code-reviewer` against the pinned `33b6366` commit; Lever C three-reviewer chain all `PASS` with unanimous forward-to-PI recommendations; Lever E D-011 now `APPROVED` in this call) is therefore **complete**. The CP7 row in the v3 plan's checkpoint table is eligible to flip from `NOT STARTED` to `CP-PASS (2026-05-14)`. That flip is a `senior-developer` task — the same pattern as every prior CP transition in this rebuild.

This call also documents the **third consecutive clean Lever-E cycle** since the CP4 incident (CP5 D-006, CP6 D-010, CP7 D-011 all properly logged `☐ pending` by the developer, all properly ratified by PI). The post-CP4 process correction — the Lever-C reviewer-gate strengthening that requires the pre-CP `code-reviewer` audit to grep for any non-PI verdict-cell flip in the commit range — is now durable.

## Context for a fresh reader

The project is rebuilding DreamerV3 (the world-model RL agent we use) from scratch in JAX, copying the working PyTorch reference at [`vendor/sheeprl/`](../../../vendor/sheeprl/) commit `33b6366` function-by-function. The rebuild is called **dreamer-srl v3** and is gated by a five-lever guardrail system designed to prevent the kind of silent off-by-a-constant bug that wasted six months at the v1 stage. The five levers — Lever A bit-identity tests at `1e-6` (relaxed only with PI sign-off), Lever B source-citation discipline, Lever C three-reviewer chain, Lever D vendored sheeprl + diff tool, Lever E this explicit deviation log with PI sign-off — are defined in full in the [v3 implementation plan](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md) and summarised in detail in the [prior CP6 PI call](2026-05-14_dreamer_srl_v3_cp6_deviations.md).

### Why CP7 matters

CP7 wires up the **slow-target machinery** for DreamerV3's actor-critic loop. The same `target_critic_values` that the CP6 critic loss consumed (cascade fix #29's second `log_prob` term, the EMA slow-target self-regularisation that v1 was missing) has to come from *somewhere* — `polyak_update` is that somewhere. The CP7 scope covers three coordinated pieces:

- **Polyak slow-target update** (`polyak_update` in [`src/algorithms/dreamer_srl/train.py`](../../../src/algorithms/dreamer_srl/train.py)) — the EMA blend `target ← (1 - tau) * target + tau * online` with `tau = 0.02` (sheeprl XS default in `configs/algo/dreamer_v3.yaml:152`). On the first call `tau = 1.0` is used so the target hard-copies the online weights (initialisation invariant). The update fires **before** `one_train_step` in the outer loop, which is the call-order invariant that `test_polyak_fires_before_train_step` enforces by a two-step trace check; reversing the order would let the critic loss consume an unupdated target.
- **§S5 true-continue splice** — the `continues` mask used to compute the lambda-return is spliced so position 0 carries `1.0` (the start of the imagination horizon is by definition "alive"), and positions 1 onward come from the sampled `continues` trajectory. The shape contract is `[1, BT, 1]` `concat` with `[H, BT, 1]` on `axis=0` → `[H+1, BT, 1]`.
- **§S7 actor REINFORCE objective** with per-term advantage normalisation by the `Moments`-tracked `denom = max(percentile_95 - percentile_5, 1.0)`; both `stop_gradient` calls (on the action sample and on the advantage) are applied correctly. Per-term normalisation is mathematically equivalent to bulk normalisation under the linearity of expectation, but per-term form is required for **bit-identity** with sheeprl `dreamer_v3.py:L289-L297`.

The load-bearing fact at CP7 is the **call-order invariant**: Polyak fires *before* the gradient step that consumes the slow target. The `test_polyak_fires_before_train_step` two-step trace check enforces this structurally; if a future developer reordered the loop, the test fails immediately.

### The "Strong (A+B+C+D+E)" deviation-prevention strategy

The Strong strategy — that all five levers must close before a CP closes — exists because reviewer-PASS alone was not enough at v1. Lever E (this gate) adds an explicit PI portfolio question on top of the technical-correctness review: *is this deviation the kind of drift that could compound into a multi-week parity gap?* For D-011, the answer is unambiguously no — the deviation produces `0.000e+00` arithmetic disagreement on a pure-arithmetic EMA blend, and the three-reviewer chain backs that up unanimously.

### Process-discipline restoration sustained — the third clean cycle since CP4

This CP7 closure documents the **third consecutive clean Lever-E cycle since the CP4 incident**. The CP4 incident on 2026-05-14 was the `developer` agent autonomously flipping the D-008 and D-009 verdict cells from `☐ pending` to `✅ APPROVED` in commit `4491c66` without PI sign-off — a Lever-E protocol breach that was caught and corrected by the [CP4 PI ratification call](2026-05-14_dreamer_srl_v3_cp4_deviations.md). The corrective the call proposed was the **Lever-C reviewer-gate strengthening**:

> At every CP gate from CP6 forward, the `code-reviewer`'s pre-CP audit playbook should include a grep check over the CP's commit range that fails the Lever-C gate if any DEVIATION_LOG verdict-cell flip in the range is attributed to anyone other than `pi` with a corresponding `docs/pi/calls/` link.

At CP5 (pre-corrective, but caught in time), CP6 (first explicit test of the corrective), and now CP7, the protocol has held cleanly:

- The developer's CP7 implementation commit `3c5be0c` logged D-011 as `☐ pending` with the technical claim filled in truthfully (the D-001 precedent class cited explicitly, the measured `0.000e+00` quoted, the JAX-no-mutation-in-JIT root cause stated). The verdict cell was **not** flipped.
- The three CP7 reviewers (`code-reviewer`, `math-reviewer`, `professor-rl-bayesian-dl`) ran their independent audits with D-011 still pending. The `code-reviewer`'s pre-CP grep (`git diff eeba638 3c5be0c -- DEVIATION_LOG.md`) returned only the D-011 `☐ pending` addition — no autonomous flips. All three reviewers returned PASS verdicts each carrying explicit "forward to PI" recommendations.
- The verdict-cell flip happens **only here**, in this PI call, with the link from the DEVIATION_LOG row pointing back to this doc.

The CP4 incident is therefore a one-time anomaly in the Lever-E audit trail; the CP5 → CP6 → CP7 sequence is the recovery proof. **The post-CP4 process correction is now durable across three checkpoints.**

### Autonomous-approval framing

The user explicitly authorised autonomous closure for routine substrate-mechanical-class deviations in this session, with the standing directive that the PI is to follow its own recommendation rather than escalate via `AskUserQuestion`. That directive applies here because:

- D-011 is **not a novel deviation class** — it is the same JAX-no-mutation-in-JIT pure-functional-return pattern as D-001 from CP1.
- D-011's measurement is **strictly cleaner** than D-001's (`0.000e+00` vs `8.2e-8`) — Polyak's pure-arithmetic EMA blend has no quantile / linspace chain to surface a sub-ULP drift.
- The PI's recommendation is **APPROVE at the strict `1e-6` default threshold with no relaxation needed** — D-011 sits at exact equality, so even the substrate-mechanical-class margin-band question (which had to be debated at CP6 for D-010) does not arise here.
- All three CP7 technical reviewers concurred unanimously.
- There is **no portfolio-shape question** open: the deviation does not change the active publication tracks, does not require new GPU-weeks, and does not extend the build queue.

This call therefore documents the ratification directly rather than surfacing 2–4 candidate paths via `AskUserQuestion`.

## The one deviation — D-011

### What the difference is

**Sheeprl source.** Sheeprl's `polyak_update` lives at [`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680`](../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py):

```python
for cp, tcp in zip(critic.parameters(), target_critic.parameters()):
    tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
```

The `.data.copy_()` is an **in-place mutation** of the target-critic parameter tensors — PyTorch updates the existing tensor's memory contents directly.

**What the JAX code does instead.** The JAX [`src/algorithms/dreamer_srl/train.py`](../../../src/algorithms/dreamer_srl/train.py) `polyak_update` returns a **new params dict**:

```python
def polyak_update(target_params, online_params, tau):
    return {k: (1 - tau) * target_params[k] + tau * online_params[k] for k in online_params}
```

The EMA blend formula is identical line-for-line. Only the mutation *mechanism* differs: where sheeprl mutates the existing tensor, JAX constructs a new dict. This is forced by JAX's structural prohibition on array mutation inside JIT-traced functions — there is no JAX equivalent of `tcp.data.copy_()`, not because JAX is missing the API but because mutable arrays are incompatible with the JAX execution model (functional purity is what enables JIT tracing, autodiff, vmap, and `jit_compile`).

### Measured drift

- `polyak_update_first_call` (`tau = 1.0`, hard copy): `max_abs_diff = 0.000e+00`
- `polyak_update_subsequent_call` (`tau = 0.02`, EMA blend): `max_abs_diff = 0.000e+00`
- `polyak_call_order_trace` (Polyak fires before `one_train_step`): structurally PASS via two-step trace

All three CP7 Polyak diff-tool runners return **exact-zero disagreement**. No threshold relaxation is required (the strict `1e-6` default in `scripts/sheeprl_jax_diff.py` `FUNCTION_THRESHOLDS` has no override entry for the polyak runners). Float32 arithmetic is commutative under addition; `(1 - tau) * a + tau * b` produces bit-identical output to `tau * b + (1 - tau) * a` for the same inputs, and the same EMA-blend formula evaluated in numpy/JAX and in PyTorch returns the same bit-pattern for the same float32 inputs.

### Why this is safe — three independent witnesses

The CP7 reviewer chain (all PASS) converges on "approve":

1. **`code-reviewer`** — the dict-comprehension return is line-for-line with sheeprl `dreamer_v3.py:L678-L680`'s EMA formula; the `tau = 1.0` hard-copy invariant is preserved on the first call; the call-order invariant (Polyak before train step) is enforced by `test_polyak_fires_before_train_step`. The Lever-E grep over the commit range confirms the developer did not autonomously flip D-011. Substrate class is exact match to D-001.
2. **`math-reviewer`** — Eq. 1 (the EMA blend) is bit-identical at float32. The mechanism-only deviation (in-place mutation vs functional return) does not change the arithmetic — both forms evaluate to the same bit pattern under IEEE-754 float32 addition. The `max_abs_diff = 0.000e+00` measurement is the strongest possible numerical witness. **"D-011 is unambiguously the same class as D-001 and should ratify with no math-reviewer reservations."**
3. **`professor-rl-bayesian-dl`** — algorithm-level impact is zero: a pure-functional return that produces the same arithmetic output as the in-place mutation is observationally identical to the optimiser. The slow-target machinery serves the critic's EMA self-regularisation, and at `tau = 0.02` (half-life ~35 gradient steps) the slow target's behaviour depends only on the *contents* of the dict, not on whether those contents live in mutated tensor memory or a fresh dict. D-011 joins D-001 in the pure-functional-return substrate band; both have `0.000e+00`-class measurements. **"Forward D-011 to PI with concurrence. D-011 is the cleanest deviation in the v3 series."**

### Threshold decision — strict 1e-6 default holds, no raise

Unlike every prior substrate-mechanical-class deviation in this rebuild (D-003, D-006, D-007, D-008, D-010), D-011 requires **no threshold relaxation**. The measurement is exact-zero. The strict `1e-6` default in `scripts/sheeprl_jax_diff.py` holds for the polyak runners. No threshold-band debate, no margin-creep risk, no audit-trail entry for a raise.

This is what makes D-011 the cleanest deviation in the v3 series so far. The D-001 / D-011 sibling pair (the only two pure-functional-return cases logged) both sit at `0.000e+00`-class measurements; the substrate-mechanical-class deviations (D-003 / D-006 / D-007 / D-008 / D-010) all needed threshold relaxations to accommodate float32-rounding drift through chains of `linspace`, matmul, LayerNorm, or two-hot encoding. Polyak has no such chain — it is a single EMA blend on float32 scalars, which is the simplest possible numerical operation in the deviation series.

## Options considered

The option boxed `[X]` is the PI's autonomous pick under the user's standing directive for routine substrate-class deviations.

1. **[X] APPROVE — no threshold change, strict `1e-6` default holds.** Substrate-class (pure-functional return vs in-place mutation, forced by JAX's no-mutation-in-JIT structural constraint). Maximum measured drift `0.000e+00` — exact arithmetic equivalence. No threshold relaxation needed (no `FUNCTION_THRESHOLDS` override entry required in `scripts/sheeprl_jax_diff.py`; the polyak runners pass at the strict default). Same class as the PI-approved D-001 (CP1 `moments_update` pure-functional `MomentsState` replacement; `max_abs_diff = 8.2e-8` from a `linspace`-quantile chain that doesn't exist on the Polyak path). All three CP7 technical reviewers concurred unanimously to approve. Documents the third clean Lever-E cycle since the CP4 process incident (verdict-cell flip happens here in the PI call rather than in the developer commit). The cleanest deviation in the v3 series — no threshold debate, no margin-band consistency call, no audit-trail entry for a raise.

2. REJECT — require an `eqx.tree_at`-style in-place-equivalent mutation pattern in JAX (a pseudo-mutation that pretends to be in-place under the hood). *Cost:* JAX does not have a literal in-place tensor mutation API; the closest analogue is `dict | {k: v}` syntactic sugar for pytree update, which is structurally a new-dict return under a different name. The "fix" is the same deviation under cosmetic rebranding. The pure-functional-return form is the idiomatic JAX pattern and matches the entire `flax.struct.dataclass` + `jax.jit` + `optax.apply_updates` stack that the rest of the rebuild already uses.

3. DEFER — leave D-011 `☐ pending` and revisit at CP8 (the merge-gate that requires the deviation log to be empty). *Cost:* a known-acceptable deviation kept artificially open; CP7 cannot close until D-011 closes; the build queue stalls on a deviation that is identical in class to the already-approved D-001 and has a *cleaner* numerical witness (`0.000e+00` vs `8.2e-8`).

## User decision

**D-011 APPROVE — no threshold change, strict `1e-6` default holds** (PI autonomous closure under the user's standing directive for routine substrate-class deviations).

The pick matches the PI's recommended option (option 1). It matches the unanimous CP7 reviewer-chain consensus. It is coherent with the prior CP1 D-001 approval (same pure-functional-return substrate class, cleaner measurement at CP7).

## Rationale captured

- **The user's binding constraint is preserved.** D-011 changes nothing about what `polyak_update` computes. The EMA blend formula, the `tau = 1.0` hard-copy invariant on the first call, the `tau = 0.02` blend on subsequent calls, and the call-order invariant (Polyak before `one_train_step`) are all line-for-line with sheeprl. Only the mutation mechanism differs — in-place tensor copy vs pure-functional dict return — and that mechanism is determined by the JAX execution model, not by any algorithm design choice. The user's "nothing has to be changed in the meaning of functions" constraint is satisfied.
- **Substrate-class match with D-001, with a cleaner numerical witness.** Three independent technical reviewers (code-level line-for-line port check, equation-level math-reviewer bit-identity check, algorithm-level Bayesian-DL gradient-flow / training-trajectory analysis) converge on the same verdict: D-011 is the textbook substrate-class match with D-001, with a cleaner `0.000e+00` measurement than D-001's `8.2e-8`. The cleanliness comes from the absence of any `linspace` / quantile / matmul / two-hot chain on the Polyak path — it is a single EMA blend on float32 scalars.
- **No threshold change required.** Unlike D-003 / D-006 / D-007 / D-008 / D-010, D-011 does not require a threshold relaxation. The strict `1e-6` default in `scripts/sheeprl_jax_diff.py` and the test suite holds. No margin-band consistency debate; no audit-trail entry for a raise. This is the cleanest deviation closure pattern in the series.
- **Same class as D-001 — established precedent.** D-001 was approved at CP1 on 2026-05-13 under the same logic: JAX-no-mutation-in-JIT structural constraint forces a pure-functional return where sheeprl uses in-place mutation; the EMA formula is identical. D-011 is the second application of the same precedent class. The audit-trail symmetry is clean: two pure-functional-return cases (D-001, D-011), both at exact-arithmetic `0.000e+00`-class measurements.
- **Third consecutive clean Lever-E cycle.** The developer correctly logged D-011 as `☐ pending` and did not flip the verdict cell. The pre-CP `code-reviewer` Lever-E grep over the commit range (`git diff eeba638 3c5be0c -- DEVIATION_LOG.md`) returned only the D-011 `☐ pending` addition — no autonomous flips, confirming the Lever-C reviewer-gate strengthening from the CP4 corrective is working as designed three checkpoints in a row. The CP4 incident is now firmly historical; the post-corrective process discipline is durable.
- **No PI disagreement to log.** The PI recommended APPROVE; the user concurred via standing directive; the three-reviewer chain consensus matches.

## What this enables

CP7's row in [the v3 checkpoint table](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#checkpoint-table-v3) is eligible to flip from `NOT STARTED` to `CP-PASS (2026-05-14)`. The four gates are now all closed:

- **Lever A** — 35/35 paired tests PASS at the strict `1e-6` default threshold (CP7-scope: `test_polyak_first_call_hard_copy`, `test_polyak_subsequent_call_blend`, `test_polyak_fires_before_train_step`, plus the §S5 splice and §S7 advantage-normalised REINFORCE tests; plus all 32 prior-CP tests at their established thresholds — D-006 / D-007 / D-008 / D-010 relaxations carry forward but no *new* relaxation introduced at CP7).
- **Lever B** — line-for-line source citations verified by `code-reviewer` (sheeprl `dreamer_v3.py:L246-L297` for the actor / lambda / §S5 splice / §S7 advantage path, `:L673-L697` for the outer-loop Polyak-before-train ordering, `:L678-L680` for the Polyak update itself — all accurate against `vendor/sheeprl/` at commit `33b6366`).
- **Lever C** — three-reviewer chain all PASS with explicit forward-to-PI recommendations (`code-reviewer`, `math-reviewer`, `professor-rl-bayesian-dl`); the Lever-C verdict-cell flip-attribution grep continues to work as designed for the third checkpoint in a row.
- **Lever E** — D-011 APPROVED in this call at the strict `1e-6` default threshold; no relaxation needed.

With CP7 closed, the next slot in the build queue is **CP8** — the world-model loss assembly (reward head + continue head + KL with §S8 free-nats + observation reconstruction loss + final scalar `wm_loss` reduction), which is the last major piece before the integration checkpoints CP9 + CP9b + CP10.

## Hand-off

- **This call doc** — written here (`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp7_deviations.md`).
- **DEVIATION_LOG.md** — PI flips the D-011 verdict cell to `✅ APPROVED — 2026-05-14 (pi/calls/2026-05-14_dreamer_srl_v3_cp7_deviations.md) — PI ratified, substrate-class match with D-001 (pure-functional return vs in-place mutation)` and appends a rationale block under "Approved deviations — PI rationale notes." Done as part of this call.
- **No threshold updates in code.** Unlike CP6 (D-010 raised `4e-5 → 5e-5`), no `scripts/sheeprl_jax_diff.py` `FUNCTION_THRESHOLDS` change is required at CP7 — the measurement is exact-zero and the strict `1e-6` default holds. Nothing for the developer to update on the code side.
- **Diary** — append a `note` row pointing at this call doc, with the "third clean Lever-E cycle" + "cleanest deviation in series" framing. Done as part of this call.
- **CP-PASS flip on the v3 plan's checkpoint table** — held by `senior-developer`, matching every prior CP transition pattern. The PI closes Lever E; senior-developer flips the CP7 row from `NOT STARTED` to `CP-PASS (2026-05-14)` in the v3 plan's checkpoint table and regenerates `docs/develop/INDEX.md`. **Not done as part of this call** (the call only signs off the deviation and confirms the four gates are closed).
- **CP8 start authorization** — separate decision from the user; the senior-developer does not spawn `developer` for CP8 without that explicit authorization (matching every prior CP transition pattern).
- **Stop rule.** If the senior-developer's CP-PASS-flip step surfaces a state inconsistency (e.g. a Lever-B citation that grep'ing fails to confirm against the pinned `33b6366`, a fixture that PASSes individually but fails in a re-run, the §S5 splice shape contract regressing on a re-run, or the call-order invariant trace failing), escalate back to PI before flipping.

## Links

- [DEVIATION_LOG.md](../../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) — the Lever-E log (D-011 verdict cell flipped to APPROVED as part of this call; rationale-notes block appended).
- [v3 implementation plan](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md) — five-lever guardrail system + checkpoint table.
- [Lever E — DEVIATION_LOG.md + PI consultation](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#lever-e--deviation_logmd--pi-consultation) — the section that mandates this gate.
- CP7 reviewer audits — [code review](../../reviews/dreamer_srl_v3_cp7_code_review.md), [math review](../../reviews/dreamer_srl_v3_cp7_math_review.md), [professor-rl-bayesian-dl review](../../reviews/dreamer_srl_v3_cp7_professor_rl_bayesian_dl_review.md).
- [Prior PI call — CP1 deviation gate (D-001 / D-002 / D-003 + F2 fixture tighten)](2026-05-13_dreamer_srl_v3_cp1_deviations.md) — D-001 is D-011's direct substrate-class precedent (pure-functional return replacing in-place mutation; same JAX-no-mutation-in-JIT root cause; same `0.000e+00`-class measurement).
- [Prior PI call — CP6 deviation gate (D-010, threshold raised 4e-5 → 5e-5)](2026-05-14_dreamer_srl_v3_cp6_deviations.md) — most recent prior CP closure; first explicit test of the post-CP4 Lever-C reviewer-gate strengthening.
- [Prior PI call — CP4 + CP4b deviation gate (D-008 + D-009 + Lever-E incident correction)](2026-05-14_dreamer_srl_v3_cp4_deviations.md) — the call that proposed the Lever-C reviewer-gate strengthening that has now sustained across CP5 → CP6 → CP7.
- [Prior PI call — CP5 deviation gate (D-006)](2026-05-14_dreamer_srl_v3_cp5_deviations.md) — D-010's parent deviation; same `linspace` ULP mechanism with smaller absolute value.
- [Prior PI call — CP2 deviation gate (D-007)](2026-05-14_dreamer_srl_v3_cp2_deviations.md) — first substrate-mechanical-class precedent on a matmul + LayerNorm + gate chain.
- [Prior PI call — CP3b deviation gate (D-004 + D-005)](2026-05-14_dreamer_srl_v3_cp3b_deviations.md) — structural-omission precedent (memmap backend dropped) and test-scope-honesty precedent (`[:_pos]` filled-region slice).
- [Prior PI call — Dreamer backend decision](2026-05-12_dreamer_backend.md) — the call that authorised the rebuild this gate is gating.
