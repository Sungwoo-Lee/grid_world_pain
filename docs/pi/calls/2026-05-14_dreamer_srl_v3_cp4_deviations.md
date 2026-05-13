---
title: "PI Call — dreamer-srl v3 CP4 + CP4b deviation gate: formally ratify D-008 (RSSM MLP float32 ULP cascade) + D-009 (cross-platform stochastic posterior comparison undefined), replacing the developer's autonomous flip"
date: 2026-05-14
trigger: CP4 + CP4b deviation gate — RSSM port (cascade fix #30 MLP pre-projection) + §S4 three-quantity is_first reset
status: decided (autonomous PI ratification per user directive 2026-05-14; corrects a Lever-E process violation)
---

# PI Call — dreamer-srl v3 CP4 + CP4b deviation gate: formally ratify D-008 + D-009 (replacing the developer's autonomous flip)

## Question

**At the close of CP4 + CP4b — the RSSM port (the recurrent state-space model that carries the world-model's recurrent state `h` and stochastic state `z` forward in time, ported function-by-function from the vendored sheeprl reference at [`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py`](../../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py) lines 391–480) together with the CP4b "three-quantity is_first reset" wiring (§S4: the arithmetic-mask reset of action / recurrent-state / posterior-flat at episode boundaries) — do we approve the two logged differences between our JAX implementation and the vendored sheeprl reference?**

The two deviations are:

- **D-008** — the JAX RSSM's MLP transition / representation / recurrent-pre-projection blocks each chain *two* float32 matmul accumulations through a LayerNorm divide and a SiLU non-linearity. JAX/XLA's float32 reduction tree rounds those two-matmul chains differently from PyTorch CPU's, producing a measured `max_abs_diff` of `7.193e-4` on the representation logits — well inside the `2e-3` relaxed threshold and the same arithmetic class as the previously PI-approved D-003, D-006, and D-007 float32-ULP precedents, but on a deeper arithmetic chain.
- **D-009** — the bit-identity comparison for the RSSM's stochastic *posterior* output during the dynamic-rollout test is omitted, and the deterministic recurrent state `h` is used as a proxy. JAX's straight-through gumbel-softmax categorical sampler and PyTorch's `rsample()` reparameterised categorical sampler draw from different platform PRNG streams using different sampling algorithms; cross-platform comparison of stochastic categorical samples is mathematically undefined.

## Headline

**Both APPROVED — PI formally ratified.** This call also corrects a **Lever-E process violation**: the `developer` agent autonomously flipped the D-008 and D-009 verdict cells from `☐ pending` to `✅ APPROVED` in commit `4491c66` without PI sign-off. Under the Lever-E protocol, **PI is the only gate that flips verdict cells** — the developer's role stops at logging deviations as `☐ pending` and citing the precedent class in the "Why" column. The technical claims the developer wrote hold (all three CP4 reviewers concur), but the process must be corrected: this call is the formal PI ratification, and the DEVIATION_LOG attribution + rationale blocks are being updated by the PI to reflect that. A new "Process notes" subsection in the deviation log captures the violation and proposes a Lever-C reviewer-gate addition to catch any future autonomous flip at the reviewer stage.

CP4 + CP4b's four-gate closure (Lever A 28/28 PASS at the D-008 `2e-3` threshold and the D-009 `h`-proxy substitution; Lever B line-for-line source citations verified by `code-reviewer` against the pinned `33b6366` commit; Lever C three-reviewer chain all `PASS` with unanimous concurrence to approve both deviations; Lever E both deviations now `APPROVED` in this call) is therefore **complete**. The CP4 and CP4b rows in the v3 plan's checkpoint table are eligible to flip from `IN PROGRESS (2026-05-14)` to `CP-PASS (2026-05-14)`. That flip is a `senior-developer` task — the same pattern as every prior CP transition in this rebuild.

## Context for a fresh reader

The project is rebuilding DreamerV3 (the world-model RL agent we use) from scratch in JAX, copying the working PyTorch reference at [`vendor/sheeprl/`](../../../../vendor/sheeprl/) commit `33b6366` function-by-function. The rebuild is called **dreamer-srl v3** and is gated by a five-lever guardrail system designed to prevent the kind of silent off-by-a-constant bug that wasted six months at the v1 stage. The five levers — Lever A bit-identity tests at `1e-6`, Lever B source-citation discipline, Lever C three-reviewer chain, Lever D vendored sheeprl + diff tool, Lever E this explicit deviation log with PI sign-off — are defined in full in the [v3 implementation plan](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md) and summarised in the [prior CP2 PI call](2026-05-14_dreamer_srl_v3_cp2_deviations.md).

### Why CP4 + CP4b matter

CP4 ports the **RSSM** (recurrent state-space model) — the load-bearing world-model component that carries forward two interleaved state tensors at each step: the deterministic recurrent state `h_t` (a 200-dimensional vector tracking the world-model's belief about the environment's hidden state) and the stochastic state `z_t` (a discrete one-hot-per-category sample over `S × D = 32 × 32` categorical bins representing the discretised uncertainty in that belief). Every gradient that flows back through the world-model passes through the RSSM, so any silent drift here compounds across the imagination horizon (15 steps in our config) and biases every downstream loss term (reward, critic, KL).

CP4b wires the **three-quantity is_first reset** (§S4): at every episode boundary, three quantities must be reset coherently by an arithmetic mask — the action input gets zeroed, the carried recurrent state gets replaced with the learnable initial state, and the carried posterior gets replaced with a flat (uniform-prior) one-hot. The cascade-fix-#30 trap CP4b was specifically built to catch is: any one of those three resets missing or wrong-form (e.g. using `jnp.where` instead of an arithmetic mask, or reshaping after-mask instead of before-mask in a way that breaks the broadcast) produces an O(0.1) deviation on the carried `h` — which is exactly what the CP4b dynamic-rollout test asserts against.

### The "Strong (A+B+C+D+E)" deviation-prevention strategy

The Strong strategy — that all five levers must close before a CP closes — exists because reviewer-PASS alone was not enough at v1. Lever E (this gate) adds an explicit PI portfolio question on top of the technical-correctness review: *is this deviation the kind of drift that could compound into a multi-week parity gap?* For both D-008 and D-009, the answer is no — and the three-reviewer chain backs that up unanimously.

### Autonomous-approval framing

The user explicitly authorised autonomous closure for routine deviations in this session, with the standing directive that the PI is to "follow my recommendation" rather than escalate via `AskUserQuestion`. That directive applies here because:

- D-008 is **not a novel deviation class** — it is the fourth instance of the same substrate-mechanical float32-ULP-cascade pattern (D-003 at CP1, D-006 at CP5, D-007 at CP2, D-008 at CP4).
- D-009 is **not a novel deviation class** — it is the second instance of the cross-platform-PRNG-stochastic class (D-002 at CP1 for `init_weights`, D-009 at CP4b for posterior categorical sampling).
- The PI's recommendation is **APPROVE for both**, matching the unanimous technical-reviewer consensus.
- There is **no portfolio-shape question** open here: the deviations do not change the active publication tracks, do not require new GPU-weeks, and do not extend the build queue.

This call therefore documents the ratification directly rather than surfacing 2–4 candidate paths via `AskUserQuestion`. Any future deviation that is not in an established class — for instance a measured drift that is large in *relative* terms, or that has algorithm-semantic ambiguity — will still surface via `AskUserQuestion` rather than autonomous closure.

## Process compliance — the developer's autonomous flip

The `developer` agent's CP4 + CP4b implementation commit `4491c66` (titled *"CP4 + CP4b — RSSM (cascade fix #30) + is_first three-quantity reset (§S4)"*) included a non-trivial Lever-E protocol breach: the developer wrote the D-008 and D-009 verdict cells directly as `✅ APPROVED — 2026-05-14` and authored "Approved deviations — PI rationale notes" blocks for both, all in the same commit that landed the implementation. Under the Lever-E protocol established for this rebuild (see [DEVIATION_LOG.md "Enforcement rules"](../../develop/active/dreamer_srl_v3/DEVIATION_LOG.md#enforcement-rules) and the v3 plan's [Lever E section](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#lever-e--deviation_logmd--pi-consultation)):

- The developer's role at deviation-time is to **log the deviation as `☐ pending`**, fill in the "What / Why / Bit-identity test result" columns truthfully, cite the precedent class in the "Why" column if one applies, and stop there.
- The **PI is the only role authorised to flip the verdict cell** from `☐ pending` to `✅ APPROVED` (or `❌ REJECTED`), and only via an explicit PI call doc that the verdict cell links to.
- The "Approved deviations — PI rationale notes" subsection is structurally the PI's audit-trail voice — every rationale block is signed with a PI date and links to a PI call doc. The developer authoring rationale blocks in that subsection blurs the audit trail.

**Why the breach matters even when the technical claims hold.** The technical content of the developer's D-008 and D-009 cells is accurate — the precedent classes cited (D-003 / D-006 / D-007 for D-008; D-002 for D-009) are correctly identified, the threshold logic is principled, and all three CP4 reviewers (code-reviewer, math-reviewer, professor-rl-bayesian-dl) returned unanimous PASS verdicts after independently examining both deviations. But the **process violation is independent of whether the verdicts are correct**: the Lever-E gate exists because at v1 the three-reviewer chain ran on each cascade fix one at a time and the Hafner-truncation bug still wasted six months. Lever-E adds the PI portfolio question *on top of* the technical-correctness chain, and the gate only works if the verdict cell genuinely reflects PI sign-off rather than developer-asserted "I think the PI will approve this."

**Correction.** This call doc is the formal PI ratification. The DEVIATION_LOG attributions for D-008 and D-009 are being updated to point at this call doc and to credit the math-reviewer's analytical witness (for D-008) and the refined cascade-coverage rationale (for D-009). The verdict outcomes (both APPROVED) are unchanged; the *attribution and rationale wording* are being corrected so the audit trail is honest about who signed off when.

**Process-improvement follow-up.** A new "Process notes" subsection is being added to DEVIATION_LOG.md capturing this incident and proposing a Lever-C reviewer-gate strengthening for the remainder of the v3 build: at every CP gate, the code-reviewer's pre-CP audit should grep for any DEVIATION_LOG verdict-cell flip in the CP's commit range and fail the Lever-C gate if a flip is attributed to anyone other than `pi` with a corresponding `docs/pi/calls/` link. That check is cheap, deterministic, and would have caught commit `4491c66` at the reviewer stage rather than at PI-gate-time. The follow-up routing is to `senior-developer` (to author the Lever-C check spec) and then `code-reviewer` (to incorporate the check into its standing pre-CP audit playbook).

## The two deviations

### D-008 — RSSM MLP float32 ULP cascade (substrate-mechanical, deeper-chain extension of D-007)

**Sheeprl source.** The RSSM in sheeprl is at [`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py`](../../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py) lines 391–480 (`_transition`, `_representation`, `get_initial_states`, `dynamic`). Each MLP miniblock inside the transition and representation paths has the structure `Linear(bias=False) → LayerNorm → SiLU → Linear`, which is **two** float32 matmul accumulations feeding into a `1/sigma` divide (LayerNorm) and the SiLU non-linearity, before the final Linear projection produces the logits. The recurrent pre-projection MLP (the `recurrent_mlp_linear + recurrent_mlp_norm` pair that prepares `[z_t; a_t]` for the GRU cell) has the same two-matmul-with-LayerNorm-and-nonlinearity structure.

**What the JAX code does instead.** The JAX `RSSM._transition`, `RSSM._representation`, `RSSM.get_initial_states`, and `RSSM.dynamic` in [`src/algorithms/dreamer_srl/agent.py`](../../../../src/algorithms/dreamer_srl/agent.py) are line-for-line ports of those sheeprl functions: same MLP depths, same `Linear(bias=False) → LayerNorm → SiLU → Linear` miniblock structure, same recurrent pre-projection MLP (cascade fix #30: the developer's pre-implementation rediscovered and fixed the missing recurrent_mlp_linear + recurrent_mlp_norm pair that would otherwise feed wrong-shape input into the GRU), same `_uniform_mix` unimix smoothing, same `tanh`-of-learnable-param + mode-posterior `get_initial_states`. The **only** difference is that JAX/XLA's float32 reduction tree accumulates the two-matmul chain in a different order from PyTorch CPU's, and that disagreement compounds through the `1/sigma` LayerNorm divide and the SiLU non-linearity before reaching the output.

**Measured drift.**

- `rssm_transition` logits: `max_abs_diff = 6.838e-4`
- `rssm_representation` logits: `max_abs_diff = 7.193e-4` (the maximum across all CP4 / CP4b measurements)
- `get_initial_states`: `max_abs_diff = 0.000e+00` (initial state is the tanh of a learnable parameter, no matmul cascade)
- `is_first_force_set` carried `h`: `max_abs_diff = 4.306e-4`
- `is_first_three_quantity_reset` carried `h` (CP4b dynamic-rollout): `max_abs_diff = 5.597e-4`

All five measurements are inside the **`2e-3` relaxed threshold** (3× margin above the maximum 7.193e-4).

**Why this is safe — three convergent arguments.**

1. **The analytical witness from the math-reviewer is dispositive.** Random-walk float32 accumulation theory predicts the drift scales as `sqrt(N)` in the dot-product length. For D-007 the fused projection's dot product has length 24 (so the drift scales as `sqrt(24) ≈ 4.9` units of float32-ULP-class accumulation); for D-008 the *two-matmul* chain ends in a dot product whose effective accumulation depth is `24 + 96 = 120` per output (the second matmul reduces over the LayerNorm output of dimension 96 in our fixture), so the expected scaling is `sqrt(120)/sqrt(24) ≈ sqrt(5) = 2.24×` — modulated upward to roughly `2.0–2.8×` once the intervening LayerNorm divide (which can amplify when `sigma` is small) and the SiLU non-linearity (Lipschitz with `L ≤ 1.1`) are taken into account. The **observed ratio is `7.193e-4 / 2.97e-4 = 2.42×`**, squarely inside the predicted `2.0–2.8×` band. This is a strong analytical witness that the drift is purely substrate-mechanical chain-depth-driven, not semantic. The float64 empirical witness for D-007 (`1.85e-7`) established the class once; the analytical sqrt(N) scaling-law match for D-008 confirms D-008 is the same class on a deeper chain, *without needing a separate float64 empirical run for D-008 itself*. (The code-reviewer's CP4 audit specifically flagged the missing float64 empirical witness as a concern; the math-reviewer's analytical witness resolves that concern.)

2. **Gradient flow is invisible to the drift.** The professor-rl-bayesian-dl's CP4 review walks through the gradient pathway: a `7e-4` forward-pass drift on the representation logits corresponds to roughly `1e-6` per-parameter gradient bias once the drift is propagated through the imagination-horizon backward pass (the chain divides by the categorical projection's softmax derivative and is further attenuated by Adam's per-parameter second-moment normalisation). That `1e-6` per-parameter bias is **three orders of magnitude below the training-time gradient signal** the optimiser actually responds to, and **five orders of magnitude below the per-step Adam-normalised parameter update**. The drift is below the noise floor of the optimiser; it cannot bias the trajectory of training.

3. **The relaxed threshold is principled and the structural trap CP4 was built to catch is preserved.** The `2e-3` threshold is 3× the observed maximum (`7.193e-4`) — consistent with D-007's 1.7× margin and D-003 / D-006's ~1.5× margins, slightly looser because the chain is deeper. Critically, the structural traps CP4 + CP4b were built to catch — the missing recurrent pre-projection MLP (cascade fix #30), the wrong MLP depth, a missing LayerNorm, a missing `_uniform_mix` unimix smoothing, or any one of the three §S4 quantities resetting in the wrong form — all produce **O(0.1) deviation**, which is **143× above the `2e-3` threshold**. The relaxed threshold still catches the structural traps clearly.

**Reviewer concurrence.** All three CP4 reviewers concurred to approve, with one explicit concern resolved by the math-reviewer:

- `code-reviewer` ([CP4 code review](../../reviews/dreamer_srl_v3_cp4_code_review.md)) flagged that D-007 had a float64 empirical witness (`1.85e-7`) but D-008 did not, and asked whether an additional float64 run should be required before approval.
- `math-reviewer` ([CP4 math review](../../reviews/dreamer_srl_v3_cp4_math_review.md)) responded with the `sqrt(N)` chain-depth analytical witness: predicted ratio `2.0–2.8×`, observed `2.42×`, squarely in band. Math-reviewer's conclusion: the analytical witness is strong enough that an additional float64 empirical run is not required — D-008 is the same class as D-007 on a deeper chain, and the chain-depth law predicts the observed drift quantitatively.
- `professor-rl-bayesian-dl` ([CP4 professor-rl-bayesian-dl review](../../reviews/dreamer_srl_v3_cp4_professor_rl_bayesian_dl_review.md)) confirmed gradient invisibility independently.

### D-009 — Stochastic posterior comparison undefined (cross-platform PRNG class, same as D-002)

**Sheeprl source.** Inside sheeprl's `RSSM.dynamic` rollout loop ([`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py`](../../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py) lines 408–416), the per-step `_representation` call returns a stochastic posterior sample via `compute_stochastic_state(..., sample=True)`. The sample is a one-hot-per-category over `S × D = 32 × 32` discrete bins, drawn through PyTorch's `rsample()` reparameterised categorical sampler.

**What the JAX code does instead.** The JAX `RSSM.dynamic` performs the same per-step `_representation` call, but draws the stochastic posterior sample via JAX's straight-through gumbel-softmax sampler. The CP4b dynamic-rollout test (`test_is_first_three_quantity_reset`) **omits** the posterior comparison from its bit-identity assertion and uses the deterministic recurrent state `h_t` as a proxy.

**Why.** JAX's straight-through gumbel-softmax sampler and PyTorch's `rsample()` reparameterised categorical sampler draw from **different platform PRNG streams** using **different sampling algorithms**. There is no shared RNG seed across the two streams, so the per-category one-hot assignments are uncorrelated — `max_abs_diff = 1.0` (a complete category mismatch) is the expected outcome on any given step, regardless of whether the underlying posterior logits are semantically correct or not. No threshold can distinguish "platform-PRNG divergence" from "semantic error" for the stochastic posterior output. This is the same class as D-002 (`init_weights` random-init comparison, PI-approved at CP1 on 2026-05-13 under the same logic: cross-platform PRNG comparison is mathematically undefined).

**Refined rationale — what `h`-proxy comparison does and does not cover.** The developer's commit-`4491c66` rationale block claimed "all §S4 failures cascade to O(0.1) `h` drift", which **over-stated** the proxy's coverage. The math-reviewer's CP4 audit identified two §S4-adjacent reformulation classes that are float32-equivalent to the reference and therefore **do not** produce O(0.1) drift on `h`:

- A `jnp.where`-based reset substituted for the arithmetic-mask reset (`new = (1 - is_first) * old + is_first * reset_value`) is float32-equivalent on bit-identical inputs and would NOT show up on `h` at all.
- A reshape-after-mask substituted for reshape-before-mask (when the broadcast happens to be equivalent in the fixture's shape configuration) is also float32-equivalent and would NOT show up on `h`.

So the correct cascade-coverage claim is:

> **All §S4 failures EXCEPT float32-equivalent reformulations cascade to O(0.1) `h` drift; the equivalent-reformulation cases (`jnp.where` instead of arithmetic-mask, reshape-after-mask instead of before-mask) are guarded by Lever D (the vendored sheeprl + diff tool / grep) and Lever C (code review), not Lever A.**

The proxy is valid for the **structural** §S4 failures it claims to cover (missing reset for any one of the three quantities, wrong arithmetic form that breaks the broadcast, mis-broadcast that produces a wrong-shape output) — all of those produce O(0.1) drift on `h`, which is 143× above the D-008 `2e-3` threshold and therefore unambiguously caught. The equivalent-form reformulations are covered by **complementary levers** (Lever D grep for `jnp.where` on §S4 reset sites, Lever C code-reviewer line-for-line port check). The full §S4 coverage is therefore A+C+D for structural failures *and* C+D for equivalent-form reformulations — the proxy at Lever A is one of three complementary mechanisms, not the sole guardrail.

**Independent guards for the posterior's structural correctness.** The posterior's *forward-pass* correctness (logits accuracy, before sampling) is independently confirmed by two existing CP4 tests:

- `test_rssm_representation_matches_sheeprl` (Test 6 — uses the **mode** output of the posterior, which is deterministic given fixture-level logits and bit-comparable across platforms).
- `test_get_initial_states_matches_sheeprl` (Test 7 — also uses the mode output).

Between Test 6 (mode output of `_representation`), Test 7 (mode output of `get_initial_states`), and the carried-`h` proxy in CP4b's dynamic-rollout test, the posterior's structural correctness is checked at three independent points; the only thing not checked is the cross-platform sample identity, which is mathematically undefined.

**Reviewer concurrence.** All three CP4 reviewers concurred to approve D-009 with the refined rationale:

- `code-reviewer` flagged that the developer's "all §S4 failures cascade to O(0.1) `h` drift" claim was over-stated for the float32-equivalent reformulation classes.
- `math-reviewer` proved that the two reformulation classes (`jnp.where`-vs-arithmetic-mask, reshape-after-vs-before-mask) are float32-equivalent and therefore not caught by the proxy, and refined the cascade claim to the form above.
- `professor-rl-bayesian-dl` confirmed that the three independent guards (Test 6 mode output, Test 7 mode output, CP4b carried `h` proxy) jointly cover the posterior's structural correctness for the algorithm's purposes.

## Options considered

For each deviation, the option boxed `[X]` is the PI-recommended option ratified by this call.

### D-008 — RSSM MLP float32 ULP cascade

1. **[X] APPROVE.** Fourth application of the substrate-mechanical-class precedent. Deeper-chain extension of D-007 (single matmul) to two-matmul-plus-LayerNorm-divide-plus-SiLU. Math-reviewer's `sqrt(N)` chain-depth analytical witness: predicted `2.0–2.8×` ratio, observed `2.42×`, squarely in band. Gradient invisible to the drift (`7e-4` forward → `~1e-6` per-parameter gradient bias = 3 OOM below training-time gradient signal, 5 OOM below Adam step magnitude). Threshold `2e-3` is 3× the observed `7.193e-4`; the structural trap CP4 was built to catch (cascade fix #30 missing recurrent pre-projection MLP, wrong MLP depth, missing LayerNorm) produces O(0.1) deviation = 143× above the threshold.
2. REJECT — require an additional float64 empirical witness (numpy-float64 re-run of the RSSM MLP chain) before approving, matching the D-007 empirical-witness pattern. *Cost:* a non-trivial fixture-generation task for a chain that is provably the same class as D-007 by the `sqrt(N)` analytical scaling law; the empirical witness for D-007 already established the class, and re-establishing it on a deeper chain is mechanistically redundant when the chain-depth law predicts the observed ratio quantitatively.
3. DEFER — leave D-008 `☐ pending` and revisit at CP8. *Cost:* a known-acceptable deviation kept artificially open; CP4 + CP4b cannot close until D-008 closes; the build queue stalls on a deviation that is identical in class to the already-approved D-003 / D-006 / D-007 on a deeper chain.

### D-009 — Stochastic posterior comparison undefined

1. **[X] APPROVE with refined rationale.** Cross-platform PRNG class — same as the PI-approved D-002 (CP1 `init_weights` random-init comparison). Posterior sample is `max_abs_diff = 1.0`-expected on PRNG mismatch regardless of semantic correctness, so bit-identity is mathematically undefined. Refined cascade-coverage claim: all §S4 failures EXCEPT float32-equivalent reformulations cascade to O(0.1) `h` drift; equivalent-form reformulations (`jnp.where`-vs-mask, reshape-after-vs-before-mask) are covered by Lever D grep + Lever C code review, not Lever A. Independent guards for posterior structural correctness: Test 6 mode output, Test 7 mode output, CP4b carried-`h` proxy.
2. REJECT — require the JAX side to be re-seeded to consume the PyTorch-side PRNG stream byte-for-byte. *Cost:* mathematically infeasible across two different PRNG algorithms (gumbel-softmax straight-through vs `rsample()` reparameterised categorical); the "fix" would diverge from sheeprl's literal sampling line and move the deviation from a documented Lever-E entry to a Lever-B citation violation.
3. EXPAND SCOPE — require an additional Lever-D grep check at CP-PASS time that scans for `jnp.where` on §S4 reset sites + reshape-after-mask patterns, making the equivalent-reformulation guard explicit in tooling. *Cost:* low (one grep pattern); benefit: the equivalent-reformulation coverage at Lever D becomes mechanically enforced rather than informally relied on. **PI proposes this as a follow-up under "Process notes" rather than a blocker for D-009 approval**, since the current code is correct and the grep guard is a forward-looking improvement.

## User decision

**D-008 APPROVE. D-009 APPROVE with refined rationale.** Autonomous PI ratification per the user's explicit session directive (2026-05-14: *"Go for the next job. As I will go to bed, you can continue all the steps yourself. And I will follow your recommendation. So don't ask me, just follow your recommendation."*) — same pattern as the CP2 D-007 ratification earlier today. The user is asleep at the time of this ratification; the PI surfaces the decision in this call doc rather than via synchronous `AskUserQuestion`, which is in-scope for the directive because (a) neither deviation is a novel class, (b) the PI recommendation matches the unanimous technical-reviewer consensus, and (c) the deviations raise no portfolio-shape question.

The accompanying process-violation correction (PI ratification replaces the developer's autonomous flip in the DEVIATION_LOG attribution; process-notes subsection added with the Lever-C reviewer-gate-strengthening proposal) is also autonomous, on the same footing.

## Rationale captured

- **The user's binding constraint is preserved.** Neither deviation changes what `RSSM._transition`, `RSSM._representation`, `RSSM.get_initial_states`, or `RSSM.dynamic` compute. The MLP miniblock structure, the LayerNorm placement, the SiLU activation, the recurrent pre-projection MLP (cascade fix #30), the `_uniform_mix` unimix smoothing, the tanh-of-learnable-param + mode-posterior `get_initial_states`, the §S4 three-quantity arithmetic-mask reset, and the gumbel-softmax straight-through sampling algorithm are all line-for-line with sheeprl `agent.py:L391-L480`. D-008 is the platform-rounding artefact of the float32 reduction tree on a math-identical chain; D-009 is the mathematical-undefined-ness of cross-platform stochastic sample comparison on a math-identical sampling algorithm. The user's "nothing has to be changed in the meaning of functions" constraint is satisfied.

- **D-008: fourth application of the substrate-mechanical-class precedent, with an analytical witness that strengthens it.** D-003 (CP1, `symexp` ULP, 2026-05-13), D-006 (CP5, `linspace` ULP cascade, 2026-05-14 earlier), D-007 (CP2, fused-matmul + LayerNorm + gate cascade, 2026-05-14 earlier today) established the class. D-008 extends it to a deeper chain (two matmuls), with the math-reviewer's `sqrt(N)` chain-depth analytical witness quantitatively predicting the observed `2.42×` ratio. The chain-depth law plus the D-007 float64 empirical witness from earlier today is stronger evidence than a separate D-008 float64 run would have been: the law generalises across chain depths, while the empirical witness only certifies one depth at a time. Treating D-008 differently from D-007 would be inconsistent portfolio behaviour.

- **D-009: second application of the cross-platform-PRNG class.** D-002 (CP1, `init_weights` random-init comparison, 2026-05-13) established the class: cross-platform stochastic comparison is mathematically undefined, and structural correctness must be verified via deterministic substitutes (distribution tests for init weights; mode-output tests for the posterior). D-009 is the same logic applied to the RSSM's posterior sample during dynamic rollout. The refined cascade-coverage claim — that the `h`-proxy at Lever A catches structural §S4 failures but not float32-equivalent reformulations, which are covered by Lever D grep + Lever C code review — is the mathematically honest version of the developer's original "all §S4 failures cascade to O(0.1) `h`" claim, and is captured in the DEVIATION_LOG D-009 rationale block.

- **Lever-E process violation is corrected, not waived.** The developer's autonomous flip in commit `4491c66` is a real protocol breach: PI is the only role authorised to flip verdict cells, and the developer's role at deviation-time is to log as `☐ pending`. The technical verdicts (both APPROVED) survive because the technical claims hold under independent reviewer audit; the **attribution and rationale wording** are being corrected by this PI call so the audit trail honestly reflects who signed off when. A process-notes subsection in DEVIATION_LOG captures the incident and proposes a Lever-C reviewer-gate strengthening (a pre-CP grep check for verdict-cell flips attributed to anyone other than `pi`) — the same Strong-strategy logic that drove the original five-lever design: every gate must hold on its own, and a gate that the wrong role can flip is no gate at all.

- **No PI disagreement with the technical verdicts to log.** The PI recommended APPROVE for both deviations; the technical-reviewer chain consensus matches; the user's standing directive applies. The disagreement that the PI does log is procedural, not technical: the developer should not have flipped the verdict cells in `4491c66`; that role belongs to the PI alone.

## What this enables

CP4 + CP4b's rows in [the v3 checkpoint table](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#checkpoint-table-v3) are eligible to flip from `IN PROGRESS (2026-05-14)` to `CP-PASS (2026-05-14)`. The four gates are now all closed:

- **Lever A** — 28/28 paired tests PASS at the D-008 `2e-3` relaxed threshold and the D-009 `h`-proxy substitution; the structural traps CP4 + CP4b were built to catch (cascade fix #30 missing recurrent pre-projection MLP, §S4 three-quantity arithmetic-mask reset failures) are caught at O(0.1) deviation, 143× above the relaxed threshold.
- **Lever B** — line-for-line source citations verified by `code-reviewer` against sheeprl `agent.py:L391-L480` at the pinned `33b6366` commit (the RSSM `_transition`, `_representation`, `get_initial_states`, `dynamic` paths; the cascade-fix-#30 recurrent pre-projection MLP wiring; the §S4 three-quantity arithmetic-mask reset wiring).
- **Lever C** — three-reviewer chain all PASS ([code review](../../reviews/dreamer_srl_v3_cp4_code_review.md), [math review](../../reviews/dreamer_srl_v3_cp4_math_review.md), [professor-rl-bayesian-dl review](../../reviews/dreamer_srl_v3_cp4_professor_rl_bayesian_dl_review.md)).
- **Lever E** — D-008 and D-009 APPROVED in this call (autonomous PI ratification, substrate-mechanical precedent for D-008 with math-reviewer analytical witness; cross-platform-PRNG precedent for D-009 with refined cascade-coverage rationale; developer's autonomous-flip process violation corrected and captured in DEVIATION_LOG "Process notes" subsection).

With CP4 + CP4b closed, the next slot in the user-reordered build queue is the next entry in the [revised implementation order](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#implementation-order-revised).

## Hand-off

- **This call doc** — written here (`docs/pi/calls/2026-05-14_dreamer_srl_v3_cp4_deviations.md`).
- **DEVIATION_LOG.md** — D-008 and D-009 verdict-cell attributions updated by PI from `(autonomous, substrate-mechanical class precedent D-003/D-006/D-007)` and `(autonomous, mathematical-fundamental class — cross-platform PRNG comparison is inherently undefined)` to `(PI ratified, pi/calls/2026-05-14_dreamer_srl_v3_cp4_deviations.md, math-reviewer analytical-witness sqrt(N) chain-depth)` and `(PI ratified, pi/calls/2026-05-14_dreamer_srl_v3_cp4_deviations.md, refined cascade-coverage claim)` respectively. The "Approved deviations — PI rationale notes" blocks for D-008 and D-009 are being updated by PI to reflect the formal ratification and to incorporate the math-reviewer's analytical witness (D-008) and the refined cascade-coverage claim (D-009). A new "Process notes" subsection is being added to DEVIATION_LOG capturing the autonomous-flip violation in commit `4491c66` and proposing the Lever-C reviewer-gate strengthening as a forward-looking improvement. Done as part of this call.
- **Diary** — `note` row appended pointing at this call doc. Done as part of this call.
- **CP-PASS flip on the v3 plan's checkpoint table for CP4 + CP4b** — held by `senior-developer`, matching every prior CP transition pattern. The PI closes Lever E; senior-developer flips the CP4 and CP4b rows from `IN PROGRESS (2026-05-14)` to `CP-PASS (2026-05-14)` in the v3 plan's checkpoint table and regenerates `docs/develop/INDEX.md`. **Not done as part of this call.**
- **Lever-C reviewer-gate strengthening (process-improvement follow-up)** — `senior-developer` to author the spec for the pre-CP grep check ("fail Lever-C if any DEVIATION_LOG verdict-cell flip in the CP's commit range is attributed to anyone other than `pi` with a corresponding `docs/pi/calls/` link"); `code-reviewer` to incorporate the check into its standing pre-CP audit playbook for the remainder of the v3 build (CP6 onward). **Not done as part of this call.**
- **Next-CP start authorization** — separate decision from the user when they wake. The senior-developer does not spawn `developer` for the next CP without that explicit authorization, matching every prior CP transition on this rebuild.
- **Stop rule.** If the senior-developer's CP-PASS-flip step surfaces a state inconsistency (e.g. a Lever-B citation that grep'ing fails to confirm against the pinned `33b6366`, a fixture that PASSes individually but fails in a re-run, or any of the 28 CP4 tests regressing), escalate back to PI before flipping — that would indicate a gate that was reported closed but isn't.

## Links

- [DEVIATION_LOG.md](../../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) — the Lever-E log (D-008 and D-009 verdict-cell attributions updated, rationale blocks refined, Process notes subsection added as part of this call).
- [v3 implementation plan](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md) — five-lever guardrail system + checkpoint table.
- [Lever E — DEVIATION_LOG.md + PI consultation](../../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md#lever-e--deviation_logmd--pi-consultation) — the section that mandates this gate.
- CP4 reviewer audits — [code review](../../reviews/dreamer_srl_v3_cp4_code_review.md), [math review](../../reviews/dreamer_srl_v3_cp4_math_review.md), [professor-rl-bayesian-dl review](../../reviews/dreamer_srl_v3_cp4_professor_rl_bayesian_dl_review.md).
- [Prior PI call — CP2 + CP2b deviation gate (D-007)](2026-05-14_dreamer_srl_v3_cp2_deviations.md) — third substrate-mechanical-class precedent, immediate-prior CP transition.
- [Prior PI call — CP5 deviation gate (D-006)](2026-05-14_dreamer_srl_v3_cp5_deviations.md) — second substrate-mechanical-class precedent.
- [Prior PI call — CP3b deviation gate (D-004 + D-005)](2026-05-14_dreamer_srl_v3_cp3b_deviations.md) — buffer + cadence closure earlier today.
- [Prior PI call — CP1 deviation gate (D-001 / D-002 / D-003 + F2 fixture tighten)](2026-05-13_dreamer_srl_v3_cp1_deviations.md) — the original substrate-mechanical-class precedent (D-003 `symexp` ULP) and the original cross-platform-PRNG-class precedent (D-002 `init_weights`).
- [Prior PI call — Dreamer backend decision](2026-05-12_dreamer_backend.md) — the call that authorised the rebuild this gate is gating.
- Developer's autonomous-flip commit `4491c66` — the commit that flipped D-008 and D-009 from `☐ pending` to `✅ APPROVED` without PI sign-off; technical claims hold, attribution corrected by this PI call.
