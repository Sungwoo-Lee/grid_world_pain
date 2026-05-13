---
title: "dreamer-srl v3 CP8 — math review"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: scripts/dreamer_srl_offline_check.py, scripts/fixtures/gen_cp8_fixtures.py, src/algorithms/dreamer_srl/train.py
---

# CP8 Math Review — PASS (composition theorem) WITH CONCURRENCE ON F1 + F3

## Plain-language verdict

**Question.** CP8 is the **integration gate** of the dreamer-srl v3 rebuild. The seven prior checkpoints (CP1–CP7) each shipped a single mathematical piece — a discount mask, a two-hot encoding, a lambda-return, a slow-target EMA update, an actor REINFORCE objective, etc. — and each piece was independently verified to be **bit-faithful** to the reference PyTorch implementation we are porting from (Hafner's DreamerV3 as packaged in sheeprl). CP8's job is not to verify any new math, but to verify that wiring those bit-faithful pieces together **in the correct order** produces a bit-faithful end-to-end pipeline.

**Headline.** **PASS.** The composition theorem holds trivially: a chain of deterministic JAX functions that each match their sheeprl counterpart up to a measured per-function budget will, when fed shared fixture inputs, produce outputs whose drift is bounded by the sum (or square-root-of-sum-of-squares for independent-error chains) of per-function budgets. The 18/18 integration checks land at `0.000e+00` because the fixture's "reference pipeline" calls the same JAX production functions the offline-check then re-calls — so the composition is verifying $f(x) == f(x)$ at the JAX layer, which is trivially true for any deterministic pipeline. **The cross-framework parity claim (JAX-vs-sheeprl) is carried by the per-function CP1–CP7 tests, not by CP8.**

**The one striking finding.** The `|neg_lp1 − neg_lp2| = 4.77 \times 10^{-7}` quoted in the offline-check output as "informational, not budget-bearing" is **exactly 1 ULP at float32 magnitude 4** ($4 \cdot \epsilon_{\text{f32}} = 4.768 \times 10^{-7}$). It is **statistically indistinguishable from zero** at float32 precision. The numerical guard on this quantity therefore provides **zero real protection** against the cascade-fix-#29 regression class. The only protection is the source-inspection guard at offline-check L501-L509, which the code-reviewer's F3 flagged as also weak (counts substrings without excluding docstrings). I **concur with F3** and recommend the developer harden the guard via AST parsing or grep-with-comment-exclusion.

**Verdict.** Composition math holds. Concurrence on F1 (scope re-statement needed: CP8 verifies JAX-composition determinism, not cross-framework parity). Concurrence on F3 (numerical guard is at float32 noise floor; source-inspection guard needs hardening). F2 is a documentation issue (the source check labelled "sg(action)" actually checks "sg(advantage)") — agree it should be renamed. Forward to PI with concurrence to close CP8 once the process blocker (premature CP-PASS flip, reverted at `f5a0313`) and the three technical concerns are addressed.

## Scope

| Aspect | Path |
|---|---|
| Implementation under review | `scripts/dreamer_srl_offline_check.py` (18-check integration harness) |
| Fixture generator | `scripts/fixtures/gen_cp8_fixtures.py` |
| Production code being composed | `src/algorithms/dreamer_srl/train.py` (CP6 + CP7 functions) |
| Sheeprl reference | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` L82–L358 (the `train` function) |
| Code review (gate 1) | `docs/reviews/dreamer_srl_v3_cp8_code_review.md` (PASS WITH PROCESS BLOCKER + 3 technical concerns) |
| Deviation log | `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` — no new D-### in CP8 |
| Process incident | premature CP-PASS flip in implementation commit `e8d05b0`, reverted at `f5a0313` |

## Composition theorem — the math claim CP8 actually verifies

### Claim (informal)

Each of CP1–CP7's production functions is a **deterministic, pure JAX function** of its inputs (no PRNG state inside the deterministic-math branch — stochastic RSSM rollout is excluded by CP8 design and covered separately by CP4/CP4b). Therefore composing them produces a deterministic pipeline. When the fixture's reference pipeline and the offline-check's pipeline both call **the same JAX implementations** with **the same fixture inputs**, the outputs are mathematically guaranteed to be identical up to JAX/XLA reduction-tree non-determinism (which is bounded by the per-function ULP budgets).

### Claim (formal)

Let $f_1, \ldots, f_7$ be the seven production JAX functions:

- $f_1$ = `moments_update` (CP1)
- $f_2$ = `LayerNormGRUCell.__call__` (CP2)
- $f_3$ = `compute_lambda_values` (CP3)
- $f_4$ = `RSSM.dynamic`'s deterministic branch + `_transition` + `_representation` (CP4 / CP4b — deterministic logits only; stochastic sampling excluded by CP8)
- $f_5$ = `TwoHotEncoding.log_prob` / `.mean` (CP5)
- $f_6$ = `compute_critic_loss` + `compute_discount` (CP6)
- $f_7$ = `compute_imagined_returns` + `compute_actor_objective` + `polyak_update` (CP7)

Each $f_i$ is bit-faithful to sheeprl up to a measured per-function budget $\delta_i$ (the DEVIATION_LOG entries D-001 through D-011):

$$
\sup_{x \in \mathcal{X}_i} \big\| f_i(x) - f_i^{\text{sheeprl}}(x) \big\|_\infty \;\le\; \delta_i
$$

with $\delta_1 = 8.2 \times 10^{-8}$ (D-001), $\delta_2 = 2.97 \times 10^{-4}$ (D-007), $\delta_3 \le 1 \times 10^{-6}$, $\delta_4 = 7.19 \times 10^{-4}$ (D-008, deterministic logits), $\delta_5 = 1.81 \times 10^{-5}$ (D-006), $\delta_6 = 3.10 \times 10^{-5}$ (D-010), $\delta_7 = 0$ (D-011, exact).

The composed pipeline $F = f_7 \circ f_6 \circ \ldots \circ f_1$ then satisfies a Lipschitz-bounded composition inequality:

$$
\big\| F(x) - F^{\text{sheeprl}}(x) \big\|_\infty \;\le\; \sum_{i=1}^{7} L_{i+1:7}\cdot \delta_i
$$

where $L_{i+1:7}$ is the product of effective Lipschitz constants of the downstream functions $f_{i+1}, \ldots, f_7$ on the values actually flowing through the pipeline. For a sum-of-independent-errors model the tighter sqrt-of-sum-of-squares bound applies (variance addition under independence):

$$
\sigma_{\text{e2e}}\;\lesssim\;\sqrt{\sum_{i=1}^{7} L_{i+1:7}^2 \cdot \sigma_{i}^2}
$$

### What CP8 actually measures

The 18 integration checks **do not measure** $\big\| F(x) - F^{\text{sheeprl}}(x) \big\|$ — they would need a sheeprl runner to produce reference outputs, which the developer correctly declined (Lightning Fabric stack, PyTorch world models, dataloaders all required; the STOP-AND-SURFACE decision to use JAX-as-reference is defensible).

CP8's 18 checks measure $\big\| F^{\text{jax,offline\_check}}(x) - F^{\text{jax,fixture}}(x) \big\|$ — JAX-vs-JAX self-consistency. Because **both paths call the same compiled JAX functions on the same inputs**, this difference is mathematically zero modulo XLA reduction-tree non-determinism between two `jit` traces of the same function (which is itself sub-ULP and below all per-function budgets). The observed `0.000e+00` across every numerical check is therefore the **expected** result for a composition of deterministic JAX functions.

This is **trivially true** but **non-trivially useful**: it catches integration bugs (wrong call order, wrong tensor slicing, missing §S5 splice, missing cascade-fix-#29 second term) that would change the **structure** of the pipeline even though no single per-function call would change its math. The integration tests cited at offline-check L23-L28 are correctly identified:

- Wrong call-order in `one_train_step` (e.g. actor loss before imagination) ⇒ would crash or produce wrong shapes.
- Wrong tensor-slicing (e.g. `lambda_values[:-1]` instead of `lambda_values`) ⇒ would crash on shape mismatch in `compute_critic_loss`.
- Missing §S5 splice ⇒ caught by the splice-visibility check at L348-L367 (`continues_spliced[0] - splice_expected < 1e-6` and `continues_spliced[0] - continues_predicted[0] > 1e-4`).
- Missing cascade-fix-#29 second term ⇒ NOT caught numerically (see F3 below); caught only by source inspection.

### Cross-framework parity chain (the F1 concern)

The cross-framework parity claim **JAX-vs-sheeprl** decomposes as:

$$
\underbrace{\text{end-to-end JAX-vs-sheeprl}}_{\text{not directly measured at CP8}} \;\le\; \underbrace{\text{per-function CP1-CP7 deviations}}_{\text{measured by Lever-A bit-identity tests against sheeprl-side fixtures}}\;+\;\underbrace{\text{composition determinism}}_{\text{measured by CP8}}
$$

The CP1–CP7 Lever-A tests **do** compare against sheeprl-side stored fixtures (CP3b CP4 CP5 CP6 CP7 all reference vendored `sheeprl_jax_diff.py` runners), so the **per-function** half of the chain holds cross-framework. CP8's role is the **composition** half — verifying no integration bugs corrupt the wiring. Under this decomposition the F1 concern is correctly identified: CP8 does not verify cross-framework end-to-end parity in one step, but the **combination** of CP1-CP7 per-function parity + CP8 composition determinism implies it. **Math chain holds.**

**Recommendation.** Concur with F1. The developer should add a "Scope re-statement" subsection to the CP8 implementation report making explicit that CP8 verifies *composition determinism of the JAX pipeline* and that cross-framework integration drift is the **product** of per-function CP1-CP7 parity (Lever-A) and CP8 composition determinism.

## F3 — numerical noise-floor analysis (the striking finding)

### The claim under review

`scripts/dreamer_srl_offline_check.py:L510-L513` reports `|neg_lp1 − neg_lp2|_{\max} = 4.77 \times 10^{-7}` as "informational, not budget-bearing", with a comment that the inter-term diff "near-zero expected when lambda_values ~ 0". The implementation report cites this as evidence the cascade-fix-#29 two-term critic loss is present.

### The math

The two-term critic loss (sheeprl `dreamer_v3.py:L314-L315`):

$$
\mathcal{L}_V = -\,\log q_\phi(\Lambda_t) - \log q_\phi(\bar V_t)
$$

where $\Lambda_t$ is the un-normalised lambda-return target and $\bar V_t$ is the EMA target-critic's expected value. If the implementation drops the second term (cascade bug #29), the loss collapses to $\mathcal{L}_V = -\log q_\phi(\Lambda_t)$ alone, and $\text{neg\_lp2}$ does not exist.

The proposed numerical guard is: if `neg_lp1` and `neg_lp2` are returned and observably different, then both terms are being computed. The threshold check is implicit ($\text{diff} > 0$); the offline-check reports the observed diff for informational logging.

### Float32 noise-floor calculation

Float32 machine epsilon: $\epsilon_{\text{f32}} = 1.1920929 \times 10^{-7}$.

For a quantity of magnitude $|x|$, one unit in the last place (ULP) is:

$$
\text{ULP}(x) \;\approx\; |x| \cdot \epsilon_{\text{f32}}
$$

Typical $-\log q_\phi(\cdot)$ values for the twohot critic with 255 bins land in the range $|x| \sim 4$–$8$ (log-likelihood of a single bin near the centre under near-uniform logits is $\approx \log 255 \approx 5.5$). At $|x| = 4$:

$$
\text{ULP}(4) \;=\; 4 \cdot 1.1920929 \times 10^{-7} \;=\; 4.768 \times 10^{-7}
$$

The reported `4.77e-7` matches this **to four significant figures**:

$$
\frac{4.77 \times 10^{-7}}{4 \cdot \epsilon_{\text{f32}}} \;=\; 1.0003
$$

This is, to within rounding of the reported scientific-notation digits, **exactly 1 ULP at float32 magnitude 4** — the float32 noise floor for the operating range of $-\log q_\phi(\cdot)$. It is **statistically indistinguishable from zero** at single precision.

### Why this happens

The fixture's `lambda_values` for `terminated[0] = 0` and predicted rewards near zero (the encoder/reward heads are zero-init at construction) lands the lambda-return targets very close to bin 127 (the zero bin). The EMA target-critic at first call (`tau = 1.0` hard copy) is byte-identical to the online critic, so `target_critic_values` is **also** near bin 127. Both `log_prob(lambda_values)` and `log_prob(target_critic_values)` evaluate the same TwoHotEncoding distribution at near-identical inputs, both landing in bin 127's neighbourhood — so the two log_probs are mathematically equal up to the float32 ULP of the bin-lookup arithmetic.

This is **not** a bug in the cascade-fix-#29 implementation. The code at `train.py:L308-L312` correctly computes:

```
neg_lp1 = -qv.log_prob(jax.lax.stop_gradient(lambda_values))
neg_lp2 = -qv.log_prob(jax.lax.stop_gradient(target_critic_values))
```

Both terms are present, both call `log_prob`, both use `stop_gradient` on their targets. The math is correct. The **diagnostic value of the inter-term diff is zero** because the fixture happens to make both inputs nearly identical.

### Why this matters for F3

The code-reviewer's F3 flagged the **source-inspection guard** at L501-L509 as weak (counts substrings of `"-qv.log_prob("` in the source; threshold `>= 2`; a regression deleting line 312 but leaving the docstring intact would still pass because docstring matches are counted). My math-side concurrence: the **numerical** guard is even weaker than the source-inspection guard — it is **at the float32 noise floor** and cannot in principle distinguish "second term present" from "second term absent under degenerate fixture inputs".

**Recommendation.** Concur with F3. The fix should:

1. **Harden the source-inspection guard** via `ast.parse(critic_loss_src)` + visitor that counts actual `Call` nodes to `qv.log_prob` (not substring matches), excluding docstrings and comments. This is the **load-bearing** guard for cascade-fix-#29 regression detection.
2. **Regenerate the fixture** with a non-degenerate PRNG seed that produces `lambda_values` and `target_critic_values` that differ by more than 1 ULP at the bin-lookup magnitude — making `|neg_lp1 − neg_lp2|` observably non-zero. The current value `4.77e-7` is mathematically meaningless for guard purposes.

**Either fix alone is sufficient.** The code-reviewer's primary recommendation (AST parsing) is the simpler and more durable option.

## F2 — misnamed source check (composition irrelevant; documentation only)

The check at offline-check L600-L609 inspects `compute_actor_objective` source for `"stop_gradient"` + `"advantage"`. This correctly verifies the **§S7 advantage** stop-gradient (`jax.lax.stop_gradient(advantage)` at `train.py:L589`). The CP8 forward-looking item #1 from CP7's hand-off is the **sg(action)** discipline at the actor forward pass — which lives at the **caller** site (not yet implemented; deferred to CP9's `one_train_step`).

This is a documentation issue, not a math issue. The check is **valid** (it does verify a real `stop_gradient`); it is just **mis-labelled** in the offline-check's print statement and the implementation report. Concur with code-reviewer's F2 fix: either rename to "sg_advantage" or flag as deferred to CP9.

## D-### budget composition — does the 0.000e+00 result violate any analytical bound?

The composition theorem above predicts end-to-end drift in JAX-vs-sheeprl space bounded by:

$$
\delta_{\text{e2e,JAX-vs-sheeprl}} \;\lesssim\; \sqrt{\sum_{i=1}^{7} L_i^2 \cdot \delta_i^2}
$$

For the seven CP per-function budgets:

| CP | $\delta_i$ | D-### |
|---|---|---|
| CP1 | $8.2 \times 10^{-8}$ | D-001 |
| CP2 | $2.97 \times 10^{-4}$ | D-007 |
| CP3 | $\le 1 \times 10^{-6}$ | (no deviation) |
| CP4/4b | $7.19 \times 10^{-4}$ | D-008 (deterministic logits only; CP8 excludes stochastic sampling per design) |
| CP5 | $1.81 \times 10^{-5}$ | D-006 |
| CP6 | $3.10 \times 10^{-5}$ | D-010 |
| CP7 | $0$ | D-011 (exact arithmetic) |

If the Lipschitz constants $L_i \approx 1$ (the actual values depend on the pipeline; for log-prob terms $L \approx 1$ to within an order of magnitude), the sqrt-sum-of-squares bound predicts:

$$
\delta_{\text{e2e,JAX-vs-sheeprl}} \;\lesssim\; \sqrt{(8.2\!\cdot\!10^{-8})^2 + (3\!\cdot\!10^{-4})^2 + (7\!\cdot\!10^{-4})^2 + (1.8\!\cdot\!10^{-5})^2 + (3.1\!\cdot\!10^{-5})^2} \approx 7.6 \times 10^{-4}
$$

The CP8 offline-check's measured drift is `0.000e+00` across all 18 numerical checks. This is **not in violation** of the analytical bound: the bound is on **JAX-vs-sheeprl** drift, but CP8 measures **JAX-vs-JAX** drift. The two quantities are different by construction. The JAX-vs-JAX drift is zero because the same compiled functions are being called on the same inputs; the JAX-vs-sheeprl drift is bounded by per-function deviations but **is not measured at CP8**.

**Consistency check.** Is `0.000e+00` across all 18 checks consistent with composition-of-deterministic-pipeline? Yes — and **stronger than that**: it is the **expected** result. The fixture-generator and offline-check both call the same `compute_imagined_returns`, `compute_actor_objective`, `compute_critic_loss`, `polyak_update`, `TwoHotEncoding.mean`/`.log_prob`, etc. JAX's deterministic functions are deterministic (in the absence of XLA reduction-tree non-determinism between traces, which is below the per-function ULP budgets and not visible in the `0.000e+00` print precision). The composition result is consistent with the theorem.

The sole non-zero — `|neg_lp1 − neg_lp2| = 4.77 \times 10^{-7}` — is **not a composition drift** but an inter-term diff between two evaluations of the same `log_prob` on nearly-identical inputs, sitting at the float32 noise floor (see F3 analysis above).

## Findings table

| # | Severity | Location | Issue | Math claim |
|---|---|---|---|---|
| 1 | 🟢 PASS | Composition theorem | Deterministic chain $\Rightarrow$ zero JAX-vs-JAX drift | Verified |
| 2 | 🟢 PASS | All 18 integration checks | $0.000\text{e}{+}00$ consistent with theorem | Verified |
| 3 | 🟡 Concur F1 | Scope claim | CP8 measures JAX-vs-JAX, not JAX-vs-sheeprl; chain holds via CP1-CP7 + CP8 | Documentation fix needed |
| 4 | 🟡 Concur F3 | offline-check L501-L513 | $4.77\times10^{-7}$ inter-term diff is **exactly 1 ULP at float32 magnitude 4** — at noise floor, NOT a regression guard | AST guard or non-degenerate fixture seed |
| 5 | 🟢 Concur F2 | offline-check L600-L609 | Source check is valid (`sg(advantage)`) but **mislabelled** as `sg(action)` | Rename or defer to CP9 |
| 6 | 🟢 PASS | D-### budget composition | End-to-end bound $\sim 7.6\times10^{-4}$ in JAX-vs-sheeprl space; CP8 measures JAX-vs-JAX so this bound is not directly tested | No action |

## Conclusion

CP8 is **mathematically correct at the scope it verifies**: the composition of CP1–CP7's bit-faithful deterministic JAX functions produces a deterministic end-to-end pipeline whose JAX-vs-JAX self-consistency lands at `0.000e+00`. The cross-framework parity claim is carried by the per-function CP1–CP7 Lever-A tests (which compare against sheeprl-side fixtures); CP8 closes the **composition determinism** half of the chain.

**One striking finding**: the `4.77 \times 10^{-7}` "informational" inter-term diff for the cascade-fix-#29 guard is **exactly 1 ULP at float32 magnitude 4** ($4 \cdot \epsilon_{\text{f32}} = 4.768 \times 10^{-7}$). This is at the float32 noise floor and provides **zero diagnostic value** for regression detection. The source-inspection guard at L501-L509 (also weak per F3) is the only real protection against cascade-fix-#29 regression and must be hardened via AST parsing (or the fixture regenerated with a non-degenerate seed).

**Verdict: PASS the composition math. Concur with F1, F2, F3. Forward to PI once (i) the process blocker (premature CP-PASS flip, reverted at `f5a0313`) is closed by senior-developer's verdict-cell flip, and (ii) F1/F2/F3 are addressed under senior-dev direction.**

Reviewed by: math-reviewer
