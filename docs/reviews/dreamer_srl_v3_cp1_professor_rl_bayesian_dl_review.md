---
title: "dreamer-srl v3 — CP1 Algorithm-Fidelity Review (professor-rl-bayesian-dl)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-13
last_updated: 2026-05-13
audited_doc: src/algorithms/dreamer_srl/utils.py
audited_commit: 46a18cb
---

# dreamer-srl v3 — CP1 Algorithm-Fidelity Review

## Plain-language verdict

**What this gate is.** The dreamer-srl v3 rebuild ports a community PyTorch
DreamerV3 reference (`sheeprl`, pinned to commit `33b6366`) into JAX, in
checkpoints (CP). The first checkpoint, CP1, ports eight leaf utility
functions in `src/algorithms/dreamer_srl/utils.py`: two symmetric-log
transforms, two weight initialisers, the lambda-return computation, the
percentile-EMA value-normaliser called *Moments*, the replay-ratio scheduler
*Ratio*, and the observation-tensor reshape helper *prepare_obs*. CP1 closes
behind a three-reviewer chain: a code reviewer (JAX/Flax correctness), a math
reviewer (per-equation fidelity), and me (algorithm-level fidelity — the
silent training-loop semantics the first two cannot see in isolation).

**Headline.** The first two gates passed; I confirm the third. CP1's eight
functions are leaf utilities — they do not by themselves trigger any of the
silent training-loop traps cached in the v2 plan as §S1–§S10. Of those ten
silent semantics, only two touch CP1 outputs at all (§S6, the discount
weighting on the actor / critic losses, consumes `compute_lambda_values`'s
return; §S7, the advantage-baseline offset cancellation, consumes
`moments_update`'s return). Both CP1 outputs match the contract the
downstream CP6 caller will need: the lambda function returns a length-T
tensor of un-discounted, un-normalised lambda-targets compatible with both
the critic's regression target *and* the actor's normalised advantage; the
Moments function returns `(new_state, new_low, invscale)` where `new_low`
is the post-EMA-update scalar that algebraically cancels in the
advantage subtraction. The F3 fix in commit `46a18cb` aligned the docstring
naming with this contract without changing semantics.

**Outcome.** PASS. The three logged deviations (D-001 / D-002 / D-003) are
not algorithm deviations under the lens of this review — see §4. The portfolio-level
PI gate can fire next.

---

## 1. Algorithm-fidelity audit

### 1.1 `compute_lambda_values` contract vs. downstream consumer

Sheeprl's `dreamer_v3.py:train()` consumes the output of `compute_lambda_values`
in two places: critic regression at `dreamer_v3.py:314`
(`−qv.log_prob(lambda_values.detach()) * discount[:-1]`, §S6) and actor
advantage at `dreamer_v3.py:277` (`(lambda_values − offset) / invscale`,
§S7). Both consume the **same** un-normalised, un-discounted, length-T
tensor.

CP1's `compute_lambda_values` (utils.py:124–166) meets this contract:

- **Caller-shifted input view.** `rewards : [T, B]`, `values : [T+1, B]`,
  `continues : [T, B]`. Sheeprl's caller indexes
  `compute_lambda_values(predicted_rewards[1:], predicted_values[1:], continues[1:] * γ, λ)`
  before calling — the function's local `values[-1]` is the terminal bootstrap.
  The JAX port preserves this via `vals = [values[-1:]]` and the trailing `[:-1]`
  after `reversed`. **✓**
- **Length-T return, aligned with `predicted_values[:-1]` baseline.** The
  returned tensor maps `[1..H]` in the imagination frame so the actor's
  advantage element-wise subtraction holds:
  $$\text{advantage}_t = \frac{G^\lambda_{t+1} - v_t}{\text{invscale}}, \quad t \in [0, H-1].$$
  **✓**
- **Singleton-axis bootstrap.** `values[-1:]` (not `values[-1]`) keeps the
  leading axis as `1`, so the final `jnp.concatenate(..., axis=0)` composes
  without reshape. Math-reviewer's critical point 2 verified this; the
  algorithmic consequence is that the actor's element-wise subtraction is
  well-defined. **✓**
- **Single un-normalised tensor, reused by both consumers.** No pre-normalise,
  no pre-discount, no pre-slice. CP6 will apply §S5/§S6/§S7. **✓**

**Verdict on 1.1.** Correct leaf primitive for both downstream consumers.
§S5 and §S6 are correctly deferred to CP6.

### 1.2 `Moments` algorithmic role — actor-only normalisation

Sheeprl calls `Moments` only at `dreamer_v3.py:276` (actor advantage); the
critic at L314 uses **un-normalised** `lambda_values.detach()` directly.
This actor-only separation is §S7.

CP1's `moments_update` (utils.py:217–263) meets the contract:

- **`(new_state, new_low, invscale)` signature.** Docstring names
  `new_low` and notes "sheeprl names this `offset` at the call site."
  F3 (`46a18cb`) aligned the docstring with the body without changing
  the algorithmic invariant — the returned scalar is the post-EMA-update
  low. **✓**
- **Pure-functional state thread (D-001).** Caller is responsible for
  threading `MomentsState` across iterations; F1's `@flax.struct.dataclass`
  form makes that pytree-explicit. The sequential EMA dynamic is
  preserved as long as the caller actually threads — and CP6 will. **✓**
- **No critic conflation.** Single update path, no actor-vs-critic branch.
  `compute_lambda_values` does not call `Moments`; `Moments` does not wrap
  `compute_lambda_values`. They are independent leaf primitives, composed at
  the §S7 call site in CP6. **✓**
- **Post-update invscale.** `new_high − new_low` (utils.py:262) uses the
  post-update values, mirroring sheeprl's in-place rebind semantic at L60–L62.
  Math-reviewer's critical point 4 verified this. Algorithmic consequence:
  on early training (when `low=high=0`), the floor `1/max_` applies as
  expected; on warm batches the observed spread drives invscale. **✓**

**Verdict on 1.2.** Correct leaf primitive for the actor-only normalisation
path. The `(new_state, new_low, invscale)` triple matches §S7's needs.

### 1.3 `Ratio` scheduler semantics

CP1's `Ratio` (utils.py:270–354) preserves sheeprl's replay-ratio contract
line-for-line:

- **Fractional-debt accumulator** (utils.py:331–333 ↔ sheeprl L289–L290): byte-identical;
  integer-exact match confirmed by `test_ratio_matches_sheeprl`. **✓**
- **First-call branch and pretrain burst** (utils.py:316–330 ↔ sheeprl L274–L288):
  identical. With `pretrain_steps=0` (XS default), the pretrain branch is dormant. **✓**
- **Replay-ratio=1.0 → 1 grad per env step.** Walking the scheduler at
  `replay_ratio=1.0` (the matched-config SPS measurement): first call at
  `n₀` returns `n₀`; every subsequent unit env-step returns `1`. Matches the
  1-to-1 cadence the SPS test assumed. **✓**
- **Python-side, never crosses JIT.** Docstring's GOTCHA correctly forbids
  passing `Ratio` inside `jax.jit` / `nnx.jit` — the accumulator `_prev`
  would be stale-captured. CP6+ must call `ratio(env_step)` in plain Python
  and execute `one_train_step` `repeats` times. **✓**

**Verdict on 1.3.** Faithful Python-side port; JIT-boundary discipline correct.

### 1.4 `init_weights` algorithmic role and call signature

Sheeprl's `init_weights` is a module-walking initialiser (`module.apply(init_weights)`);
the Linear branch reads `m.in_features`, `m.out_features`, and mutates
`m.weight.data` in place. CP1's `init_weights` (utils.py:53–88) refactors
this to a functional `(in_features: int, out_features: int, key: jax.Array)
→ kernel` signature returning the sampled kernel.

- **Right call signature for CP4+ network construction.** Downstream
  `nnx.Linear` construction (the `MLP` body, `LayerNormGRUCell`, RSSM heads,
  actor, critic) can either pass `init_weights` as a `kernel_init` partial
  or build the kernel and inject via `kernel_init=lambda *_: kernel`. The
  (in, out, key) signature is what those call sites need. **✓**
- **Bias zero-init on caller.** Sheeprl L151–L152 zeros the bias separately
  from the kernel init. The CP1 docstring documents this caller obligation
  explicitly. **✓**
- **Hafner constant `0.87962566103423978` (full precision).** Closes the
  cascade-bug truncation `0.8796` that the in-house `src/models/dreamer_v3_util.py`
  was carrying. Math-reviewer's critical point 1 verified this; the
  algorithmic consequence is that the initial activation statistics
  (truncated-normal with corrected variance) match sheeprl exactly, so the
  optimisation trajectory starts in the same regime. **✓**
- **Truncated-normal vs trunc-and-scale equivalence.** `jax.random.truncated_normal`
  on `[-2, +2]` times `std` is distributionally identical to PyTorch's
  `nn.init.trunc_normal_(..., std=std, a=-2*std, b=2*std)`. **✓**

**Verdict on 1.4.** Right call signature; clean functional refactor; Hafner
precision verified.

### 1.5 §S1–§S10 touchpoints for CP1 outputs

Only §S6 and §S7 intersect CP1 outputs; the other eight fire at downstream
CP4/CP4b/CP6/CP9b/loss.py call sites. CP1 does not preemptively absorb any
of them — the right factoring. See §2 table for the full enumeration with
call sites.

### 1.6 `prepare_obs` MLP vs CNN branching

CP1's `prepare_obs` (utils.py:361–400) is a line-for-line port of sheeprl
L80–L91: MLP path `arr.reshape(1, num_envs, -1)` matches L89; CNN path
`arr.reshape(1, num_envs, -1, H, W) / 255.0 - 0.5` matches L87. The
explicit `jnp.asarray(v, dtype=jnp.float32)` cast mirrors sheeprl's
`.float()` at L85. The XS config exercises only the MLP path (vector-obs);
the CNN branch is present-but-untested as designed. The docstring's GOTCHA
correctly forbids adding a guard that errors on non-empty `cnn_keys`. **✓**

---

## 2. §S1–§S10 cross-reference table

| § | Silent semantic | Sheeprl call site | CP1 output consumed? | Contract that must hold at downstream call site |
|---|---|---|---|---|
| S1 | Force-set `is_first[0]=1` on every sampled chunk | `dreamer_v3.py:100` | No | CP6 `one_train_step` mutates `batch["is_first"]` before dynamic-learning rollout |
| S2 | Prepend-zero-action shift | `dreamer_v3.py:104` | No | CP6 `one_train_step` constructs `batch_actions = cat([zeros[:1], actions[:-1]])` before rollout |
| S3 | `learning_starts` random-action prefill | `dreamer_v3.py:604–617, 706` | Indirect (via `Ratio` outer loop gate) | CP9b `collect_step` branches on `iter_num <= learning_starts`; CP6+ outer loop gates `one_train_step` on the same predicate. The `Ratio` scheduler at CP1 supplies the per-iter gradient count — it does NOT itself enforce the learning-starts gate; the outer loop does |
| S4 | `is_first` arithmetic-mask reset of three quantities (action, recurrent, posterior with reshape-flatten) | `agent.py:423–429` | No | CP4b `RSSM.dynamic` applies `(1 − is_first) * x + is_first * init` to all three; posterior pre-flattened from `[B, S, D]` to `[B, S*D]` before masking |
| S5 | True-continue splice at imagination step 0 | `dreamer_v3.py:247–248` | No (the splice happens **before** `compute_lambda_values`, on the caller's tensor) | CP6 `one_train_step` imagination phase: `continues = cat([true_continue[None], predicted_continues[1:]])` before passing `continues[1:] * γ` to `compute_lambda_values` |
| S6 | Discount weighting on actor AND critic losses | `dreamer_v3.py:297, 316` | **YES — consumes `compute_lambda_values` output** | CP6 computes `discount = stop_gradient(cumprod(continues * γ, axis=0) / γ)`, then multiplies `policy_loss` and `value_loss` by `discount[:-1]`. CP1's lambda-values are the un-discounted scalar input to both losses. **Contract held by CP1: length-T un-discounted return.** |
| S7 | Advantage `low`-offset cancellation | `dreamer_v3.py:276–279, 314` | **YES — consumes `moments_update` output** | CP6 does `(λ − offset) / invscale` and `(baseline − offset) / invscale`, subtracts to cancel offset. Critic uses **un-normalised** `lambda_values.detach()`. CP1's `moments_update` returns `(new_state, new_low, invscale)` with `new_low` = `offset`. **Contract held by CP1: separable scalar offset + scale; pure-functional state thread.** |
| S8 | Per-element free-nats floor before mean | `loss.py:100–106` | No | CP5/`loss.py` applies `jnp.maximum(kl_per_element, free_nats)` *before* mean-reduction |
| S9 | `Independent(BernoulliSafeMode, 1)` continue wrap + decoder `dims=1` | `dreamer_v3.py:167, 246` | No | CP4 distribution constructors apply `Independent(..., 1)` wrap; `log_prob` returns `[T, B]` (event-axis summed) |
| S10 | Continue target = `1 − terminated` (no γ) | `dreamer_v3.py:168` | No | CP6 `continues_targets = 1 − data["terminated"]`; no γ multiplier on the target |

**Two-of-ten rule.** Only §S6 and §S7 are CP1-touching. Both are satisfied
by CP1's return contracts; the downstream weighting and normalisation logic
is correctly deferred to CP6.

---

## 3. Deviation review (algorithm lens)

### D-001 — `moments_update` `fabric.all_gather` omitted

**Algorithmic claim.** Sheeprl's `Moments.forward` calls
`fabric.all_gather(x)` before the percentile compute. On a single-process
Lightning Fabric, `all_gather` returns the input unchanged. We run
single-process; the JAX port drops the call.

**Algorithm lens.** The deviation is a no-op in our deployment regime.
The percentile-EMA dynamic is unchanged: at each call, the function
computes quantiles over the local batch, updates the EMA, returns the
new state and the scalar `(offset, invscale)`. Across calls, the sequential
EMA preserves the temporal smoothing exactly as sheeprl does. The
pure-functional rewrite (D-001's structural aspect) changes the JAX
call signature but not the math — the caller threads `MomentsState`
across iterations, identical to sheeprl's in-place `nn.Module` mutation
once unrolled.

**Risk.** If we ever go multi-rank (multi-GPU data-parallel), the
percentile bounds would diverge from sheeprl's multi-rank gathered
percentile. This is a deployment-regime guard, not an algorithm
deviation. The XS config is single-rank; the deviation is benign
at current scope.

**Algorithm verdict: PASS — no algorithm deviation.**

### D-002 — `init_weights` / `uniform_init_weights` stochastic init

**Algorithmic claim.** PyTorch's `nn.init.trunc_normal_` and JAX's
`jax.random.truncated_normal` draw from the *same target distribution*
(truncated normal with the same `std`, same truncation bounds `[-2σ, +2σ]`)
but use different underlying RNG streams (Mersenne Twister vs Threefry).
Individual draws are not bit-comparable; the test verifies distribution
properties.

**Algorithm lens.** The downstream algorithmic effect of an initialisation
is on (a) the initial forward-pass output distribution and (b) the initial
gradient magnitudes. Both are determined by the target distribution
(truncated normal at std `√(1/d_avg)/c_Hafner`, truncation `[-2σ, +2σ]`),
not by the specific RNG stream. As long as the target distribution
matches, the initial network behaves statistically identically — the
optimisation trajectory will differ in the first few steps but converge
to the same basin under the same Adam dynamics. Math-reviewer confirmed
the formula is identical; I confirm the algorithmic consequence (no
systematic bias in initial activations / gradients).

**Algorithm verdict: PASS — no algorithm deviation.**

### D-003 — `symexp` 1-ULP slack

**Algorithmic claim.** Float32 `exp(|x|)` on CUDA differs between JAX
(XLA `__nv_expf`) and PyTorch (cuDNN) by up to 1 ULP for large `|x|`.
At `|x|=5`, `exp(5) ≈ 148`, 1 ULP ≈ `1.77×10⁻⁵`. Observed
`max_abs_diff = 1.526×10⁻⁵` is 0.86 ULP — within hardware precision.

**Algorithm lens.** Trivial sign-off. `symexp` is used at two algorithmic
sites:
1. The two-hot distribution's `.mean` / `.mode` consumer (sheeprl
   `distribution.py`, used in CP5/CP6). The `.mean` is a sum over 255
   bins weighted by softmax probabilities; a 0.86-ULP rounding on each
   bin contributes a vanishingly small bias.
2. `SymlogDistribution.log_prob` (CP4 decoder), which applies symlog to
   targets and observations. A 0.86-ULP rounding on a target
   transformation flows through a quadratic-error loss and produces a
   loss-scale bias `≈ 2 × |x| × ULP × 1` — at most `~10⁻⁴` on the
   loss, vanishingly small relative to the loss magnitudes (~unit).

Neither has a meaningful algorithmic effect.

**Algorithm verdict: PASS — no algorithm deviation.**

---

## 4. Verdict

**PASS.**

All eight CP1 leaf utilities are algorithmically faithful to sheeprl. The
return contracts match what the downstream CP6 caller will need at the
§S6 (discount weighting) and §S7 (Moments offset cancellation) sites.
The remaining eight silent training-loop semantics (§S1, §S2, §S3, §S4,
§S5, §S8, §S9, §S10) have no CP1 touchpoint and will be reviewed at
their own CP gates (CP4b for §S4; CP6 for §S1/§S2/§S5; CP9b for §S3;
CP4 / `loss.py` for §S8/§S9/§S10).

The three deviations are not algorithm-level — D-001 is a deployment-regime
no-op, D-002 is an RNG-stream mismatch with identical target distribution,
D-003 is a transcendental-function rounding gap below 1 ULP.

The F1 (`MomentsState` → `flax.struct.dataclass`) and F3 (docstring naming
alignment) fixes in `46a18cb` did not change semantics; the
`(new_state, new_low, invscale)` algorithmic invariant is preserved.

**PI gate may fire next** to portfolio-review the three deviations
(D-001 / D-002 / D-003) and the F2 test-threshold question raised by
code-reviewer.

---

## Next steps

- **PI** (`pi` agent): portfolio-level sign-off on D-001 / D-002 / D-003.
  Each is algorithm-benign per my review and math-reviewer's review;
  approval is the expected outcome but is the PI's call.
- **senior-developer**: when PI closes the deviations, mark CP1 status
  `CP-PASS` in `IMPLEMENTATION_PLAN.md` and unblock CP5 (`loss.py`
  two-hot distribution) per the revised implementation order.
- **professor-rl-bayesian-dl** (me, future CP gates): the §S items I
  flagged for downstream CPs are tracked above. At CP4b I will audit
  §S4 (the three-quantity `is_first` reset); at CP6 I will audit §S1,
  §S2, §S5, §S6, §S7 wired in `one_train_step`; at CP9b I will audit
  §S3 (learning_starts prefill); at CP4 I will audit §S9 distribution
  wraps; at CP5/CP6 I will audit §S8 and §S10 in `loss.py`.

---

## Links

- [v3 implementation plan](../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
- [v3 deviation log](../develop/active/dreamer_srl_v3/DEVIATION_LOG.md)
- [v2 plan (algorithmic backbone with §S1–§S10)](../develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md)
- [v2 professor-rl-bayesian-dl review (stylistic template)](../develop/archive/dreamer_srl/review_professor_rl_bayesian_dl_v2.md)
- [CP1 math review](dreamer_srl_v3_cp1_math_review.md)
- [pre-CP0 code review](dreamer_srl_v3_pre_cp0_code_review.md)
- [Vendored sheeprl utils.py](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/utils.py)
- [Vendored sheeprl Ratio](../../vendor/sheeprl/sheeprl/utils/utils.py)
- [Vendored sheeprl train() — actor/critic loss assembly](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py)
- [Audited code](../../src/algorithms/dreamer_srl/utils.py)
- [Audited tests](../../tests/algorithms/dreamer_srl/test_utils.py)

Reviewed by: professor-rl-bayesian-dl
