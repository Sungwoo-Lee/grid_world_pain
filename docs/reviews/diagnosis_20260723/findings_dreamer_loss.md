# Diagnosis findings — dreamer_srl loss.py + utils.py (2026-07-23)

Unit: `src/algorithms/dreamer_srl/loss.py`, `src/algorithms/dreamer_srl/utils.py`
Reviewer scope: world-model/actor/critic loss math + return/lambda-target/symlog/two-hot utilities, compared against DreamerV3 (Hafner et al. 2023) semantics and the vendored sheeprl@33b6366 reference. Call sites in `train.py` / `dreamer_srl_main.py` / `agent.py` traced end-to-end to determine live impact.

**Headline: no P0/P1 defects found.** The unit is a faithful sheeprl port; every numeric spot-check passed. Three P2 (latent/doc-quality) findings below. Both FIXED bugs touching this unit are verified still fixed.

---

## Findings

### P2-1 — `utils.py:196` (docstring of `compute_lambda_values`) — wrong shape claim for `values`

- **Claim**: The docstring says `values : [T+1, B] — bootstrap values (values[-1] is the terminal bootstrap)`. The implementation requires `values` to be the **same length T** as `rewards`/`continues` — `interm = rewards + continues * values * (1 - lmbda)` (utils.py:224) broadcasts element-wise, and a `[T+1,B]` values array raises `TypeError` (verified numerically: shapes `(4,1)/(5,1)/(4,1)` raise).
- **Actual semantics** (matches sheeprl utils.py:66-77 exactly): caller passes `predicted_values[1:]` of length T; the bootstrap is `values[-1]`, the value at the last imagined state — the docstring's intent is right but the declared shape is wrong.
- **Failure scenario**: none at runtime today — the live call site (`train.py:488` via `compute_imagined_returns`) passes `[H, BT, 1]` for all three, and a wrong-shaped call crashes loudly rather than silently. Pure future-maintainer trap: someone "fixing" the code to accept T+1 values per the docstring would change the recursion alignment.
- **Fix direction**: correct the docstring to `values: [T, B] (same length as rewards; values[-1] doubles as the horizon bootstrap)`.

### P2-2 — `utils.py:284` (`moments_update` default `max_=1e8`) — default encodes the wrong return-normalization clip; live path safe

- **Claim**: DreamerV3 divides returns by `max(1, S)` where S = 95th-5th percentile range — the denominator is clipped at **1** so small returns are not amplified. sheeprl realizes this by config: `configs/algo/dreamer_v3.yaml:134 -> moments.max: 1.0`, giving `invscale = max(1/max, high-low) = max(1, S)` (vendored sheeprl `algos/dreamer_v3/utils.py`, Moments.forward). Our `moments_update` default `max_ = 1e8` copies the sheeprl *class* default, not the *effective recipe* — with the default, `invscale = max(1e-8, S)`, so a near-zero return range (early training, sparse reward) amplifies advantages by up to 1e8x.
- **Live impact: none.** `train.py:887-894` passes `max_=moments_max`, `dreamer_srl_main.py:594` reads it as `get_mandatory("algo.actor.moments.max")`, and every config under `configs/models/dreamer_srl/` sets `moments.max: 1.0`. Actor gradients are also clipped at norm 100.
- **Failure scenario**: any *future* caller (test, notebook, new algorithm variant) calling `moments_update(state, x)` bare gets the paper-violating `max(1e-8, S)` — advantage explosion whenever imagined returns are nearly constant.
- **Fix direction**: change the default to `max_=1.0` (the effective sheeprl-recipe value) or remove the default entirely (project no-fallback-defaults spirit); add a comment distinguishing sheeprl class-default (1e8) from config-effective (1.0).

### P2-3 — `utils.py:315-325` (`moments_update`) — missing `stop_gradient` that the sheeprl reference has; safe only by call-site discipline

- **Claim**: sheeprl `Moments.forward` detaches at three points: `gathered_x = ....detach()`, and `return self.low.detach(), invscale.detach()`. Our port has zero `jax.lax.stop_gradient` — `x` flows into `jnp.quantile` (differentiable in JAX) and the returned `(new_low, invscale)` carry gradients from the lambda-values.
- **Live impact: none.** `train.py:887` calls `moments_update` *outside* any `value_and_grad` closure, and inside `actor_loss_fn` the offset/invscale are re-wrapped with `jax.lax.stop_gradient` before `compute_actor_objective` (train.py:946-949), so no gradient leaks today.
- **Failure scenario**: a refactor that moves the moments update inside the actor's grad closure (a natural "fuse the losses" cleanup) would silently backprop through the percentile normalizer — the actor would receive a spurious gradient term through offset/invscale (quantile subgradients), diverging from both paper and sheeprl.
- **Fix direction**: mirror the reference — `x = jax.lax.stop_gradient(x)` at the top and `stop_gradient` on the two returned values, making the function safe regardless of call context.

---

## Fixed-bug regression check

1. **Recon loss half-weighted + missing symlog — STILL FIXED.** `loss.py:49-86 SymlogDistribution`: `log_prob = -(mode - symlog(value))^2`, no 0.5 factor, decoder output treated as the symlog-space prediction, `tol=1e-8` clamp present (loss.py:85), `symexp` only at mean/mode consumption. `train.py:720` routes the WM obs term through it (`po = {"obs": SymlogDistribution(...)}`) and `reconstruction_loss` (loss.py:613) sums the obs NLL at weight 1.0 — no residual 0.5 anywhere in the unit.
2. **Gradient clipping — STILL FIXED.** `utils.py:45-55`: `WM_CLIP_NORM=1000.0`, `ACTOR_CLIP_NORM=100.0`, `CRITIC_CLIP_NORM=100.0` (matching sheeprl dreamer_v3.yaml:52/127/154); `make_optim_tx` chains `optax.clip_by_global_norm(clip_norm)` **before** `optax.adam` (clip-then-step, matching sheeprl's fabric.clip_gradients order). Wired live at `dreamer_srl_main.py:734-736` for all three optimizers.

---

## Reviewed but clean

- **symlog/symexp** (utils.py:89-108): exact `sign*log1p(|x|)` / `sign*(exp(|x|)-1)` pair; round-trip error at 1e6 is 0.125 absolute = 1.25e-7 relative (fp32, expected).
- **Two-hot encoding** (loss.py:89-285): bins stored as `linspace(-20,20,255)` in SYMLOG space (historical-bug guard intact); target symlog-encoded before bin lookup; cross-weight interpolation (`weight_below = dist_to_above/total`) correct; boundary clamp + `equal` branch correct for out-of-range targets; `symexp` applied only at mean/mode. Numeric check vs. hand-computed log_prob (incl. x=1e6 out-of-range): max diff 5.1e-6 (fp32).
- **Lambda-return recursion** (utils.py:186-228): `interm[t] = r[t] + c[t]*v[t]*(1-lambda)`; `R[t] = interm[t] + c[t]*lambda*R[t+1]`; bootstrap `R[T-1] = r + c*v_last`. With the caller's alignment (`rewards[1:]`, `values[1:]`, `continues[1:]*gamma`, train.py:485-493) this is exactly `R_t = r_{t+1} + gamma*c_{t+1}[(1-lambda)v_{t+1} + lambda*R_{t+1}]` — the DreamerV3 lambda-return. Verified against an independent hand recursion incl. a mid-sequence `continue=0`: max diff 2.2e-8.
- **Continue head** (loss.py:292-457): `BernoulliSafeMode` uses the stable log-sigmoid BCE (equals PyTorch BCE-with-logits), mode `(p>0.5)`; `IndependentBernoulli` sums the size-1 event dim; continue target `1 - terminated` with **no** gamma multiplier (matches sheeprl code, §S10).
- **reconstruction_loss / KL balancing** (loss.py:464-622): `dyn = KL(sg(post)||prior)`, `rep = KL(post||sg(prior))` — stop_gradient sides correct; coefficients 0.5/0.1 from config (verified in configs); free-nats floor is **per-[T,B]-element on the full KL** (summed over the categoricals) applied BEFORE the mean — matches the paper's 1-nat free-bits clip and sheeprl L68-74; total = `(kl_reg*kl + obs + rew + cont).mean()`. KL inputs are unimix'd logits (agent.py:758/809 `_uniform_mix` inside `_representation`/`_transition`), matching sheeprl's KL-over-unimix-probs.
- **Critic loss** (train.py:231-320 compute_critic_loss + call site): two-term NLL — `-log q(sg(lambda-targets)) - log q(sg(EMA-target-critic mean))`, both discount-weighted; lambda-targets are **raw** (not Moments-normed); live critic supplies the lambda bootstrap/baseline; target critic reserved for the regularizer term; polyak EMA fires before the train step with hard copy (tau=1) on gradient step 0, tau=0.02 after (dreamer_srl_main.py:1154-1166).
- **Actor loss** (train.py:504-615 + actor_loss_fn): REINFORCE for discrete actions — `log pi(sg(a_rollout)) * sg(advantage)` (no dynamics backprop, matching DreamerV3 discrete recipe); logits recomputed with unimix (agent.py:1475-1487); advantage = per-term percentile-normalized `(lambda-offset)/invscale - (v-offset)/invscale`; denominator clip `max(1, S)` effective via config `moments.max: 1.0`; entropy term `ent_coef*H` added inside the discount-weighted mean; entropy's `log(p+1e-8)` epsilon deviates from exact log-softmax entropy negligibly given the unimix probability floor.
- **Discount weighting** (train.py:170-227): `cumprod(c*gamma)/gamma`, stop_gradient'd; `discount[0]=continues[0]`; true-continue splice grounds step 0 in the observed `1-terminated`.
- **Moments EMA numerics** (utils.py:279-325): quantile -> EMA(0.99) -> `invscale = max(1/max_, high-low)` — matches sheeprl Moments.forward line-for-line (single-process, all_gather no-op); zero-init buffers match.
- **Initializers** (utils.py:115-179): full-precision Hafner constant 0.87962566103423978; trunc-normal +/-2 sigma; `uniform_init_weights(0.0)` zero-init for reward/critic output heads (cascade fix #27 intact).
- **Ratio / derive_prefill / prepare_obs** (utils.py:62-82, 332-462): faithful ports; Ratio kept Python-side (JIT hazard documented); prefill off-by-one intentional per sheeprl L511.

## Method notes

- Numeric checks run with `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`: lambda-return vs. independent hand recursion (diff 2.2e-8), two-hot log_prob vs. manual symlog+neighbor-interpolation (diff 5.1e-6), symlog/symexp round trip, `[T+1]`-values shape crash confirmation.
- Reference: vendored `vendor/sheeprl/sheeprl/` (Moments source + `configs/algo/dreamer_v3.yaml` moments.max: 1.0 read directly).
- Report-only pass: no code, config, or script modified.
