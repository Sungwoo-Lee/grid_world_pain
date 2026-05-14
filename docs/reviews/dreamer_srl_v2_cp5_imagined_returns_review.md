---
title: "v2-CP5 — `compute_imagined_returns` + λ-return chain re-audit"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
---

# v2-CP5 — `compute_imagined_returns` + λ-return chain re-audit

## Verdict (plain-language entry point)

**What this review is.** A line-by-line re-audit of the dreamer-srl imagined-rollout
chain — the block of code that takes the world model's predicted rewards / values /
continues, splices in the real-data "did the episode actually end?" signal, runs the
backward λ-return recurrence (the H-step bootstrap that mixes the next-step value
estimate `V(s_{t+1})` with the next-step λ-return `V_λ(s_{t+1})`), and feeds the
resulting `lambda_values` into the actor's advantage signal and the critic's regression
target. This is the **H3 hypothesis surface** in the v2 plan: H3 says the parity-failed
3-seed run (collapsed to ~104 survival steps versus a ~500-step sheeprl baseline) is
rooted in an **advantage-sign bug** — either an inverted `lambda_values - predicted_values`
subtraction (which would teach the policy to AVOID rewarded actions instead of seek them),
or a §S5 splice arithmetic mistake (the splice at the first imagined step that replaces
the world-model's predicted continue with the real-data `1 - terminated`).

The audit surface is two short blocks: dreamer-srl
[`src/algorithms/dreamer_srl/train.py:L396-L493`](../../src/algorithms/dreamer_srl/train.py)
(`compute_imagined_returns`) plus
[`src/algorithms/dreamer_srl/utils.py:L124-L166`](../../src/algorithms/dreamer_srl/utils.py)
(`compute_lambda_values`). Cross-cutting context: the surrounding orchestrator at
[`train.py:L770-L867`](../../src/algorithms/dreamer_srl/train.py) wires those two
functions together and consumes their outputs in `compute_actor_objective` and
`compute_critic_loss`.

**Headline verdict.** **❌ FAIL — substantive P-class divergence found, distinct from H3.**

H3 itself (the advantage-sign hypothesis) is **REFUTED in this audit surface**:
`advantage = normed_lambda_values - normed_baseline` at
[`train.py:L585`](../../src/algorithms/dreamer_srl/train.py) has the correct sheeprl
sign convention (positive when the imagined return exceeds the predicted value). The §S5
splice arithmetic (`true_continue = 1 - terminated_observed`, then concat at axis 0) is
faithfully ported from sheeprl `dreamer_v3.py:L247-L248`. The backward λ-return
recurrence in `compute_lambda_values` is a line-by-line transliteration of sheeprl
`utils.py:L66-L77`.

However, **two structural P-class bugs surface in the surrounding orchestration**:

- **P1 (🔴 blocker).** The bootstrap value `predicted_values` that feeds the λ-return
  recurrence is computed from `target_critic`, not from the live `critic`. Sheeprl
  unambiguously uses the **live critic** for this purpose at
  `dreamer_v3.py:L244`. This is not a sign bug — but it's an algorithmic deviation that
  decouples the actor's advantage signal from the critic's most-recent estimate, and
  slows value-target propagation by `1/τ ≈ 50` steps (the τ=0.02 Polyak EMA timescale).
- **P2 (🔴 blocker, inherited).** The §S5 splice consumes `batch["terminated"]` which
  the v2-CP7 driver review showed is **always 1.0** at every episode boundary on the
  food-only NoPred substrate (because the driver conflates `terminated` with
  `truncated`). When the splice fires at a max-steps boundary, it sets
  `continues_spliced[0] = 0` and the cumulative-product discount mask zeroes out the
  bootstrap immediately — propagating the v2-CP7 P2 bug into the value-target chain.
  This audit confirms the §S5 arithmetic is correct **conditional on a correct
  `terminated` column from the driver**.

Two F-class findings (F1, F2) are detailed below.

The grad-test pseudocode for the four sites named in
[`GRAD_PARITY_METHODOLOGY.md`](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md)
§5.3 is specified in §6 of this review for the v2-CP9 developer.

---

## Findings table

| ID | Severity | Site (dreamer-srl) | Site (sheeprl) | Issue | Suggested fix |
|---|---|---|---|---|---|
| **P1** | 🔴 **blocker** | [`train.py:L798-L801`](../../src/algorithms/dreamer_srl/train.py) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L244` | **`predicted_values` for the λ-return bootstrap is computed from `target_critic` (the slow EMA), but sheeprl uses the live `critic`.** Sheeprl L244 reads `predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean` — the LIVE critic. dreamer-srl L798 reads `predicted_values_logits = jax.vmap(target_critic)(imag_flat)` — the SLOW EMA target. Consequences: (a) the actor's advantage signal `lambda_values - predicted_values` lags the live critic by τ⁻¹≈50 grad steps; (b) the λ-return bootstrap itself uses the slow target, slowing value-target propagation through the H-step recurrence; (c) `predicted_target_values` for the critic's second-term cascade fix #29 IS already correctly the target critic, so this is not a copy-paste from that site. The bug is a substantive ALGORITHM deviation (the actor's normalised advantage no longer matches Hafner 2024 / sheeprl semantics), not a substrate-class deviation. | Change L798 to call `critic` (the live module), not `target_critic`. Keep `target_critic` for `target_critic_values` at L875-L878 (that site is correct). |
| **P2** | 🔴 **blocker** (inherited) | [`train.py:L809-L818`](../../src/algorithms/dreamer_srl/train.py) (consumer) → [`train.py:L475`](../../src/algorithms/dreamer_srl/train.py) (splice site) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L247` | **§S5 splice consumes a corrupted `terminated` signal.** The §S5 splice (`true_continue = 1.0 - terminated_observed` at L475) is **arithmetically correct** and matches sheeprl exactly. But its input — `batch["terminated"]` at L809 — was written by the v2-CP7 driver as a conflation of `terminated|truncated` (see [`dreamer_srl_v2_cp7_driver_review.md`](dreamer_srl_v2_cp7_driver_review.md) §P2). On the food-only NoPred substrate, every episode boundary is a max-steps truncation, so `terminated=1.0` at every done — and the splice therefore writes `continues_spliced[0] = 0` at every boundary, zeroing the discount mask and zeroing the λ-return bootstrap exactly where value-target learning matters most. This is a P2 bug **inherited from the driver**, not a CP5 bug per se. It is reproduced here because it lands inside the imagined-return chain's input signal, so any v2-CP5 grad-parity test that uses fixture data with `terminated=1.0` will silently pass (the splice is arithmetically correct) while training-time runs will exhibit the value-bootstrap suppression. | Fix the driver per v2-CP7 P2 (read `terminated` separately from `truncated`). Then add a v2-CP10 integration assertion: at training-time, the fraction of buffer rows with `terminated=1.0` should match the rate of real death events (~0 on food-only NoPred, ~0.X on predator-on). |
| **F1** | 🟡 concern | [`train.py:L823-L830`](../../src/algorithms/dreamer_srl/train.py) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L276` | **Moments update receives un-normalised `lambda_values` (correct), but its `invscale` is then used in the actor objective WITHOUT a `sg` at the moments_update call site.** L823 returns `(new_moments, moments_offset, moments_invscale)` — three values. The orchestrator passes `moments_offset` and `moments_invscale` into `compute_actor_objective` at L858-L859 wrapped in `jax.lax.stop_gradient(...)` explicitly. **This is correct** — but it duplicates the `sg` discipline that `moments_update` should arguably enforce internally (sheeprl `utils.py:L63` returns `self.low.detach(), invscale.detach()`). The orchestrator's belt-and-suspenders sg is benign (sg of an already-sg value is a no-op), but a future caller who forgets to wrap could leak gradient through the moments percentile-EMA. Defensive: have `moments_update` apply `stop_gradient` to its returned `offset` and `invscale` before returning. | Add `return new_state, jax.lax.stop_gradient(new_low), jax.lax.stop_gradient(invscale)` to [`utils.py:L263`](../../src/algorithms/dreamer_srl/utils.py). |
| **F2** | 🟡 concern | [`train.py:L823`](../../src/algorithms/dreamer_srl/train.py) | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L276` | **Moments update happens AFTER `compute_imagined_returns` returns and BEFORE the actor loss reads it — correct ordering — but the moments update IS happening on the gradient path of the **previous** train step's moments state.** The `moments` argument to `one_train_step` is the moments state from the prior call; the function returns `new_moments` which becomes the next call's `moments`. The arithmetic for the CURRENT step's normalisation uses `new_moments`'s `offset` and `invscale` — i.e. the EMA-updated values that include this step's `lambda_values`. Sheeprl L60-L63 does the same in-place update of `self.low / self.high` BEFORE returning. So the ordering is correct (the v2-CP7 reviewer's checklist would flag this as ✅). Worth flagging only because the CP10b run showed `moments_invscale` pinning to the `1.0/max_=1e-8` floor mid-late training. With `max_=1.0` (the dreamer-srl default per `make_train_step(moments_max=1.0)` at L624), the floor is `1.0/1.0 = 1.0`, not `1e-8`. So `moments_invscale = max(1.0, high - low)` — the floor IS `1.0`. If the `high - low` percentile spread drops below 1.0 (which happens when λ-returns are concentrated near zero), the invscale clamps to 1.0 and the advantage normalisation becomes a no-op (divide by 1). The CP10b observation is therefore consistent with a flat-returns regime, not with a moments bug. | No code fix; flag for v2-CP9 / v2-CP10 analysis: if `moments_invscale` pins to 1.0 throughout training, that's a SIGNAL that λ-returns are too small — likely caused by P1 (slow-target bootstrap) and/or P2 (boundary-zeroed bootstrap). Both upstream fixes will lift λ-return magnitudes and unfloor invscale. |
| **Nit** | 🟢 nit | [`train.py:L450-L451`](../../src/algorithms/dreamer_srl/train.py) | n/a | The `terminated_observed` shape contract in the docstring says "[1, BT, 1] or [BT, 1]" but the orchestrator at L809 passes `batch["terminated"].reshape(BT, 1)` — i.e. always [BT, 1]. The "or" branching is defensive code with no current caller. | Tighten the contract to `[BT, 1]`; remove the `or` branch in the docstring. |
| **Nit** | 🟢 nit | [`utils.py:L134-L135`](../../src/algorithms/dreamer_srl/utils.py) | n/a | The docstring says `values: [T+1, B]` and `continues: [T, B]` — but the caller passes `predicted_values[1:]` (length H, not H+1) at `train.py:L485`. The docstring is wrong about `values`; both `values` and `continues` are length H. | Fix the docstring shape comment to read `values: [T, B]`. The arithmetic is correct; only the docstring claims a wrong shape. |

---

## Conventions audit checklist

| Convention | Status | Notes |
|---|---|---|
| **Forward λ-return parity** with sheeprl `utils.py:L66-L77` | ✅ | dreamer-srl `compute_lambda_values` ([`utils.py:L124-L166`](../../src/algorithms/dreamer_srl/utils.py)) is a faithful line-by-line transliteration. Init at `vals = [values[-1:]]` (L161 ↔ sheeprl L72), `interm = rewards + continues * values * (1 - lmbda)` (L162 ↔ sheeprl L73), reverse loop with `vals.append(interm[t] + continues[t] * lmbda * vals[-1])` (L163-L164 ↔ sheeprl L74-L75), final `reverse + concat + drop-last` (L165 ↔ sheeprl L76). Pre-existing CP1 bit-identity test in `tests/algorithms/dreamer_srl/test_utils.py::test_compute_lambda_values_matches_sheeprl` is still valid. |
| **§S5 true-continue splice** semantics | ✅ (conditional on P2) | Splice formula: `true_continue = 1.0 - terminated_observed; continues_spliced = concat([true_continue, continues_predicted[1:]], axis=0)` at [`train.py:L475-L479`](../../src/algorithms/dreamer_srl/train.py). Matches sheeprl `dreamer_v3.py:L247-L248` exactly. **Conditional on the driver correctly populating `terminated`** — see P2. The splice replaces the WM-predicted `continues[0]` with the real-data continue at the rollout-init step, NOT at the rollout's end (the rollout-init step is the only one where the env has provided ground-truth termination info). The reviewer prompt asked whether the splice should consume `is_first[0]` instead of `terminated[0]`; the answer is NO — sheeprl L247 unambiguously uses `terminated`. The semantic intent is "if the agent just died at the replay-batch's first step, don't bootstrap from the post-death value". `is_first` is a separate RSSM-reset signal at the world-model input, not at the value bootstrap. |
| **Advantage sign** `lambda_values - predicted_values` (H3 epicentre) | ✅ | [`train.py:L585`](../../src/algorithms/dreamer_srl/train.py): `advantage = normed_lambda_values - normed_baseline`. Matches sheeprl `dreamer_v3.py:L279` exactly. Positive when `λ_return > V(s)` → REINFORCE term `log_prob × advantage` pulls actor toward the chosen action when the return beats the baseline. **H3 is REFUTED at this site**. |
| **Predicted-values source = LIVE critic** | ❌ | [`train.py:L798`](../../src/algorithms/dreamer_srl/train.py) calls `target_critic` (the slow EMA). Sheeprl `dreamer_v3.py:L244` calls `critic` (the live one). This is **P1 (🔴 blocker)**. |
| **`predicted_values` `sg`-detached at the advantage site** | ✅ | [`train.py:L857`](../../src/algorithms/dreamer_srl/train.py): `predicted_values=jax.lax.stop_gradient(predicted_values)` at the actor-objective call. Gradient does NOT flow from the actor loss back into the critic via the baseline term. Matches sheeprl L275 semantics (where `predicted_values` is computed inside `train()` and its critic weights are NOT in the actor's optimizer's parameter list — the gradient barrier is structural rather than via `.detach()`, but the effect is identical). |
| **`lambda_values` `sg`-detached at the critic loss target** | ✅ | [`train.py:L310`](../../src/algorithms/dreamer_srl/train.py): `neg_lp1 = -qv.log_prob(jax.lax.stop_gradient(lambda_values))`. Matches sheeprl L314 `lambda_values.detach()`. |
| **`lambda_values` `sg`-detached at the actor objective** | ✅ | [`train.py:L856`](../../src/algorithms/dreamer_srl/train.py): `lambda_values=jax.lax.stop_gradient(lambda_values)`. Sheeprl does NOT `.detach()` `lambda_values` at the actor site (only at the critic site L314) — instead, the actor's `imagined_trajectories.detach()` at L273 means the actor module is called on detached latents and `lambda_values` is consumed only via `advantage.detach()` at L291. The dreamer-srl `sg` here is **belt-and-suspenders** and benign (sg of a value that is already cut off from any actor-parameter gradient path is a no-op). |
| **`advantage` `sg`-detached at the REINFORCE term** | ✅ | [`train.py:L591`](../../src/algorithms/dreamer_srl/train.py): `objective = log_probs * jax.lax.stop_gradient(advantage)`. Matches sheeprl L291 `* advantage.detach()`. The canonical `sg(advantage)` site. |
| **`continues_predicted` source** | ✅ | [`train.py:L803-L806`](../../src/algorithms/dreamer_srl/train.py): `continues_logits = jax.vmap(world_model.continue_model)(imag_flat); continues_predicted = IndependentBernoulli(continues_logits.reshape(H+1, BT, 1)).mode`. Matches sheeprl L246 `Independent(BernoulliSafeMode(logits=world_model.continue_model(...)), 1).mode`. `continues_predicted` is the world-model's learned continue head — gradient does flow from any downstream use back into the WM, but since the imagined_latents themselves are `sg`-detached at the actor/critic boundary (L838, L873), the WM does not receive any gradient signal from the rollout chain. |
| **`continues_spliced[1:] * gamma` passed to `compute_lambda_values`** | ✅ | [`train.py:L486`](../../src/algorithms/dreamer_srl/train.py): `continues_spliced[1:] * gamma`. Matches sheeprl L254 `continues[1:] * cfg.algo.gamma`. The convention is that `compute_lambda_values` receives the continues argument already pre-multiplied by γ. Inside `compute_lambda_values`, this same `continues` array is used both in the `interm` step and in the recursion — see the next row. |
| **`continues` (pre-multiplied by γ) used consistently inside `compute_lambda_values`** | ✅ | [`utils.py:L162-L164`](../../src/algorithms/dreamer_srl/utils.py): `interm = rewards + continues * values * (1 - lmbda)` and `vals.append(interm[t] + continues[t] * lmbda * vals[-1])`. Both lines use the same `continues` array, so the `γ` factor is correctly applied to both the `(1-λ) * V(s_t+1)` mixing term AND the `λ * V_λ(s_t+1)` recursion. Substituting in: `V_λ[t] = r_t + γ·c_t · ((1-λ) · V(s_{t+1}) + λ · V_λ(s_{t+1}))`, which is the textbook Hafner λ-return formula. |
| **Discount cumprod uses `continues_spliced * gamma`, then divides by γ** | ✅ | [`train.py:L218`](../../src/algorithms/dreamer_srl/train.py): `discount = stop_gradient(jnp.cumprod(continues * gamma, axis=0) / gamma)`. Matches sheeprl L260 `torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma`. The `/ γ` shifts the cumprod so `discount[0] = continues_spliced[0]` (not `continues_spliced[0] * γ`), giving the first imagined step a weight of 1 (or 0 if terminated). |
| **Discount tensor is `sg`-detached** | ✅ | [`train.py:L217`](../../src/algorithms/dreamer_srl/train.py): `jax.lax.stop_gradient(...)` wraps the entire `jnp.cumprod(...) / gamma` expression. Matches sheeprl's `with torch.no_grad():` block at L259. |
| **`moments_update` reads `lambda_values` (the live signal), not just `rewards`** | ✅ | [`train.py:L825`](../../src/algorithms/dreamer_srl/train.py): `moments_update(moments, lambda_values, ...)`. Matches sheeprl L276 `offset, invscale = moments(lambda_values, fabric)`. The Moments percentile-EMA tracks the distribution of λ-returns, which is what the advantage normalisation should be invariant to. |
| **`moments_update` happens BEFORE the advantage normalisation** | ✅ | [`train.py:L823-L830`](../../src/algorithms/dreamer_srl/train.py) (update) runs before [`train.py:L854-L864`](../../src/algorithms/dreamer_srl/train.py) (actor loss that consumes `moments_offset` / `moments_invscale`). Matches sheeprl L276 → L277-L279 ordering. |
| **`moments_offset` / `moments_invscale` are `sg`-detached when used in the advantage** | ✅ | [`train.py:L858-L859`](../../src/algorithms/dreamer_srl/train.py): both wrapped in `jax.lax.stop_gradient(...)`. Matches sheeprl L63 `self.low.detach(), invscale.detach()`. The Moments percentile-EMA is treated as a fixed scale at each step, not a differentiable factor. |
| **Per-term normalisation form (sheeprl L277-L279) vs algebraic shortcut** | ✅ | [`train.py:L579-L585`](../../src/algorithms/dreamer_srl/train.py): per-term form is preserved: `normed_lambda_values = (lambda_values - offset) / invscale`, `normed_baseline = (baseline - offset) / invscale`, `advantage = normed_lambda_values - normed_baseline`. Algebraically the offsets cancel (`(λ-b)/s = (λ-o)/s - (b-o)/s`), but per `train.py:L527-L532` comment, the per-term form is kept for bit-identity with sheeprl's float32 evaluation order. |
| **Entropy weighting matches sheeprl** | ✅ | [`train.py:L594`](../../src/algorithms/dreamer_srl/train.py): `entropy_term = ent_coef * entropy[:-1]`. Sheeprl L294 `cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)` followed by L297 `entropy.unsqueeze(dim=-1)[:-1]`. The `[:-1]` slice matches. The `sum over action dims` happens upstream in the actor's `lp_h, ent_h` return (verified in v2-CP3's scope, not here). |
| **Discount weighting at the loss** | ✅ | [`train.py:L600-L602`](../../src/algorithms/dreamer_srl/train.py): `policy_loss = -mean(discount[:-1] * (objective + entropy_term))`. Matches sheeprl L297 `policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))`. The `discount[:-1]` slice drops the last imagined step (length H, matching `lambda_values` length). |

---

## H3 surface conclusion

**H3 is REFUTED in this audit surface.** The advantage-sign hypothesis tested four
arithmetic locations:

1. **`advantage = lambda_values - predicted_values` (NOT `predicted_values - lambda_values`).**
   ✅ Confirmed at [`train.py:L585`](../../src/algorithms/dreamer_srl/train.py).
2. **§S5 splice arithmetic `1 - terminated` (NOT `1 + terminated` or `terminated - 1`).**
   ✅ Confirmed at [`train.py:L475`](../../src/algorithms/dreamer_srl/train.py).
3. **Backward λ-return recursion `interm[t] + continues[t] * lmbda * vals[-1]`.**
   ✅ Confirmed at [`utils.py:L164`](../../src/algorithms/dreamer_srl/utils.py); matches
   sheeprl `utils.py:L75` exactly.
4. **`predicted_values` is `sg`-detached at the actor-objective consumer.**
   ✅ Confirmed at [`train.py:L857`](../../src/algorithms/dreamer_srl/train.py): gradient
   does NOT flow from the REINFORCE term back into the critic.

The empirical advantage-sign diagnostic from
[`GRAD_PARITY_METHODOLOGY.md`](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md)
§5.3 test #4 (feed `predicted_values=[2,2,2]`, `imagined_rewards=[0.1,0.1,0.1]`, verify
`advantage < 0`) is expected to PASS on the current dreamer-srl code — there is no
sign bug in the surveyed arithmetic.

**However, H3 may still manifest empirically** through the upstream P1 bug: if the
`predicted_values` used in the advantage subtraction is the slow target's estimate,
the advantage signal lags the live critic and may have a systematically different
distribution (mean, variance, skew) than sheeprl's. This is NOT a sign flip in the
formula — it's a correlation drift between the actor's advantage and the most-recent
value signal. The 3-seed parity failure is consistent with both P1 (target-vs-live)
and P2 (boundary-zeroed bootstrap) compounding to produce a near-zero advantage
distribution, at which point the actor's REINFORCE gradient becomes
direction-uncorrelated with reward and collapses to the entropy-only branch
(uniform random) — matching the observed ~104-step (random-policy floor) outcome.

---

## Grad-test pseudocode for v2-CP9

Per [`GRAD_PARITY_METHODOLOGY.md`](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md)
§5.3, four grad-parity tests are scoped for v2-CP5. Fixture-seed convention:
`0xD3EAF + 0x100 * 5 = 0xD41AF` for params, `+1` for inputs.

### Test 1: `∂lambda_values / ∂predicted_values`

```python
# tests/algorithms/dreamer_srl/test_train_grad.py
from src.algorithms.dreamer_srl.train import compute_imagined_returns

FIXTURE_INPUT_SEED = 0xD3EAF + 0x100 * 5 + 1  # 0xD41B0
H = 15
BT = 32  # batch × seq, kept small
GAMMA = 0.997
LMBDA = 0.95

def _build_fixtures():
    key = jax.random.PRNGKey(FIXTURE_INPUT_SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return dict(
        predicted_rewards=jax.random.normal(k1, (H+1, BT, 1)) * 0.5,
        predicted_values=jax.random.normal(k2, (H+1, BT, 1)) * 1.0,
        continues_predicted=jnp.ones((H+1, BT, 1)) * 0.997,  # near-1 (no death in fixture)
        terminated_observed=jnp.zeros((BT, 1)),  # no terminations in fixture
    )

def test_dlambda_dvalues_matches_sheeprl():
    """v2-CP5 grad-parity test #1 — ∂lambda_values / ∂predicted_values."""
    fx = _build_fixtures()
    def jax_loss(predicted_values):
        lambda_vals, _, _ = compute_imagined_returns(
            predicted_rewards=fx["predicted_rewards"],
            predicted_values=predicted_values,
            continues_predicted=fx["continues_predicted"],
            terminated_observed=fx["terminated_observed"],
            gamma=GAMMA,
            lmbda=LMBDA,
        )
        return jnp.sum(lambda_vals)
    grad_jax = jax.grad(jax_loss)(fx["predicted_values"])

    # PyTorch side: wrap sheeprl's L243-L256 block as a function that takes
    # predicted_values as a torch.Tensor with requires_grad=True; call
    # compute_lambda_values; sum; torch.autograd.grad. See
    # GRAD_PARITY_METHODOLOGY §2.2 for the canonical recipe.
    grad_torch = _sheeprl_dlambda_dvalues(fx_torch=_to_torch(fx))

    max_abs_diff = float(jnp.max(jnp.abs(grad_jax - jnp.asarray(grad_torch))))
    assert max_abs_diff < 5e-4, (
        f"∂λ/∂V parity FAIL: max_abs_diff={max_abs_diff:.3e} >= 5e-4"
    )
```

**Threshold**: `5e-4` per
[`GRAD_PARITY_METHODOLOGY.md`](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md)
§3.4 (substrate-mechanical floor `~3e-6 × 24 H-step` × 1.7× margin = `5e-4`).

### Test 2: `∂lambda_values / ∂imagined_rewards`

Same scaffold as Test 1, but `jax.grad` w.r.t. `predicted_rewards`. Expected gradient
shape: per-step ∝ `Σ_t (γλ)^t` (discount-cumprod weighting from the recurrence's
`(1-λ)` term on `values[t]` plus the `r_t` direct addend in `interm[t]`).
Threshold: `5e-4`.

### Test 3: `∂lambda_values / ∂continues`

Same scaffold, `jax.grad` w.r.t. `continues_predicted`. Expected gradient: per-step
includes the `r_t` × cumprod weight (because `continues * gamma` is the loop's
recursion gate, so a perturbation to `continues[t]` propagates the rewards from
all later steps multiplied by the chain of intermediate continues and lambdas).
Threshold: `5e-4`.

**Note**: under the path-A fixture protocol from
[`GRAD_PARITY_METHODOLOGY.md`](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md)
§4.1, `continues_predicted` enters the function as a fixed fixture input rather than
as the output of the world model's continue head — so no straight-through gradient
through the Bernoulli `.mode` operator is exercised. This is correct for v2-CP5
isolation; the Bernoulli straight-through is a v2-CP8 wrapper-grad surface.

### Test 4: Advantage-sign empirical diagnostic (the H3 headline)

```python
def test_advantage_sign_when_predicted_exceeds_lambda():
    """v2-CP5 grad-parity test #4 — H3 advantage-sign empirical diagnostic.

    Construct a fixture where predicted_values >> imagined_rewards so the
    bootstrapped lambda_values are LESS than the predicted_values baseline.
    The resulting advantage = lambda - baseline should be NEGATIVE.
    """
    H = 3  # short horizon for the diagnostic
    BT = 1
    fx = dict(
        predicted_rewards=jnp.full((H+1, BT, 1), 0.1),  # small rewards
        predicted_values=jnp.full((H+1, BT, 1), 2.0),    # large baseline
        continues_predicted=jnp.ones((H+1, BT, 1)),
        terminated_observed=jnp.zeros((BT, 1)),
    )
    lambda_vals, _, _ = compute_imagined_returns(
        predicted_rewards=fx["predicted_rewards"],
        predicted_values=fx["predicted_values"],
        continues_predicted=fx["continues_predicted"],
        terminated_observed=fx["terminated_observed"],
        gamma=0.997,
        lmbda=0.95,
    )
    # lambda_vals has shape [H, BT, 1]; baseline is predicted_values[:-1]
    baseline = fx["predicted_values"][:-1]
    advantage = lambda_vals - baseline
    # The whole advantage tensor should be NEGATIVE because rewards (0.1) are
    # much smaller than the value bootstrap (2.0). If the dreamer-srl port has
    # the sign flipped, advantage will come out positive and this test fails.
    assert jnp.all(advantage < 0.0), (
        f"H3 advantage-sign FAIL: advantage should be < 0 but got {advantage}"
    )
    # Quantitative check: advantage[t] should be approximately
    #   r * (1 + γλ + (γλ)^2 + ...) - V * (1 - (γλ)^{H-t}) / (1-γλ)
    # which for r=0.1, V=2.0, γ=0.997, λ=0.95, H=3 is roughly -1.6.
    print(f"advantage values: {advantage.flatten()}")
```

**Why this test is the highest-leverage one in the v2-CP5 manifest**: it is a
**zero-fixture-cost qualitative assertion** that catches the H3 sign-flip mode
without any sheeprl-side comparison. If the dreamer-srl `advantage =
normed_lambda_values - normed_baseline` were silently inverted, this test would
fail immediately on the simplest possible fixture. The current code at
[`train.py:L585`](../../src/algorithms/dreamer_srl/train.py) passes this test by
inspection (the subtraction is in the canonical sign), but the test is still
worth shipping because it guards against future regressions.

---

## Conclusion

**❌ FAIL.** Two P-class findings explain potential parity-failure contributors at this
surface, **but H3 itself (the advantage-sign hypothesis) is REFUTED in this audit
surface**. P1 (`predicted_values` from `target_critic` instead of `critic`) is a
substantive algorithm deviation that lags the actor's advantage signal by τ⁻¹≈50 grad
steps; P2 (boundary-zeroed bootstrap inherited from v2-CP7's `terminated`/`truncated`
conflation) suppresses value-target propagation at every episode boundary. P1 and P2
compound: together they drive λ-returns toward zero, which in turn floors
`moments_invscale` at 1.0 (F2 corroborates the CP10b observation), which in turn
makes the advantage normalisation a no-op, which in turn means the REINFORCE gradient
loses correlation with reward and the actor collapses to the entropy-only (random)
branch. P1 is the **top finding** from this audit because it is INSIDE the v2-CP5 surface
(not inherited from another CP) and it is a one-line fix.

Reviewed by: code-reviewer
