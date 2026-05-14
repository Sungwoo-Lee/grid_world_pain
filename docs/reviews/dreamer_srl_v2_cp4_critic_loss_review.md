---
title: "v2-CP4 — `compute_critic_loss` forward + gradient re-audit (H3 hypothesis-locus)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
---

# v2-CP4 — `compute_critic_loss` forward + gradient re-audit

## Verdict (plain-language entry point)

**What this review is.** A fresh re-read of the dreamer-srl critic-loss function
([`src/algorithms/dreamer_srl/train.py:L227-L324`](../../src/algorithms/dreamer_srl/train.py))
against the canonical sheeprl reference
([`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L306-L316`](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py)),
done without trusting v1's CP6 verdict. The critic block is the **H3 hypothesis surface** in the
v2 plan — H3 says the v1 parity failure (3-seed run collapsed to the random-policy floor) could
be rooted in a sign or stop-gradient bug in the value-function loss that corrupts the
advantage signal the actor consumes. The experiment-analyzer rated H3 at MED-LOW after H1 (actor
REINFORCE) and H2 (driver `is_first` / terminated/truncated). v2-CP7 has since elevated H2 to
HIGH (P1 missing reset_data write, **P2 terminated/truncated conflation kills the value
bootstrap at every truncation**). The job of v2-CP4 is to ask: *does the critic-loss function
itself have a separate bug?* — meaning, if v2-CP9 fixes P2 in the driver, is the critic loss
still going to behave correctly?

**Headline verdict.** **✅ PASS — the function is a faithful sheeprl port; no semantic
deviation found.** The two-term form (cascade-fix-#29) is present, both stop-gradient sites are
correctly placed, the sign is correct (negative log-probability minimised against
detached targets), the discount weighting matches sheeprl's `[:-1].squeeze(-1)` shape exactly,
the reduction is `jnp.mean` over [H, BT] matching `torch.mean` over the same axes, and the
function does not compute or consume `advantage` (advantage lives in `compute_actor_objective`,
which is v2-CP3's surface).

H3 is therefore **NOT** rooted in `compute_critic_loss`. If H3 turns out to be a real
contributor to the parity failure, the bug lives somewhere else — most likely the
imagined-return chain in `compute_imagined_returns` (v2-CP5's surface), where the λ-return
sign and the advantage construction at `compute_actor_objective` interact. v2-CP4's only
deliverable beyond the PASS verdict is the **grad-test pseudocode + thresholds + seeds** for
v2-CP9 developer to implement, per
[`GRAD_PARITY_METHODOLOGY.md`](../develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md) §5.2.

One **F-finding (F1)** is flagged: the `target_critic_values` input is *expected* to arrive
already detached (the train-step orchestrator at `train.py:L878` constructs it from the
target-critic forward over a stop-gradient'd latent), but the function itself wraps it with a
second `stop_gradient` at L314. That's defensive and harmless — but worth a one-line audit
note for v2-CP6 because it means a future caller passing a non-detached `target_critic_values`
would still be safe by construction.

---

## Findings table

| ID | Severity | Site (dreamer-srl) | Site (sheeprl) | Issue | Suggested fix |
|---|---|---|---|---|---|
| (none P) | — | — | — | No P-class findings. The function ports sheeprl faithfully. | — |
| **F1** | 🟢 nit | `train.py:L314` | `dreamer_v3.py:L315` | **Double stop_gradient on target_critic_values.** The caller at `train.py:L878` computes `target_critic_values = TwoHotEncoding(target_critic_logits, dims=1).mean` where `target_critic_logits` was computed by `jax.vmap(target_critic)(sg_latents_h.reshape(...))` — i.e. the input latents are already stop_gradient'd but the target-critic params themselves are differentiable in the JAX graph (they're not separately detached). Sheeprl's reference uses a separate `target_critic` *module* (with its own param tree), and the gradient never flows there in the first place because the autograd graph is module-scoped — the critic optimizer only steps the online critic. dreamer-srl has the same module-scoping (`critic_loss_fn(critic_module)` only differentiates the online critic). So the L314 `stop_gradient(target_critic_values)` is **redundant** — but it is also **correct** and matches sheeprl's L315 `.detach()` belt-and-braces pattern. | No action required. Document the redundancy in the function's docstring for the next reviewer. |
| **F2** | 🟢 nit | `train.py:L322` | `dreamer_v3.py:L316` | **`jnp.mean` reduces over both axes implicitly.** sheeprl's `torch.mean(value_loss * discount[:-1].squeeze(-1))` reduces over all dims (both `H` and `BT`). dreamer-srl's `jnp.mean((neg_lp1 + neg_lp2) * discount_weights)` does the same. **This is correct** — both reduce over all dims because `(H, BT)` has 2 dims and neither `torch.mean` nor `jnp.mean` is given an `axis` argument. Flagged as a nit because the bit-identity test (`test_critic_loss_two_terms`) should explicitly verify the shape of the unreduced product is `[H, BT]` before the mean, so a future shape regression (e.g. accidentally introducing a trailing `1` axis from a future change to `TwoHotEncoding.log_prob`) doesn't silently change the reduction's denominator. | No action required for v2-CP4 PASS. v2-CP9 grad-test #1 (below) covers the shape verification. |

---

## §S Audit checklist (Critic-loss-specific)

| Convention | Status | Evidence |
|---|---|---|
| **Two-term form (cascade-fix-#29)** present | ✅ | `train.py:L310, L314`: `neg_lp1 = -qv.log_prob(sg(lambda_values))`; `neg_lp2 = -qv.log_prob(sg(target_critic_values))`. Matches sheeprl `dreamer_v3.py:L314-L315`. |
| **`sg(lambda_values)`** wrapping the first NLL term | ✅ | `train.py:L310`: `jax.lax.stop_gradient(lambda_values)` inside `qv.log_prob(...)`. Sheeprl `L314`: `lambda_values.detach()`. |
| **`sg(target_critic_values)`** wrapping the second NLL term | ✅ | `train.py:L314`: `jax.lax.stop_gradient(target_critic_values)`. Sheeprl `L315`: `predicted_target_values.detach()`. (Per F1, this is defensive — caller already detached.) |
| **Sign**: `value_loss = -(log_prob(λ) + log_prob(target))` (minimising NLL maximises log-prob) | ✅ | `train.py:L310, L314`: both `neg_lpN` are prefixed with `-`; summed at L322 inside `jnp.mean(...)`. Sheeprl `L314-L316`: same. **No sign flip.** |
| Gradient flows ONLY through `qv` (the online critic's predicted-value distribution) | ✅ | `qv = TwoHotEncoding(qv_logits, ...)` at L306; `qv_logits` is the only un-detached argument to the function. The sg on both target inputs blocks every other path. |
| Live critic is NOT called on target features | ✅ | The function takes `target_critic_values` as a **pre-computed input**; no live-critic re-evaluation on target features happens inside this function. (Polyak target-critic forward happens at `train.py:L875-L878` in the train-step orchestrator, OUTSIDE `compute_critic_loss`. Verified separately under v2-CP6/CP7.) |
| **Discount weighting (§S6)** correctly broadcast | ✅ | `train.py:L318`: `discount_weights = discount[:-1].squeeze(-1)`. `discount` shape is `[H+1, BT, 1]` per `compute_discount`'s contract; `[:-1]` → `[H, BT, 1]`; `.squeeze(-1)` → `[H, BT]`. Matches `neg_lp1` and `neg_lp2` shape `[H, BT]` exactly. Sheeprl `L316`: same. |
| **Reduction**: `mean` over horizon × batch | ✅ | `train.py:L322`: `jnp.mean((neg_lp1 + neg_lp2) * discount_weights)` — reduces over both axes (no `axis=` arg). Sheeprl `L316`: `torch.mean(value_loss * discount[:-1].squeeze(-1))` — same. |
| **Discount is already detached** (no gradient into continues) | ✅ | `compute_discount` at `train.py:L217-L219` wraps the cumprod in `jax.lax.stop_gradient`. The caller passes the detached tensor. Sheeprl `L259`: same via `with torch.no_grad():`. |
| **No advantage computation or consumption** inside `compute_critic_loss` | ✅ | `train.py:L227-L324`: the function signature takes `qv_logits, lambda_values, target_critic_values, discount` — no `advantage` argument. The function body computes `neg_lp1, neg_lp2, value_loss` only — no advantage construction or normalisation. (Advantage lives in `compute_actor_objective` at `train.py:L500-L604`, v2-CP3's surface.) |
| **H3 cross-check — sign of advantage is NOT this function's surface** | ✅ | Confirmed: `compute_critic_loss` is gradient-symmetric in the two target terms (both negative log-probs of detached scalars). If H3 is a real bug, it lives in `compute_imagined_returns` (the λ-return sign) or in `compute_actor_objective` (the `normed_lambda - normed_baseline` direction). v2-CP5's GRAD_PARITY §5.3 test #4 is the canonical H3 diagnostic. |
| **Two-hot bin grid in SYMLOG space** (cascade-fix-#2 / triple-consistency) | ✅ | `loss.py:L49-L100` and `loss.py:L157-L246` confirm the bin grid is `jnp.linspace(-20, 20, 255)` stored in symlog space; targets are symlog-encoded inside `log_prob`. The historical cascade bug (#2 — bin grid in real space) is not present. |
| **dims=1 reduces the trailing event dim** correctly in `TwoHotEncoding.log_prob` | ✅ | `loss.py:L245`: `return (target * log_pred).sum(axis=self.dims)`. With `dims=1` on a target of shape `[H, BT, 1]` and a 255-bin distribution, the trailing event dim (size 1) is summed out → `log_prob` returns `[H, BT]`. Matches sheeprl `TwoHotEncodingDistribution` with `dims=1`. |

All checklist items pass. **No P-finding; no F-finding requiring action.**

---

## v2-CP9 grad-test manifest (per `GRAD_PARITY_METHODOLOGY.md` §5.2)

The four sites below are the v2-CP4 deliverable for the developer to implement under v2-CP9.
Per `GRAD_PARITY_METHODOLOGY.md` §5.2 (the v2-CP4 row of the manifest), the fixture-seed
convention is `0xD3EAF + 0x100 * 4 = 0xD40AF` for params and `+1` for inputs. No path-A action
draw is needed (the critic loss does not consume actions).

### Test 1 — `∂value_loss / ∂(critic params)` two-term grad (normal gradient)

**Purpose**: verify the two-term loss produces a non-zero, sheeprl-bit-identical gradient
through the online critic's MLP weights. This is the headline grad-parity test for the
function.

**Sheeprl reference**: `dreamer_v3.py:L307-L316`.

**Threshold**: `2e-3` (D-008 substrate-class band — critic MLP cascades through `mlp_layers`
Linear+LayerNorm+SiLU blocks; matches the `D-008 × 2.78` margin per `GRAD_PARITY_METHODOLOGY.md` §3.4).

**Fixture**:
- Param seed: `0xD40AF` (= 866095 decimal)
- Input seed: `0xD40AF + 1` = `0xD40B0`
- Fixture path: `tests/fixtures/dreamer_srl/grad_cp4_critic_weights.npz`
- Shapes (XS recipe): `H=15`, `BT = batch_size × seq_len = 16 × 64 = 1024`
  - `qv_logits`: `[15, 1024, 255]` standard-normal at scale 1.0 (matches typical post-LayerNorm logits)
  - `lambda_values`: `[15, 1024, 1]` uniform in `[-2.0, 2.0]` (typical λ-return range post-symlog)
  - `target_critic_values`: `[15, 1024, 1]` uniform in `[-2.0, 2.0]`
  - `discount`: cumprod of `0.997 × continues` with `continues = ones([16, 1024, 1])` (all-non-terminal fixture so discount monotonically decreases from 1.0)

**Test pseudocode**:

```python
# tests/algorithms/dreamer_srl/test_loss_grad.py

FIXTURE_PARAM_SEED = 0xD40AF      # 866095
FIXTURE_INPUT_SEED = 0xD40AF + 1  # 866096


def test_critic_loss_grad_matches_sheeprl_weights():
    """v2-CP4 grad-test #1: ∂value_loss/∂(critic params) two-term."""
    params_jax, inputs_jax, params_torch, inputs_torch = _build_critic_fixtures(
        FIXTURE_PARAM_SEED, FIXTURE_INPUT_SEED
    )
    # inputs_jax contains: qv_logits, lambda_values, target_critic_values, discount
    # but qv_logits is computed from critic(latent) — so we feed in `latent` and
    # build `qv_logits` inside the loss closure to be differentiable w.r.t. params.

    # --- JAX side ---
    def jax_loss(critic_params):
        qv_logits = apply_critic_mlp(critic_params, inputs_jax["latent"])  # [H, BT, 255]
        value_loss, _, _ = compute_critic_loss(
            qv_logits=qv_logits,
            lambda_values=inputs_jax["lambda_values"],
            target_critic_values=inputs_jax["target_critic_values"],
            discount=inputs_jax["discount"],
        )
        return value_loss
    jax_grads = jax.grad(jax_loss)(params_jax)

    # --- PyTorch side ---
    # Run sheeprl's critic loss block on the same latent and target tensors
    qv_logits_torch = critic_torch_module(inputs_torch["latent"])
    qv = TwoHotEncodingDistribution(qv_logits_torch, dims=1)
    value_loss_torch = -qv.log_prob(inputs_torch["lambda_values"].detach())
    value_loss_torch = value_loss_torch - qv.log_prob(
        inputs_torch["target_critic_values"].detach()
    )
    value_loss_torch = torch.mean(
        value_loss_torch * inputs_torch["discount"][:-1].squeeze(-1)
    )
    torch_grads_tuple = torch.autograd.grad(
        value_loss_torch,
        list(critic_torch_module.parameters()),
        allow_unused=False,  # load-bearing
    )
    torch_grads = _name_torch_grads(torch_grads_tuple, critic_torch_module)

    # --- Compare leaf-by-leaf ---
    max_abs_diff = 0.0
    for name, jax_g in _flatten_pytree(jax_grads):
        torch_g = torch_grads[name].numpy()
        diff = float(np.max(np.abs(np.asarray(jax_g) - torch_g)))
        max_abs_diff = max(max_abs_diff, diff)

    threshold = 2e-3  # D-008 class
    assert max_abs_diff < threshold, (
        f"critic_loss grad-parity FAIL: max_abs_diff={max_abs_diff:.3e}"
    )
```

**Expected result**: PASS (well under `2e-3`; the critic MLP is 1-layer wide in the XS recipe).

### Test 2 — `∂value_loss / ∂lambda_values` sg-leak check (should be ~0)

**Purpose**: verify `sg(lambda_values)` at `train.py:L310` correctly blocks gradient from
flowing through the λ-return target into the critic loss. A failing test here means the
critic is being trained to AVOID the λ-return (gradient through λ from the negative log-prob
would adversarially push the critic away from the bootstrapped target).

**Sheeprl reference**: `dreamer_v3.py:L314` (`lambda_values.detach()`).

**Threshold**: `1e-7` (tight — a correctly placed sg produces exactly zero gradient; the only
allowed non-zero is sub-eps numerical noise).

**Fixture**: same as Test 1 (`grad_cp4_sg_lambda.npz`).

**Test pseudocode**:

```python
def test_critic_loss_lambda_values_is_sg_detached():
    """v2-CP4 grad-test #2: ∂value_loss/∂lambda_values should be ~0."""
    params_jax, inputs_jax, *_ = _build_critic_fixtures(
        FIXTURE_PARAM_SEED, FIXTURE_INPUT_SEED
    )

    # Build qv_logits at the fixture params (no params in the jax.grad)
    qv_logits = apply_critic_mlp(params_jax, inputs_jax["latent"])

    def jax_loss_wrt_lambda(lambda_vals):
        value_loss, _, _ = compute_critic_loss(
            qv_logits=qv_logits,
            lambda_values=lambda_vals,
            target_critic_values=inputs_jax["target_critic_values"],
            discount=inputs_jax["discount"],
        )
        return value_loss
    grad_wrt_lambda = jax.grad(jax_loss_wrt_lambda)(inputs_jax["lambda_values"])

    max_leak = float(jnp.max(jnp.abs(grad_wrt_lambda)))
    assert max_leak < 1e-7, (
        f"sg(lambda_values) leak: max |∂L/∂lambda| = {max_leak:.3e}"
    )
```

**Expected result**: PASS at exactly 0.0 (`jax.lax.stop_gradient` returns 0 gradient).

### Test 3 — `∂value_loss / ∂target_critic_values` sg-leak check (should be ~0)

**Purpose**: verify `sg(target_critic_values)` at `train.py:L314` correctly blocks gradient
from flowing through the EMA slow-target prediction. A failing test here means gradient leaks
backwards from the critic loss into the target critic chain, which would couple the online
critic's update to the target critic's evaluation noise — explicitly NOT what Hafner's
slow-target regularisation prescribes.

**Sheeprl reference**: `dreamer_v3.py:L315` (`predicted_target_values.detach()`).

**Threshold**: `1e-7` (tight; same rationale as Test 2).

**Fixture**: same as Test 1 (`grad_cp4_sg_target.npz`).

**Test pseudocode**:

```python
def test_critic_loss_target_values_is_sg_detached():
    """v2-CP4 grad-test #3: ∂value_loss/∂target_critic_values should be ~0."""
    params_jax, inputs_jax, *_ = _build_critic_fixtures(
        FIXTURE_PARAM_SEED, FIXTURE_INPUT_SEED
    )

    qv_logits = apply_critic_mlp(params_jax, inputs_jax["latent"])

    def jax_loss_wrt_target(target_vals):
        value_loss, _, _ = compute_critic_loss(
            qv_logits=qv_logits,
            lambda_values=inputs_jax["lambda_values"],
            target_critic_values=target_vals,
            discount=inputs_jax["discount"],
        )
        return value_loss
    grad_wrt_target = jax.grad(jax_loss_wrt_target)(
        inputs_jax["target_critic_values"]
    )

    max_leak = float(jnp.max(jnp.abs(grad_wrt_target)))
    assert max_leak < 1e-7, (
        f"sg(target_critic_values) leak: max |∂L/∂target| = {max_leak:.3e}"
    )
```

**Expected result**: PASS at exactly 0.0.

### Test 4 — Sign-consistency check (qualitative)

**Purpose**: verify the loss has the correct sign by construction — at a non-trivial fixture
the `value_loss` must be **POSITIVE** because it is the sum of two negative log-probabilities
(`log_prob ≤ 0` for any probability mass function, so `-log_prob ≥ 0`).

**Sheeprl reference**: `dreamer_v3.py:L314-L316`.

**Threshold**: qualitative — assert `value_loss > 0.0` for the fixture.

**Fixture**: same as Test 1 (`grad_cp4_sign_check.npz`).

**Test pseudocode**:

```python
def test_critic_loss_is_positive_under_nontrivial_inputs():
    """v2-CP4 grad-test #4: value_loss must be positive (negative log-prob)."""
    params_jax, inputs_jax, *_ = _build_critic_fixtures(
        FIXTURE_PARAM_SEED, FIXTURE_INPUT_SEED
    )

    qv_logits = apply_critic_mlp(params_jax, inputs_jax["latent"])
    value_loss, neg_lp1, neg_lp2 = compute_critic_loss(
        qv_logits=qv_logits,
        lambda_values=inputs_jax["lambda_values"],
        target_critic_values=inputs_jax["target_critic_values"],
        discount=inputs_jax["discount"],
    )
    assert float(value_loss) > 0.0, (
        f"value_loss should be positive (negative log-prob) but got {value_loss}"
    )
    # Stronger: each per-step neg_lp should also be non-negative
    assert float(jnp.min(neg_lp1)) >= 0.0
    assert float(jnp.min(neg_lp2)) >= 0.0
```

**Expected result**: PASS (`value_loss` is a positive scalar; both `neg_lp1` and `neg_lp2`
are element-wise non-negative because `log_prob ≤ 0` always).

**Note on the qualitative sign-flip hazard.** A flipped sign (`value_loss = +qv.log_prob(...)`)
would minimise a negative scalar — the critic would be trained to assign **low** probability
to the λ-return and target predictions. This sign-flip is **not present** in dreamer-srl's
port (verified by inspection at `train.py:L310, L314`). Test 4 is the regression guard so a
future refactor cannot silently introduce one.

---

## Forward-parity audit (verify v1's CP6 forward test still holds)

v1 shipped `tests/algorithms/dreamer_srl/test_train.py::test_critic_loss_two_terms` against the
fixture seed `0xD3EAF` (the v1 convention). Re-reading the production code at
`train.py:L227-L324` against sheeprl `dreamer_v3.py:L307-L316` line-by-line:

| Sheeprl line | Sheeprl operation | Dreamer-srl line | Dreamer-srl operation | Match |
|---|---|---|---|---|
| L307 | `qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)` | L306 | `qv = TwoHotEncoding(qv_logits, dims=1)` (caller passes the post-detach slice already) | ✅ |
| L308-L310 | `predicted_target_values = TwoHotEncodingDistribution(target_critic(imagined_trajectories.detach()[:-1]), dims=1).mean` | (caller, `train.py:L875-L878`) | `target_critic_values = TwoHotEncoding(target_critic_logits, dims=1).mean` on sg'd latents | ✅ |
| L314 | `value_loss = -qv.log_prob(lambda_values.detach())` | L310 | `neg_lp1 = -qv.log_prob(jax.lax.stop_gradient(lambda_values))` | ✅ |
| L315 | `value_loss = value_loss - qv.log_prob(predicted_target_values.detach())` | L314 | `neg_lp2 = -qv.log_prob(jax.lax.stop_gradient(target_critic_values))` (combined at L322 via `(neg_lp1 + neg_lp2)`) | ✅ |
| L316 | `value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))` | L318, L322 | `discount_weights = discount[:-1].squeeze(-1)`; `value_loss = jnp.mean((neg_lp1 + neg_lp2) * discount_weights)` | ✅ |

All 5 load-bearing lines port faithfully. The forward bit-identity test from v1 (`test_critic_loss_two_terms`) is still valid against sheeprl `33b6366` — no source-change between v1 and v2 invalidated it.

---

## Conclusion

**✅ PASS.** `compute_critic_loss` is a faithful sheeprl port. The two-term form is present,
both stop-gradient sites are correctly placed (`L310` and `L314`), the sign is correct (the
function minimises negative log-probability), the discount weighting matches sheeprl's
`[:-1].squeeze(-1)` shape exactly, and the reduction is `jnp.mean` over `[H, BT]` matching
`torch.mean` over the same axes. The function does not compute or consume `advantage` —
advantage lives in `compute_actor_objective` (v2-CP3's surface) and is constructed from
`lambda_values` and `predicted_values` in `compute_imagined_returns` (v2-CP5's surface).
**H3 is not rooted in this function.** If H3 turns out to be a real contributor to the v1
parity failure, the bug lives in the imagined-return / advantage chain — see v2-CP5's
GRAD_PARITY §5.3 test #4 for the canonical advantage-sign diagnostic.

Two F-class nits (F1 redundant target sg, F2 unreduced shape assertion) — neither requires
action; both documented for the next reviewer.

The v2-CP9 developer implementing the grad-parity tests should consume the manifest above
(4 tests, threshold `2e-3` for the weights grad, `1e-7` for the two sg-leak checks,
qualitative for the sign check). Fixture-seed convention: `0xD40AF` for params,
`0xD40AF + 1` for inputs. All fixtures land under
`tests/fixtures/dreamer_srl/grad_cp4_*.npz`. The test file is
`tests/algorithms/dreamer_srl/test_loss_grad.py` (shared with v2-CP3).

Reviewed by: code-reviewer
