---
title: "v2-CP2 — Gradient-parity test methodology spec"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
---

# v2-CP2 — Gradient-parity test methodology spec

> **Doc role.** This is the canonical recipe for **"given a sheeprl PyTorch
> function and a dreamer-srl JAX analog, how do you test that their *gradients*
> match within an approved deviation budget?"**. Every downstream v2 audit
> checkpoint (v2-CP3 actor, v2-CP4 critic, v2-CP5 imagined returns, v2-CP6 train
> orchestrator) consumes this spec when writing its new Lever-A grad tests.
> Authored by `professor-rl-bayesian-dl` per
> [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §2 Lever-A extension and
> v2-CP2 row. Cites
> [`SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md) (v2-CP1) for the
> sheeprl side, and v1's
> [`DEVIATION_LOG.md`](../dreamer_srl_v3/DEVIATION_LOG.md) for the substrate-class
> precedent that sets the threshold band.

---

## 1. Context (plain-language entry point)

**What gradient parity is.** A "gradient-parity test" asks: given a loss
function and a fixed set of inputs, does the **gradient** that JAX computes
match the **gradient** that PyTorch's sheeprl reference computes, to within an
allowed numerical band? **`jax.grad(loss)(params)`** is JAX's reverse-mode
differentiation entry point — it returns a pytree (a nested dict of arrays)
of partial derivatives of `loss` with respect to every leaf array in
`params`. **`torch.autograd.grad(loss, params)`** is PyTorch's analog — it
walks the **autograd graph** (the implicit DAG of tensor operations that
PyTorch records during the forward pass) and returns gradients with respect
to the requested tensors.

**Why it's distinct from forward parity.** A function can produce a
bit-identical *forward* output and still produce a wrong gradient. The two
classic ways this happens:

- A misplaced **`stop_gradient`** (in JAX, `jax.lax.stop_gradient`; in
  PyTorch, `.detach()`; abbreviated **`sg`** in this doc). The forward value
  of `sg(x)` is identical to `x`, so a forward test cannot detect a missing
  or extra `sg`. The backward gradient, however, differs: gradient flows
  through `x` but is blocked by `sg(x)`. Forward parity is blind to this.
- A **reparameterised sampler** mis-port. Hafner's actor uses
  `OneHotCategoricalStraightThrough`, where the forward sample is a discrete
  one-hot vector but the backward gradient is the **softmax Jacobian** of the
  pre-mix logits. The forward sample is even *intentionally* non-deterministic
  across PRNG streams — bit-identity is mathematically impossible — so the
  forward test cannot anchor the comparison; only the gradient can.
  Algorithms that use a **REINFORCE** estimator (the
  `log_prob(action) × advantage` policy-gradient term) or a **score-function**
  estimator are entirely defined by their backward gradients; the forward
  sample only carries the random draw.

**Why v1's forward-only discipline missed the bug.** v1's Lever-A tests
verified `compute_actor_objective`, `compute_critic_loss`, and
`compute_imagined_returns` forward outputs to `1e-6` of sheeprl on **fixture
seed** `0xD3EAF` (a fixed PRNG key for fixture-input generation, hex notation
chosen to be visually distinct from any random integer). They never compared
gradients. The leading H1 hypothesis — that the actor REINFORCE gradient is
mis-ported at the `unimix=0.01` mixture site, the entropy-bonus site, or a
`stop_gradient(advantage)` site — is therefore **invisible** to the v1 test
suite. v2 extends Lever-A from forward-only to forward-plus-gradient
bit-identity.

**What this spec produces.** §2 gives the test pattern as runnable
pseudocode. §3 derives the threshold band per gradient site, anchored to v1's
substrate-class deviation precedent (D-003 through D-010). §4 handles the
**cross-PRNG-stochastic** boundary classes from
[`SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md) §8 (the 9 sites
where PyTorch and JAX legitimately diverge at the bit level). §5 is the
per-CP test-site manifest for v2-CP3 / v2-CP4 / v2-CP5 / v2-CP6. §6 names the
test files and the diff-tool extension. §7 catalogs failure modes. §8 fixes
the execution order. §9 is honest about scope limits.

**Reading order for downstream CPs.** v2-CP3 reads §2, §3, §4.1 (fixed-sample
path A), §5.1. v2-CP4 reads §2, §3, §5.2. v2-CP5 reads §2, §3, §5.3. v2-CP6
reads §2, §3, §4.3 (Adam-state boundary), §5.4.

---

## 2. The test pattern

### 2.1 Generic gradient-parity test recipe

For any gradient-producing function `f(params, fixture_inputs) -> loss`, the
test is:

1. **Construct fixture inputs at a fixed PRNG seed.** Use seed `0xD3EAF` for
   model parameters and seed `0xD3EAF + 1` for input tensors (matching v1's
   diff-tool convention from
   [`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py)).
   New v2 grad fixtures use the convention `0xD3EAF + 0x100 * <cp_id>` for
   per-CP isolation per the v2 IMPLEMENTATION_PLAN. Fixture inputs are the
   things `f` consumes: parameter pytree, action one-hot tensor, observation
   tensor, hidden-state tensor, target tensor, advantage tensor, discount
   tensor — whatever the function's signature demands.
2. **Compute the PyTorch gradient** by running the sheeprl analog of `f` and
   calling `torch.autograd.grad`.
3. **Compute the JAX gradient** by running `f` and calling `jax.grad` (or
   `jax.value_and_grad` if the loss value is also needed for assertion
   bookkeeping).
4. **Compare element-wise** by `max_abs_diff = max(|jax_grads - torch_grads|)`
   over corresponding leaves of the pytree (and the same shape on the torch
   side).
5. **Assert** `max_abs_diff < threshold` where the threshold is set per §3.

### 2.2 Runnable pseudocode

The following is the canonical template. Future v2 grad tests copy-paste-adapt
this skeleton.

```python
# tests/algorithms/dreamer_srl/test_grad_parity.py (skeleton)

import jax
import jax.numpy as jnp
import numpy as np
import torch

from src.algorithms.dreamer_srl.train import compute_actor_objective
from vendor.sheeprl.sheeprl.algos.dreamer_v3 import dreamer_v3 as sheeprl_dv3

# v2-CP3 fixture seed convention
FIXTURE_PARAM_SEED = 0xD3EAF + 0x100 * 3   # 0xD3FAF
FIXTURE_INPUT_SEED = 0xD3EAF + 0x100 * 3 + 1


def _build_fixtures():
    """Construct parameter pytree + input tensors at fixed seeds.
    Returns (params_jax, inputs_jax, params_torch, inputs_torch).
    """
    # Build JAX side
    jax_key_params = jax.random.PRNGKey(FIXTURE_PARAM_SEED)
    jax_key_inputs = jax.random.PRNGKey(FIXTURE_INPUT_SEED)
    params_jax = init_actor_params(jax_key_params)  # mirrors sheeprl init
    inputs_jax = {
        "imagined_trajectories": jax.random.normal(jax_key_inputs, (H+1, B, D)),
        "imagined_actions":      _sample_actions_at(jax_key_inputs),  # see §4.1
        "advantage":             jax.random.normal(jax_key_inputs, (H, B, 1)),
        "discount":              jnp.cumprod(jnp.full((H+1, B, 1), 0.997), axis=0),
    }

    # Mirror onto torch side, parameter-by-parameter and tensor-by-tensor,
    # so the two sides see byte-identical inputs (modulo D-007 / D-008
    # substrate-class accumulation drift).
    params_torch = _jax_params_to_torch(params_jax)
    inputs_torch = {k: torch.from_numpy(np.asarray(v)).requires_grad_(False)
                    for k, v in inputs_jax.items()}
    # The "_action" tensor is the cross-PRNG boundary — see §4.1 path A
    return params_jax, inputs_jax, params_torch, inputs_torch


def test_actor_objective_grad_matches_sheeprl():
    """v2-CP3 grad-parity test #1 — ∂L_actor/∂(actor MLP weights) via REINFORCE.

    Threshold: 5e-5 (substrate-mechanical band, cf. v1 D-007/D-008/D-010
    precedent; see GRAD_PARITY_METHODOLOGY.md §3).
    """
    params_jax, inputs_jax, params_torch, inputs_torch = _build_fixtures()

    # --- JAX side ---
    def jax_loss(params):
        return compute_actor_objective(params, **inputs_jax).policy_loss
    jax_grads = jax.grad(jax_loss)(params_jax)

    # --- PyTorch side ---
    # Run sheeprl's actor objective block on the SAME inputs.
    # Convention: sheeprl_actor_objective_loss is a thin wrapper around
    # vendor/sheeprl/.../dreamer_v3.py:L272-L297 that returns the scalar
    # `policy_loss` without calling fabric.backward.
    policy_loss_torch = sheeprl_actor_objective_loss(
        actor_module=_load_torch_actor(params_torch),
        **inputs_torch,
    )
    torch_grads_tuple = torch.autograd.grad(
        policy_loss_torch,
        list(actor_module.parameters()),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,    # explicit: every param must be reached
    )
    torch_grads = _name_torch_grads(torch_grads_tuple, actor_module)

    # --- Compare leaf-by-leaf ---
    max_abs_diff = 0.0
    for name, jax_g in _flatten_pytree(jax_grads):
        torch_g = torch_grads[name].numpy()
        diff = float(np.max(np.abs(np.asarray(jax_g) - torch_g)))
        max_abs_diff = max(max_abs_diff, diff)

    # Threshold per §3
    threshold = 5e-5  # substrate-mechanical band; see §3 derivation
    assert max_abs_diff < threshold, (
        f"actor REINFORCE grad-parity FAIL: "
        f"max_abs_diff={max_abs_diff:.3e} >= threshold={threshold:.3e}"
    )
```

Two practical notes for whoever ports this:

- **`allow_unused=False`** is the load-bearing flag. If a parameter is
  reachable in the *forward* graph but unreachable from the *backward*
  graph — i.e. a `sg` placement somewhere along the chain blocks the gradient —
  PyTorch returns `None` for that param's grad. The default `allow_unused=True`
  silently substitutes a zero, masking the bug. v2 grad tests must pass
  `allow_unused=False` so a structural sg leak surfaces as an exception, not
  a silent pass.
- **`retain_graph=False`** keeps the test cheap; the graph is rebuilt per
  test. For tests that compare multiple grad-w.r.t. targets (e.g. grad w.r.t.
  params AND grad w.r.t. input advantage), use `retain_graph=True` on the
  first call and `False` on the last.

### 2.3 Comparing grad-w.r.t.-inputs (the sg verification pattern)

To verify a `sg` placement, take the gradient of the loss w.r.t. the INPUT
that should be detached, not the parameters. If `sg` is correctly placed,
this gradient should be **exactly zero**:

```python
def test_actor_objective_advantage_is_sg_detached():
    """v2-CP3 grad-parity test #4 — ∂L_actor/∂advantage should be zero.

    sheeprl: dreamer_v3.py:L291 `* advantage.detach()` — the canonical
    sg(advantage) site. If the dreamer-srl port drops the sg, gradient leaks
    backwards through advantage into the critic / lambda_values chain. This
    test asserts the leak does not happen.
    """
    params_jax, inputs_jax, *_ = _build_fixtures()

    # JAX grad w.r.t. the advantage input
    def jax_loss_wrt_advantage(advantage):
        new_inputs = dict(inputs_jax, advantage=advantage)
        return compute_actor_objective(params_jax, **new_inputs).policy_loss
    grad_wrt_advantage = jax.grad(jax_loss_wrt_advantage)(inputs_jax["advantage"])

    # Should be all zeros if sg is correctly placed
    max_leak = float(jnp.max(jnp.abs(grad_wrt_advantage)))
    assert max_leak < 1e-7, (
        f"sg(advantage) leak detected: max |∂L/∂advantage| = {max_leak:.3e}"
    )
```

This same pattern verifies `sg(action)` at `dreamer_v3.py:L286` (the
`p.log_prob(imgnd_act.detach())` site), `sg(lambda_values)` at
`dreamer_v3.py:L314`, and `sg(target_critic_values)` at
`dreamer_v3.py:L315`. The `1e-7` threshold is tight because a correctly-placed
sg produces **exactly zero** gradient — not a small float; the only allowed
non-zero is sub-eps numerical noise from float32 arithmetic in upstream graph
nodes, which is bounded by `~1e-8`.

---

## 3. Threshold-band derivation

Every gradient site gets a threshold band with two components: a
**substrate-class floor** (the 1-ULP-at-magnitude floor below which we cannot
detect drift even in principle) and a **cascade margin** (the multiplier
above the floor that absorbs accumulated rounding through the gradient
chain).

### 3.1 Substrate-class floor

For a forward output at magnitude `M_f` and a parameter at magnitude `M_p`,
the gradient `∂loss/∂param` has typical magnitude `M_f / M_p` (chain rule with
unit-modulus intermediate Jacobians as a baseline). In float32:

- 1 ULP at magnitude `M_g = M_f / M_p` is `M_g * 2^{-23} ≈ M_g * 1.19e-7`.
- For actor REINFORCE: `M_f ≈ |policy_loss| ≈ 1.0` (post `torch.mean`); `M_p ≈
  1.0` (LayerNorm output scale + zero-mean init). So `M_g ≈ 1.0` and the
  ULP floor is `~1.2e-7`.
- For critic two-hot log-prob: `M_f ≈ |value_loss| ≈ 5.5` (log-softmax across
  255 bins with target near bin 127, `log(softmax)[127] ≈ -log(255) ≈ -5.5`);
  `M_p ≈ 1.0`. ULP floor at gradient: `~6.5e-7`.
- For imagined-return λ-recurrence: `M_f ≈ |λ-return| ≈ 1.0`; gradient
  w.r.t. `predicted_values` has magnitude bounded by
  `Σ (γλ)^t ≤ 1/(1-γλ) ≈ 1/(1-0.997 × 0.95) ≈ 24` over a `horizon=15`
  rollout. So the per-step gradient floor is `~1.2e-7` but the accumulated
  floor over the chain is `~3e-6`.

### 3.2 Cascade margin (anchored to v1 precedent)

v1's substrate-class deviation log establishes an empirical margin band of
**1.5–2.8×** the floor for substrate-mechanical drift through deep arithmetic
chains. The precedent is:

| v1 deviation | Site | Floor | Margin | Approved threshold |
|---|---|---|---|---|
| **D-003** | `symexp` 1-ULP `exp` drift at `\|x\|=5` | `~1.5e-5` | **1.5×** | `2e-5` |
| **D-006** | `linspace` midpoint 1-ULP × two-hot log-prob cascade | `~1.81e-5` | **1.65×** | `3e-5` |
| **D-007** | Fused matmul + LayerNorm + tanh/sigmoid gate cascade | `~2.97e-4` | **1.68×** | `5e-4` |
| **D-008** | Two-matmul + LayerNorm + SiLU MLP stack | `~7.19e-4` | **2.78×** | `2e-3` |
| **D-010** | TwoHotEncoding log-prob with seed near bin boundary | `~3.10e-5` | **1.61×** | `5e-5` |

The math-reviewer's analytical witness for D-008 — a `sqrt(N)` random-walk
chain-depth scaling argument — predicts a **2.0–2.8×** ratio when chain depth
roughly doubles, and the observed ratios sit squarely in that band.

**For v2 gradient tests we adopt the same band:**

- **Site is a single matmul + activation** (e.g. actor MLP last-layer
  gradient): use **1.5–1.7× margin** above the substrate floor. Default
  threshold `5e-5` for actor params, matching D-010.
- **Site cascades through 2+ MLP miniblocks** (e.g. critic loss gradient
  back through the critic MLP stack of `mlp_layers=1`, post-LayerNorm,
  SiLU): use **2.0–2.8× margin**. Default threshold `2e-3` for critic
  params, matching D-008.
- **Site cascades through the horizon-H imagined recurrence** (e.g.
  `compute_imagined_returns` gradient w.r.t. `predicted_values`): use
  **2.8× margin**, but multiplied by the H-step recurrence factor
  `Σ (γλ)^t ≈ 24` for `horizon=15`, `γ=0.997`, `λ=0.95`. Default threshold
  `5e-3` for λ-return gradients.

### 3.3 Gradients vs forward thresholds — why they differ

A naive expectation is "gradient tests should use the same threshold as
forward tests". This is wrong in two opposite directions:

- **Gradients can be TIGHTER than forward.** The chain rule is *linear* in
  each Jacobian — `∂L/∂x_0 = J_n × J_{n-1} × ... × J_1`. Each Jacobian's
  error accumulates *predictably* (multiplicatively in magnitude, additively
  in 1-ULP relative terms). So if the forward `max_abs_diff` is `2e-3`
  through a depth-2 MLP, the gradient `max_abs_diff` is bounded by `~2e-3`
  if the Jacobians have unit modulus, NOT by `2 × 2e-3 = 4e-3`.
- **Gradients can be LOOSER than forward.** Gradients accumulate through
  *more* multiplications than the forward pass — the backward pass has the
  same depth but each backward Jacobian-vector product has its own ULP
  drift. For a forward `f = g(h(x))` with `h` and `g` each introducing 1 ULP
  drift, the forward output has `~2 ULP` accumulated drift, but the gradient
  `∂f/∂x = g'(h(x)) × h'(x)` introduces 4 ULPs: 1 from `h(x)`, 1 from
  `g'(h(x))`, 1 from `h'(x)`, 1 from the product. **Practical heuristic:
  gradient threshold = forward threshold × 1.5–2×** unless the site is
  dominated by a sg-detached input (in which case it's exactly zero, see
  §2.3).

The v2 grad-parity tests will adopt **forward × 1.5×** as the default
threshold and tighten or relax per the site-specific derivation in §5.

### 3.4 Threshold-band table (provisional, refined at each CP)

| Site | Substrate floor estimate | Margin | Threshold | Anchored to v1 deviation |
|---|---|---|---|---|
| Actor MLP last-layer weights (REINFORCE) | `~3e-5` | 1.65× | **`5e-5`** | D-006 / D-010 class |
| Actor MLP trunk weights (cascaded through LayerNorm + SiLU) | `~7e-4` | 2.8× | **`2e-3`** | D-008 class |
| Actor head pre-mix logits gradient | `~5e-5` | 1.5× | **`8e-5`** | D-007 / D-010 class |
| Entropy-only gradient w.r.t. actor params | `~3e-5` | 1.65× | **`5e-5`** | D-006 / D-010 class |
| `sg(advantage)` leak check (should be 0) | `~1e-8` | n/a | **`1e-7`** | tight |
| `sg(action)` leak check (should be 0) | `~1e-8` | n/a | **`1e-7`** | tight |
| Critic MLP weights (two-hot log-prob loss) | `~7e-4` | 2.8× | **`2e-3`** | D-008 class |
| `sg(lambda_values)` leak check | `~1e-8` | n/a | **`1e-7`** | tight |
| `sg(target_critic_values)` leak check | `~1e-8` | n/a | **`1e-7`** | tight |
| λ-return gradient w.r.t. `predicted_values` | `~3e-6` × 24 (H-step) | 1.7× | **`5e-4`** | D-006 × H-step scaling |
| λ-return gradient w.r.t. `imagined_rewards` | `~3e-6` × 24 | 1.7× | **`5e-4`** | D-006 × H-step scaling |
| Full train-step grad on actor params (end-to-end) | `~7e-4` | 2.8× | **`2e-3`** | D-008 class |
| Full train-step grad on critic params | `~7e-4` | 2.8× | **`2e-3`** | D-008 class |
| Full train-step grad on world-model params | `~7e-4` | 2.8× | **`3e-3`** | D-008 × 1.5 (deeper WM) |

These are **provisional** thresholds. Each downstream CP refines them
empirically: if a site's measured `max_abs_diff` is comfortably under the
provisional, the test ships at the provisional; if it's `~1–10×` over, the
threshold is raised inline as a new D-015+ deviation row per §7.

---

## 4. Cross-PRNG-stochastic gradient handling

[`SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md) §8 identified
**9 cross-substrate boundary classes** where PyTorch and JAX diverge at the
bit level. For gradient tests, the load-bearing ones are:

### 4.1 Categorical-sample boundary (path A vs path B)

**The boundary**: sheeprl's actor draws actions via
`OneHotCategoricalStraightThrough.rsample()` (`agent.py:L834`). The forward
sample is a discrete one-hot vector drawn from the post-`unimix`
probabilities; PyTorch's global RNG and JAX's explicit-key RNG draw
**different samples** even from identical logits. So `action_torch` and
`action_jax` are not equal — bit-identity is impossible.

The downstream consequence is that the actor REINFORCE term
`log_prob(action) × sg(advantage)` evaluates `log_prob` on different
arguments on the two sides, producing different forward losses and different
gradients. The gradient signal we want to test (the gradient flow through
the actor's logits) is **drowned out** by the action-sample mismatch.

**Two approaches**:

- **(A) Fix the SAMPLE, not the seed.** Pre-compute the one-hot action
  tensor *outside* the gradient test (e.g. by drawing from a numpy RNG once
  and saving as a fixture `.npz`) and pass it as a fixture input to BOTH
  sheeprl and dreamer-srl. Both sides receive the *same* action tensor; the
  gradient through `log_prob(action)` is then deterministic w.r.t. the logits
  and the test becomes a clean comparison of `∂(log_prob(a) × adv) / ∂logits`
  across the two substrates. **Trade-off**: the test no longer exercises the
  sampler itself — the straight-through gradient through the *forward
  sampling* operation is bypassed. But the sampler is structurally simple
  (`OneHotCategoricalStraightThrough.rsample` is bit-equivalent to
  `softmax(logits) - sg(softmax(logits)) + sg(one_hot(argmax(softmax(logits))))`
  in expectation, with the sample randomness orthogonal to the gradient
  path); the gradient flow it implements is what we actually want to test,
  and that flow goes through `log_prob`, not through the sample draw.

- **(B) Test the EXPECTED gradient.** For REINFORCE, the expected gradient
  is the **score function**: `E[∂ log p(a|θ) / ∂θ] × A` where the
  expectation is over `a ~ p(·|θ)`. Compute this analytically by
  enumerating over all `num_actions` (only ~5 in the food task), weighted
  by the probabilities `p(a|θ)`. The resulting expected gradient is
  exactly comparable across substrates (it's a deterministic function of
  the logits and the advantage). **Trade-off**: more involved to implement
  (requires writing a new closed-form-gradient routine that does not match
  any line in sheeprl), and only valid for small action spaces. For larger
  action spaces it's a Monte-Carlo estimate over many samples, which is
  itself stochastic.

**Recommendation for v2-CP3: Path A.** The path-of-least-resistance for an
empirical bug-hunt is to fix the sample and test `∂(log_prob(a_fixed) × adv) /
∂θ` deterministically. The theoretical purity of path B is not needed until
v2 ships the parity launch — at which point the *whole-training* gradient is
what matters, not the per-step expectation. If v2-CP3's path-A test passes
but v2-CP10 (the 20k-step policy-learning gate) fails, we can add path-B
expected-gradient tests as a deeper diagnostic in the v2-CP9 fix bundle.

**Concrete fixture protocol for path A:**

```python
def _sample_actions_at(jax_key):
    """Draw a fixed batch of one-hot action tensors from a numpy RNG so both
    sheeprl and dreamer-srl receive the same actions. The actions are
    used as `imagined_actions` input — the same tensor passed to both
    log_prob calls.
    """
    rng = np.random.RandomState(0xD3EAF + 0x100 * 3 + 2)  # path-A seed
    probs = np.asarray([0.18, 0.22, 0.20, 0.24, 0.16])    # uniform-ish
    indices = rng.choice(5, size=(H+1, B), p=probs)
    one_hot = np.eye(5)[indices]                          # shape (H+1, B, 5)
    return jnp.asarray(one_hot)
```

### 4.2 Straight-through gradient boundary

**The boundary**: `OneHotCategoricalStraightThrough.rsample()` returns a
one-hot vector whose **backward gradient** is the softmax Jacobian of the
pre-mix logits. PyTorch implements this via a custom autograd Function;
JAX needs `jax.custom_vjp` or equivalent. v1's port at
[`src/algorithms/dreamer_srl/agent.py`](../../../../src/algorithms/dreamer_srl/agent.py)
needs to be re-read against this exact gradient contract.

**Under path A** (fixed action), the straight-through sampler is **not
exercised** in the gradient test — the action is a fixture input, so no
gradient flows through the sample operation. The gradient flows through
`log_prob(fixed_action)`, which is just a categorical log-prob lookup
followed by a sum over the `num_actions` dimension; no straight-through is
involved. So path A bypasses this boundary entirely.

**If a separate test for the straight-through sampler is required** (it is
NOT required for v2-CP3; flagged here as a future task), the pattern is:

```python
def test_one_hot_categorical_straight_through_grad():
    """Verify that the straight-through gradient through rsample matches
    softmax(logits) Jacobian on the JAX side.
    """
    # JAX side
    def jax_sample_then_sum(logits):
        # one_hot sample with straight-through gradient
        sampled = one_hot_categorical_straight_through(logits, key)
        return sampled.sum()  # arbitrary scalar; the gradient is what we want
    grad_jax = jax.grad(jax_sample_then_sum)(logits_jax)

    # Analytical grad: ∂ sum(sample) / ∂ logits = ∂ sum(softmax) / ∂ logits
    # (under straight-through)
    softmax = jax.nn.softmax(logits_jax)
    grad_analytic = jax.grad(lambda l: jax.nn.softmax(l).sum())(logits_jax)

    assert jnp.max(jnp.abs(grad_jax - grad_analytic)) < 1e-7
```

This is a unit test on the JAX sampler implementation, NOT a parity test
against sheeprl. The sheeprl side's `OneHotCategoricalStraightThrough` is
PyTorch-builtin and treated as the reference; the JAX implementation must
match its gradient contract.

### 4.3 Adam-state boundary (out of scope for grad parity)

**The boundary**: `optax.adam(eps=1e-5)` and `torch.optim.Adam(eps=1e-5)`
differ at the eps placement inside the sqrt — both libraries place eps
inside (`update = lr * m_hat / (sqrt(v_hat) + eps)`) by default, but the
order of operations and the bias-correction interleaving differ.

**Critical scoping observation**: the gradient-parity test runs `jax.grad`
and `torch.autograd.grad` **directly** — it does NOT call
`optimizer.step()`. The optimizer-state evolution is a separate boundary
that v2-CP6 tests with its own optimizer-step equivalence test. So eps
placement is **NOT in scope** for v2-CP3 / v2-CP4 / v2-CP5 grad-parity
tests. It IS in scope for v2-CP6's full-step grad check (which composes
grad + apply_updates), where it is tested via two dedicated optimizer-step
parity tests per v2 `IMPLEMENTATION_PLAN.md` §3 row v2-CP6.

### 4.4 Other §8 boundaries — handling per gradient test

| §8 boundary class | Path-A handling |
|---|---|
| In-place vs functional state updates (D-001, D-011) | Outside gradient graph (Moments and Polyak are sg-detached); no impact on grad tests. |
| `torch.quantile` vs `jnp.quantile` | Inside `Moments.forward`, detached output; no gradient flow. |
| `F.softplus` / `F.silu` ULP drift (D-003 / D-006 / D-007 / D-008 / D-010) | Built into the threshold band per §3. |
| `probs_to_logits` numerical floor | Active only when `unimix=0`; with `unimix=0.01` the floor `unimix / num_classes ≈ 3.1e-4` is well above the tiny-clamp; no impact. |
| `torch.quantile` vs `jnp.quantile` | Same as above; detached. |
| `fabric.all_gather` (multi-GPU) | Single-device runs; no-op. |
| Categorical sampling RNG | Path A: fixed; no impact. |
| One-hot straight-through | Path A: bypassed; no impact. |
| Adam eps placement | Out of scope for grad parity; v2-CP6 owns. |

Under path A with detached non-gradient sites (Moments, polyak, discount,
quantile), the only ULP-class drift that propagates into the gradient
parity comparison is the **substrate-mechanical** drift through the actor /
critic / WM MLP stacks — exactly what §3's threshold band absorbs.

---

## 5. Sites to test (the v2-CP3/CP4/CP5/CP6 manifest)

Each row below names the **gradient quantity**, the **fixture seed**, the
**threshold band** (anchored to §3), and the **H-hypothesis** it would catch.
Fixture-input file paths follow the convention
`tests/fixtures/dreamer_srl/grad_<cp>_<site>.npz` and are generated by
`scripts/fixtures/gen_dreamer_srl_grad_fixtures.py` (a new helper added at
v2-CP2 to mirror the v1 fixture-generator pattern).

### 5.1 v2-CP3 — `compute_actor_objective` (4 grad sites)

Sheeprl reference: `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L262-L304`.

Fixture-seed convention: `0xD3EAF + 0x100 * 3 = 0xD3FAF` for params,
`0xD3FAF + 1` for inputs, `0xD3FAF + 2` for the path-A action draw.

| # | Gradient quantity | Fixture path | Threshold | H-catches |
|---|---|---|---|---|
| 1 | `∂L_actor / ∂(actor MLP last-layer weights)` via REINFORCE term, entropy term zeroed | `grad_cp3_reinforce_lastlayer.npz` | **`5e-5`** | H1 — mis-port at `unimix` mixture site OR at `probs_to_logits` conversion |
| 2 | `∂L_actor / ∂(actor MLP trunk weights)` via REINFORCE term (cascaded gradient) | `grad_cp3_reinforce_trunk.npz` | **`2e-3`** | H1 — mis-port at LayerNorm placement or SiLU sign |
| 3 | `∂L_actor / ∂(actor MLP weights)` via entropy term only (REINFORCE zeroed: advantage=0) | `grad_cp3_entropy_only.npz` | **`5e-5`** | H1 — `ent_coef` mis-port; entropy formula mis-port |
| 4 | `∂L_actor / ∂advantage` — should be exactly 0 (sg verification) | `grad_cp3_sg_advantage.npz` | **`1e-7`** | H1 — missing `sg(advantage)` at line `dreamer_v3.py:L291` |
| 5 | `∂L_actor / ∂imagined_actions` — should be exactly 0 (sg verification) | `grad_cp3_sg_action.npz` | **`1e-7`** | H1 — missing `imgnd_act.detach()` at `dreamer_v3.py:L286` |

Test file: `tests/algorithms/dreamer_srl/test_loss_grad.py` (new).
Sheeprl wrapper: `scripts/sheeprl_jax_diff.py` extension via
`register_grad_diff('compute_actor_objective', ...)`.

### 5.2 v2-CP4 — `compute_critic_loss` (3 grad sites)

Sheeprl reference: `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L306-L327`.

Fixture-seed convention: `0xD3EAF + 0x100 * 4 = 0xD40AF` for params, `+1`
inputs. No path-A action draw (the critic loss does not consume actions).

| # | Gradient quantity | Fixture path | Threshold | H-catches |
|---|---|---|---|---|
| 1 | `∂L_critic / ∂(critic MLP weights)` via the two-term loss (`-qv.log_prob(λ-return) - qv.log_prob(target_critic_pred)`) | `grad_cp4_critic_weights.npz` | **`2e-3`** | H3 — wrong sign in either term, or missing the target-critic term |
| 2 | `∂L_critic / ∂lambda_values` — should be exactly 0 (sg verification, line `L314`) | `grad_cp4_sg_lambda.npz` | **`1e-7`** | H3 — missing `sg(lambda_values)`; the canonical diagnostic for "advantage leaks gradient into critic" |
| 3 | `∂L_critic / ∂target_critic_values` — should be exactly 0 (sg verification, line `L315`) | `grad_cp4_sg_target.npz` | **`1e-7`** | H3 — missing `sg(predicted_target_values)` |
| 4 | Sign check: at fixture where `predicted_value > λ-return`, verify the gradient signs are consistent with `value_loss = -qv.log_prob(target)` minimisation (gradient pulls predicted_value DOWN toward target, not UP) | `grad_cp4_sign_check.npz` | (qualitative) | H3 — sign flip in either log-prob term |

Test file: `tests/algorithms/dreamer_srl/test_loss_grad.py` (same file as
CP3 above). Sheeprl wrapper: register via
`register_grad_diff('compute_critic_loss', ...)`.

### 5.3 v2-CP5 — `compute_imagined_returns` (3 grad sites)

Sheeprl reference: `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L202-L260`
and `utils.py:L66-L77` (`compute_lambda_values`).

Fixture-seed convention: `0xD3EAF + 0x100 * 5 = 0xD41AF` for params, `+1`
inputs.

| # | Gradient quantity | Fixture path | Threshold | H-catches |
|---|---|---|---|---|
| 1 | `∂λ_values / ∂predicted_values` — over the H-step backward recurrence | `grad_cp5_dlambda_dvalues.npz` | **`5e-4`** | sign / index mis-port in the `vals.append(interm[t] + continues[t] * lmbda * vals[-1])` recursion |
| 2 | `∂λ_values / ∂imagined_rewards` — should produce per-step gradients ∝ `Σ (γλ)^t` | `grad_cp5_dlambda_drewards.npz` | **`5e-4`** | sign / index mis-port in the `interm = rewards + continues * values * (1 - lmbda)` interm computation |
| 3 | `∂λ_values / ∂continues` — verifies the `continues * γ` factor handling | `grad_cp5_dlambda_dcontinues.npz` | **`5e-4`** | mis-port of the `continues[1:] * cfg.algo.gamma` boundary at the caller (`dreamer_v3.py:L254`) |
| 4 | **Advantage sign empirical check (the H3 diagnostic):** with fixture `predicted_values = jnp.array([2.0, 2.0, 2.0])` and fixture `imagined_rewards = jnp.array([0.1, 0.1, 0.1])`, the resulting `λ_values` should be `< 2.0` (because the rewards are too small to maintain the value bootstrap), and `advantage = λ_values - predicted_values` should be **NEGATIVE**. If the dreamer-srl port has the sign flipped, advantage comes out positive and the actor learns to AVOID the (bad) action. | `grad_cp5_advantage_sign.npz` | (qualitative — `advantage[t] < 0` for all `t`) | **H3 (the headline)** |

Test file: `tests/algorithms/dreamer_srl/test_train_grad.py` (new).
Sheeprl wrapper: `register_grad_diff('compute_imagined_returns', ...)`.

### 5.4 v2-CP6 — `one_train_step` orchestrator (3 grad sites + 2 optimizer-step)

Sheeprl reference: the composite `train` function at
`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358`.

Fixture-seed convention: `0xD3EAF + 0x100 * 6 = 0xD42AF` for params, `+1`
inputs.

| # | Gradient quantity | Fixture path | Threshold | H-catches |
|---|---|---|---|---|
| 1 | `∂total_loss / ∂(actor params)` after one full train-step composition (WM update + rollout + actor loss + critic loss combined) | `grad_cp6_full_actor.npz` | **`2e-3`** | spurious sg leak between modules; cross-module gradient interference |
| 2 | `∂total_loss / ∂(critic params)` | `grad_cp6_full_critic.npz` | **`2e-3`** | sg leak from rollout into critic; missed `imagined_trajectories.detach()` |
| 3 | `∂total_loss / ∂(world-model params)` | `grad_cp6_full_wm.npz` | **`3e-3`** | WM is the deepest stack — D-008 × 1.5; sg leak from actor / critic into WM |
| 4 | Optimizer-step parity #1: one `optax.adam.update + apply_updates` step on the actor params should produce post-step params within `5e-5` of `torch.optim.Adam.step()` on the same gradient. | `grad_cp6_adam_actor.npz` | **`5e-5`** | optimizer eps placement, bias-correction order |
| 5 | Optimizer-step parity #2: one `optax.adam` step on the critic params (deeper stack, larger update magnitudes) | `grad_cp6_adam_critic.npz` | **`5e-5`** | same as #4, with larger numerical surface |

Test file: `tests/algorithms/dreamer_srl/test_train_grad.py` (same file as
CP5). Sheeprl wrappers: 3 grad-diff registrations + 2 optimizer-step
parity tests written outside the `register_grad_diff` framework (the
optimizer step is a separate diff-tool path, see §6).

### 5.5 Manifest summary

| CP | Sites | Sg-leak checks | Threshold range |
|---|---|---|---|
| v2-CP3 | 5 (3 grad-w.r.t.-params + 2 sg-leak) | 2 | `5e-5` – `2e-3` (+ `1e-7` for sg) |
| v2-CP4 | 4 (1 grad-w.r.t.-params + 2 sg-leak + 1 sign) | 2 | `2e-3` (+ `1e-7` for sg) |
| v2-CP5 | 4 (3 grad-w.r.t.-inputs + 1 sign) | 0 | `5e-4` |
| v2-CP6 | 5 (3 grad-w.r.t.-params + 2 optimizer-step) | 0 | `2e-3` – `3e-3` (+ `5e-5` for optimizer) |
| **Total** | **18** | **4** | spread per row above |

This is the full v2-CP3–v2-CP6 grad-parity manifest. v2-CP7 (driver) does
not consume this spec — it's gradient-invariant tests, not grad-parity.
v2-CP8 (wrappers) uses the same threshold band but its tests are *structural*
(every param reached by the loss), not bit-identity.

---

## 6. Test-file naming and diff-tool extension

### 6.1 Test file layout

Single grad-parity test file under
`tests/algorithms/dreamer_srl/test_grad_parity.py` is **NOT** the
recommendation — splitting along the same lines as v1's forward tests is
cleaner. Per the v2 IMPLEMENTATION_PLAN §9 deliverables:

- `tests/algorithms/dreamer_srl/test_loss_grad.py` — CP3 + CP4 (actor and
  critic loss grads; ~8 tests).
- `tests/algorithms/dreamer_srl/test_train_grad.py` — CP5 + CP6 (imagined
  returns grads + train-step orchestrator + optimizer step; ~8 tests).
- `tests/algorithms/dreamer_srl/test_dreamer_srl_main_invariants.py` —
  CP7 driver invariants (not grad parity; out of this spec's scope).
- `tests/algorithms/dreamer_srl/test_agent_grad.py` — CP8 wrapper
  structural-gradient-flow checks (this spec's threshold band applies; the
  test pattern is from §2.2).

For developer convenience, every grad test follows
`def test_<site>_grad_matches_sheeprl():` so `pytest -k grad` selects the
full grad suite, and `pytest -k "actor and grad"` selects only the actor
grad subset.

### 6.2 Diff-tool extension

[`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py) is
extended at v2-CP2 with the following new API (skeleton; the developer
implementing it follows the existing `register_diff(...)` pattern in the
script):

```python
def register_grad_diff(
    name: str,                              # e.g. "compute_actor_objective_grad"
    fixture_seed: int,                      # e.g. 0xD3EAF + 0x100 * 3
    fixture_input_builder: Callable[[int], dict],  # builds fixture dict at seed
    jax_loss_fn: Callable[[dict, dict], float],    # (params, inputs) -> scalar
    torch_loss_fn: Callable[[Any, dict], torch.Tensor],   # (module, inputs) -> scalar
    grad_target: Literal["params", "input:<name>"],  # what to take grad w.r.t.
    threshold: float,                       # per §3
    cite_sheeprl: str,                      # e.g. "dreamer_v3.py:L262-L304"
):
    """Register a gradient-parity diff. Computes jax.grad and torch.autograd.grad
    on the same fixture and writes max_abs_diff + margin to
    tmp/diff_runs/<name>_grad.json.
    """
```

Output JSON schema (per `name`):

```json
{
  "name": "compute_actor_objective_grad",
  "fixture_seed": 866223,
  "sheeprl_cite": "dreamer_v3.py:L262-L304",
  "threshold": 5e-5,
  "max_abs_diff": 3.1e-5,
  "max_rel_diff": 2.4e-6,
  "margin": 1.61,
  "verdict": "PASS",
  "grad_target": "params",
  "n_params_compared": 14,
  "leaf_breakdown": [
    {"leaf_name": "actor.mlp.0.kernel", "max_abs_diff": 2.8e-5},
    ...
  ]
}
```

The diff-tool extension lands as an inline modification of
`sheeprl_jax_diff.py` per v2 IMPLEMENTATION_PLAN §8 — "originals stay,
corrections are additive". An inline correction note at the function-table
comment block names this v2-CP2 spec as the authorising doc.

---

## 7. Failure-mode catalog

When a grad-parity test fails, four diagnostic shapes are possible. The
catalog below routes each one.

### 7.1 Substantial diff (`max_abs_diff > 10 × threshold`)

**This is a real bug.** A 10× breach of the substrate-mechanical band is
outside the ULP-drift envelope and indicates a **semantic** mis-port (wrong
operator, wrong sign, wrong index, missing term, missing `sg`).

**Action**: stop the audit chain. The v2-CPN developer drops the audit and
hands off to `senior-developer` to write a fix plan, then to `developer` to
implement at v2-CP9. The fix lands as a single named commit with a paired
Lever-A test that fails on the pre-fix code and passes on the post-fix
code. **Do not raise the threshold to make the test pass.**

Concrete example: if v2-CP3 test #4 (`∂L_actor / ∂advantage`) returns
`max_abs_diff = 1.4e-3` (well over `1e-7`), the `sg(advantage)` is missing.
This is the H1 root cause — a single-line fix at the dreamer-srl port site,
adding `jax.lax.stop_gradient(advantage)` at the appropriate line.

### 7.2 Marginal diff (`1× < max_abs_diff < 10× threshold`)

**This is a substrate-class deviation candidate.** A 2–10× breach is in the
range where substrate-mechanical drift through a slightly-deeper chain than
the threshold anticipated could explain it.

**Action**: document as a v2 deviation row (`D-015` onwards). The format
mirrors v1's
[`DEVIATION_LOG.md`](../dreamer_srl_v3/DEVIATION_LOG.md) schema — anchor to a
substrate-class precedent (D-003 / D-006 / D-007 / D-008 / D-010 / D-011 are
the gradient-relevant classes), measure the analytical margin (see
math-reviewer's `sqrt(N)` chain-depth witness for D-008), and approve inline
by `senior-developer` (no PI consult — per the v2 IMPLEMENTATION_PLAN §2
A+B+C+D framing, substrate-class deviations flip inline by senior-developer
without escalation to Lever E).

Concrete example: v2-CP4 test #1 returns `max_abs_diff = 5.2e-3` (against a
`2e-3` threshold). The ratio is 2.6× — within the D-008 2.78× band. Log as
`D-015` with the anchor "substrate-class match with D-008; gradient
amplification through critic two-hot log-prob chain"; raise the threshold
to `8e-3` (= 4× the original measured floor); rerun and pass.

### 7.3 NaN or inf on either side

**This is numerical instability.** Possible causes: log-softmax overflow at
saturated logits, divide-by-zero in a Jacobian, missing eps clamp in
`probs_to_logits`, `1/x` where `x ≈ 0` upstream.

**Action**: investigate. Common fixes:

- Reduce fixture-input magnitudes (the test should be run at training-
  realistic scale; if fixture randomness lands far outside the realistic
  range, that's a fixture bug, not a code bug — fix the fixture).
- Add a `jnp.clip` / `torch.clamp` to a known-saturating site. If sheeprl
  has the clip and dreamer-srl doesn't, that's a port bug.
- Check whether the `probs_to_logits(probs)` post-`unimix` is hitting the
  tiny-clamp: at `unimix=0.01, num_classes=5`, the floor is `0.002`, well
  above `tiny`; if the fixture has `unimix=0`, the floor disappears and
  log can underflow.

A NaN never gets logged as a deviation. It always indicates either a
fixture-construction error or a real bug; fix and re-run.

### 7.4 One side returns `None` / `0` grad, the other doesn't

**This is structural sg mis-placement** — the H1 / H3 canonical diagnostic.

**Diagnosis path**: the `allow_unused=False` flag on the PyTorch side (§2.2
note) causes `torch.autograd.grad` to **raise** when a target param is not
reached by the backward graph. The JAX side, by contrast, returns a zero-leaf
for unreachable params (the pytree leaf is zero-initialised by `jax.grad`).
So the failure surface is asymmetric: PyTorch raises, JAX returns zero.

**Action**: identify which side has the extra/missing sg by walking the
graph. The dreamer-srl side can be inspected with `jax.make_jaxpr` to see
which params actually have a path to the loss. The sheeprl side can be
inspected by removing `allow_unused=False` and checking which entries in
the returned tuple are `None`.

Concrete example: v2-CP3 test #4 returns: PyTorch raises
`RuntimeError: One of the differentiated Tensors appears to not have been
used in the graph`. This means the sheeprl reference has correctly placed
`sg(advantage)` AND the test asked `torch.autograd.grad(loss, [advantage])`
expecting a non-None result — that's the **test** bug: the sg-leak check
should request `grad_outputs=torch.ones_like(loss)` and use a tensor that
has `requires_grad=True` set explicitly, then check whether the returned
grad is non-zero. Refactor the test as §2.3.

---

## 8. Recommended order of testing

The v2 IMPLEMENTATION_PLAN §5 hand-off chain sets v2-CP3 → v2-CP4 → v2-CP5
→ v2-CP6 → v2-CP7 → v2-CP8 → v2-CP9. The grad-parity tests within that
ordering should be executed as:

1. **v2-CP3 runs FIRST.** The experiment-analyzer's H1 ranking puts the
   actor REINFORCE gradient at HIGH confidence as the v1-parity root
   cause. v2-CP3 has the highest probability of surfacing the bug; landing
   it first concentrates reviewer attention on the most-likely-to-fail
   site.
2. **v2-CP4 (critic) and v2-CP5 (imagined returns) can run in PARALLEL
   after v2-CP3.** Both are MED-LOW confidence on the H-ranking; running
   them concurrently smooths reviewer load without staking parity on
   either. Note that v2-CP5's test #4 — the H3 advantage-sign empirical
   check — is the **single highest-leverage test in the manifest**; it's a
   qualitative one-line assertion that detects the sign-flip mode at zero
   threshold cost.
3. **v2-CP6 (orchestrator) runs LAST.** It is the integration test — it
   composes the three loss heads + WM + optimizer into a single train
   step. If v2-CP3/v2-CP4/v2-CP5 all PASS individually but v2-CP6
   FAILs, the bug is in the composition (cross-module sg leak, optimizer-
   state mishandling, JIT-vs-eager semantic divergence). v2-CP6 is the
   net for what the per-function tests miss.
4. **Within each CP**, run the sg-leak tests (the `1e-7`-threshold checks)
   BEFORE the gradient-magnitude tests. A failing sg-leak test
   immediately localises the bug to a specific line in the port;
   gradient-magnitude tests have wider diagnostic scope.

---

## 9. What this methodology does NOT cover

Honest scope limits — flagged so downstream CPs do not over-claim what
their tests prove.

- **Parameter-update parity**. The grad-parity test compares gradients
  before any optimizer step. It does NOT compare the *post-step parameter
  values* between `optax.adam.update + apply_updates` and
  `torch.optim.Adam.step()`. v2-CP6 includes two dedicated optimizer-step
  parity tests; the bulk of this methodology covers only the gradient
  itself.

- **Stochastic-gradient variance**. REINFORCE has a notoriously high
  gradient variance (`Var[grad] ∝ E[advantage^2 × ||∂log_prob/∂θ||^2]`).
  Path A (fixed-sample) bypasses variance entirely — the test is asking
  "given THIS sample, do the gradients match?", not "is the *expected*
  gradient consistent?". A separate variance-estimation test would be
  required to claim variance parity; that test is **out of scope** for v2
  (would belong in a future "training-dynamics parity" gate).

- **Full-training-trajectory parity**. Even if every per-function grad
  test passes, the 200k-step training dynamics can still diverge due to
  accumulated ULP drift, JIT-vs-eager rounding differences, or
  numerical-stability differences in the optimiser state evolution. The
  **policy-learning gate (v2-CP10)** is the empirical instrument that
  measures this; the grad-parity tests alone cannot.

- **Optimizer-state semantic parity**. `optax.adam` and `torch.optim.Adam`
  differ in (a) eps placement inside vs outside the sqrt, (b) bias-
  correction interleaving, (c) the order of moment updates. v2-CP6's two
  optimizer-step parity tests measure the post-step parameter drift; they
  do NOT verify state-evolution semantics over many steps. A separate
  multi-step optimizer-state test (e.g. 100 steps of identical inputs,
  compare moment buffers) is out of scope for v2 — flagged for the v2-CP9
  fix bundle if v2-CP10 fails.

- **Cross-platform PRNG-induced learning-dynamics drift.** Path A fixes
  the action sample to bypass the categorical-sampling PRNG boundary.
  Over a 200k-step training run, the actual PRNG draws between the two
  substrates produce *different* trajectories — even with bit-identical
  per-step gradients. This is exactly the **D-002 / D-009 cross-PRNG
  stochastic** class from v1's deviation log. The grad-parity tests
  cannot detect this; only v2-CP10 / v2-CP11 can.

- **Recurrent-state-history sensitivity**. The grad-parity tests use a
  single fixture batch. If the bug surfaces only when the recurrent state
  has accumulated a specific pattern of inputs over many time steps, the
  per-step grad test cannot reach it. v2-CP10's 20k-step policy-learning
  gate is the diagnostic for this class.

---

## Authoring history

- **2026-05-14**: Spec authored by `professor-rl-bayesian-dl` per v2-CP2 of
  [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). No source-tree or
  `vendor/sheeprl/` modifications. Foundation for v2-CP3–v2-CP6 grad-parity
  test implementations.

## Verification Report

> **Verified by**: `math-reviewer` — scheduled per v2-CP2's Lever-C row in
> [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §3 (math-reviewer
> validates the threshold table against the forward thresholds in v1's
> `DEVIATION_LOG.md`).
> **Date**: TBD (this spec is authored; the math-reviewer pass is the
> v2-CP2 verdict step).

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| [`docs/develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md`](GRAD_PARITY_METHODOLOGY.md) | New methodology spec (this file) | ☐ pending review | Threshold band anchored to v1 D-003/D-006/D-007/D-008/D-010/D-011 precedent; recommends path A (fixed-sample) for cross-PRNG categorical boundary; 18 test sites manifest across v2-CP3/CP4/CP5/CP6. |

**Conclusion**: v2-CP2 spec authored. Ready to be cited by v2-CP3 (actor
re-audit, recommended FIRST per §8) and v2-CP4/v2-CP5/v2-CP6 (parallel /
sequential per §8). The diff-tool extension at
[`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py) is
specified in §6.2 but **not implemented in this spec** — implementation is
deferred to v2-CP3 first-touch (the `developer` agent at v2-CP3 implements
the `register_grad_diff(...)` helper as a single inline modification under
the v2-CP3 plan).
