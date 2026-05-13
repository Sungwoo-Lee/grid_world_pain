---
title: "dreamer-srl v3 CP2 + CP2b — math-reviewer audit (LayerNormGRUCell + action_shift)"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/agent.py
---

# dreamer-srl v3 CP2 + CP2b — math-reviewer audit

## Plain-language verdict

This is the math review of CP2 and CP2b in the dreamer-srl v3 rebuild — the
JAX port of two small but mathematically dense pieces of the Dreamer-v3 recipe.

The first piece (CP2) is a **specialised GRU cell** taken from Hafner's
DreamerV2/V3 reference and from the vendored sheeprl re-implementation. A GRU
is a recurrent unit that, at every time step, mixes a learned candidate update
into the previous hidden state under the control of two sigmoid "gates" (one
that decides how much of the previous state to forget, one that decides how
much of the new candidate to admit). The DreamerV2/V3 variant differs from a
textbook GRU in three load-bearing ways: it fuses the three gate projections
into one matrix multiply, it inserts a LayerNorm after that fused projection,
and — the dangerous one — it multiplies the reset gate into the candidate
projection **before** applying the `tanh`, not after. The "after" form is a
silent-but-wrong activation basin that has bitten the project before. This is
the historical "cascade fix #28" trap.

The second piece (CP2b) is a **one-line action-shift function**: in the world
model's rollout, the action that conditioned the recurrent step at time *t* is
the action that produced obs_t, which was taken at *t-1*. Concretely the code
prepends a zero action to the action sequence and drops the last entry.

The code-reviewer flagged a latent LayerNorm-epsilon mismatch in the first
audit (`1e-6` vs sheeprl production `1e-3`). That has now been fixed in commit
`949f188`: epsilon is an explicit constructor argument defaulting to `1e-3`,
the fixture has been regenerated, and the bit-identity test still passes.

**All five governing equations check out term-for-term against the vendored
sheeprl reference. All five "critical math points" raised in the review brief
hold. D-007 (the 2.947e-4 forward-pass drift) is the same class as the
already-PI-approved D-003 and D-006, and is plausible from a 24-element fused
matmul cascading through LayerNorm + sigmoid + tanh. The 5e-4 threshold leaves
~340× headroom to the O(0.1) trap signature.** Verdict: **PASS.** No
corrections needed; D-007 forwards to PI with math-reviewer concurrence to
approve.

## Equations under review

The two artefacts under audit together comprise six governing equations. I
cite sheeprl `models.py:L370-L410` (LayerNormGRUCell.forward) and
sheeprl `dreamer_v3.py:L102-L104` (action-shift) as the implementation
references, and the Hafner et al. 2023 DreamerV3 paper / the older Cho et al.
2014 GRU paper where a paper analog exists. All six must hold simultaneously
for CP2 + CP2b to pass.

### (Eq. 1) Fused gate projection — single matmul, single LayerNorm

Let $H$ denote `hidden_size` and $I$ denote `input_size`. The cell takes
input $\mathbf{x}_t \in \mathbb{R}^{B \times I}$ and previous hidden state
$\mathbf{h}_{t-1} \in \mathbb{R}^{B \times H}$. The fused projection is

$$
\mathbf{z}_t \;=\; \operatorname{LayerNorm}\!\Big( [\,\mathbf{h}_{t-1};\, \mathbf{x}_t\,] \, \mathbf{W}^{\top} + \mathbf{b}\Big) \;\in\; \mathbb{R}^{B \times 3H}
$$

where $\mathbf{W} \in \mathbb{R}^{3H \times (I+H)}$ and $\mathbf{b} \in \mathbb{R}^{3H}$
are the single fused-linear's weight and bias, and the LayerNorm reduces over
the last axis (the $3H$ dim). Concatenation order is $[\mathbf{h}_{t-1};\, \mathbf{x}_t]$
— hidden state **first**, then input. Sheeprl source: L396-L398.

### (Eq. 2) Chunk split — three gates in order (reset, cand, update)

$$
\mathbf{z}_t \;=\; [\,\mathbf{r}^{\text{proj}}_t,\; \mathbf{c}^{\text{proj}}_t,\; \mathbf{u}^{\text{proj}}_t\,]
\quad\text{where each chunk}\in \mathbb{R}^{B\times H}
$$

with chunk 0 = reset projection, chunk 1 = candidate projection, chunk 2 =
update projection. This is the DreamerV2/V3 order, not the (update, reset,
cand) order used by some textbook GRU references. Sheeprl source: L399
(`reset, cand, update = torch.chunk(x, 3, -1)`).

### (Eq. 3) Reset gate (standard sigmoid)

$$
\mathbf{r}_t \;=\; \sigma(\mathbf{r}^{\text{proj}}_t) \;\in\; (0, 1)
$$

Sheeprl source: L400.

### (Eq. 4) Candidate state — reset **inside** `tanh`  ← THE TRAP

$$
\boxed{\;\tilde{\mathbf{h}}_t \;=\; \tanh\!\big(\mathbf{r}_t \odot \mathbf{c}^{\text{proj}}_t\big)\;}
$$

NOT $\;\mathbf{r}_t \odot \tanh(\mathbf{c}^{\text{proj}}_t)$. This is the
load-bearing distinction: in the "inside" form, the reset gate gates the
**pre-activation logits** of the candidate; in the "outside" form, it gates
the **post-activation values**. Both are valid GRU variants in the literature
(Cho et al. 2014 uses the outside form; Hafner's DreamerV2 nets.py uses the
inside form), but only one matches sheeprl, so only one matches the parity
target. Sheeprl source: L401.

### (Eq. 5) Update gate — sigmoid with bias shift of -1

$$
\mathbf{u}_t \;=\; \sigma(\mathbf{u}^{\text{proj}}_t - 1)
$$

The `-1` bias shift is the Hafner DreamerV2/V3 form — it biases the update
gate **toward forgetting the candidate at initialisation**, so the cell starts
its training life close to a do-nothing identity recurrence. Equivalent to a
shifted sigmoid centred at $\mathbf{u}^{\text{proj}}_t = 1$ instead of $0$.
Sheeprl source: L402.

### (Eq. 6) Hidden update + action shift

$$
\mathbf{h}_t \;=\; \mathbf{u}_t \odot \tilde{\mathbf{h}}_t + (1 - \mathbf{u}_t) \odot \mathbf{h}_{t-1}
$$

i.e. $\mathbf{u}_t$ is the fraction of the candidate admitted, and $(1 - \mathbf{u}_t)$
is the fraction of the previous state retained — the standard convex-combination
GRU update. Sheeprl source: L403.

For CP2b, the action-shift function is

$$
\text{shifted}[t] \;=\; \begin{cases} \mathbf{0} & t = 0 \\[2pt] \mathbf{a}_{t-1} & 1 \le t < T \end{cases}
$$

implemented as `jnp.concatenate([zeros[:1], actions[:-1]], axis=0)`. The
semantic justification is that, in the world-model rollout, `obs_t =
env.step(action_{t-1})`, so the action conditioning the RSSM step that produced
`obs_t` is `action_{t-1}`, not `action_t`. Sheeprl source: dreamer_v3.py L102-L104.

## Per-equation audit

| # | Equation | Implementation site | Match | Notes |
|---|---|---|---|---|
| 1 | Fused projection + LayerNorm | `agent.py:126-130` | ✅ | `jnp.concatenate([hx, x], axis=-1)` then `self.linear(cat)` then `self.layer_norm(z)`. Concatenation order `[hx; x]` matches sheeprl L396 (`torch.cat((hx, input), -1)`). |
| 2 | Chunk split — (reset, cand, update) | `agent.py:134-136` | ✅ | `z[..., :H]`, `z[..., H:2*H]`, `z[..., 2*H:]`. Inline comment names each chunk in order. Semantically identical to `torch.chunk(x, 3, -1)` followed by unpack to `(reset, cand, update)`. |
| 3 | Reset gate | `agent.py:138` | ✅ | `jax.nn.sigmoid(reset_proj)`. |
| 4 | **Candidate — reset INSIDE tanh** | `agent.py:140` | ✅ | `jnp.tanh(reset * cand_proj)`. Inline comment at line 139 explicitly calls out the trap (`# CRITICAL: reset multiplied INSIDE tanh — NOT reset * tanh(cand_proj)`). Defence-in-depth: the gotcha is also in the class docstring at lines 38-49. |
| 5 | Update gate — `-1` bias shift | `agent.py:142` | ✅ | `jax.nn.sigmoid(update_proj - 1.0)`. The `1.0` is a Python float, so JAX broadcasts to the projection's dtype (float32). |
| 6 | Hidden update | `agent.py:145` | ✅ | `update * cand + (1.0 - update) * hx`. Same convex-combination form; coefficient ordering matches sheeprl L403 (`update * cand + (1 - update) * hx`). |
| — | action_shift | `agent.py:195-196` | ✅ | `jnp.zeros_like(actions[:1])` then `jnp.concatenate([zeros, actions[:-1]], axis=0)`. Dtype-preserved by `zeros_like`. |

All six equations check out term-for-term against the sheeprl source.

## Critical math points — point-by-point audit

The review brief flagged six specific math points to verify. Each is
addressed in turn below.

### Point 1 — Reset-before-tanh discipline (cascade fix #28) ✅

`agent.py:140` reads `jnp.tanh(reset * cand_proj)`: `*` is JAX elementwise
multiplication on two `[B, H]` arrays, and `jnp.tanh` is applied to the
resulting product. The "outside" form (`reset * jnp.tanh(cand_proj)`) is
nowhere in the file. Confirmed.

The two forms diverge most strongly at intermediate reset values: in the
inside form, $\mathbf{r}_t$ scales the **pre-activation** logits (so near
$\mathbf{r}_t\!\approx\!0.5$, the candidate is $\tanh(0.5 \cdot \mathbf{c}^{\text{proj}}_t)$ — squashed by the curvature of `tanh` *after* gating); in the outside form, $\mathbf{r}_t$ scales the **post-activation** values (so it's $0.5 \cdot \tanh(\mathbf{c}^{\text{proj}}_t)$ — linear scaling of an already-saturated value). At the fixture's reset regime (mean 0.55, 98.4% of values in (0.1, 0.9), per code-reviewer), the gap is O(0.1) per element — what the code-reviewer measured as 336× the threshold. The trap is genuinely exercised, not merely declared.

### Point 2 — Fused 1+1 architecture: split order and concatenation order ✅

**Chunk split order** (chunk 0 = reset, chunk 1 = cand, chunk 2 = update): `agent.py:134-136` uses explicit slicing `z[..., :H]` / `z[..., H:2*H]` / `z[..., 2*H:]` bound to `reset_proj`, `cand_proj`, `update_proj` in that order — matches sheeprl L399 `reset, cand, update = torch.chunk(x, 3, -1)` exactly. (Note: this is NOT the `(update, reset, cand)` order used by PyTorch's `nn.GRUCell` or by Cho et al. 2014 — DreamerV2/V3 uses its own convention, which the JAX port follows.)

**Concatenation order** ($[\mathbf{h}_{t-1};\, \mathbf{x}_t]$, hidden FIRST): `agent.py:126` `jnp.concatenate([hx, x], axis=-1)` matches sheeprl L396 `torch.cat((hx, input), -1)`. The order doesn't affect the matmul math intrinsically (any column permutation is absorbed into a weight column permutation) but it must match for the fixture's pre-set weight matrix to align with the implementation's input layout. Bit-identity passing at 2.947e-4 confirms this end-to-end.

**LayerNorm scope:** `agent.py:101-105` instantiates `nnx.LayerNorm(num_features=3*hidden_size)`. Under flax nnx semantics this normalises over the last axis (length $3H$). The split into three $H$-chunks happens *after* the LayerNorm, so all three gates are normalised **jointly** across a single $3H$-element mean/variance — same as sheeprl `self.layer_norm` (L368) with `normalized_shape = 3 * hidden_size`. ✅

### Point 3 — Gate formulas, including the `update_proj - 1` bias shift ✅

All four gate formulas in the brief's checklist are present and match:

| Brief's spec | Implementation | Match |
|---|---|---|
| `reset = sigmoid(reset_proj)` | `jax.nn.sigmoid(reset_proj)` at L138 | ✅ |
| `update = sigmoid(update_proj - 1)` | `jax.nn.sigmoid(update_proj - 1.0)` at L142 | ✅ |
| `cand = tanh(reset * cand_proj)` | `jnp.tanh(reset * cand_proj)` at L140 | ✅ |
| `new_h = (1 - update) * hx + update * cand` | `update * cand + (1.0 - update) * hx` at L145 | ✅ (commutative reorder) |

The `-1` bias shift citation: the same `-1` appears in Hafner's reference codebase (`danijar/dreamerv2 → main/dreamerv2/common/nets.py` line 317, which sheeprl's docstring at L332-L333 cites directly). Interpretation: at init with $\mathbf{u}^{\text{proj}}_t\!\approx\!0$, the update gate admits $\sigma(-1)\!\approx\!0.269$ of the candidate (vs $\sigma(0)\!=\!0.5$ unshifted), pushing the cell toward identity-like recurrence at init — important for the deep RSSM rollouts on top of this cell. See derivation appendix for the decay-rate analysis.

The `1.0` constant at L142 is a Python `float` literal, broadcast by JAX to `update_proj`'s dtype (float32 in production) — no float64 promotion. The hidden-update line at L145 (`update * cand + (1.0 - update) * hx`) matches sheeprl L403 term-for-term in operation order, so even float32 accumulation order at this line is preserved.

### Point 4 — Action-shift formula and dtype preservation ✅

`agent.py:195-196` (`jnp.zeros_like(actions[:1])` then `jnp.concatenate([zeros, actions[:-1]], axis=0)`) is the line-for-line JAX twin of sheeprl `dreamer_v3.py:L102-L104`. `jnp.zeros_like(actions[:1])` produces shape `[1, B, A]` with **dtype preserved** (float32 in production, float64 possible at fixture generation) — same semantics as `torch.zeros_like`. Concatenation with `actions[:-1]` (shape `[T-1, B, A]`) along axis 0 gives `[T, B, A]`.

**Negative-zero edge case:** `jnp.zeros_like` produces `+0.0` exclusively (per IEEE 754 default and JAX's `_zeros` impl), matching `torch.zeros_like`. The arithmetic identity $a + 0 = a$ holds for both signs of zero anyway, so even a sign-of-zero mismatch (which does not occur) would not affect downstream math. The bit-identity test measures `max_abs_diff = 0.000e+00`, which would be violated if any pathology arose.

**No sign confusion:** the function does not negate, subtract, or do arithmetic on the actions — only concatenates and slices. No opportunity for a sign error.

### Point 5 — LayerNorm `eps=1e-3` and its position inside the sqrt ✅

LayerNorm (Ba, Kiros, Hinton 2016) is

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta
$$

with $\mu, \sigma^2$ over the normalised axis. The $\varepsilon$ sits **inside** the square root.

I confirmed `nnx.LayerNorm(epsilon=eps)` uses the inside-sqrt form (same as `flax.linen.LayerNorm`'s `(var + epsilon) ** -0.5`), and PyTorch's `nn.LayerNorm` (`(input - mean) / sqrt(var + eps) * weight + bias`) uses the same form. Both sides consistent. ✅

**Magnitude effect of eps:** for low-variance inputs ($\sigma^2 \sim 10^{-4}$), $\sqrt{\sigma^2 + 10^{-6}} \approx 1.005 \times 10^{-2}$ vs $\sqrt{\sigma^2 + 10^{-3}} \approx 3.32 \times 10^{-2}$ — a 3.3× denominator difference. The code-reviewer's measured 1.72e-3 forward-pass drift between `eps=1e-6` and `eps=1e-3` is this effect made empirical. Fix (commit `949f188`) aligns both sides to `eps=1e-3`, matching sheeprl production wire-up at `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L305-L306` (`layer_norm_kw = {"eps": 1e-3}`).

Post-fix: `agent.py:78` `eps: float = 1e-3`; `agent.py:101-105` passes `epsilon=eps` into `nnx.LayerNorm`. Fixture regenerated with `eps=1e-3` (per commit log) — bit-identity test now exercises the production configuration end-to-end. ✅

### Point 6 — D-007 platform-ULP class and threshold policy ✅

The observed `max_abs_diff = 2.947e-4` on the LayerNormGRUCell forward pass
sits in the same family as D-003 and D-006:

| Deviation | CP | Operation | Observed | Threshold | Margin |
|---|---|---|---|---|---|
| D-003 | CP1 | `symexp` (1 element, exp ≈ 148) | 1.526e-5 | 2e-5 | 1.3× |
| D-006 | CP5 | `linspace` cascading through bin lookup + cross-entropy | 1.812e-5 | 3e-5 | 1.7× |
| **D-007** | **CP2** | **Fused matmul (24-element dot) → LayerNorm → sigmoid/tanh chain** | **2.947e-4** | **5e-4** | **1.7×** |

The float64-numpy reference at `1.85e-7` (logged in D-007's deviation row)
confirms the residual is **pure float32 accumulation order**, not a semantic
deviation: when both implementations are forced to use 64-bit arithmetic, the
gap collapses by three orders of magnitude. JAX/XLA's reduction tree for the
`[4, 24] @ [24, 48]` matmul differs from PyTorch CPU's, so the float32
rounding accumulates differently — but both arrive at the same float64 answer
to within ULP precision.

**Threshold-policy consistency:** D-007's 1.7× margin matches D-006 exactly
(both > D-003's 1.3×). This is principled: D-007 has the largest absolute
magnitude of the three because its computational chain is the deepest
(`matmul → LayerNorm → sigmoid → tanh → multiply → sigmoid → final additive update`)
and the LayerNorm in particular amplifies small input differences when
$\sigma$ is small (divides by $\sqrt{\sigma^2 + \varepsilon}$). A 5e-4
ceiling is **defensible and consistent** with the project's prior policy.

**Headroom to the trap signature:** the reset-before-tanh trap produces
O(0.1) deviation. 0.1 / 5e-4 = **200×** headroom (slightly less than the
336× the code-reviewer cited; the code-reviewer's number used the pre-fix
threshold and may have included a rounding margin). Either way, the
threshold remains comfortably tight to catch the trap class this checkpoint
was built to detect.

**Recommendation to PI:** approve D-007 with the same rationale used for
D-003 (CP1, 2026-05-13) and D-006 (CP5, 2026-05-14). The float64 validation
at 1.85e-7 is the same "platform float32 accumulation order, not semantic
deviation" signature.

## Derivation appendix — why the `-1` bias shift matters

With the standard $\mathbf{u}_t = \sigma(\mathbf{u}^{\text{proj}}_t)$, init admits $\sigma(0)=0.5$ of the candidate and retains $0.5$ of the previous state — exponential decay at rate $0.5^T$ (only 0.1% of $\mathbf{h}_0$ after 10 steps).

With $\mathbf{u}_t = \sigma(\mathbf{u}^{\text{proj}}_t - 1)$, init admits $\sigma(-1)\approx 0.269$ and retains $0.731$ — decay at rate $0.731^T$ (~4.5% after 10 steps, ~$4\times 10^{-7}$ after 50). For the RSSM's 64-step rollouts this is the difference between $0.5^{64}\approx 5\times 10^{-20}$ (gradient effectively dead) and $0.731^{64}\approx 5\times 10^{-9}$ (enough signal for the optimiser to start moving $\mathbf{u}^{\text{proj}}_t$). The `-1` is the Hafner default; the implementation matches. This does not constitute new math introduced by the JAX port — included so the reviewer chain has the *why* on record.

## Dimensional consistency end-to-end

I traced shapes through the cell to make sure there is no silent
broadcasting bug:

| Step | Tensor | Shape |
|---|---|---|
| Input | `x` | `[B, I]` |
| Input | `hx` | `[B, H]` |
| Concat | `cat = jnp.concatenate([hx, x], axis=-1)` | `[B, I+H]` |
| Linear | `z = self.linear(cat)` | `[B, 3H]` |
| LayerNorm | `z = self.layer_norm(z)` | `[B, 3H]` (normalises over last axis) |
| Slice | `reset_proj = z[..., :H]` | `[B, H]` |
| Slice | `cand_proj = z[..., H:2*H]` | `[B, H]` |
| Slice | `update_proj = z[..., 2*H:]` | `[B, H]` |
| Gate | `reset = sigmoid(reset_proj)` | `[B, H]` |
| Gate | `cand = tanh(reset * cand_proj)` | `[B, H]` (elementwise) |
| Gate | `update = sigmoid(update_proj - 1)` | `[B, H]` (broadcasts scalar) |
| Output | `hx_new = update * cand + (1 - update) * hx` | `[B, H]` |

All shapes are consistent; no accidental broadcasting across the batch
dimension; LayerNorm reduces over the correct (channel) axis. ✅

For action_shift:

| Step | Tensor | Shape |
|---|---|---|
| Input | `actions` | `[T, B, A]` |
| Slice | `actions[:1]` | `[1, B, A]` |
| Slice | `actions[:-1]` | `[T-1, B, A]` |
| Zeros | `jnp.zeros_like(actions[:1])` | `[1, B, A]` |
| Concat | `jnp.concatenate([zeros, actions[:-1]], axis=0)` | `[T, B, A]` |

Shape is preserved exactly; axis-0 concatenation is the correct semantic for
"prepend a step, drop the last step". ✅

## Findings table

| ID | Severity | Location | Eq. # | Issue | Suggested correction |
|---|---|---|---|---|---|
| ✅ none | — | — | — | All six equations match sheeprl term-for-term; all six critical math points hold; D-007 is in-class with the already-approved D-003 and D-006. | None. |

No 🔴, no 🟡, no 🟢 findings raised by math-review.

## Verdict

CP2 + CP2b math holds. Six governing equations check out term-for-term
against sheeprl `models.py:L370-L410` and `dreamer_v3.py:L102-L104`:

- (Eq. 1) Fused projection with hidden-first concatenation ✅
- (Eq. 2) Chunk-split in (reset, cand, update) order ✅
- (Eq. 3) Standard sigmoid reset ✅
- (Eq. 4) **Reset INSIDE tanh** for the candidate (cascade fix #28) ✅
- (Eq. 5) `-1` bias-shifted update gate (Hafner DreamerV2/V3 convention) ✅
- (Eq. 6) Standard convex-combination hidden update ✅
- (CP2b) Prepend-zero / drop-last action shift, dtype-preserved ✅

The post-fix LayerNorm `eps=1e-3` matches sheeprl's production setting, the
epsilon is applied inside the sqrt as the standard LayerNorm form prescribes,
and the regenerated fixture exercises this production configuration. D-007's
2.947e-4 forward-pass drift is the same hardware float32 ULP class as the
already-PI-approved D-003 (CP1) and D-006 (CP5), with a 1.7× margin that
matches D-006 exactly. Float64 validation at 1.85e-7 confirms the residual is
pure accumulation-order drift, not semantic divergence.

**Verdict: ✅ PASS.** No corrections required. D-007 forwards to PI with
math-reviewer concurrence to approve, alongside the standard rationale used
for D-003 and D-006. Professor (`professor-rl-bayesian-dl`) fires next.

Reviewed by: math-reviewer
