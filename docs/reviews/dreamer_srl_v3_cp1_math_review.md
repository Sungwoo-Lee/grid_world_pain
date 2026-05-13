---
title: "dreamer-srl v3 — CP1 Math Review"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-13
last_updated: 2026-05-13
audited_doc: src/algorithms/dreamer_srl/utils.py
audited_commit: fdcbfa5
---

# dreamer-srl v3 — CP1 Math Review

## Verdict (plain language)

**Purpose.** Second of three sequential reviewer gates on the first checkpoint of the dreamer-srl v3 rebuild. The rebuild ports eight leaf utility functions from a vendored snapshot of `sheeprl` (a PyTorch DreamerV3 reference) into JAX, with the contractual promise that **the math does not change** — every JAX function is supposed to be a line-for-line translation of its PyTorch source. My job is to verify that promise. Code-reviewer already audited JAX idioms and source-citation line ranges; the next reviewer (a Bayesian-RL professor) will audit algorithm-level semantics that span multiple functions. I sit between them: equation-by-equation, constant-by-constant, axis-by-axis.

**Headline.** All eight functions are mathematically faithful to the sheeprl source. The lambda-return recursion preserves the exact backward-unroll structure including the `vals[-1:]` bootstrap and the trailing `[:-1]` drop. The percentile-EMA normalizer (called `Moments` in sheeprl) preserves the order-of-operations on the invscale floor — the floor is applied to the **post-update** high-minus-low, matching sheeprl's in-place semantics under a pure-functional rewrite. The Hafner truncated-normal constant is the full-precision value `0.87962566103423978` (not the historical `0.8796` truncation bug — that scar motivated this rebuild). The 1-ULP slack on `symexp` is hardware-precision-level on `exp(5) ≈ 148` and is mathematically defensible.

**Outcome.** No math findings. The three logged deviations (D-001, D-002, D-003) are not math deviations — they are an unused distributed-training no-op, an RNG-stream mismatch, and a transcendental-function rounding gap, in that order. The math expressed by the JAX code is identical to sheeprl. **Verdict: PASS.** Professor-rl-bayesian-dl can fire next.

## Equations under review

### Eq. 1 — symlog

$$\mathrm{symlog}(x) \;=\; \mathrm{sgn}(x) \cdot \log\bigl(1 + |x|\bigr)$$

Sheeprl source (`vendor/sheeprl/sheeprl/utils/utils.py:148-149`): `torch.sign(x) * torch.log(1 + torch.abs(x))`. JAX port: `jnp.sign(x) * jnp.log1p(jnp.abs(x))`. Mathematically identical (`log1p(z) ≡ log(1+z)`); `log1p` is preferred for numerical conditioning near `z = 0`.

### Eq. 2 — symexp

$$\mathrm{symexp}(x) \;=\; \mathrm{sgn}(x) \cdot \bigl(e^{|x|} - 1\bigr)$$

Sheeprl source (`utils.py:152-153`): `torch.sign(x) * (torch.exp(torch.abs(x)) - 1)`. JAX port: `jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)`. Identical.

### Eq. 3 — Hafner truncated-normal init

$$W_{ij} \;\sim\; \mathcal{N}_{[-2\sigma,\,+2\sigma]}\bigl(0,\,\sigma^2\bigr), \quad \sigma \;=\; \frac{1}{c_\text{Hafner}} \sqrt{\frac{1}{d_\text{avg}}}, \quad d_\text{avg} = \tfrac{1}{2}(d_\text{in} + d_\text{out}), \quad c_\text{Hafner} = 0.87962566103423978$$

The Hafner constant `c_Hafner` rescales the truncated-normal variance back to unit variance under the `[-2σ, +2σ]` truncation. Sheeprl source (`dreamer_v3/utils.py:147-150`).

### Eq. 4 — Hafner uniform init

$$W_{ij} \;\sim\; \mathcal{U}[-\ell,\,+\ell], \quad \ell \;=\; \sqrt{\frac{3 s}{d_\text{avg}}}$$

`s` is the user-supplied scale. When `s = 0`, `ℓ = 0`, so the kernel is zero-initialized (cascade fix #27 — reward head and critic output). Sheeprl source (`dreamer_v3/utils.py:175-178`).

### Eq. 5 — lambda-return recursion

$$G^{\lambda}_t \;=\; r_t + c_t \cdot \Bigl[(1-\lambda)\,v_{t+1} + \lambda\,G^{\lambda}_{t+1}\Bigr], \qquad G^{\lambda}_T = v_T$$

where `c_t = mask_t * γ` is **pre-multiplied by the caller** (v2 plan §S6, sheeprl `dreamer_v3.py:254`). Sheeprl's internal arithmetic — operating on caller-shifted inputs — is `interm[t] = rewards[t] + continues[t] * values[t] * (1-λ)` and then `V[t] = interm[t] + continues[t] * λ * V[t+1]`, which expands to `rewards[t] + continues[t] * ((1-λ)*values[t] + λ*V[t+1])` — algebraically identical to the canonical form after accounting for the caller's `[1:]` slicing on `values` (the local `values[t]` is the caller's `v_{t+1}`).

### Eq. 6 — percentile-EMA normalizer (Moments)

Given a batch of advantage samples `x`,
$$q_\text{lo} = Q_{0.05}(x),\quad q_\text{hi} = Q_{0.95}(x)$$
$$L \leftarrow \alpha L + (1-\alpha) q_\text{lo}, \quad H \leftarrow \alpha H + (1-\alpha) q_\text{hi}$$
$$\mathrm{offset} = L,\quad \mathrm{invscale} = \max\bigl(1/M,\; H - L\bigr)$$

where `α = decay`, `M = max_` is the floor parameter. The invscale floor uses the **post-update** `H - L`. Sheeprl source (`dreamer_v3/utils.py:56-63`).

### Eq. 7 — Ratio scheduler

Integer gradient-steps owed at env-step `n`, given target ratio `r` and accumulated debt:
$$k = \lfloor (n - n_\text{prev}) \cdot r \rfloor, \qquad n_\text{prev} \leftarrow n_\text{prev} + k / r$$

with a special-cased first call and a pretrain-burst branch. Sheeprl source (`utils/utils.py:259-300`).

### Eq. 8 — prepare_obs (MLP path)

For each non-CNN obs key with raw shape `[F]` or `[B, F]`, reshape to `[1, B, F]` (the RSSM's `[T, B, ...]` contract with `T = 1`). For CNN keys, additionally `/ 255 - 0.5`. Sheeprl source (`dreamer_v3/utils.py:80-91`).

## Per-function math audit

| # | Function | Math correctness | Constants checked | Reduction axes | Issues |
|---|---|---|---|---|---|
| 1 | `symlog` | Eq. 1 verified: `jnp.sign(x) * jnp.log1p(jnp.abs(x))` (utils.py:34). `log1p` is mathematically identical to `log(1 + ·)` and numerically better-behaved; sheeprl uses the latter. | none | element-wise | none |
| 2 | `symexp` | Eq. 2 verified: `jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)` (utils.py:45). Identical formula. | none | element-wise | none — see [D-003 audit](#d-003-symexp-1-ulp-slack) for the 1-ULP gap |
| 3 | `init_weights` | Eq. 3 verified: `denoms=(in+out)/2`, `scale=1/denoms`, `std=sqrt(scale)/0.87962566103423978`, truncation `[-2,+2]` on standard normal then multiply by `std` (utils.py:78-86). | `c_Hafner=0.87962566103423978` (full precision, **not** `0.8796`) | shape `(in, out)`, no reduction | none |
| 4 | `uniform_init_weights` | Eq. 4 verified: `denoms=(in+out)/2`, `scale=given_scale/denoms`, `limit=sqrt(3*scale)`, sample uniform `[-limit, +limit]` (utils.py:112-115). When `given_scale=0`, `limit=0`, so `jax.random.uniform(key, minval=0, maxval=0)` returns zeros — cascade fix #27 path verified. | factor `3` inside `sqrt` correctly recovers unit variance for `U[-1,+1]` rescaling | shape `(in, out)`, no reduction | none |
| 5 | `compute_lambda_values` | Eq. 5 verified. The Python loop traversing `reversed(range(T))`, the `vals = [values[-1:]]` bootstrap, the `interm = rewards + continues * values * (1 - lmbda)` precompute, and the trailing `list(reversed(vals))[:-1]` drop are all line-for-line identical to sheeprl (utils.py:160-164 ↔ sheeprl 72-76). `jnp.concatenate(..., axis=0)` is the JAX equivalent of `torch.cat(..., dim=0)` (the default in `torch.cat`). The fixture is generated by the caller-shifted view (rewards/values/continues all length T), so the function returns a length-T tensor with `V[T-1] = interm[T-1] + c[T-1]*λ*values[T]` (= bootstrap) and the recursion unrolls backward. | `lmbda=0.95` default; `(1-lmbda)` and `lmbda` are mixed in the canonical 1-λ / λ proportions | concat `axis=0` (the time axis); no reduction over batch | none — see [critical point #2](#critical-point-2-compute_lambda_values-recursion-direction) |
| 6 | `moments_update` | Eq. 6 verified. `x.ravel()` matches `torch.quantile`'s implicit flattening when `dim=None`. EMA on `(low, high)` precedes the invscale floor — **post-update** `(new_high - new_low)` is what enters `jnp.maximum(1/max_, ·)`, mirroring sheeprl's in-place semantic where `self.low` and `self.high` are mutated **before** L62's `torch.max`. Offset returned is the new EMA low (sheeprl L63: `self.low.detach()` post-mutation). | `decay=0.99`, `max_=1e8`, `percentile_low=0.05`, `percentile_high=0.95` all match sheeprl `__init__` defaults (L43-L46) | quantile reduces the flattened scalar; EMA + max are scalar ops | none — see [critical point #4](#critical-point-4-moments_update-invscale-order-of-operations) |
| 7 | `Ratio` | Eq. 7 verified. Special-case first call: `repeats = int(step * ratio)` with optional pretrain override. Steady state: `repeats = int((step - prev) * ratio)`, then `prev += repeats / ratio`. The accumulator uses float division so fractional debt does not round away. Integer-exact against sheeprl across the test sequence. No JIT crossing (Python-side object). | `int(...)` floor matches Python 3 / Hafner convention | n/a — scalar Python | none |
| 8 | `prepare_obs` | Eq. 8 verified. MLP key path: `arr.reshape(1, num_envs, -1)`. CNN key path: `arr.reshape(1, num_envs, -1, H, W) / 255.0 - 0.5`. Both match sheeprl L87 / L89 element-for-element. The CNN-path dtype cast happens implicitly through `jnp.asarray(v, dtype=jnp.float32)` at L389, which mirrors sheeprl's `.float()` at L85. | divisor `255`, offset `-0.5` for `[0,255] → [-0.5, +0.5]` rescale | reshape adds `T=1` leading axis; no reduction | none |

## Critical-points audit (the five named in the brief)

### Critical point 1 — Hafner constant precision

The full-precision Hafner constant `0.87962566103423978` is the truncation-correction factor for a normal distribution restricted to `[-2σ, +2σ]`: the post-truncation standard deviation is `≈ 0.8796 × σ_pre`, so dividing the target std by this factor recovers unit variance under truncation. The historical bug in the in-house `src/models/dreamer_v3_util.py:hafner_init` truncated this to `0.8796`, introducing a `7e-6` multiplicative error in every weight (small enough to compile and train, large enough to cumulatively drift over hundreds of millions of multiplications).

- JAX port (utils.py:80): `std = float(np.sqrt(scale) / 0.87962566103423978)` — full precision ✓
- Sheeprl source (`dreamer_v3/utils.py:149`): `std = np.sqrt(scale) / 0.87962566103423978` — same digit-for-digit ✓
- Test (`test_init_weights_matches_sheeprl`) recomputes `std_theoretical` from the same constant, so a regression to `0.8796` would surface as a >100x std relative error (the 12% pass margin would not absorb it).

**Verified.**

### Critical point 2 — compute_lambda_values recursion direction

The recursion unrolls **backward** from the terminal bootstrap.

```
sheeprl L72-L77                              JAX port utils.py:160-164
─────────────────────────────────            ─────────────────────────────────
vals = [values[-1:]]                         vals = [values[-1:]]
interm = rewards + continues*values*(1-λ)    interm = rewards + continues*values*(1-λ)
for t in reversed(range(len(continues))):    for t in reversed(range(len(continues))):
    vals.append(interm[t]                        vals.append(interm[t]
        + continues[t]*λ*vals[-1])                   + continues[t]*λ*vals[-1])
ret = torch.cat(list(reversed(vals))[:-1])   ret = jnp.concatenate(
                                                 list(reversed(vals))[:-1], axis=0)
```

The structural identity is line-for-line. Specifically:

- **`vals[-1:]` bootstrap**: keeps the leading singleton axis intact (shape `[1, B]`), so the subsequent appends produce a list of `[1, B]` tensors that can be `jnp.concatenate`d along `axis=0` without `reshape`. If the developer had written `values[-1]` (no slice) the result would be `[B]` and the final concat would fail or silently broadcast — neither happened.

- **Trailing `[:-1]` drop**: the list after the loop is `[V_T, G_{T-1}, ..., G_0]` (length `T+1`); after `reversed` it's `[G_0, G_1, ..., G_{T-1}, V_T]`; the `[:-1]` drops `V_T`, leaving length `T`. This is critical because sheeprl's caller at `dreamer_v3.py:251-256` expects a length-T return and further trims with `[:-1]` itself for the actor baseline (`baseline = predicted_values[:-1]`). If the port had returned length `T+1` the caller's downstream `[:-1]` would silently misalign by one timestep — exactly the kind of off-by-one that destroys credit assignment.

- **Concat axis**: `torch.cat` defaults to `dim=0` when no `dim` is specified; `jnp.concatenate` requires `axis=` explicitly. The JAX port correctly specifies `axis=0`. ✓

- **§S5 (true-continue splice) and §S6 (discount weighting)**: these are training-loop responsibilities of the **caller** of `compute_lambda_values`, not the function itself. The function takes `continues` as already-spliced-and-γ-multiplied input. Verified at sheeprl `dreamer_v3.py:247-256` — the splice happens at L247-L248, then `continues[1:] * γ` is passed in at L254. CP1's `compute_lambda_values` is correct in not re-implementing these; the §S5/§S6 obligations land on CP6 (`train.py` `one_train_step`).

**Verified.**

### Critical point 3 — moments_update percentile semantics

`torch.quantile(x)` with no `dim` argument operates on the flattened input — documented at <https://pytorch.org/docs/stable/generated/torch.quantile.html>: "If `dim` is `None`, the input is treated as flat." Sheeprl calls `torch.quantile(gathered_x, self._percentile_low)` with no `dim` (sheeprl L58-L59) — flattening is implicit.

JAX port (utils.py:248): `x_flat = x.astype(jnp.float32).ravel()` then `jnp.quantile(x_flat, percentile_low)` — flattening is explicit. The two are mathematically equivalent: both compute the empirical α-quantile over the entire input tensor.

The explicit-ravel form is also defensive against `jnp.quantile`'s axis-handling not exactly matching `torch.quantile`'s when shapes are awkward — by raveling first, no axis machinery is exercised on either side. The fixture diff of `max_abs_diff = 8.2e-8` is consistent with float32 quantile-interpolation rounding only (linear interpolation between the two adjacent ranks).

**Verified.**

### Critical point 4 — moments_update invscale order-of-operations

Sheeprl `dreamer_v3/utils.py:60-62`:

```python
self.low  = self._decay * self.low  + (1 - self._decay) * low      # mutate first
self.high = self._decay * self.high + (1 - self._decay) * high
invscale  = torch.max(1 / self._max, self.high - self.low)         # then use new values
```

In Python with PyTorch attribute mutation, `self.low` on the RHS of L60 is the **old** value; the LHS rebinds `self.low` to the **new** value. By the time L62 reads `self.high - self.low`, both have been rebound, so the invscale floor uses the **post-update** spread.

JAX port (utils.py:253-257):

```python
new_low  = decay * state.low  + (1 - decay) * low_new       # explicit new_*
new_high = decay * state.high + (1 - decay) * high_new
new_state = MomentsState(low=new_low, high=new_high)
invscale  = jnp.maximum(1.0 / max_, new_high - new_low)     # uses new_*
```

The pure-functional rewrite is **explicit** about what was implicit in sheeprl — `new_high - new_low` is unambiguously the post-update spread. This is the correct translation of sheeprl's in-place semantic.

The deviation log (D-001) frames the all_gather drop as the only difference. The order-of-operations on the invscale floor is **not** a deviation — it preserves sheeprl's semantic. The pure-functional refactor is mathematically sound. Fixture diff of `8.2e-8` would surface any order swap as a difference at the spread scale (typically `~1`), not at `~10⁻⁷`.

**Verified.**

### Critical point 5 — symexp float32 1-ULP slack

The fixture's input is `x ~ U[-5, +5]`, so `|x|` peaks near 5 and `exp(|x|)` near `e⁵ ≈ 148.41`. The float32 ULP at value `v` is `v × 2⁻²³`:

$$\text{ULP}\bigl(e^5\bigr) \;=\; 148.41 \times 2^{-23} \;\approx\; 148.41 \times 1.192 \times 10^{-7} \;\approx\; 1.77 \times 10^{-5}$$

The observed `max_abs_diff = 1.526 × 10⁻⁵` is `0.86 ULP` at `exp(5)` — strictly less than 1 ULP. The relaxed threshold `2e-5` is `1.13 ULP` — a tight, principled bound.

The relative-diff check in the test (`max_rel_diff < 1e-5`, asserted at L113-L115 of `test_utils.py`) is the semantic-correctness gate: a true math bug would manifest as a constant relative drift (e.g., wrong exponent base, missing absolute-value, sign flipped on small `x` near zero) and would push `max_rel_diff` to `~10⁻³` or larger. The observed `2.1 × 10⁻⁷` is the same scale as a single multiplication's rounding error — confirming the gap is the transcendental-function rounding of `exp` on CUDA float32, not a semantic deviation.

The CUDA `expf` implementation in JAX (XLA) routes through `__nv_expf` from libdevice, while PyTorch's CUDA `exp` uses cuDNN's implementation; both are IEEE-754-compliant within ≤1 ULP for arguments in `[0, ~88]`, and the implementations differ in their internal polynomial approximations by a few ULP on rare values. A `0.86 ULP` discrepancy is well within both implementations' documented tolerance.

**Verified.** D-003's relaxation is mathematically defensible.

## Deviation math review

### D-001 — `moments_update` `fabric.all_gather` omitted

Sheeprl L57: `gathered_x = fabric.all_gather(x).float().detach()`. On a single-process Fabric, `all_gather` is the identity function (no peer ranks to gather from) — confirmed by Lightning Fabric documentation. The JAX port drops the call and operates on the local batch.

Mathematical consequence: in multi-rank training, the percentile is computed over the union of all per-rank batches; in single-rank training (sheeprl's and our XS config both), it is computed over the local batch only. Since we run single-rank, both sides compute over the same set of values. The fixture's `max_abs_diff = 8.2 × 10⁻⁸` is at the float32-roundoff level (a single subtraction `new_high - new_low` of two `~0.05` quantities introduces error of `~5e-8` from catastrophic cancellation), consistent with bit-identity-within-rounding.

**Math verdict: equivalent. No math deviation. The structural code change (pure-functional vs in-place mutation) preserves the math, including the post-update invscale ordering verified in critical point 4.**

### D-002 — `init_weights` / `uniform_init_weights` stochastic sampler

The mathematical statement is unchanged:

- `init_weights`: draw from truncated normal with `std = √(1/d_avg) / c_Hafner`, truncation `[-2σ, +2σ]`.
- `uniform_init_weights`: draw from `U[-ℓ, +ℓ]` with `ℓ = √(3s/d_avg)`.

Both PyTorch's `nn.init.trunc_normal_` and JAX's `jax.random.truncated_normal` implement the same distribution. Both PyTorch's `nn.init.uniform_` and JAX's `jax.random.uniform` implement the same distribution. The samplers differ in their underlying RNG streams (Mersenne Twister vs Threefry counter-based), so individual draws cannot match bit-for-bit even at the same nominal seed.

Distribution-property test design is correct: shape contract, theoretical-std agreement within 15% (which is a generous bound for `76 × 256 = 19,456` samples — the standard error on the sample-std is `~1/sqrt(2N) ≈ 0.5%`, so 15% leaves room for the truncation effect to register without raising false alarms), truncation-bound respect, and explicit zero-init verification for `given_scale=0`. The 12% measured rel-err on `init_weights` is plausible (the truncated-normal sample std is systematically below the theoretical un-truncated std by the Hafner factor — `0.8796×` — which is partially what the constant `c_Hafner` is correcting for; some residual variance reduction near the bounds is expected).

**Math verdict: formula identical. The RNG-stream mismatch is implementation-level, not mathematical. The distribution test correctly verifies the math without depending on RNG correspondence.**

### D-003 — `symexp` float32 1-ULP threshold

Discussed at length in critical point 5. The formula `sign(x) * (exp(|x|) - 1)` is identical to sheeprl's. The 1-ULP gap is the float32 `exp` rounding difference between JAX (XLA `expf`) and PyTorch (cuDNN `exp`).

The relative-diff guard at `1e-5` is the math-vs-precision separator: a math bug (wrong sign, wrong base, wrong shift) cannot fit under `2.1e-7` relative on inputs up to `|x| = 5`. The threshold is mathematically defensible.

**Math verdict: formula identical. The 1-ULP gap is hardware-precision-level, not mathematical.**

## Verdict

**PASS — math is bit-faithful to sheeprl.**

All eight functions implement the equations identically to the vendored sheeprl source. The three logged deviations (D-001, D-002, D-003) are not math deviations — they are an unused distributed-training no-op, an RNG-stream difference, and a transcendental-function rounding difference, respectively, each with the correct math underlying.

The historical Hafner-constant truncation bug (`0.8796`) — one of the original motivations for the dreamer-srl rebuild and for Lever B (source-citation discipline) — is **not** present: the JAX port uses the full-precision constant `0.87962566103423978`, byte-identical to sheeprl.

The `compute_lambda_values` recursion, the `Moments` post-update invscale order, the `Ratio` integer-exact match, the `prepare_obs` `T=1` axis prepend, and the `symlog`/`symexp` formula are all line-for-line faithful. No findings to escalate; no rerouting required.

**Professor-rl-bayesian-dl may fire next.** Their scope (algorithm-level training-loop semantics §S1–S10) does not overlap with CP1's leaf utilities, but they should confirm that the caller-side `[1:]` slicing on `(rewards, values, continues)` before `compute_lambda_values`, the `continues * γ` pre-multiply, the §S5 true-continue splice, and the §S6 discount weighting all land in CP6 (`train.py`) and not anywhere upstream of it.

Reviewed by: math-reviewer
