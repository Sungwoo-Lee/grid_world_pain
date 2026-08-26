---
title: "Distance-degraded vision — point-spread mechanism study (anisotropic blur, masking, per-cell olfaction)"
topic: sensors
status: active
created: 2026-08-19
last_updated: 2026-08-19
phase: null
aliases: [visual-psf-study, anisotropic-visual-blur]
---

# Distance-degraded vision — point-spread mechanism study

**Shareable page:** https://claude.ai/code/artifact/246faf74-8c5c-48f5-a706-fff8a809f418
**Sandbox:** [`visual_psf_study/`](visual_psf_study/) — pure numpy, independent of `src/`.

## Question

The visual sensor currently reports an exact cell match, so an object two cells away is delivered
with perfect position *and* perfect identity. That is far stronger than a real retina, and it makes
the agent's world too legible. We want vision to degrade with distance the way olfaction already
does through its `1/d^γ` decay — but by corrupting the signal rather than merely dimming it.

The obvious approach, a Gaussian blur that widens with distance, has a failure mode: widen it enough
and the *north* cell and the *east* cell report the same value for an object that is plainly
up-and-to-the-right. Direction is exactly the information worth keeping, so a mechanism that
destroys it is the wrong mechanism. This study measures the candidates before any of them is built.

## Headline finding

**It is not blur that destroys bearing — it is *tangential* blur.** An isotropic kernel spreads
signal equally along the ray to the object and across it. Only the across-the-ray component carries
bearing; only the along-the-ray component carries distance. Widen them together and both are lost.
Widen them separately and one can be spent to buy the other.

The fix is an anisotropic Gaussian in a polar frame, elongated along the agent→object ray:

$$w(c) = \exp\left(-\frac{(v \cdot \hat{u})^2}{2\sigma_\parallel^2} - \frac{(v \cdot \hat{t})^2}{2\sigma_\perp^2}\right), \qquad v = c - e$$

`û` is the unit vector from agent to object, `t̂` is perpendicular to it. `σ_∥` grows quickly with
distance — *where* it is becomes vague. `σ_⊥` stays tight — *which way* it is stays sharp.

This is not a patch but the more honest abstraction. A real retina measures **angle** directly (a
pixel *is* a direction) and never measures distance at all; distance is inferred. Defining the
tangential width by a fixed angular blur, `σ_⊥ = d · sin Δθ`, gives constant angular uncertainty
with linear spread growing in cells — which is the "something is on my left, I don't know how far"
percept we are after.

Measured, for an object at distance 3, range-2 diamond — north/east signal ratio:

| object bearing | isotropic | anisotropic |
|---|---|---|
| 20° (mostly north) | 3.9× | 150× |
| 45° (exact diagonal) | 1.0× | 1.0× |
| 70° (mostly east) | 0.26× | 0.01× |

At 45° both report north = east. That is **correct**, not a bug — the object really is on the
diagonal. The pathology is that isotropic blur makes 20° and 70° read as nearly equal too.

## What each figure settles

| Fig | Question it answers |
|---|---|
| 0 | Anatomy of the kernel — a plain Gaussian, stretched along one axis and pointed at the object |
| 1 | Kernel shape on the grid — the isotropic disc vs the anisotropic cigar pointing at the object |
| 2 | What the agent actually reads, for the top-right case that motivated the study |
| 3 | Angular discriminability vs distance, for both kernels |
| 4 | Whether distant objects get fainter — depends entirely on the normalisation choice |
| 5 | One scene under all five candidate mechanisms, side by side |
| 6 | The anisotropy knob ρ = `σ_∥`/`σ_⊥`, and where its useful band lies |

## Reading it as a newcomer

The shareable page is written to be followed from first principles, for a reader who has not met a
covariance matrix inside a Gaussian exponent before. It carries a six-part **Background** section
(point-spread functions; the Gaussian and its width; covariance as two widths instead of one;
rotating the ellipse onto the ray; normalisation; angles versus lengths), Fig 0 as a visual anatomy
of the kernel, and a **worked example** that takes one object and two cells all the way from
coordinates to final weights. The worked example is the left column of Fig 2, so every number in it
can be located in the plot.

Two implementation consequences are called out there and are easy to miss:

- **It is a matmul, not a convolution.** `σ_∥` depends on each object's distance from the agent, so
  the kernel is spatially varying and the operator is not shift-invariant. Costs nothing here — the
  sensor is already a matmul over entities; the boolean match matrix simply becomes real-valued.
- **The sensor loses its hard range.** Every object then has some weight in every cell (negligible
  past a few σ, but nonzero), so `visual_sensor_range` controls how many cells are *output*, not how
  far the agent can see. A separate cutoff radius mirroring olfaction's is available if wanted.

## Two corrections the measurements forced

Recorded because both were asserted confidently before the sandbox existed, and both were wrong:

1. **"The anisotropic kernel keeps its angular sharpness at every range" — false.** Both kernels
   degrade with distance (Fig 3). The defensible claim is narrower: anisotropic stays 3–5× sharper
   at every range, and buys that without reducing the radial blur at all. Radial vagueness and
   angular sharpness are independent axes; that independence is the actual result.
2. **"Angular sharpness saturates around ρ ≈ 3–4" — false.** It rises indefinitely (Fig 6). The real
   limit is different and more useful: once `σ_⊥` falls below roughly half a cell, the blob no longer
   reaches the cells either side of the ray, so further increases buy astronomical value ratios
   rather than information. Same practical band, defensible reason.

## The normalisation trap (Fig 4)

Not raised in the original design discussion, and consequential. If each object's kernel is
normalised over the *visible* cells, distant objects stop fading entirely — renormalising puts back
exactly the mass the blur pushed outside the diamond:

| object distance | normalised over diamond | normalised by full analytic mass |
|---|---|---|
| d = 1 | 0.513 | 0.262 |
| d = 2 | 0.698 | 0.161 |
| d = 4 | 0.486 | 0.032 |

Normalising by the kernel's full analytic mass (`2π σ_∥ σ_⊥`) reports only the fraction landing
inside the diamond, so the falloff comes **for free** — vision needs no separate `1/d^γ` term. The
mechanism is real and the monotonicity holds; the *magnitude* quoted here was wrong, see the
correction below.

> **Correction (2026-08-26, math review).** This section originally claimed "~30×" from d=1 to d=5.
> Measured on the **shipped** kernel that number is **9.1×** for the in-diamond total (17× for the
> brightest cell). The 30× came from this study's sandbox, which has **no σ floor**; the
> implementation floors both widths at half a cell, which fattens the near-field kernel and flattens
> the ratio. The table above is likewise a range-1 brightest-cell quantity, not the range-2 totals
> Fig 4 plots — read it as illustrative of the *mechanism*, not as the shipped numbers.

## Cost — measured, not estimated

Benchmarked on this machine (RTX-class CUDA device, JAX 0.9.0.1) with the real environment,
`num_envs: 128`, via [`visual_psf_study/bench_aniso.py`](visual_psf_study/bench_aniso.py). That
script also contains a **complete reference implementation** of the anisotropic sensor, so the
measurement and the proposed code are the same thing.

**The kernel is free.** Replacing the boolean match matrix with the Gaussian weight matrix costs
+2 to +7 µs per batched call of 128 environments — against an `env.step` of 350–420 µs, so under 2%.
Over a 10M-step run that is **half a second of extra compute in total**:

| `visual_sensor_range` | cells | obs dim | kernel delta | % of `env.step` | total over 10M steps |
|---|---|---|---|---|---|
| 0 | 1 | 27 | +2.4 µs | 0.60% | 0.2 s |
| 1 | 5 | 59 | +7.3 µs | 1.78% | 0.6 s |
| 2 | 13 | 123 | +6.5 µs | 1.72% | 0.5 s |
| 3 | 25 | 219 | +5.6 µs | 1.24% | 0.4 s |

The delta does **not** grow with the number of cells even though the arithmetic scales 25× from
range 0 to range 3. That is the tell: at these sizes the operation is kernel-launch-bound, not
FLOP-bound. The GPU is idle inside the call either way.

**The network side is also free**, contrary to the reasonable worry that a hierarchical encoder makes
observation growth expensive. The visual unimodal MLP's first layer grows from 1,024 to 13,312
parameters at range 2, but a forward+backward over a full rollout (128 envs × 128 steps) measures
100–155 µs at *every* range — the differences are inside the launch-overhead noise. Across 10M steps
at `K_epochs: 4` that is ~0.3 s. Rollout buffers grow from 1.77 MB to 8.06 MB per rollout, which is
nothing on a 24 GB card.

**Conclusion: compute is not the constraint.** Nothing in this proposal costs measurable wall-clock at
10M steps. What growing the observation actually costs is **sample efficiency** — a 123-dimension
observation with 104 visual dimensions is a harder representation to learn than 27 with 8, and that is
paid in environment steps, not microseconds. Argue about `visual_sensor_range` on learning grounds;
the performance argument does not exist.

## Implementation variants — parity and speed, measured

[`visual_psf_study/bench_visual_variants.py`](visual_psf_study/bench_visual_variants.py) tests the two
paths separately, because they have different obligations.

**The OFF path must be bit-identical to today, and is.** The plan restructures it — the activity mask
moves from `all_props` onto the weight matrix `W`, and a mask-gate multiply is added to both paths —
so parity was a claim to verify rather than assume. Measured on GPU against the real environment:
**bit-identical at ranges 0, 1 and 2**, max absolute difference exactly 0. (The plan review had only
CPU micro-checks; this closes that gap.) The reason is that `matches` and `all_active` are exactly 0.0
or 1.0, so reordering their multiplication is exact in IEEE arithmetic.

**The ON path is new behaviour with no parity obligation**, so it can be chosen on speed and accuracy.
Four formulations, `env.step` = 352 µs for scale:

| variant | range 0 | range 1 | range 2 |
|---|---|---|---|
| exact match (today) | 8.7 µs | 12.8 µs | 13.7 µs |
| V1 — projections as `cells @ u.T`, normalise `W` | 9.9 µs | 16.1 µs | 17.0 µs |
| **V2 — `(c − e)` first, then contract** | 11.4 µs | **13.8 µs** | 17.7 µs |
| V3 — fold normalisation into `props` | 9.9 µs | 18.1 µs | 17.8 µs |
| V4 — `rsqrt` for the unit vector | 9.9 µs | 16.6 µs | 17.4 µs |

Every variant sits within a few µs of the others and within ~1% of `env.step`. There is no meaningful
speed choice to make.

### The finding that matters: GPU matmul precision

V1 and V2 are algebraically identical but disagreed by **5.4e-04** — far above float32 rounding. A
float64 reference showed the two projection formulas differ by only ~1.6× in accuracy, nowhere near
enough to explain it. The actual cause:

| `jax_default_matmul_precision` | V1 vs V2, max abs diff |
|---|---|
| default (unset) | 5.379e-04 |
| `'highest'` | 3.576e-07 |

**V1's `[C,2] @ [2,E]` projection runs in reduced precision (TF32) by default on this hardware**,
keeping roughly three decimal digits. V2 expresses the same contraction in a form XLA does not route
through the reduced-precision path.

**This reverses implementation note 1 below, which said never to materialise the `[C, E, 2]`
displacement tensor.** That advice was written to save memory traffic; measurement shows the tensor is
free at these sizes (V2 is the *fastest* variant at range 1) and that avoiding it via a matmul costs
three decimal digits. **Use V2.**

Two caveats worth carrying forward. First, 5e-04 is 400× smaller than the perceptual-noise σ of 0.2,
so this was never going to corrupt the science — but it would make a geometry unit test flaky and
could make results differ across GPU models. Second, and pre-existing: `sense_visual`'s
`matches @ all_props` runs under the same default. It is harmless today because the shipped visual
properties are one-hot 0/1 vectors, exactly representable in TF32 — but it becomes a real source of
error for anyone who configures non-trivial `visual_properties` or sets `visual_properties_std > 0`.

## Implementation notes for the JAX version

Four things the reference implementation gets right, each of which is easy to get wrong:

1. **The structure does not change.** It is still one `[C, E] @ [E, V]` matmul. Only the contents of
   the left matrix change, from `jnp.all(coords == pos)` booleans to Gaussian weights. No new pass,
   no convolution, no scatter.
2. **Never form the `[C, E, 2]` displacement tensor.** Since `v·û = c·û − e·û`, the projections come
   from two small matmuls (`cell_coords @ u.T`) minus a per-entity constant. Keeps the working set at
   `[C, E]` instead of `[C, E, 2]`.
3. **Floor both widths at about half a cell.** Mass normalisation divides by `2π σ_∥ σ_⊥`, so an
   entity standing on the agent (`d = 0` → `σ_∥ = 0`) sends the peak to infinity. A floor of ~0.5
   cells fixes it — and it is principled rather than a patch: half a cell is the grid's sampling
   limit, and Fig 6 independently found that sub-cell `σ_⊥` buys nothing but larger value ratios.

   **The floor has a consequence this study understated: it caps the effective anisotropy at short
   range.** With the shipped `scale = 0.5`, `ρ = 3.0`, `floor = 0.5`:

   | entity distance | σ_∥ | σ_⊥ | **effective ρ** |
   |---|---|---|---|
   | 1 | 0.50 | 0.50 | **1.00 — isotropic** |
   | 2 | 1.00 | 0.50 | 2.00 |
   | ≥ 3 | 0.5·d | 0.5·d/3 | 3.00 (as configured) |

   So "distance goes vague while bearing stays sharp" is true from three cells out, and **at one cell
   the kernel is isotropic no matter what ρ is set to**. That is defensible — an adjacent object's
   bearing is already unambiguous — but it is enforced, not chosen, and Fig 6's "past σ_⊥ < half a
   cell you buy nothing" is therefore a hard cap in the implementation rather than advice.

   **The figures in this study were generated without the floor** (`psf_lib.py` omits it), so panels
   at d < 3 depict the idealised kernel rather than the shipped one. Panels at d ≥ 3 — including
   Fig 2's worked case at d = 3.5 — are unaffected.
4. **Keep the new config values as traced arrays, not static.** `visual_sensor_range` is
   shape-determining and must stay static; the blur scale, anisotropy and floor must not, or every
   sweep value triggers a recompile.

## Decisions

### Settled

| Item | Decision |
|---|---|
| Olfaction | Per-cell diamond, field resampled at each cell; distances and the `sensor_radius` cutoff measured from that cell. Nothing else changes — no mask key, no added noise. `olfactory_grid_range: 0` must stay byte-identical to today. |
| Masking | Per-entity `visual_mask: none \| far \| all` |
| Blur | Deterministic. No stochastic term. |
| Support | Folded into the entity matmul — a weight matrix replaces the boolean match matrix, so the periphery is soft and objects outside the diamond bleed into its edge. |

### Open

| Item | Options |
|---|---|
| Architecture | Anisotropic kernel on the existing diamond; sharp near field + far sectors; or a log-polar map (Fig 5) |
| Width knobs | Radial scale + fixed Δθ; two independent linear scales; or one width + anisotropy ρ |
| Normalisation | Full analytic mass; over visible cells; or none (Fig 4 argues the first) |
| Mask order | Whether `far` zeroes the blurred weight at d≥1, or lets a masked object contribute undimmed to the centre cell only |
| **Sensor range** | **Blocking.** Every question above resolves differently at range 1 than at range 3. At range 0 none of this does anything. |

### Known risk, accepted

A fixed linear blur is in principle invertible: the network sees the same operator every step, the
content is sparse and non-negative, and the policy is recurrent — close to a best case for learning
an implicit deconvolution. Blur destroys information only where the operator is rank-deficient. The
decision to use deterministic blur with no stochastic term is defensible for reproducibility, but
this is the assumption most likely to be falsified by a sufficiently trained agent. Adding a small
Gaussian is what would make the deconvolution ill-posed, and remains available as a follow-up.

## Reproducing

```bash
cd docs/develop/active/sensors/visual_psf_study
PY=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$PY fig0.py           # figure 0 — kernel anatomy
$PY make_figs.py      # figures 1-5
$PY fig6.py           # figure 6
$PY worked_example.py # prints the worked-example numbers + asserts the matrix
                      # form v^T Sigma^-1 v equals the explicit two-term exponent
$PY build_page.py     # rebuild index.html
```

Every number quoted in the page's prose is **computed at build time** from the same formulas the
figures use, so the text cannot drift away from the plots.

`index.html` is generated (1.1 MB of base64-embedded PNGs) and gitignored — the PNGs and scripts are
tracked. Republishing to the same artifact URL keeps the existing link.

## Related

- [[OLFACTORY_EXPANSION_STUDY]] — companion study for the per-cell olfactory expansion. Establishes
  that the two senses become the same operator with different kernels (global `1/d^γ` versus local
  Gaussian), so one piece of machinery can serve both.
- [[09_sensors_and_observation]] — canonical sensor reference. **Its §9 is stale**: it documents the
  pre-v3.0 hardcoded one-hot visual sensor, `decay_power` 2.0 (config says 1.0), and olfactory
  signatures that no longer match `default.yaml`. Must be corrected alongside any change here.
- [[CONFIGURABLE_VISUAL_PROPERTIES]] — the v3.0 refactor that made visual properties per-entity
  vectors; this study builds directly on that matmul structure.
- [[10_perceptual_noise]] — the injury-scaled noise system, orthogonal to this blur.
- [[OLFACTORY_PROPERTY_VARIANCE]] — per-episode chemical-signature sampling.
