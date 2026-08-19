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
inside the diamond, giving a ~30× falloff from d=1 to d=5 **for free** — vision would then need no
separate `1/d^γ` term at all.

## Decisions

### Settled

| Item | Decision |
|---|---|
| Olfaction | Per-cell diamond, field resampled at each cell; distances and the `sensor_radius` cutoff measured from that cell. Nothing else changes — no mask key, no added noise. `olfactory_sensor_range: 0` must stay byte-identical to today. |
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

- [[09_sensors_and_observation]] — canonical sensor reference. **Its §9 is stale**: it documents the
  pre-v3.0 hardcoded one-hot visual sensor, `decay_power` 2.0 (config says 1.0), and olfactory
  signatures that no longer match `default.yaml`. Must be corrected alongside any change here.
- [[CONFIGURABLE_VISUAL_PROPERTIES]] — the v3.0 refactor that made visual properties per-entity
  vectors; this study builds directly on that matmul structure.
- [[10_perceptual_noise]] — the injury-scaled noise system, orthogonal to this blur.
- [[OLFACTORY_PROPERTY_VARIANCE]] — per-episode chemical-signature sampling.
