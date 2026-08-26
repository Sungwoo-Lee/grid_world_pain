# Math review — directional sensors (anisotropic visual blur, per-cell olfaction, on-source rule, occlusion)

**Reviewed:** 2026-08-26 · branch `v3.0` · code `src/environment/sensor.py` (`_psf_weights`, `_occlusion_gate`, `_visual_mask_gate`, `sense_resource`, `sense_olfaction_cells`)
**Against:** VISUAL_PSF_MECHANISM_STUDY.md · OLFACTORY_EXPANSION_STUDY.md · ONSOURCE_RULE_STUDY.md · DIRECTIONAL_SENSORS_REPORT.md
**Numerical checks:** `tmp/20260826_math_review_directional_sensors.py` (22 checks, run on CPU with the project interpreter; every check passed — the findings below are doc-vs-code and doc-vs-doc discrepancies, not code bugs).

## Verdict (plain language)

The new sensor code was reviewed against the four design documents that justify it. The question was: do the equations in the documents hold, and does the code implement those equations rather than something slightly different?

**The mathematics in the code is sound.** The elongated Gaussian "blur" that makes distant objects look vague is a genuine 2-D Gaussian (verified against the textbook matrix form), its two axes really are perpendicular unit vectors, and the constant it divides by really is the kernel's total mass — which is what makes far-away objects correctly read fainter. The rule that replaces "standing on a smell source" with "half a cell away" is exactly what the study claims: identical to the old hard-coded value at the shipped setting, bit for bit, and the feared discontinuity sits in a region an agent on a grid of whole-number cells can never reach. The line-of-sight shadow test is the correct geometric test for "inside a cone".

**One document overstates its numbers.** The visual-blur study's headline "signal falls about 30-fold from one cell away to five cells away" is not reproduced by its own sandbox (4.5–15x depending on what you measure) nor by the shipped code (9x for total signal, 17–24x for the brightest cell), and the table printed beside that claim turns out to come from a different measurement than the one the surrounding text describes. The mechanism the study argues for is real and verified — only the specific magnitude is wrong. Nothing needs re-running; the study text needs correcting.

**Severity legend — reproduce it verbatim in every report so the labels never need looking up:** 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run · 🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

---

## Equations under review

**(Eq. 1) Anisotropic point-spread weight** — VISUAL_PSF_MECHANISM_STUDY.md §Headline finding; implemented at `src/environment/sensor.py:212-217`:

$$w(c) = \exp\left(-\frac{(v\cdot\hat u)^2}{2\sigma_\parallel^2} - \frac{(v\cdot\hat t)^2}{2\sigma_\perp^2}\right),\qquad v = c - e$$

with `û = (e − a)/d` the agent→entity unit vector and `t̂ = (−û_y, û_x)`.

**(Eq. 2) Claimed equivalence to the general 2-D Gaussian:**

$$w(c) = \exp\left(-\tfrac{1}{2}\, v^{\mathsf T}\Sigma^{-1} v\right),\qquad \Sigma = R\,\mathrm{diag}(\sigma_\parallel^2, \sigma_\perp^2)\,R^{\mathsf T},\quad R = [\hat u\;\; \hat t]$$

**(Eq. 3) Mass normalisation** — `src/environment/sensor.py:217`:

$$W \leftarrow \frac{w}{2\pi\,\sigma_\parallel\,\sigma_\perp}$$

**(Eq. 4) Width laws with floor** — DIRECTIONAL_SENSORS_PLAN.md §Design (lines 124-125); `src/environment/sensor.py:208-210`:

$$\sigma_\parallel = \max(k\,d,\; f),\qquad \sigma_\perp = \max(\sigma_\parallel/\rho,\; f)$$

shipped `k = 0.5`, `ρ = 3.0`, `f = 0.5` (`configs/environment/default.yaml:223-228`).

**(Eq. 5) Olfactory decay with on-source rule** — ONSOURCE_RULE_STUDY.md §Headline finding; `src/environment/sensor.py:19-20`:

$$\mathrm{decay}(d) = \begin{cases} 1/0.5^{\gamma} & d < 0.001 \\ 1/(d^{\gamma} + 10^{-10}) & \text{otherwise}\end{cases}$$

**(Eq. 6) Occlusion predicate** — DIRECTIONAL_SENSORS_REPORT.md §Addendum; `src/environment/sensor.py:251-254`: entity `i` is hidden iff there exists an active sight-blocking entity `j` with

$$d_j < d_i \quad\text{and}\quad \hat u_i \cdot \hat u_j > \cos\theta$$

---

## Findings table

| # | Sev | Location | Claim / issue | Detail |
|---|---|---|---|---|
| 1 | 🟡 | VISUAL_PSF_MECHANISM_STUDY.md §The normalisation trap (Fig 4) | Table numbers are not what the surrounding text says they are, and "~30× falloff from d=1 to d=5" is not reproduced by any measured quantity | The table's values (0.513/0.698/0.486 and 0.262/0.161/0.032) are **exactly** reproduced only as the *brightest-single-cell* value on a **range-1** diamond at bearing **20°** (search over range/bearing/scale/quantity; residual < 0.002). The section presents them under a Fig-4 heading, and Fig 4 is a range-2 diamond at bearing 30° whose left panel plots *totals*. Measured falloff d=1→5: study's own Fig-4 setup — totals **4.5×**, peaks **10.6×**; the table's own quantity — **14.6×**; implemented code at shipped defaults — in-diamond total **9.1×** (axis) / **10.8×** (3-4-5 diagonal), brightest cell **17×** (axis) / **24×** (diagonal). No measured quantity gives ~30×. The claim also appears verbatim in the artifact page's static prose (`visual_psf_study/build_page.py:57`), so "every number computed at build time" does not cover it. **The qualitative claims all hold**: mass normalisation produces a strictly monotone distance falloff (verified 1.005 > 0.623 > 0.326 > 0.173 > 0.111 for d=1..5), diamond normalisation removes it entirely (total ≡ 1.000 at every distance), and no separate `1/d^γ` term exists in the code (`sensor.py:215-217` contains only the exponent and the mass division; asymptotically the in-diamond fraction falls as `A/(2π σ_∥ σ_⊥) ∝ 1/d²` once both floors unbind). Suggested correction: restate the table as "range-1 diamond, bearing 20°, brightest cell" or regenerate it from Fig 4's actual quantities, and replace "~30×" with the measured ~9× (total) / ~17× (peak) at shipped parameters. |
| 2 | 🟢 | `src/environment/sensor.py:208-210`; `configs/environment/default.yaml:223`; `src/environment/state.py:272` | Effective anisotropy at short range is not the configured ρ | With shipped values the floors give effective `σ_∥/σ_⊥` = **1.00 at d=1, 2.00 at d=2, 3.00 (= ρ) only from d≥3** (full ρ requires `k·d ≥ ρ·f`). So the kernel is *isotropic* at d=1 and the study's headline "σ_⊥ stays tight — which way it is stays sharp" does not describe d<3 exactly. Mitigations already in place: DIRECTIONAL_SENSORS_PLAN.md line 483 records the interaction as intended behaviour; the study's headline table is quoted at d=3, where ρ is fully expressed; and at d=1 the blur is half a cell anyway, so bearing is barely blurred in absolute terms. The plan asked for "a note in the config comment"; the shipped comment (`default.yaml:225-228`) explains the infinite-peak reason for the floor but not the effective-ρ reduction, and the one-line formulas at `default.yaml:223` / `state.py:272` (`sigma_parallel = scale * distance`) omit the `max(·, floor)`. Cosmetic doc/comment completion, no code change. |
| 3 | 🟢 | `src/environment/sensor.py:217` (Eq. 3) | "Mass conservation" on the integer lattice is approximate, not exact, at the floor width | The *integral* is exactly `2π σ_∥ σ_⊥` (verified to 8 decimals), but the sensor sums over unit cells, i.e. a Riemann/Poisson lattice sum. At the floor `σ_∥ = σ_⊥ = 0.5` the total lattice mass of a normalised kernel is **1.0290** (+2.9%), so an adjacent entity contributes an in-diamond total of 1.0049 — slightly *more* than the exact-match branch's 1.0. Deviation shrinks fast with σ (−0.00% at d=5). No document claims exactness, so this is a recorded caveat, not an error; worth knowing if anyone ever asserts `sum(W) ≤ 1` in a test. |
| 4 | 🟢 | ONSOURCE_RULE_STUDY.md §Question ("sensor.py:14"); DIRECTIONAL_SENSORS_REPORT.md §Addendum heading | Stale/cosmetic doc slips | The guard now lives at `src/environment/sensor.py:19-20` (line 14 is a comment); the report's addendum heading reads "— value mode and occlusion — value mode and occlusion" (duplicated). Neither affects any equation. |

No 🔴 findings. Everything else checked out — the verified-true claims are listed below so they are on record as *checked* rather than assumed.

---

## Claims verified true (with the check that proved each)

**1. The anisotropic Gaussian (Eqs. 1-3).**
- `û`, `t̂` as built at `sensor.py:204-206` are orthonormal with `det[û t̂] = +1` (a proper rotation, so `R` in Eq. 2 is a valid rotation matrix; the sign of `t̂` is irrelevant anyway since it enters squared). Property-tested over 1000 random directions.
- Eq. 1 equals Eq. 2 to within 1e-15 over 2000 random `(θ, σ_∥, σ_⊥, v)` draws (float64, `Σ⁻¹` via linear solve — an independent code path from the two-term exponent).
- The unnormalised integral equals `2π σ_∥ σ_⊥` to 8 decimals (numeric quadrature vs analytic), so Eq. 3 is the correct full-mass normaliser: dividing by it makes the continuous kernel integrate to 1, and the diamond readout report only the in-diamond fraction (subject to finding 3's lattice caveat).
- The `d = 0` fallback direction `[1, 0]` at `sensor.py:204-205` is harmless: when both floors bind, `σ_∥ = σ_⊥` and Eq. 1 collapses to the isotropic `exp(−|v|²/2σ²)` — because `(v·û)² + (v·t̂)² = |v|²` for any orthonormal pair — so the arbitrary direction cancels exactly.

**2. Mass normalisation ⇒ distance falloff.** Reproduced with the *actual* `_psf_weights` at shipped parameters (range-2 diamond, entity at d = 1..5): totals fall strictly monotonically, 1.005 → 0.111. The code contains no `1/d^γ` factor and needs none — the falloff is entirely the mass pushed outside the diamond, exactly the mechanism the study argues for. Only the "~30×" magnitude is wrong (finding 1).

**3. The on-source rule (Eq. 5).** All three claims hold, including the guard interaction:
- `1/0.5^γ` **is** the decay curve at half a cell, and stays on it for every γ tested (γ = 0.5 → 1.414214, γ = 2 → 4, γ = 3 → 8, matching ONSOURCE_RULE_STUDY.md §Where it does become a problem).
- At γ = 1 it is **bit-identical** to the literal 2.0 in float32 — checked bytewise in raw numpy *and* in the jitted traced form the sensor evaluates (`.tobytes()` equality), matching `tests/env/test_directional_sensors.py:58-65`.
- The `+1e-10` denominator guard does not disturb any of this: `1/(0.5^γ + 10⁻¹⁰)` is also bitwise 2.0 at γ = 1 (10⁻¹⁰ is below half an ulp of 0.5 in float32), and the guard is **bit-invisible at every reachable distance** — checked bytewise for all 105 distinct Euclidean distances realisable between integer cells within offset 14, at γ = 1 and γ = 2. The guard existed in the pre-change code too (verified at `git show 0e8a4ef~1`), so the parity claim's baseline is right.
- The discontinuity is real and unreachable: distances are norms of integer-difference vectors, whose smallest nonzero value is exactly 1, so the open interval (0.001, 1) — where the function returns up to `1/(0.001 + 10⁻¹⁰) ≈ 1000.0` at γ = 1, matching the doc's "up to 1000" — is never sampled.

**4. The occlusion cone (Eq. 6).** On unit vectors, `cos∠(û_i, û_j) > cos θ ⟺ ∠ < θ` for θ ∈ (0°, 90°), since cosine is strictly decreasing on [0°, 180°] — property-tested over 5000 random pairs; this is the correct predicate for "inside a cone of half-angle θ". Exercised the real `_occlusion_gate`: a target 9° off a nearer blocker's bearing is hidden and one at 11° is not (10° cone); a farther blocker never occludes (strict `<` at `sensor.py:252`); equal-distance entities never occlude each other; an entity never occludes itself; a nearer *non-blocking* entity never occludes; a blocker on the agent's own cell never occludes (belt and braces: the `d > 10⁻⁶` exclusion at `sensor.py:253` plus its zero unit vector giving `cos = 0`); `strength = 0.4` attenuates to exactly 0.6. The cost claim is structurally true: `θ` enters only as the scalar threshold `visual_occlusion_cos` in an elementwise comparison at `sensor.py:254` — every tensor (`[E,E]` cos matrix included) has θ-independent shape — and the report additionally measured it (§Occlusion cost). Config loader check: `visual_occlusion_cos = cos(radians(cone_deg))` with domain guard (0°, 90°) at `src/environment/config_loader.py:1016-1021`, so the predicate's precondition (cos θ > 0 ⇒ entities behind the agent's ray direction can't accidentally occlude) always holds.
  One modelling (not mathematical) note, already acknowledged in the report's §Why a cone: the cone half-angle is *fixed*, whereas a physical unit-cell blocker subtends an angle shrinking with its distance — the docs present this honestly as an artefact-avoidance choice, not as physical shadowing.

**5. The mask gate and olfactory expansion.** `_visual_mask_gate` (`sensor.py:228-232`) gates on the *entity's* Manhattan distance as DIRECTIONAL_SENSORS_REPORT.md §Two findings requires: mask `far` keeps only `d < 1` (the entity on the agent's cell), `all` hides even that, `none` keeps everything — all three verified against the real function. `sense_olfaction_cells` (`sensor.py:44-62`) matches OLFACTORY_EXPANSION_STUDY.md §Decisions term for term: each diamond cell re-runs the untouched `sense_resource` with the cell as sampling point (so both the `1/d^γ` distances and the `sensor_radius` cutoff are measured from the cell), out-of-bounds cells are zeroed across all channels, range 0 statically bypasses the vmap, and the diamond size formula `2r² + 2r + 1` matches `get_observation_breakdown` (`sensor.py:503-505`). The olfactory study's signal table is exact forward differences of `1/d` (e.g. `1 − 1/2 = 0.500`, `1/2 − 1/3 = 0.167`), consistent with its own definition.

---

## Derivation appendix — why the two-term exponent is the general Gaussian

With `R = [û t̂]` orthogonal (`RᵀR = I`, det +1) and `Σ = R D Rᵀ`, `D = diag(σ_∥², σ_⊥²)`:

$$v^{\mathsf T}\Sigma^{-1}v = v^{\mathsf T} R D^{-1} R^{\mathsf T} v = \frac{(v\cdot\hat u)^2}{\sigma_\parallel^2} + \frac{(v\cdot\hat t)^2}{\sigma_\perp^2}$$

since `Rᵀv = (v·û, v·t̂)`. Halving both sides gives Eq. 1 ≡ Eq. 2. The integral follows by the rotation change of variables (Jacobian 1, `R` orthogonal):

$$\int e^{-\frac{1}{2}v^{\mathsf T}\Sigma^{-1}v}\,dv = \int e^{-x^2/2\sigma_\parallel^2}dx \int e^{-y^2/2\sigma_\perp^2}dy = \sqrt{2\pi}\,\sigma_\parallel \cdot \sqrt{2\pi}\,\sigma_\perp = 2\pi\,\sigma_\parallel\sigma_\perp = \sqrt{(2\pi)^2\det\Sigma}$$

which is the standard normalising constant of a bivariate Gaussian with covariance Σ — confirming `sensor.py:217` divides by exactly the full analytic mass the study's mechanism requires.

---

**Conclusion:** the directional-sensor mathematics is implemented faithfully — every formula in the code matches its design document and passed independent numerical verification; the only defects found are documentation-side (a mislabelled table and an unsupported "~30×" figure in the visual-blur study, plus cosmetic comment/line-reference slips), none requiring a code change or a re-run.

Reviewed by: math-reviewer
