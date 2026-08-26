---
title: "v3.1 directional sensors — implementation report (parity, tests, throughput)"
topic: sensors
status: active
created: 2026-08-21
last_updated: 2026-08-21
phase: null
aliases: [v31-implementation-report, directional-sensors-report]
---

# v3.1 directional sensors — implementation report

**Shareable page:** https://claude.ai/code/artifact/873a463d-53ae-46a3-9ee9-08515d8182bd
**Commit:** `0e8a4ef` on `v3.0` · 153 files
**Plan:** [[DIRECTIONAL_SENSORS_PLAN]] · **Sandbox:** [`v31_implementation_report/`](v31_implementation_report/)

## What shipped

Three opt-in behaviours plus one unconditional correction.

| change | switch | ships as |
|---|---|---|
| Olfaction sampled over a Manhattan diamond, so its readings carry a direction | `sensory.olfactory_grid_range` | `0` (off) |
| Anisotropic point-spread blur on vision — distance vague, bearing sharp | `sensory.visual_blur_enabled` | `false` (off) |
| Per-entity invisibility | `visual_mask: none \| far \| all` | `none` (off) |
| On-source decay expressed as a half-cell floor `1/(0.5^γ)` | none | always |

**With the shipped configuration the observation is bit-for-bit identical to pre-change**, verified
against a fixture captured from the old code.

## Verification

| check | result |
|---|---|
| Observation identical to pre-change fixture (pinned CPU, 3 runs) | **bit-identical** |
| Full environment test suite | **226 passed, 0 failed**, 590 skipped |
| New `tests/env/test_directional_sensors.py` | **17 passed** |
| Breakdown sums to actual observation width (5 range combinations) | pass |
| Blur weights finite everywhere, including an entity on the agent's cell | pass |
| Anisotropy = 1 reproduces an isotropic kernel | pass |
| Total weight falls monotonically with distance | pass |
| Hidden entity contributes exactly zero, including the centre cell | pass |
| Out-of-bounds olfactory cells read exactly zero | pass |
| Olfactory gradient points at the source from all four bearings | pass |
| `1/(0.5**1.0)` bit-identical to the literal `2.0` in the traced form | pass |

### Two findings from the verification itself

**Byte-parity is only meaningful on a pinned backend.** Mid-implementation the parity check failed and
passed alternately on GPU (3 broken / 2 held in five runs) and failed 5/5 on CPU. Neither was the code:
the fixture had been captured on GPU, so CPU runs were a cross-backend comparison that can never match,
and repeat GPU runs differ from each other because of autotuning. `capture_sensor_baseline.py` now
forces `JAX_PLATFORMS=cpu` for both capture and comparison and asserts the backend, so the check is
deterministic and portable. This is the same class of effect as
[[20260820_1606_reset_ulp_divergence_is_compiler_fusion]].

**The `far`-mask regression test earns its place.** It fails against the plan's original specification,
which gated on the cell's distance rather than the entity's and leaked ~13% of a visible entity's peak
into the agent's own cell under blur.

## Throughput

`num_envs: 128`, jitted `lax.scan` of 200 steps, best of 7.

| configuration | observation | steps/sec | µs/step | vs baseline |
|---|---|---|---|---|
| baseline (all off) | 27 | 1,668,946 | 76.7 | — |
| olfaction range 1 | 47 | 1,574,770 | 81.3 | −5.6% |
| vision range 2, blur off | 123 | 1,550,693 | 82.5 | −7.1% |
| vision range 2 + blur | 123 | 1,526,021 | 83.9 | −8.6% |
| **both (chosen settings)** | **143** | **1,442,930** | **88.7** | **−13.5%** |

**Free when off**: the all-off baseline after the change (1,668,946 SPS) matches the baseline measured
before any code was written (1,662,434 SPS).

**The chosen settings cost 13.5%**, which is more than the sensor arithmetic — micro-benchmarks put the
kernels at 1–2% ([[VISUAL_PSF_MECHANISM_STUDY]] §Cost, [[OLFACTORY_EXPANSION_STUDY]] §Cost). The
balance is the observation growing 27 → 143 numbers, which the whole rollout carries. **The lever is
diamond size, not sensor code.** Ten million environment steps: ~115 min at the chosen settings versus
~100 min at baseline.

## v3.2 addendum — value mode and occlusion

Two further visual options, added 2026-08-21 (commit `9771e98`) after the first five training
arms were already running. Both are config knobs rather than baked-in choices, so they can be
swept like the blur.

| key | default | meaning |
|---|---|---|
| `sensory.visual_value_mode` | `sum` | `sum` = weighted sum (two rocks read 2.0); `clamp` = per-channel presence capped at 1.0. Entity contribution only — terrain keeps its own value. |
| `sensory.visual_occlusion_enabled` | `false` | A nearer `blocks_sight` entity hides anything inside the shadow cone of the ray to it. |
| `sensory.visual_occlusion_cone_deg` | *(conditional)* | Half-angle of the cone. Read only when occlusion is on. |
| `sensory.visual_occlusion_strength` | *(conditional)* | `1.0` hides fully; lower attenuates. |
| per-entity `blocks_sight` | `false` | Independent of `blocking` (movement) and `hides_agent` (concealment). |

**Single-channel presence needed no new code** — `visual_vector_size: 1` with zeroed
`visual_background_properties` already worked via the v3.0 configurable-properties system. What
v3.2 adds is the sum-vs-clamp choice.

### Occlusion cost, measured properly

First attempt reported the isolated gate at 11.8 µs and a batch sweep put the end-to-end effect
anywhere from −1% to −11% — **the sweep contained an impossibility** (blur-*on* measuring faster
than blur-off), which meant min-of-7 across separately-compiled configs could not resolve a 1%
effect. Re-measured **paired and interleaved**, 40 rounds, same process:

| configuration | median µs/step | IQR | vs occlusion off |
|---|---|---|---|
| occlusion off (`visual_sensor_range: 2`) | 80.81 | 1.00 | — |
| occlusion on, 5° | 84.50 | 0.95 | **+4.57%** |
| occlusion on, 15° | 84.58 | 0.64 | **+4.67%** |

**+3.7 µs per step, about 4.6%.** The **cone angle does not affect cost** — identical tensor
shapes either way, only the comparison threshold moves — so the angle is a science knob, not a
budget one. Note the isolated-gate figure (11.8 µs) *over*-estimated the end-to-end cost, which
is what you would expect once XLA fuses the gate into surrounding work.

### Why a cone and not a grid line

Measured on the real scene: exact collinearity fires on 2.76% of occluder/target pairs versus
16.7% for a 15° cone, and it fires almost only along axes and perfect diagonals. Strict occlusion
would make the sensor blind north-south and clear-sighted obliquely — a worse artefact than the
thing it models.

### Calibration

This scene holds up to 36 entities on 100 cells, so occlusion bites hard. Share of live entities
hidden, obstacles-only blocking: **5° → 31%, 10° → 47%, 15° → 59%, 30° → 75%**. Even the
narrowest sensible setting hides a third of the world.

---

## Deviations from the plan

- **Tests consolidated into one module**, not the six the plan named; sections map 1:1 onto the plan's
  test table.
- **Config sweep was 146 files, not the ~98 estimated.** Affected files were found empirically by
  loading every config and catching failures rather than by pattern. Only 11 standalone *configs*
  needed the keys; a further set of test fixtures and **archived configs still loaded by the test
  suite** did too, which the estimate had missed.

## Known consequence

Eleven configs use `decay_power: 2.0`; their on-source reading changes 2.0 → 4.0, the accepted
consequence of applying the fix unconditionally. Two are the olfactory parity-verification configs,
whose stored expectations need regenerating if re-run. Everything at the shipped `decay_power: 1.0` is
untouched.

## Not done

Environment side only. Nothing trained, no experiment configs written, no claim that any of this helps
the agent. Next step is choosing an experiment, not more code.

## Related

- [[DIRECTIONAL_SENSORS_PLAN]] · [[VISUAL_PSF_MECHANISM_STUDY]] · [[OLFACTORY_EXPANSION_STUDY]] · [[ONSOURCE_RULE_STUDY]]
- [[09_sensors_and_observation]] — **still needs updating** for v3.1, and still carries the pre-v3.0
  staleness in §6 and §9 found during the studies.
