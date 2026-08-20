---
title: "Directional sensors — per-cell olfaction + anisotropic visual point-spread (implementation plan)"
topic: sensors
status: active
created: 2026-08-19
last_updated: 2026-08-19
phase: null
aliases: [directional-sensors-plan, visual-blur-plan, olfactory-diamond-plan]
---

# Directional sensors — implementation plan

> **Status**: PLANNED
> **Opened**: 2026-08-19
> **Related**: [[VISUAL_PSF_MECHANISM_STUDY]], [[OLFACTORY_EXPANSION_STUDY]], [[09_sensors_and_observation]]

---

## Context

The agent's two outward-facing senses each have a defect that this plan fixes.

**Olfaction cannot point.** It returns five numbers sampled at the agent's own cell — how much of
each smell is present, and nothing about where it comes from. An agent standing in a food gradient
cannot tell which way is uphill without walking and comparing across time.

**Vision is too good.** It reports an exact cell match, so anything inside the sensor's diamond
arrives with perfect position and perfect identity. That is not an abstraction of a retina; it is a
database lookup, and it makes distant threats as legible as adjacent ones.

Both fixes are the same idea applied twice: a weight per (cell, entity) pair, matmul'd against the
entities' property vectors. Olfaction gets a diamond of sampling points so its readings carry a
spatial gradient. Vision gets a Gaussian blur that is *stretched along the line of sight*, so how far
away something is becomes vague while which direction it lies in stays sharp. Two numerical studies
established the mechanisms and measured their cost; this plan turns them into code.

Everything here is **off by default**. The shipped configuration must produce byte-identical
observations after this change, and the new behaviour is opt-in per experiment.

## Analysis

### What the studies established

From [[VISUAL_PSF_MECHANISM_STUDY]]:

- An **isotropic** blur destroys bearing: widen it and the north and east cells report the same thing
  about an object plainly up-and-right. An **anisotropic** kernel elongated along the agent→object ray
  keeps bearing 3–5× sharper at every range while blurring distance just as much. The two axes are
  independent.
- **Normalising by the kernel's full analytic mass** (`2π σ_∥ σ_⊥`) is what makes distant objects
  fade. Normalising over the visible cells instead removes the distance falloff entirely — a predator
  five cells away reads as loud as one adjacent.
- Useful anisotropy sits around ρ = 2–4; past the point where `σ_⊥` drops below half a cell, further
  increases buy value ratios rather than information.
- **Measured cost**: +2 to +7 µs per batched call of 128 envs, ~1.5% of `env.step`, ≈0.5 s across a
  full 10M-step run. Launch-bound, not FLOP-bound. The network side and buffer growth are likewise
  immeasurable. Compute is not a constraint on this change.

From [[OLFACTORY_EXPANSION_STUDY]]:

- Per-cell sampling recovers source bearing to within ~2° noiselessly.
- **At the configured olfactory σ = 0.2 the directional signal falls below the noise floor between two
  and three cells.** The level goes as `d^-γ` but the quantity carrying direction is its derivative,
  `γ·d^-(γ+1)` — one power steeper. Direction always fades faster than presence.
- A larger diamond materially helps *under noise* (median bearing error at four cells: 59° at range 1,
  30° at range 2, 14° at range 3) because it supplies redundant samples. Noiseless the ordering
  reverses past ~3 cells as curvature bias grows.
- Cost is dimensional, not computational: range 2 takes the observation from 27 to 87 dims.

### Decisions this plan implements

| Decision | Choice | Source |
|---|---|---|
| Olfactory sampling | Resample the whole field at each diamond cell; distances and the `sensor_radius` cutoff measured **from that cell** | user, confirmed |
| Olfactory scope | Diamond only — no mask key, no added noise | user, confirmed |
| Visual blur | Deterministic, no stochastic term | user, confirmed |
| Visual kernel support | Folded into the entity matmul; soft periphery | user, confirmed |
| Visual kernel shape | Anisotropic, elongated along the ray | user, confirmed |
| Visual masking | Per-entity `visual_mask: none \| far \| all` | user, confirmed |
| Width parameterisation | One radial scale + anisotropy ratio ρ | **plan's choice** — ρ=1 is exactly the isotropic kernel, so the ablation is one config value |
| Normalisation | Full analytic mass | **plan's choice** — Fig 4; the alternative silently removes distance falloff |
| Mask ordering | Zero the **entity's whole column** whenever that *entity* is at Manhattan distance ≥ 1 from the agent | **plan's choice, revised after review** — see below |

The three plan's-choice rows are the author's calls from the study evidence, not user decisions.
They are the rows most worth arguing with.

**Mask ordering was wrong in the first draft and is corrected here.** The original spec gated on the
*cell's* distance from the agent — zero every cell at distance ≥ 1. Under exact matching that is
equivalent to gating on the entity, because an entity only ever writes into its own cell. Under blur
the equivalence breaks: a `far`-masked entity three cells away still deposits weight in the agent's
own cell, which the gate leaves untouched. At the shipped knobs that leak is 0.086 at d=1, 0.043 at
d=2 and 0.029 at d=3 — 13.5%, 6.8% and 4.5% of a visible adjacent entity's peak, deterministic and
noise-free. A recurrent policy would learn it, and the first masking experiment would read
leak-driven detection as anticipatory avoidance of an unseen threat: a paper-level wrong conclusion
shaped exactly like the hoped-for result. Gating on the **entity's** distance makes `far` mean what
it says — visible only when the agent is standing on it.

### What is NOT in scope

Values. `visual_sensor_range`, `olfactory_sensor_range`, γ, ρ and the blur scale are experiment
configuration, chosen per run. This plan ships them all at parity-preserving defaults and does not
recommend training values.

## Implementation Plan

### Design

Both sensors converge on one shape:

```
weights [C, E]  @  properties [E, V]   ->   per-cell readings [C, V]
```

- **Vision** already has this matmul. Only the left matrix changes, from
  `jnp.all(cell_coords == entity_pos)` booleans to Gaussian weights. No new pass, no convolution.
- **Olfaction** currently evaluates its sum at one point. It gains a diamond of sampling points; the
  per-source `1/d^γ` decay is unchanged.

The visual kernel, per (cell `c`, entity `e`), with `û` the unit vector agent→entity and `t̂`
perpendicular to it:

```
d      = ||e - agent||
σ_∥    = max(radial_scale * d, sigma_floor)
σ_⊥    = max(σ_∥ / rho,        sigma_floor)
v∥     = (c - e)·û          v⊥ = (c - e)·t̂
w      = exp( -v∥²/(2σ_∥²) - v⊥²/(2σ_⊥²) ) / (2π σ_∥ σ_⊥)
```

Four implementation constraints, each of which is a real trap:

1. **Compute `v = c − e` first, then contract it** — do *not* use the `cells @ u.T` matmul trick.
   The first draft said the opposite, to avoid materialising `[C, E, 2]`. Measurement reversed it:
   the tensor is free at these sizes (it is the *fastest* variant at range 1), while the matmul form
   silently runs in reduced precision (TF32) on GPU and loses three decimal digits — 5.4e-04 against
   3.6e-07 with `jax_default_matmul_precision='highest'`. Harmless for the science, fatal for a
   geometry unit test. See [[VISUAL_PSF_MECHANISM_STUDY]] §Implementation variants.
2. **`sigma_floor` is mandatory, not cosmetic.** Mass normalisation divides by `2π σ_∥ σ_⊥`; an entity
   on the agent's own cell gives `d = 0`, `σ_∥ = 0`, and an infinite peak. A floor of half a cell is
   the grid's sampling limit and independently the point past which anisotropy stops buying
   information.
3. **`û` needs a guarded divide** at `d = 0`. With the floor in place the direction is irrelevant
   there, but the NaN is not.
4. **Static versus traced.** `visual_sensor_range` and `olfactory_sensor_range` are shape-determining
   and must stay `pytree_node=False`. `visual_blur_enabled` is a trace-time Python branch. The three
   continuous blur knobs **must be traced arrays**, or every sweep value recompiles `jax_step`.

### File Changes

#### `configs/environment/default.yaml` (`sensory:` block, ~line 186)

```yaml
# AFTER — added to sensory:
  # --- Olfactory directional sampling (v3.1) -------------------------------
  # 0 = today's single sample at the agent's cell (byte-parity). >0 samples the
  # same field at every cell of a Manhattan diamond, so readings carry a gradient.
  olfactory_sensor_range: 0

  # --- Visual point-spread blur (v3.1) ------------------------------------
  # false = exact cell match, byte-identical to pre-v3.1 behaviour.
  visual_blur_enabled: false
  visual_blur_radial_scale: 0.5    # sigma_parallel = scale * distance
  visual_blur_anisotropy: 3.0      # rho = sigma_par / sigma_perp; 1.0 == isotropic
  visual_blur_sigma_floor: 0.5     # cells; the grid's sampling limit
```

Per-entity, optional, on every resource / entity / obstacle entry (default `none`):

```yaml
    visual_mask: none    # none | far | all
```

#### `src/environment/state.py` — `EnvParams` (~line 256)

```python
# BEFORE:
    sensor_radius: float
    sensor_decay: float
    ...
    visual_sensor_range: int = struct.field(pytree_node=False)

# AFTER:
    sensor_radius: float
    sensor_decay: float
    ...
    visual_sensor_range: int = struct.field(pytree_node=False)
    # v3.1 directional sensors ------------------------------------------------
    olfactory_sensor_range: int = struct.field(pytree_node=False)   # shape-determining
    visual_blur_enabled: bool = struct.field(pytree_node=False)     # trace-time branch
    visual_blur_radial_scale: float      # traced — sweeping must not recompile
    visual_blur_anisotropy: float        # traced
    visual_blur_sigma_floor: float       # traced
    res_visual_mask: jnp.ndarray         # [num_res]    int 0=none 1=far 2=all
    animal_visual_mask: jnp.ndarray      # [N]          int
    obs_visual_mask: jnp.ndarray         # [num_obs]    int
```

#### `src/environment/config_loader.py`

- Read the five new `sensory.*` keys with `get_mandatory` (project rule: no fallback defaults for
  critical config).
- Parse per-entity `visual_mask` into int arrays alongside the existing `visual_properties` parsing
  (`_read_visual_properties`, lines ~829, ~1032, ~1175). Optional per entity, defaulting to `0`
  (`none`) — same precedent as `visual_properties_std`. An unrecognised string must raise, not
  silently become `none`.
- Empty entity lists produce `jnp.zeros(0, dtype=jnp.int32)`, matching the existing zero-row pattern.

#### `src/environment/sensor.py` — on-source rule (option B)

```python
# BEFORE (sensor.py:14):
decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, decay_power) + 1e-10))

# AFTER: the same value, with its meaning stated -- "on the source" == "half a cell away".
# At decay_power=1.0 this is bit-identical to the literal 2.0 in float32 (verified), so
# byte-parity holds and no existing test changes. At any other gamma it stays on the curve
# where the constant does not.
_ON_SOURCE = 1.0 / jnp.power(0.5, decay_power)
decay = jnp.where(dist < 0.001, _ON_SOURCE, 1.0 / (jnp.power(dist, decay_power) + 1e-10))
```

Add to the parity test suite: assert `1/(0.5**1.0) == 2.0` bitwise in float32, so the equivalence
this rests on is checked rather than remembered.

#### `src/environment/sensor.py`

**(a) new — per-cell olfaction**, called from `get_observation`:

```python
def sense_olfaction_cells(state, params):
    """Olfactory field sampled at every cell of a Manhattan diamond.

    olfactory_sensor_range == 0 reproduces the pre-v3.1 single sample exactly:
    the diamond is [[0,0]], so the sampling point IS the agent's cell. The
    range-0 case takes a STATIC fallback to the original un-vmapped expression
    so parity does not depend on vmap-of-one compiling identically -- the same
    belt-and-braces the visual path gets from visual_blur_enabled.
    """
    if params.olfactory_sensor_range == 0:          # static branch, trace time
        return _sense_olfaction_point(state.agent_pos, state, params)   # today's code, verbatim
    offsets = get_visual_offsets(params.olfactory_sensor_range)      # [C,2]
    cells = state.agent_pos + offsets

    def at(p):
        return (sense_resource(p, state.res_pos,    state.res_active,    state.res_property_sampled,
                               params.sensor_radius, params.sensor_decay)
              + sense_resource(p, state.animal_pos, state.animal_active, state.animal_property_sampled,
                               params.sensor_radius, params.sensor_decay)
              + sense_resource(p, state.obs_pos,    state.obs_active,    state.obs_property_sampled,
                               params.sensor_radius, params.sensor_decay))

    return jax.vmap(at)(cells).flatten()      # [C*V]
```

**Specification details that must not drift:**

1. **Offsets come from the existing `get_visual_offsets(r)`** — the same function vision uses, so the
   two senses share a cell ordering and the renderer can draw both with one routine. Order at r=1 is
   `[centre, north, east, south, west]`; at r=2 it is `[centre, N, E, W, S, then the outer ring]`.
   Note the inner ring's order **differs between r=1 and r=2** (S and W swap), because `r == 1` takes
   a hard-coded early return. That is pre-existing and already baked into vision's layout; olfaction
   inherits it deliberately rather than diverging.
2. **The three-pool structure is preserved at every range** — `res_chem + animal_chem + obs_chem`,
   in that order, evaluated per cell. Not flattened into one concatenated matmul. This keeps float
   accumulation order identical to today, so the centre cell is bit-identical to the current single
   sample at *every* range, not just at range 0. Verified: `[0.5, 0, 0, 0, 0]` from both paths,
   byte-for-byte.
3. **Distances and the `sensor_radius` cutoff are measured from each cell**, not from the agent. On a
   10×10 grid with `sensor_radius: 20` the cutoff never binds, but the per-cell form is what makes
   the sensor correct if the grid or the radius ever changes.
4. **Out-of-bounds cells are zeroed** across all `vector_size` channels, using the same
   `is_in_bounds` mask `sense_visual` already builds. Per the user's decision; see the note above
   about the resulting wall cue.
5. **Flatten cell-major**: `[C, V] -> C*V` with cell 0's channels first. `get_observation_breakdown`
   reports `C * V` for Olfaction, and the noise system and renderer both slice on that assumption.

**(b) `sense_visual` (line ~145)** — add the blur branch. Note the false branch is **not** untouched:
the activity mask moves from `all_props` onto `W`, and a gate multiply is added to both paths. Parity
is therefore a claim the tests must establish, not a structural guarantee — the first draft's "parity
by construction" language was wrong:

```python
    # BEFORE:
    matches = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :], axis=-1)
    vis_entities = jnp.matmul(matches.astype(jnp.float32), all_props)

    # AFTER:
    if params.visual_blur_enabled:                       # static branch, trace time
        W = _psf_weights(cell_coords, all_pos, agent_pos, offsets, params)
    else:
        W = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :], axis=-1).astype(jnp.float32)
    W = W * all_active[None, :]                          # activity mask, both paths
    W = W * _visual_mask_gate(offsets, all_mask)         # none/far/all
    vis_entities = jnp.matmul(W, all_props)
```

Today the activity mask is applied to `all_props` (`sensor.py:230`); moving it onto `W` is
algebraically identical for the boolean path and is what lets the blur path share it. Bitwise
micro-checks during review found both transformations bit-identical, including for negative
properties — but that is evidence, not proof across backends. **Parity is established by the tests
below, on a pinned backend**; if the test fails, keep the mask on `all_props` for the boolean path and
apply it separately in the blur path.

`_visual_mask_gate` builds `[C, E]` from the per-entity mask code and each **entity's** Manhattan
distance from the agent (`d_e = |Δrow| + |Δcol|`, a `[E]` vector), broadcast across cells:

```python
def _visual_mask_gate(agent_pos, all_pos, all_mask):
    d_e = jnp.sum(jnp.abs(all_pos - agent_pos), axis=-1)          # [E]
    keep = jnp.where(all_mask == 2, 0.0,                          # all  -> never visible
            jnp.where(all_mask == 1, (d_e < 1).astype(jnp.float32),  # far -> only when co-located
                      1.0))                                        # none -> always
    return keep[None, :]                                          # [1, E], broadcasts over cells
```

Gating the entity's whole column is what makes `far` leak-free under blur. Gating on the *cell's*
distance instead — the first draft's spec — leaves the agent's own cell open to a distant masked
entity's blur tail.

**(c) `get_observation` (line ~315)** — olfaction branch calls `sense_olfaction_cells`.

**(d) `get_observation_breakdown` (line ~379)**:

```python
    # BEFORE:
    breakdown["Olfaction"] = int(params.res_property.shape[-1])
    # AFTER:
    n_olf_cells = 2 * (params.olfactory_sensor_range**2) + 2 * params.olfactory_sensor_range + 1
    breakdown["Olfaction"] = int(n_olf_cells * params.res_property.shape[-1])
```

The perceptual-noise system reads this dict, so its per-modality σ arrays follow automatically. **No
change needed in `apply_perceptual_noise`** — but this must be verified, not assumed, because the
noise arrays are padded to 13 modality slots and the olfaction slot now covers many more dimensions.

**(e) `build_sensory_viz` (line ~415)** — olfaction becomes a diamond when range > 0. Also fixes a
live bug: the olfaction pod currently passes the **eight visual channel labels**
(`['GRS','SND','PLN','FOD','DNG','PRD','RCK','NEU']`) for a five-dimensional chemical vector. It is
inert today because `draw_spectrum_pod` ignores `labels`, but a diamond renderer will not.

#### `train.py` (line 784) and `src/algorithms/dreamer_srl/dreamer_srl_main.py` (line 774)

**Corrected after review — the first draft misdiagnosed this.** Both files carry a duplicated
13-field modality fingerprint that gates whether curriculum stages share an observation layout, and
it has no olfactory range field because none existed. The first draft claimed two stages differing
only in `olfactory_sensor_range` would pass validation. **That is false**: both validators check
obs_dim equality *first* (`train.py:816-822`, `dreamer_srl_main.py:806-812`) and an olfactory-range
change always changes obs_dim, so it is already rejected today.

The fingerprint addition is still correct, but for a different and narrower reason:
**`visual_blur_enabled` is the only genuinely new hazard** — it changes observation *semantics* at an
*identical* dimension count, which is precisely the case obs_dim equality cannot catch. Adding
`olfactory_sensor_range` is defence in depth rather than a fix. The test must therefore target the
fingerprint-specific rejection (see Test Plan), or it passes without the change and proves nothing.

```python
# BEFORE (both files):
    return (
        p.visual_sensor_enabled,
        p.visual_sensor_range,
        ...
        p.sensor_range,
    )

# AFTER (both files, identically):
    return (
        p.visual_sensor_enabled,
        p.visual_sensor_range,
        ...
        p.sensor_range,
        p.olfactory_sensor_range,   # v3.1 — changes obs_dim
        p.visual_blur_enabled,      # v3.1 — changes obs SEMANTICS at identical dim
    )
```

The "13-field" wording becomes 15 — but only where it actually appears, which is
`dreamer_srl_main.py:771,775`. `train.py`'s docstring does not carry the phrase, and neither does
`tests/env/test_no_recompile.py` (whose docstring is about animal-class recompiles). The first draft
named two edit sites that do not exist.

#### Standalone configs — the undisclosed blast radius (added after review)

Making the five `sensory.*` keys mandatory has consequences the first draft did not state.

**Historical runs become un-evaluatable.** Every run dumps its fully merged config to
`models/config.yaml` (`train.py:881-883`), and the evaluation path loads that snapshot and merges
only the *evaluation* and *visualization* defaults on top of it (`evaluation.py:_load_eval_config`,
line ~132-140) — never `configs/environment/default.yaml`. A pre-change snapshot therefore lacks the
new keys, and `load_env_params` raises `ValueError` on every historical run.

Required, in the same change:

- Sweep the new keys into every standalone config that does not inherit from
  `configs/environment/default.yaml` (~98 files; `test_backward_compat_configs.py:66-90` otherwise
  demotes them to a silent "stale-skip" rather than failing loudly). Precedent: the interoceptive
  keys forced a 48-config sweep.
- Update `scripts/verification/check_olfaction_parity.py` and its standalone configs.
- Document the one-line remedy for existing run snapshots (append the five keys to
  `results/.../models/config.yaml`) in the plan's Implementation Report and in
  [[CONFIG_CRITICAL_SETTINGS]]'s change-log entry.

Keeping the keys mandatory is the project rule (`get_mandatory`, no fallback defaults); the cost is
disclosed here rather than discovered by whoever next re-evaluates an old run.

#### Docs (maintenance contracts)

- [[09_sensors_and_observation]] — document both changes, **and repair the existing staleness** found
  during the studies: §9 still describes the pre-v3.0 hardcoded one-hot visual sensor; §6 records
  `decay_power` 2.0 where the config ships 1.0 and olfactory signatures that no longer match
  `default.yaml`.
- `docs/environment/02_config_schema.md` **and** [[CONFIG_GUIDE]] — five new `sensory.*` keys plus
  per-entity `visual_mask`. Both are required by the schema maintenance contract; the atomic-landing
  recipe is at `CONFIG_GUIDE.md:208-211`. The first draft listed only the schema doc.
- [[CONFIG_CRITICAL_SETTINGS]] — register the new keys with canonical values, plus a dated change-log
  entry (required by that doc's maintenance contract).

## Checkpoints

- [ ] **CP0 — OFF-path parity is already verified.** The restructured exact-match path (activity mask
      moved onto `W`, plus the mask gate) measured **bit-identical** to today's `sense_visual` on GPU
      at ranges 0, 1 and 2. Re-confirm after implementation with
      `visual_psf_study/bench_visual_variants.py`; it should stay exact, because the values involved
      are exactly 0.0 and 1.0.
- [ ] **CP1 — parity before anything else.** With stock `default.yaml`, dump the full observation
      vector for a fixed seed and 20 steps, before and after the change. Must be **bit-identical**.
      Compare saved arrays, not a recomputation from source.
- [ ] **CP2 — breakdown arithmetic.** `sum(get_observation_breakdown(params).values()) ==
      get_observation(...).shape[0]` at `olfactory_sensor_range` ∈ {0,1,2} × `visual_sensor_range` ∈
      {0,1,2}. Catches slice drift that silently mis-assigns noise.
- [ ] **CP3 — kernel sanity on known geometry.** ρ=1 reproduces an isotropic Gaussian; an object at
      exactly 45° gives equal north and east weights; an object on the agent's own cell produces a
      finite value (the `sigma_floor` guard); no NaN or Inf anywhere in the weight matrix.
- [ ] **CP4 — distance falloff exists.** Total in-diamond weight for one object must *decrease*
      monotonically with its distance. If it does not, the normalisation is wrong — this is the exact
      failure Fig 4 documents.
- [ ] **CP5 — no recompile from sweeping.** Changing the three continuous blur knobs must not
      recompile `jax_step`; changing either range must.
- [ ] **CP6 — cost, measured on the shipped code.** `bench_aniso.py` currently benchmarks its own
      embedded `sense_visual_aniso` prototype, so re-running it as written would re-measure the
      prototype and reproduce the study's numbers no matter what was built — circular. Repoint it at
      the real `sense_visual` with `visual_blur_enabled: true`, then confirm <3% of `env.step`.

## Test Plan

New, under `tests/env/`:

| Test | Asserts |
|---|---|
| `test_olfaction_range0_parity.py` | `olfactory_sensor_range: 0` gives observations bit-identical to a stored pre-change reference |
| `test_visual_blur_disabled_parity.py` | `visual_blur_enabled: false` likewise, **including** the moved activity mask |
| `test_visual_psf_kernel.py` | CP3 + CP4 as unit assertions on the weight matrix. **Must pin `jax_default_matmul_precision`**, or exact-geometry assertions are flaky at the 1e-4 level on GPU |
| `test_visual_mask.py` | With blur ON, a `far`-masked entity at distance ≥ 1 contributes **exactly zero to every cell including the centre** (the leak regression test); it contributes normally when the agent stands on it; `all` zeroes everything; `none` unchanged; an unknown string raises at load |
| `test_olfaction_diamond.py` | At range 1 with a single source, the cell toward the source reads higher than the cell away from it, for several bearings |
| `test_modality_fingerprint.py` | Two configs with **identical obs_dim** but different `visual_blur_enabled` are rejected by the curriculum-stage validator, in both `train.py` and `dreamer_srl_main.py`. Must assert the *fingerprint* error, not the obs_dim error — a test built on `olfactory_sensor_range` passes without the change and proves nothing |

Extend: `test_no_recompile.py` (CP5), `test_visual_parity.py` / `test_visual_properties.py` /
`test_visual_sampling.py` (must still pass unchanged — they are the existing parity net).

Parity fixtures must **pin the JAX backend** (CPU) so a bitwise comparison cannot pass or fail on
which device the suite happens to run.

## Risks

| Risk | Mitigation |
|---|---|
| Moving the activity mask from `all_props` to `W` changes float accumulation order | Explicit parity test; if it fails, keep the mask on `all_props` for the boolean path and apply it separately in the blur path |
| The modality fingerprint is duplicated in two files and drifts | Change both in the same commit; `test_modality_fingerprint.py` exercises both |
| Noise arrays are padded to 13 modality slots | Verify slot indices are unchanged (they are keyed by name order, not by dimension count), and assert dimensions in CP2 |
| `sigma_floor` interacts with anisotropy — flooring `σ_⊥` silently reduces effective ρ at short range | Documented as intended behaviour; CP3 asserts the value stays finite. Worth a note in the config comment |
| Olfactory range ≥ 2 tripling the observation | Not an implementation risk; a training-cost decision, deferred to experiment configs |

## Decisions still needing the user

Four questions the review surfaced that the plan should not answer on its own. **Three are now
answered by the user (2026-08-20); the fourth is still open.**

### Answered

| # | Question | Decision |
|---|---|---|
| — | `visual_sensor_range` for blur runs | **2** (13 cells, 104 visual dims, observation 27 → 123). The smallest diamond with room for the kernel to place an off-axis lobe, and the range both Fig 2 and Fig 3 were measured on. |
| — | `olfactory_sensor_range` | **1** (5 cells, 25 olfaction dims, observation 27 → 47). Cheapest range that gives direction, and per Fig 3 the *most accurate* one while perceptual noise is off — which is the default. |
| 2 | The on-source decay rule | **Option B — express the constant as a half-cell floor, `1 / (0.5^γ)`.** Verified bit-identical to the hard-coded `2.0` in float32 at the shipped γ=1, so parity holds and no test changes; correct at every other γ, where the constant silently is not. Study: [[ONSOURCE_RULE_STUDY]]. |
| 1 | Out-of-bounds olfactory cells | **Zero them, matching the visual sensor.** This overrides the plan's recommendation to sample anyway. Consequence to record: the resulting asymmetry is a usable wall cue carried in a chemical channel, so any behaviour analysis attributing wall-avoidance or edge-hugging to olfaction must account for it. Implement with the same `is_in_bounds` mask `sense_visual` already builds. |

Combined observation with both settings: 27 − 8 − 5 + 104 + 25 = **143 dims**.

### Still open

Each has a recommendation; none is implemented until confirmed.

| # | Question | Recommendation |
|---|---|---|
| 3 | **The three continuous blur knobs are unfingerprinted.** Curriculum stages could differ in ρ or radial scale — a large same-dimension semantics change — without rejection, while a `visual_blur_enabled` flip is rejected. (Pre-existing sibling: `visual_vector_size` is also unfingerprinted.) | **Accept, and say so in the code comment.** Fingerprinting floats is brittle and would forbid legitimate schedules. But the asymmetry should be deliberate rather than accidental. |
| 4 | **Fingerprinting `visual_blur_enabled` forecloses a sharp→blurred curriculum.** That is correct per the check's stated purpose — semantics must not change mid-run — but it removes an experiment someone might want. | **Accept.** A perceptual-degradation curriculum would need its own weight-compatibility story anyway. |

## Author response to plan-reviewer

Reviewed by `plan-reviewer` (Fable) on 2026-08-19; full report at
[`docs/reviews/plan_directional_sensors.md`](../../../reviews/plan_directional_sensors.md). Verdict:
**NOT READY**. All seven findings accepted; the two most consequential were independently verified
against the code before acting.

| Finding | Disposition |
|---|---|
| 🔴 1 — `far` mask leaks under blur | **Accepted and fixed.** Gate now zeroes the entity's column by *entity* distance. The leak arithmetic reproduces: 0.086 at d=1 with the shipped knobs. The proposed regression test was itself enshrining the leak ("leaves the centre") and has been rewritten to assert exactly zero everywhere. |
| 🟡 2 — mandatory keys break historical runs | **Accepted.** Verified: `evaluation.py:_load_eval_config` merges only evaluation + visualization defaults over a run's snapshot, never the environment default, so old snapshots do raise. Added a File Changes section covering the ~98-config sweep, the verification script, and the remedy for existing snapshots. Keys stay mandatory per project rule. |
| 🟡 3 — fingerprint misdiagnosed, test vacuous | **Accepted.** Verified: `train.py:816-822` checks obs_dim before the fingerprint, so an olfactory-range change is already caught. My "highest-risk change" framing was wrong — it is the *safest*. The addition survives on `visual_blur_enabled` alone, and the test now targets the same-dim case. |
| 🟡 4 — CP6 circular | **Accepted.** CP6 now requires repointing the benchmark at the real `sense_visual`. |
| 🟡 5 — "parity by construction" overstated | **Accepted.** Language dropped; olfaction gains a static range-0 fallback matching vision's; parity fixtures pin the backend. |
| 🟡 6 — CONFIG_GUIDE.md missing | **Accepted.** Added. |
| 🟢 7 — stale line references | **Accepted.** `train.py:784` corrected; two claimed edit sites that do not exist removed. |
| ❓ a–d | **Promoted** to "Decisions still needing the user" above, each with a recommendation. |

On the three unilateral decisions the review was asked to attack: full-mass normalisation and the
one-width-plus-ρ parameterisation were upheld as evidence-backed — the review additionally observed
that `σ_⊥ = (scale/ρ)·d` is algebraically the study's own fixed-angular-blur form under different
knob names. Mask-after-blur was **not** upheld, and is the Critical finding above.

Status remains **PLANNED**. It should not move to IN PROGRESS until the four user decisions are
answered.

## Implementation Report

> **Implemented by**: _(not yet implemented)_
> **Date**: —

## Verification Report

> **Verified by**: _(not yet verified)_
> **Date**: —

---

## Feedback from plan-reviewer

**Date**: 2026-08-19 · **Verdict**: **NOT READY** — 1 Critical, 5 Moderate, 1 Low, 4 Open. Full review: [[plan_directional_sensors]] (`docs/reviews/plan_directional_sensors.md`).

- 🔴 **`far` mask leaks by its own spec**: the per-cell gate leaves the centre cell open, so a "hidden" entity deposits ~13.5% (d=1) → 1.6% (d=5) of an adjacent visible entity's peak into the agent's own cell, deterministically — contradicting the decision row's rationale ("'hidden' should mean no leakage") and detectable by a recurrent policy. Gate on the entity's distance, or re-specify the semantics with user sign-off.
- 🟡 Five unconditional `get_mandatory` keys break re-evaluation/resume of **every pre-change run** (merged `models/config.yaml` snapshots lack them) plus `check_olfaction_parity.py` and its configs; the standalone-config sweep is missing from File Changes.
- 🟡 The fingerprint motivation is misdiagnosed — obs_dim validation already rejects olfactory-range mismatches (train.py:816-822, dreamer_srl_main.py:806-812) — and `test_modality_fingerprint.py` as specified passes without the change; the genuinely new case (`visual_blur_enabled`, same dim) is untested.
- 🟡 CP6 is circular: `bench_aniso.py` benchmarks its own embedded prototype, not the implemented code.
- 🟡 "Parity by construction" claims are overstated (the false branch *does* change; range-0 olfaction goes through vmap) — CPU bitwise micro-checks pass, but keep the tests as the guarantee and add an olfaction fallback. 🟡 CONFIG_GUIDE.md missing from the maintenance-contract doc list.
- ❓ OOB olfactory sampling cells; the study's Open "on-source decay 2.0" rule silently resolved to "keep"; continuous blur knobs unfingerprinted; blur-curriculum foreclosed.

— plan-reviewer
