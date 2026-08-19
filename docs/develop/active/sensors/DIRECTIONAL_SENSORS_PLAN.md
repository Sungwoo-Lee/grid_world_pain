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
| Mask ordering | Zero the entity's weight in every cell at Manhattan distance ≥ 1 (blur first, mask after) | **plan's choice** — "hidden" should mean no leakage |

The three plan's-choice rows are the author's calls from the study evidence, not user decisions.
They are the rows most worth arguing with.

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

1. **Never materialise `[C, E, 2]`.** Since `v·û = c·û − e·û`, both projections come from small
   matmuls (`cell_coords @ u.T`) minus a per-entity constant. Working set stays `[C, E]`.
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

#### `src/environment/sensor.py`

**(a) new — per-cell olfaction**, called from `get_observation`:

```python
def sense_olfaction_cells(state, params):
    """Olfactory field sampled at every cell of a Manhattan diamond.

    olfactory_sensor_range == 0 reproduces the pre-v3.1 single sample exactly:
    the diamond is [[0,0]], so the sampling point IS the agent's cell.
    """
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

Note the summation order (`res + animal + obs`) is preserved from the current code so float
accumulation is bit-identical at range 0.

**(b) `sense_visual` (line ~145)** — add the blur branch. The exact-match path stays untouched, so
`visual_blur_enabled: false` is byte-parity by construction rather than by test:

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

Today the activity mask is applied to `all_props`; moving it onto `W` is algebraically identical for
the boolean path (both are pure scaling of the same product) and is what lets the blur path share it.
**This must be parity-tested, not assumed** — float multiplication order changes.

`_visual_mask_gate` builds `[C, E]` from the static per-cell Manhattan distances and the per-entity
mask code: `all` → 0 everywhere, `far` → 0 wherever cell distance ≥ 1, `none` → 1.

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

#### `train.py` (~line 485) and `src/algorithms/dreamer_srl/dreamer_srl_main.py` (~line 774)

**The highest-risk change in this plan.** Both files carry a duplicated "13-field modality
fingerprint" that gates whether curriculum stages share an observation layout. It contains
`visual_sensor_range` and `olfactory_vector_size` but has no olfactory range field, because none
existed. Two stages differing only in `olfactory_sensor_range` would produce different observation
dimensions and **pass validation**, loading weights shaped for the wrong observation.

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

The "13-field" wording in both docstrings and in `tests/env/test_no_recompile.py`'s module docstring
becomes 15.

#### Docs (maintenance contracts)

- [[09_sensors_and_observation]] — document both changes, **and repair the existing staleness** found
  during the studies: §9 still describes the pre-v3.0 hardcoded one-hot visual sensor; §6 records
  `decay_power` 2.0 where the config ships 1.0 and olfactory signatures that no longer match
  `default.yaml`.
- `docs/environment/02_config_schema.md` — five new `sensory.*` keys plus per-entity `visual_mask`.
- [[CONFIG_CRITICAL_SETTINGS]] — register the new keys with canonical values, plus a dated change-log
  entry (required by that doc's maintenance contract).

## Checkpoints

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
- [ ] **CP6 — cost.** Re-run `bench_aniso.py` after implementation and confirm the measured delta is
      in the same range as the prototype (<3% of `env.step`).

## Test Plan

New, under `tests/env/`:

| Test | Asserts |
|---|---|
| `test_olfaction_range0_parity.py` | `olfactory_sensor_range: 0` gives observations bit-identical to a stored pre-change reference |
| `test_visual_blur_disabled_parity.py` | `visual_blur_enabled: false` likewise, **including** the moved activity mask |
| `test_visual_psf_kernel.py` | CP3 + CP4 as unit assertions on the weight matrix |
| `test_visual_mask.py` | `far` zeroes every cell at distance ≥ 1 and leaves the centre; `all` zeroes everything; `none` unchanged; an unknown string raises at load |
| `test_olfaction_diamond.py` | At range 1 with a single source, the cell toward the source reads higher than the cell away from it, for several bearings |
| `test_modality_fingerprint.py` | Two configs differing **only** in `olfactory_sensor_range` are rejected by the curriculum-stage validator, in both `train.py` and `dreamer_srl_main.py` |

Extend: `test_no_recompile.py` (CP5), `test_visual_parity.py` / `test_visual_properties.py` /
`test_visual_sampling.py` (must still pass unchanged — they are the existing parity net).

## Risks

| Risk | Mitigation |
|---|---|
| Moving the activity mask from `all_props` to `W` changes float accumulation order | Explicit parity test; if it fails, keep the mask on `all_props` for the boolean path and apply it separately in the blur path |
| The modality fingerprint is duplicated in two files and drifts | Change both in the same commit; `test_modality_fingerprint.py` exercises both |
| Noise arrays are padded to 13 modality slots | Verify slot indices are unchanged (they are keyed by name order, not by dimension count), and assert dimensions in CP2 |
| `sigma_floor` interacts with anisotropy — flooring `σ_⊥` silently reduces effective ρ at short range | Documented as intended behaviour; CP3 asserts the value stays finite. Worth a note in the config comment |
| Olfactory range ≥ 2 tripling the observation | Not an implementation risk; a training-cost decision, deferred to experiment configs |

## Implementation Report

> **Implemented by**: _(not yet implemented)_
> **Date**: —

## Verification Report

> **Verified by**: _(not yet verified)_
> **Date**: —
