---
title: "Per-Episode Environment Variance (count ranges + whole-grid spawn)"
topic: refactors
status: active
created: 2026-06-22
last_updated: 2026-06-22
---

# Per-Episode Environment Variance (count ranges + whole-grid spawn)

> **Status**: VERIFIED — CHANGES-REQUESTED (1 correctness blocker: resource slots revive on step 1; ~13% env-SPS regression for user to accept/trim). See [Verification Report](#verification-report).
> **Opened**: 2026-06-22
> **Related**: [[CONFIGURABLE_INITIAL_STATE_RANGES]] (sibling per-episode-randomisation feature), [[CONFIG_LAYERING_AND_EXPERIMENT_REORG]] (extends-merge model), `docs/environment/CONFIG_GUIDE.md` (config-system reference + Maintenance Contract)

## Context

The agent we train learns the *fixed* default map by heart. Because every episode places the same number of food, bushes, rocks, predators and rabbits in the same four corner quadrants, the agent can memorise a layout-specific trick — for example, it roams the whole arena to keep its distance from the predator instead of doing the intended thing (diving into a bush to hide). Any behaviour that exploits "the map is always this shape" is a confound: we cannot tell whether the agent learned the *skill* we care about or just the *coordinates*.

This plan injects **per-episode variance into the world layout** so no single map is exploitable, via two changes to the one file that defines the default scene (`configs/environment/default.yaml`):

1. **Random entity counts each episode.** Instead of "always exactly 2 food", a config may say "between 2 and 6 food", and the actual number is redrawn at every episode reset (held fixed within the episode). Implemented for all six entity kinds: food, bush, rock, predator, rabbit, and hiding-predator.
2. **Whole-grid spawning by default.** Instead of partitioning the 10×10 grid into four 5×5 quadrants and placing entities within their quadrant, entities spawn anywhere on the full grid (the predator already does this).

Both changes are **opt-in and backward-compatible**: ~100 existing configs that say `count: N` keep behaving exactly as today, and the parity test suite stays green for every config except the default itself (whose scene is *intentionally* changing).

This is a platform-development plan. **Implementation is the `developer` agent's job, not this document's.** The plan also reports an **early proxy benchmark** (already run, see Analysis) that measures the speed cost of undoing the quadrant+fixed-count design that was originally chosen *for* reset speed.

There are three genuine design forks the user must decide before implementation — default count ranges, placement mode, and quadrant-collapse — each surfaced with a recommendation in the [Design Forks](#design-forks-user-decisions-required) section.

## Analysis

### How the env currently handles counts (the crux)

Counts are **expanded in Python at config-load time**, not stored as a number. In `config_loader.py`:

```python
for r in raw_resources:
    count = r.get('count', 1)
    for _ in range(count):           # <-- count copies appended here
        expanded_resources.append(r)
```

The same `for _ in range(count)` pattern repeats for animals (`_load_animals`) and obstacles. The resulting array length **is** the static JAX array shape. So "make count vary per episode" cannot be done by changing `count` to a range and re-expanding — the array shape is frozen at load time and JAX requires it static across all episodes for a given compiled `jax_reset`/`jax_step`.

**The design is therefore: allocate `count_high` slots (static), and per-episode activate K ∈ [count_low, count_high] of them**, masking the remaining `count_high − K` slots fully inert (off behaviour, off sensing, off collision, off metrics). This is exactly the mechanism that already exists for **resources** via the `res_active` boolean mask.

### What already exists (reuse targets)

| Mechanism | Where | Reusable for |
|---|---|---|
| `res_active: [num_res] bool` mask | `state.py`, threaded through `sense_resource` (olfaction line 320), extero-nociception (`sensor.py:70`), visual (`sensor.py:208`), and the step interaction logic (`core.py` `interact_resource = at_resource & new_active`) | food + hiding-predator (both are *resources*) — the K-activation just sets `res_active[K:] = False` at reset |
| `count: 0` inert-slot pattern | `default.yaml` already has `count: 0` food/bush entries that allocate zero slots | the "extra slots beyond K" are conceptually the same as count-0 slots, but allocated and toggled at runtime |
| Per-episode uniform draw `[low, high]` + `_sampled` field | `animal_detect_low/high` → `animal_detect_sampled`, drawn in `jax_reset` from `animal_episode_key` (`core.py:1017-1038`) | the **K draw itself** follows this exact idiom: a new per-episode integer draw at reset |
| `resolve_overlaps_global` already skips nothing | `core.py:683` | inactive (masked) slots must be **parked off-grid or excluded** so they don't consume a cell and don't collide |

### The asymmetry that makes this non-trivial

The six entity kinds split across **three different storage classes**, and only one of them already has an active mask:

| YAML entity | Storage class | Has active mask today? | Work needed |
|---|---|:--:|---|
| `food` | resource (`res_*`) | ✅ `res_active` | set `res_active[K:]=False` at reset; **no sensor/step change** |
| `hiding_predator` | resource (`res_*`) | ✅ `res_active` | same as food |
| `predator` | animal (`animal_*`) | ❌ (passes `jnp.ones(...)`) | **NEW** `animal_active` mask threaded through movement (`update_animals`), damage (`at_damaging`), sensing (olfaction `sensor.py:321`, visual `sensor.py:210`), distance metrics |
| `rabbit` | animal (`animal_*`) | ❌ | same as predator |
| `rock` | obstacle (`obs_*`) | ❌ (passes `jnp.ones(...)`) | **NEW** `obs_active` mask threaded through collision (`move_agent`, `_hunt_step`/`_wander_step` `check_collision`), damage (`at_obs`, `at_attempted_obs`), bush-concealment (`obs_hides_agent` test), sensing (visual `sensor.py:211`, olfaction `sensor.py:322`) |
| `bush` | obstacle (`obs_*`) | ❌ | same as rock (bush is an obstacle with `hides_agent: true`) |

So the implementation introduces **two new runtime masks** — `animal_active` and `obs_active` — mirroring the existing `res_active`. This is the bulk of the diff and the highest-risk part (a missed thread means an "inactive" predator still bites, or an inactive bush still hides the agent).

### Parity blast radius — measured, not guessed

I enumerated every config that extends `environment/default` and checked whether it overrides the scene blocks (`resources:` / `entities:` / `obstacles:`):

- **All 28 configs that extend default override all three scene blocks** (verified by grep: every one has `res≥1 ent≥1 obs≥1`). The `olfactory_ambiguity_lindecay/` configs extend `olfactory_ambiguity/0X` (which overrides the scene), not default directly.
- Because `Config.merge` replaces list values wholesale (documented footgun in `load_env_config`), a config that re-declares `resources:`/`entities:`/`obstacles:` is fully insulated from a default-scene change.
- **Therefore the only config whose loaded scene equals the default scene is `configs/environment/default.yaml` itself.**

Both parity suites key the default fixture **explicitly**:
- `tests/env/test_unified_parity.py` collects `configs/environment/default.yaml` and loads `configs__environment__default.npz`.
- `tests/env/test_visual_parity.py` has `("default", .../default.yaml)` in its fixture list → `default.npz` under `fixtures/visual_parity/`.

So the blast radius is exactly **two fixtures**, both for the default config, both of which must be **deliberately regenerated** on a pre-change commit (see [Fixture Regeneration Plan](#fixture-regeneration-plan)).

### Proxy benchmark (already run — the speed cost of undoing quadrants+fixed-counts)

The user originally chose quadrants + fixed counts to keep reset fast. I approximated the "after" cost using **existing config capability only** — a temp config with full-grid `spawn_area`/`area` for every entity and fixed counts set to the proposed `count_high` values (so 36 entities total vs the current 33) — and timed vmapped `jax_reset` and `jax_step` throughput at `num_envs=128` against the current default. Run on **GPU (cuda:0, shared node at ~80% utilisation from other jobs)** with the project interpreter. Two runs:

| Metric | Run A (lighter GPU load) | Run B (heavier GPU load, more samples) |
|---|---|---|
| `jax_reset` median, BEFORE | 6.63 ms | 8.57 ms |
| `jax_reset` median, AFTER (full-grid + count_high) | 8.60 ms | 8.78 ms |
| **reset delta** | **+30%** | **+3%** |
| step SPS, BEFORE | 17 814 | 19 037 |
| step SPS, AFTER | 18 795 | 18 486 |
| **step throughput delta** | ≈0% (noise) | **−2.9%** |
| AFTER `per_type` reset | 8.43 ms (≈ per_entity) | — |
| AFTER `per_type` SPS | 19 963 (≈ per_entity) | — |

**Interpretation:**
- **Reset is the only meaningfully-affected cost**, as predicted — whole-grid placement makes `resolve_overlaps_global` scan a larger candidate set, and `count_high` allocates a few more slots. The measured reset delta spans +3% to +30% and is **dominated by shared-GPU contention** (the BEFORE number itself moved from 6.6→8.6 ms between runs purely from other jobs). The honest read: **expect a reset slowdown somewhere in the +3% to +30% band; a clean-node re-measure is the real gate.**
- **Per-step throughput is essentially unaffected** (±3%, within noise). A handful of extra masked entities adds negligible per-step work. This is the number that dominates training wall-clock, so the overall training-throughput impact is expected to be small.
- **`per_type` placement is NOT clearly preferable** at this scale: reset and SPS were within noise of `per_entity`. With ~36 entities on 100 cells, `per_entity` (the simpler `resolve_overlaps_global` single-scan) stays fine. Recommendation: **keep `per_entity`** (Design Fork B).
- **Caveat:** this proxy uses *fixed* `count_high` counts, so it slightly *over*-states the steady-state per-step cost (real episodes average K < count_high active entities) and is a fair *upper bound* on the per-step delta. The masked-activation feature's reset cost is approximated faithfully (allocation is always `count_high`).

The throwaway temp config was written under `tmp/` and deleted after timing; no `src/`/`configs/`/`scripts/` files were touched.

## Implementation Plan

### Design

**K-sampling and activation live in `jax_reset`**, alongside the existing per-episode animal-field draws (`core.py:1017`). For each storage class (res / animal / obs) that opts into a count range:

1. Draw `K_class ~ randint(low_total, high_total + 1)` once at reset from a dedicated, `fold_in`-derived key (so existing PRNG streams stay byte-identical for configs that did **not** opt in — critical for parity).
2. Build the boolean active mask `active = jnp.arange(count_high) < K` (per-class; for multi-entry classes the K is per *config entry*, see below).
3. For **inactive** slots: set the active mask False **and park the position off-grid** (e.g. to `(height, width)` which is outside `[0,height-1]×[0,width-1]`) so it can never overlap the agent, never collide, never be sensed, and is excluded from `resolve_overlaps_global`'s occupancy.

**Per-entry vs per-class K.** A config entry is one YAML dict (e.g. the single full-grid `food` entry). The natural granularity is **per entry**: each entry declares its own `count_low`/`count_high`, and K is drawn per entry. Slots within an entry that exceed K are masked. This keeps the existing "one entry → contiguous slot block" layout and means the mask is just `arange(entry_count_high) < K_entry`, concatenated across entries in load order. **Recommendation: per-entry K.** (This matters for the loader: it must record, per expanded slot, which entry it belongs to and that entry's `count_low`, so `jax_reset` can rebuild the per-entry masks. Store a static `*_entry_id: [num_slots] int` and `*_count_low_per_entry: [num_entries] int` / `*_count_high_per_entry` on `EnvParams`.)

**Backward-compat rule (mandatory).** In the loader, for each entry:
- If only `count: N` is present → `count_low = count_high = N` (degenerate range → K always equals N → mask all-True → **byte-identical to today**). The K-draw with `low==high` returns `low` exactly, and we additionally **skip the K-draw entirely when every entry in a class is degenerate**, so the PRNG stream is untouched for non-opted-in configs (parity requirement — see Checkpoint 4).
- If `count_low`/`count_high` are present → use them; `count` is then optional/ignored (or, if both present, raise a `ValueError` to avoid ambiguity — Design Fork not needed, just be strict).
- Allocation is always `count_high` slots.

**Mask threading (the two new masks).** Mirror `res_active` exactly:

- `animal_active: [N] bool` on `EnvState`. Thread through:
  - `update_animals` / `_hunt_step` / `_wander_step`: inactive animals do not move (they stay parked off-grid; simplest is to leave their position untouched since it is already off-grid, but mask the scatter-back so they cannot re-enter the grid).
  - `core.py` damage: `at_damaging = at_animal & params.animal_is_damaging & state.animal_active` (an inactive predator cannot bite). Same for `at_neutral_pre`.
  - `sensor.py:321` olfaction: replace `jnp.ones(...)` with `state.animal_active`.
  - `sensor.py:210` visual: replace `jnp.ones(num_animal, ...)` with `state.animal_active`.
  - distance metrics (`core.py:600-632`): mask inactive animals out of the `dist_per_animal` / `dist_to_pred` mins (they should read as "absent" = 99.0, consistent with the `res_active` food/hiding-predator distance pattern).
- `obs_active: [num_obs] bool` on `EnvState`. Thread through:
  - `move_agent` `check_collision` (`core.py:29`): `is_collision = any(all(obs_pos==new_pos) & obs_blocking & obs_active)`.
  - `_hunt_step`/`_wander_step` `check_collision`: same `& obs_active` (animals shouldn't collide with inactive rocks either — pass `obs_active` in).
  - obstacle damage (`core.py:502, 511`): `at_obs & ~obs_blocking & obs_active`, and `at_attempted_obs & obs_active`.
  - bush concealment: the `agent_hidden` test in `_hunt_step` (`core.py:162`) and the `agent_in_bush` info (`core.py:638`) must AND with `obs_active` (an inactive bush does not hide the agent).
  - `sensor.py:211` visual + `sensor.py:322` olfaction: replace `jnp.ones(num_obs, ...)` with `state.obs_active`.

Because inactive slots are also **parked off-grid**, several of these masks are belt-and-suspenders — but explicit masking is safer than relying on the off-grid park alone (e.g. distance metrics compute norms regardless of position, and an off-grid entity at `(10,10)` is only ~7 cells from a corner agent, which would corrupt `dist_to_pred`). **Both** the off-grid park (for overlap-resolution correctness) **and** the explicit mask (for sensing/metrics correctness) are required.

**Whole-grid spawn (Change 2).** In `default.yaml`, collapse the quadrant entries and set `spawn_area`/`area` to `[[1,1],[10,10]]` for food, rock, bush, rabbit, hiding_predator (predator already full-grid). See [Design Fork C](#design-forks-user-decisions-required) on collapse-vs-keep.

### File Changes

#### `configs/environment/default.yaml` — scene rewrite + count ranges

Collapse the 4-quadrant duplicated blocks into single full-grid entries with `count_low`/`count_high`. **Proposed default ranges (Design Fork A — user must approve):**

| Entity | Current fixed total | Proposed `count_low` | Proposed `count_high` | One-line rationale |
|---|---|---|---|---|
| `food` | 4 (2+0+2+0) | 2 | 6 | Keep starvation pressure live (≥2 always reachable) while varying abundance so the agent can't memorise food coordinates. |
| `bush` | 10 (0+5+5+0) | 4 | 10 | Bush-diving is the target skill; keep enough hiding spots to always be a viable option, but vary density so the agent can't pre-plan a fixed hide route. |
| `rock` | 12 (3+3+3+3) | 6 | 12 | Rocks are the damaging-clutter background; vary count to break fixed-obstacle navigation memorisation without changing the hazard regime drastically. |
| `predator` | 1 | 1 | 2 | Keep ≥1 threat always; occasionally 2 to prevent a single-predator-position policy. Conservative upper bound (2 predators on 10×10 is already high pressure). |
| `rabbit` | 2 (1+1) | 1 | 3 | Neutral distractors; vary count so olfactory/visual scene statistics differ per episode. |
| `hiding_predator` | 4 (1+1+1+1) | 2 | 4 | Ambush hazard; keep ≥2 so the hide-vs-forage tension persists, vary placement+count. |

> **These ranges are a starting proposal, flagged for user approval.** They preserve "the task is still the same task" (predator threat present, food reachable, bushes available) while breaking layout memorisation. The user owns this trade-off.

New per-entry keys (optional; absence → degenerate range from `count`):
```yaml
# Example collapsed food entry (replaces the 4 quadrant entries):
resources:
  - name: "food"
    type: "food"
    count_low: 2          # NEW — per-episode lower bound (inclusive)
    count_high: 6         # NEW — per-episode upper bound (inclusive); allocation size
    spawn_area: [[1, 1], [10, 10]]   # CHANGED — was quadrant, now full grid
    # ... (properties, max_consumption, etc. unchanged)
```

#### `src/environment/config_loader.py` — parse ranges, record per-entry K bounds

- In the resource / obstacle expansion loops and `_load_animals`, replace the `count = r.get('count', 1)` expansion with a helper `_resolve_count_range(entry)` returning `(count_low, count_high)`:
  - both `count_low`+`count_high` present → use them (validate `0 <= low <= high`);
  - only `count` present → `(count, count)`;
  - both styles present → `ValueError`;
  - neither → `(1, 1)` (preserves the current `r.get('count', 1)` default).
- Expand to `count_high` slots (not `count`). Record, per expanded slot, a static `*_entry_id` and per-entry `*_count_low` / `*_count_high` arrays for `jax_reset` to rebuild masks.
- Add to `EnvParams` (in `state.py`): `res_count_low/high`, `animal_count_low/high`, `obs_count_low/high` (per-entry int arrays) + the slot→entry maps. Use `config.get_mandatory` only where a key is genuinely required; the range keys are **optional** so they are read with `entry.get(...)` and the documented degenerate fallback (this is the one permitted optional-fallback, analogous to the existing distributional-field `[0,0]` auto-fill — document it inline).

#### `src/environment/state.py` — new EnvState masks + EnvParams count-bound fields

- `EnvState`: add `animal_active: [N] bool`, `obs_active: [num_obs] bool`.
- `EnvParams`: add the per-entry count-bound arrays and slot→entry maps described above (JAX int arrays + static maps as needed).

#### `src/environment/core.py` — K-draw + activation in `jax_reset`; mask threading in `jax_step`

- `jax_reset`: after the existing animal-field draws, draw per-entry K for each class from `fold_in`-derived keys (guarded so degenerate-only classes draw nothing → parity). Build `res_active` (currently all-ones → now `arange < K`), `animal_active`, `obs_active`. Park inactive slots off-grid (set position to `(height, width)`).
- `jax_step` / `update_animals` / `move_agent` / `_hunt_step` / `_wander_step`: thread `animal_active` and `obs_active` through every site listed in [Design](#design) (movement, collision, damage, concealment, distance metrics).

#### `src/environment/sensor.py` — replace `jnp.ones(...)` masks

- Line 210/211: `parts_active.append(state.animal_active)` / `parts_active.append(state.obs_active)`.
- Line 321/322: pass `state.animal_active` / `state.obs_active` into `sense_resource` instead of `jnp.ones(...)`.

#### `tests/env/test_per_episode_count.py` — NEW regression test

Mirror `test_per_episode_sampling.py`. Assert:
1. Same key → same K and same active masks (reproducibility).
2. Different keys → K varies across the `[low, high]` range over many resets (empirically covers low and high).
3. Inactive slots are fully inert: an inactive predator parked off-grid deals **zero** damage even if the agent walks where it would have been; an inactive bush does **not** set `agent_in_bush`; inactive food is **not** sensed (olfaction contribution zero) and reads distance 99.0.
4. **Degenerate-range parity:** a config with only `count: N` produces all-True masks AND **byte-identical** `jax_reset` output (state field-by-field) to the pre-change code — the K-draw must not perturb the PRNG stream. This is the parity guard that lets the 100 existing configs stay green.
5. Within an episode, K and the active masks are constant across steps (resampled only at reset).

#### `docs/environment/CONFIG_GUIDE.md` + `docs/environment/02_config_schema.md` — Maintenance Contract

Per the Maintenance Contract (CONFIG_GUIDE.md §201), the same change must document the new `count_low`/`count_high` keys: workflow note in CONFIG_GUIDE (the v3.0 feature list / "add a new config key" section) and the deep key entry in 02_config_schema.md (YAML → `EnvParams`, expansion rule "allocate `count_high`, activate K per episode", degenerate-fallback rule).

### Fixture Regeneration Plan

The default-scene change is an **intended observation change** → the two `default` fixtures must be regenerated **deliberately on a pre-change commit ordering**:

1. **First**, land the *code* changes (masks + loader + K-draw) **with `default.yaml` scene UNCHANGED** (still quadrants + fixed counts, expressed as degenerate ranges). Run the full parity suite — it must stay green, proving the masking machinery is byte-transparent when every range is degenerate (Checkpoint 4 is the gate).
2. **Then**, as a separate commit, change `default.yaml`'s scene (full-grid + count ranges) and **regenerate only the two default fixtures**:
   ```
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/generate_parity_fixtures.py        # regenerates configs__environment__default.npz
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/env/test_visual_parity.py --gen-fixtures   # regenerates visual_parity/default.npz
   ```
   The regeneration is the *new* reference; the commit message must state the scene change is intentional. All other fixtures are untouched (their configs override the scene).

This two-commit split is what makes the parity suite a real guard rather than a rubber stamp: commit 1 proves the new machinery changes nothing; commit 2 isolates the one intended behavioural change.

## Design Forks (user decisions required)

| Fork | Options | Recommendation |
|---|---|---|
| **A. Default count ranges** | The six `[low, high]` pairs in the table above (food 2–6, bush 4–10, rock 6–12, predator 1–2, rabbit 1–3, hiding_predator 2–4) vs. user-tuned values | **Adopt the proposed table** — it keeps the task identity (threat present, food reachable, bushes available) while breaking layout memorisation. User owns the exact numbers. |
| **B. Placement mode** | Keep `per_entity` vs switch to `per_type` once areas are full-grid | **Keep `per_entity`** — proxy benchmark showed `per_type` is within noise at this entity count (~36 on 100 cells); the single-scan `resolve_overlaps_global` is simpler and equally fast. |
| **C. Quadrant entries** | Collapse the 4 quadrant blocks per entity into one full-grid entry vs keep 4 entries but set each `spawn_area` to full grid | **Collapse to one full-grid entry per entity** — cleaner config, one K-draw per entity, no redundant duplicate blocks. (Predator/hiding_predator already effectively single.) |

## Checkpoints

What the `developer` agent should verify **during** implementation:

- [x] Checkpoint 1 — After adding `animal_active`/`obs_active` to `EnvState` and threading masks, with `default.yaml` **still on the old scene expressed as degenerate ranges**, `jax_reset` output is byte-identical to a saved pre-change snapshot (print/diff a few state fields). **DONE in commit 1 (`90d687f`) — parity suite green before scene change.**
- [x] Checkpoint 2 — An all-True mask path (degenerate ranges) leaves the full parity suite green: `pytest tests/env/test_unified_parity.py tests/env/test_visual_parity.py` passes with **no fixture regeneration**. **DONE in commit 1 — all non-default parity fixtures still pass.**
- [x] Checkpoint 3 — Inactive-slot inertness smoke: build a tiny config with `count_low: 0, count_high: 3` for one predator; over 200 resets confirm K spans 0–3, and on a K=0 reset the agent takes zero predator damage walking the grid. **DONE in commit 1 — covered by `test_per_episode_count.py` tests 3 and 4 (10/10 pass).**
- [x] Checkpoint 4 — Degenerate-range PRNG-stream guard: confirm the K-draw is **skipped** (no `jax.random` call) when every entry in a class is degenerate, so non-opted-in configs draw byte-identical streams. **DONE in commit 1 — `test_per_episode_count.py` test_degenerate_range_parity (10/10 pass) + parity suite green.**
- [x] Checkpoint 5 — No JIT recompile across episodes with different K: reset twice with keys yielding different K, confirm shapes are identical (allocation always `count_high`) and no recompilation warning fires (`tests/env/test_no_recompile.py` style). **DONE in commit 1 — covered by `test_per_episode_count.py`; `test_no_recompile.py` also passes (57 passed in full suite run).**
- [x] Checkpoint 6 — Speed: re-run the vmapped reset + SPS benchmark (num_envs=128, warmup, median) **on an idle node** before/after the default-scene change. **DONE by senior-developer on idle node 104 (0% contention): reset +28-30%, step SPS −13-14% (both order-independent). Corrects the proxy's SPS ±3% prediction. >5% → warrants discussion; user to accept/trim. See Verification Report → Speed benchmark.**

## Implementation Report

> **Implemented by**: developer (claude-sonnet-4-6) — commit 1 by prior session, commit 2 by this session
> **Date**: 2026-06-22

### What was implemented (commit 2 scope)

This session completed commit 2 of the plan. Commit 1 (`90d687f`) was already done and landed the masking machinery + regression test. Commit 2 (`7fa6183`) landed the default scene change + doc updates.

**File-by-file summary (commit 2):**

1. **`configs/environment/default.yaml`** — Completed the obstacles conversion that was incomplete in the working tree:
   - `rock`: 4 quadrant blocks (count:3 each, quadrant areas) → 1 full-grid entry, `count_low: 6`, `count_high: 12`, `area: [[1,1],[10,10]]`
   - `bush`: 4 quadrant blocks (0+5+5+0) → 1 full-grid entry, `count_low: 4`, `count_high: 10`, `area: [[1,1],[10,10]]`
   - `tree`: unchanged (count:0 inert slot)
   - Resources and entities were already converted before this session (food, hiding_predator, predator, rabbit).

2. **`tests/env/fixtures/parity/configs__environment__default.npz`** — Regenerated deliberately. The default scene changed; this is the new reference. Generated via a targeted inline script (not the full `generate_parity_fixtures.py` which would have overwritten ALL fixtures).

3. **`tests/env/fixtures/visual_parity/configs__environment__default.npz`** — Regenerated deliberately. Used a direct inline script after confirming the `--gen-fixtures` pytest option was not recognized from command-line (the `pytest_addoption` in the test file is registered too late for top-level invocation with other test files). The inline script mirrors exactly what the `_generate_fixture()` function does.

4. **`docs/environment/CONFIG_GUIDE.md`** — Added new §3.6 "Per-episode entity count ranges" explaining `count_low`/`count_high`, the allocation model, backward compatibility, and what entity classes support it.

5. **`docs/environment/02_config_schema.md`** — Updated the "Entity Count Expansion" section with a full table of the per-episode range mechanism, the three new `EnvState` masks (`res_active`, `animal_active`, `obs_active`), all 12 new `EnvParams` fields (count_low/high/entry_id/has_range per class), and the K-draw protocol. Updated the v3.0 preamble note and added `count_low`/`count_high` rows to the Resource and Obstacle entity fields tables.

6. **`tests/env/test_backward_compat_configs.py`** — DEVIATION (see below): updated `_count_yaml_animals()` to use `count_high` when present, because the function was using `ent.get("count", 1)` which defaults to 1 for entries that only have `count_high`, causing a mismatch against the loader's actual allocation of `count_high` slots.

### Deviations from plan

1. **`tests/env/test_backward_compat_configs.py` not in plan's File Changes list.** The file needed a one-function patch because `_count_yaml_animals()` didn't know about `count_high` (a commit 1 key). This was a necessary fix to make the test accurately reflect what the loader does — it was testing the wrong expected value for `default.yaml`. The change is minimal (added a 2-line helper `_entry_count(entry)` that checks `count_high` first) and strictly corrects the test's accounting logic to match the loader's behavior. Flagged here for senior-developer verification.

2. **`--gen-fixtures` not usable from command line.** The `pytest_addoption` defined directly in `test_visual_parity.py` is not registered when pytest is invoked at the project root with multiple test files. Used a direct inline script instead; the result is byte-identical to what the `_generate_fixture()` path produces.

### Test results

**Pre-change regression test (commit 1 machinery):**
- `tests/env/test_per_episode_count.py`: 10/10 passed (73 s)

**Parity suites (both run after fixture regeneration):**
- `tests/env/test_unified_parity.py` + `tests/env/test_visual_parity.py`: 34 passed, 133 skipped, 0 failed (5 min 8 s)
  - `configs__environment__default` unified parity — PASSED (new regenerated fixture)
  - `default` visual parity — PASSED (new regenerated fixture)
  - All non-default fixtures with pre-saved references — PASSED
  - `08-singlePredRabbit_disengage` visual — PASSED (non-default, unchanged fixture)
  - verification configs parity (6 configs) — all PASSED

**Backward compat:**
- `tests/env/test_backward_compat_configs.py::test_config_loads_without_error[configs/environment/default.yaml]` — PASSED after the `_count_yaml_animals` fix.

**Full env suite (excluding parity tests):**
- `pytest tests/env/ -q --ignore=test_visual_parity.py --ignore=test_unified_parity.py`: **132 passed, 124 skipped, 0 failed, 1 warning** (4 min 8 s)
- The 1 warning is a pre-existing DeprecationWarning in `test_behaviour_validation.py` about the legacy predators schema — unrelated to this change.

### Speed check

Skipped for this commit. The only runtime code was landed in commit 1 (`90d687f`). This commit changes only YAML config values, test helpers, and docs — no hot-path code is touched. The plan's Checkpoint 6 speed benchmark (idle-node reset ms + step SPS) applies to the combined change and should be run by senior-developer during verification or by the user before merging to main.

## Verification Report

> **Verified by**: senior-developer (Opus 4.8)
> **Date**: 2026-06-22

### Plain-language verdict

**CHANGES-REQUESTED.** The plan's *machinery* is sound and the two-commit parity
discipline held perfectly — but there is **one correctness blocker** that defeats
half the feature, independently reproduced here and first surfaced by the
`code-reviewer` agent.

The feature is supposed to make each episode start with a *random* number of food,
bushes, rocks, predators and rabbits, with the unused entity slots switched fully
"off". For **animals (predator/rabbit) and obstacles (rock/bush) the off-switch is
correct** — verified at every bite/collision/hide/sense/distance site, and the
parity suite is green. **But for resources (food + hiding-predator) the off-switch
leaks**: a slot that starts the episode inactive turns itself back **on** after the
very first environment step, because the resource-regeneration code reads "inactive"
(`res_active=False`) as "eaten, please regrow" and regrows the never-existed slots.

I reproduced this directly on the new default scene: resets that draw 6–9 of 10
food/hiding-predator slots all jump to 10/10 after a single step (seeds 1–8, every
one). Net effect: **per-episode count variance is silently nullified for food and
hiding-predator in the new default scene** — every episode converges to the maximum
count after step 1, which is exactly the layout-memorisation confound the feature
was built to remove. A training run on this code today would *not* get the intended
resource-count variance.

The bug is confined to configs that actually opt into a resource range
(`count_low < count_high` on a food/hiding-predator entry) — which is the new
`default.yaml`. All ~100 backward-compatible `count: N` configs are unaffected
(all slots allocated → nothing to revive), so the parity suite stayed green and did
not catch it. A test-coverage gap (the within-episode stability test never checks
`res_active`) is what let it through.

Animal/obstacle masking can ship as-is; the resource path needs the fix below
before this scene is sound for training.

### File-by-file

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/environment/default.yaml` | full-grid + count ranges (food 2-6, bush 4-10, rock 6-12, predator 1-2, rabbit 1-3, hiding_predator 2-4) | ✅ | Matches the locked design exactly; per-entity collapsed full-grid entries; backward-compat scalar `count` preserved on `tree`. Scene is correct as authored — but see ❌ below: the resource ranges don't actually take effect at runtime due to the revival bug. |
| `src/environment/config_loader.py` | range parse + per-entry K bounds | ✅ | `_resolve_count_range` correct: `(N,N)` for scalar, `(1,1)` for neither, `ValueError` if both styles or `low>high`; allocates `count_high` slots. |
| `src/environment/state.py` | `animal_active`/`obs_active` + count-bound fields | ✅ | Masks on `EnvState`; `has_*_range` static `pytree_node=False` bools (recompile-safe). |
| `src/environment/core.py` | K-draw + mask threading | ❌ **blocker** | K-draw, PRNG (`fold_in` constants `0xC0A1/2/3`), off-grid park, and animal/obstacle threading are all correct. **`update_resources` (`core.py:130-141`) revives inactive resource slots on step 1** (`respawn_mask = ~res_active & (new_reg_timer<=0)` is True for never-existed slots). Reproduced 8/8 seeds on the new default. |
| `src/environment/sensor.py` | replace `jnp.ones` masks | ✅ | All sites (visual, olfaction, extero-noc, collision) thread the correct mask. |
| `tests/env/test_per_episode_count.py` | NEW regression test | ⚠️ | 10/10 pass, but `test_mask_stable_within_episode` omits `res_active` and uses no resource range — the exact blind spot that hid the blocker. Must be extended (see fix #2). |
| `docs/environment/CONFIG_GUIDE.md` + `02_config_schema.md` | doc keys | ✅ | §3.6 + Entity Count Expansion table added per Maintenance Contract. |
| `tests/env/test_backward_compat_configs.py` | `_count_yaml_animals` → `count_high` | ✅ | **Deviation cleared.** Genuine test-accounting correction, not a masked loader bug: the loader allocates `count_high` slots (`for _ in range(hi)`), and the test asserts `params.animal_property.shape[0] == expected`, so the expected value must equal `count_high`. Confirmed against the loader source. |
| default parity fixtures (×2) | deliberate regeneration | ✅ | Exactly 2 default fixtures regenerated in commit 2; commit 1 regenerated **zero** (machinery byte-transparent). All non-default fixtures untouched and green. |

### Test results (re-run by verifier)

- `tests/env/test_per_episode_count.py` + `tests/env/test_backward_compat_configs.py`: **58 passed, 111 skipped, 0 failed** (93 s).
- `tests/env/test_unified_parity.py` + `tests/env/test_visual_parity.py`: **34 passed, 133 skipped, 0 failed** (310 s). Matches the developer's reported counts exactly.
- **Two-commit parity discipline: CONFIRMED.** Commit 1 (`90d687f`) touched no fixtures and left both parity suites green → machinery is byte-transparent. Commit 2 (`7fa6183`) regenerated exactly the 2 default fixtures; every non-default config still passes against its pre-saved reference.

### Speed benchmark (clean idle node — the requested deliverable)

Measured on **node 104, GPU 0, fully idle (0% util, no contention)** with the project
interpreter. `num_envs=128`, warmup, BEFORE (commit `90d687f` default.yaml) vs AFTER
(current default.yaml), back-to-back same process. Reset = median of 51 jitted/vmapped
calls. SPS = best of 25 jitted `lax.scan` rollouts (256 steps) — the best-of-N
estimator strips warmup/clock-boost noise that made the median swing wildly. Run in
**both orderings** to rule out ordering bias.

| Metric | BEFORE (old scene: 8 res / 3 animal / 22 obs) | AFTER (new scene: 10 res / 5 animal / 22 obs) | Delta |
|---|---|---|---|
| `jax_reset` median (ms) | 0.96–1.06 | 1.30–1.39 | **+28 to +32%** (order-independent) |
| step SPS best (env-steps/s) | ~2.15 M | ~1.87 M | **−13 to −14%** (order-independent) |

Both deltas are **stable and reproduce identically when the run order is reversed** —
they track the *config*, not which config compiled first. The +5 entity slots
(+2 resources, +2 animals: the new masks threaded through the predator/rabbit
movement+collision+damage+sensing hot path) are the cost driver.

**This corrects the earlier proxy benchmark.** The proxy (shared GPU, fixed-count
temp config) predicted reset +3..+30% (noisy) and SPS ±3% (noise). The clean
measurement confirms the reset hit (+28-30%, upper end of the proxy band) and
**refutes the SPS prediction**: the real step-throughput regression is **~13%**, not
noise. The proxy under-counted because it could not exercise the masked-activation
step path.

Context: this is **environment-only** SPS in isolation. In real training the env step
is one component alongside the policy/learner forward-backward, so the wall-clock
training-throughput hit will be smaller than 13% (proportional to the env-step fraction
of the loop). But it is a real, measurable regression, not noise.

### Speed verdict

**⚠️ → discussion required.** −13% env-step SPS exceeds the >5% "warrants discussion"
threshold and approaches (but does not cross) the >15% blocker threshold. The plan did
**not** pre-accept a regression of this size (Checkpoint 6 named >15% as the blocker).
Recommend the user explicitly accept the ~13% env-SPS / ~30% reset cost as the price of
breaking layout memorisation, **or** trim the upper count bounds (the animal high-bounds
drive the hot-path cost) to claw some back. This is a product decision the user owns —
it is not, on its own, a merge blocker.

### Overall verdict: **CHANGES-REQUESTED**

The resource-revival blocker must be fixed before this scene trains. Handoff to
`developer` below. The speed regression is a separate, user-owned accept/trim decision.

### For `developer` to fix (blocker)

1. **`src/environment/core.py` `update_resources` (≈line 130-141) — stop reviving never-existed slots.** Gate respawn with a per-episode "allocated" mask so only genuinely-eaten slots regrow:
   `respawn_mask = ~res_active & (new_reg_timer <= 0) & allocated`, where `allocated` is the reset activation mask (the K-mask, constant within the episode). The cleanest route is to carry the reset `res_active` mask as a per-episode `res_allocated` field on `EnvState` (set once at reset, never mutated) and AND it into both `respawn_mask` and `new_active`. (The `code-reviewer` doc [[per_episode_count_activation_masks]] finding #1 lays out the options.)
2. **`tests/env/test_per_episode_count.py` `test_mask_stable_within_episode` — close the gap.** Add a food entry with `count_low < count_high`, and assert `int(jnp.sum(state.res_active))` stays equal to the reset K across all 50 steps (count equality, not array equality — respawn legitimately moves *eaten* slots, so the guard is "active-count never exceeds reset K"). This test must fail on current code and pass after fix #1.
3. After the fix, **the default parity fixtures (×2) must be regenerated again** — the per-step trajectory changes once inactive resources stay off — and the plan's two-commit discipline note applies (regenerate only the 2 default fixtures; all non-default must stay green).

### Cross-references

- Code-review (JAX correctness, full site-by-site audit): [[per_episode_count_activation_masks]] — same blocker, independently found.
