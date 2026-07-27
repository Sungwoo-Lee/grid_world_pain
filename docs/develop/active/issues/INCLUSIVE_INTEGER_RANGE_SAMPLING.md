---
title: "Inclusive-integer sampling for attack_range and detection_range"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Inclusive-integer sampling for `attack_range` and `detection_range`

> **Status**: PLANNED
> **Opened**: 2026-07-04
> **Related**: [[KNOWN_BUGS]] — row "Fractional `attack_range` bound silently has no effect" · memory insight `20260703_0344_attack_range_float_threshold_gotcha` · [[PREDATOR_JUMP_MECHANISM]] (jump feature) · [[UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING]] (per-episode sampling design)

---

## Context

Two predator-behaviour settings in the environment take a range written as `[low, high]` in the config — **`attack_range`** (how close the agent must be before a predator lunges/pounces) and **`detection_range`** (how close before a resting predator switches into the HUNT chase state). The intent of `[2, 3]` is "the value is 2 or 3". But the code today samples these as **floating-point numbers drawn uniformly** from `[2.0, 3.0)`, and then compares that float against the agent's **integer** grid distance (Manhattan distance, always a whole number) with `distance <= value`. Because a uniform draw from `[2.0, 3.0)` is always **less than 3**, the predator fires at distance 2 but **never** at distance 3 — so `[2, 3]` behaves identically to `[2, 2]`, silently. To actually reach distance 3 today you have to write `[2, 4]`, which is a confusing, undocumented workaround. The same bug hits `detection_range: [1, 7]` — the predator can never actually detect the agent at exactly 7 cells away; the top of the range is dead.

This plan changes both fields to sample as **inclusive integers**, exactly mirroring how `move_interval` and `attack_delay` already work (they draw `randint(low, high+1)`). After the fix, `[2, 3]` means the set {2, 3}, and `[1, 7]` means {1, 2, 3, 4, 5, 6, 7}. Single-value (scalar) configs like `detection_range: 5` are **completely unchanged** — they already collapse to a degenerate range and sample exactly 5 either way. Only configs that write a genuine `[low, high]` range with `low != high` change meaning.

**Do NOT implement from this doc directly** — it is a plan for the `developer` agent to execute after user approval. The plan carries a **scope toggle** (Part 1 = `attack_range` only; Part 2 = both fields) — see [Scope decision](#scope-decision).

---

## Analysis

### The two footguns, verified in code

Both fields compare a **float** per-episode sample against an **integer** Manhattan distance.

`dist` is integer Manhattan distance — [`src/environment/core.py:201-202`](../../../../src/environment/core.py):
```python
# Manhattan distance
dist = jnp.sum(jnp.abs(hunt_pos - agent_pos), axis=-1)
```

**`attack_range`** — jump/pounce trigger, [`src/environment/core.py:292-294`](../../../../src/environment/core.py):
```python
jump_attempted = (
    (next_state == 1) & (dist <= attack_range_s) & jnp.logical_not(agent_hidden)
    & (new_attack_timer <= 0) & (attack_range_s > 0) & hunt_active
)
```
`attack_range_s` is a **float** uniform draw — [`src/environment/core.py:1298-1301`](../../../../src/environment/core.py):
```python
attack_range_key = jax.random.fold_in(animal_episode_key, 0xA77AC7)
animal_attack_range_sampled = jax.random.uniform(
    attack_range_key, (N,), minval=params.animal_attack_range_low,
    maxval=jnp.maximum(params.animal_attack_range_high, params.animal_attack_range_low))
```

**`detection_range`** — HUNT-state transition, [`src/environment/core.py:218-222`](../../../../src/environment/core.py):
```python
rested_enough = hunt_stamina >= (hunt_max_stamina * hunt_thresh)
become_hunt = jnp.logical_and(
    jnp.logical_and(dist <= hunt_detect, rested_enough),
    jnp.logical_not(agent_hidden)
)
```
`hunt_detect` is `state.animal_detect_sampled`, a **float** uniform draw sharing key `ep_keys[0]` — [`src/environment/core.py:1277-1279`](../../../../src/environment/core.py):
```python
ep_keys = jax.random.split(animal_episode_key, 7)
animal_detect_sampled = jax.random.uniform(
    ep_keys[0], (N,), minval=params.animal_detect_low, maxval=jnp.maximum(params.animal_detect_high, params.animal_detect_low))
```

`detection_range` also appears in the lose-interest test at [`core.py:224-225`](../../../../src/environment/core.py) as `dist > hunt_detect * hunt_lose_interest`. Under the fix `hunt_detect` becomes an int and `hunt_lose_interest` stays float, so the product stays float — the comparison is unaffected in kind (its numeric value shifts only for range-form configs, exactly as intended).

**Confirmed: `detection_range` has the identical `<=`-on-integer-distance footgun and is in scope.** It is not a different formula.

### Why the four sibling float fields must NOT change

`detection_range` currently lives in `DISTRIBUTIONAL_FIELDS` ([`config_loader.py:111-117`](../../../../src/environment/config_loader.py)) and in the shared `dist_field_names` build loop ([`config_loader.py:717-734`](../../../../src/environment/config_loader.py)) alongside four genuinely continuous fields — `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier`. Those four are **not** compared against an integer distance and must keep float uniform sampling. The fix must peel `detection_range` out of the shared float machinery without touching the other four or the size-7 `ep_keys` split.

### The `move_interval` / `attack_delay` precedent (the exact pattern to copy)

Integer inclusive sampling already exists and is tested — [`core.py:1288-1293`](../../../../src/environment/core.py):
```python
# Integer sampling: randint half-open [lo, hi+1) is inclusive on both ends.
# Degenerate range (lo == hi): randint([s, s+1)) always yields s — byte-identical to scalar.
animal_move_int_sampled = jax.random.randint(
    ep_keys[5], (N,), params.animal_move_int_low, params.animal_move_int_high + 1)
animal_attack_delay_sampled = jax.random.randint(
    ep_keys[6], (N,), params.animal_attack_delay_low, params.animal_attack_delay_high + 1)
```
Their config parse loops ([`config_loader.py:620-643`](../../../../src/environment/config_loader.py) and [`654-677`](../../../../src/environment/config_loader.py)) do `int(mi[0]), int(mi[1])` with a `low <= high` guard and store `int32` low/high arrays. This is the template.

### PRNG parity — why scalar configs stay byte-identical

- `detection_range` keeps key `ep_keys[0]` and the split stays size-7, so `move_interval` (`ep_keys[5]`) and `attack_delay` (`ep_keys[6]`) draws are **unchanged**.
- `attack_range` keeps its independent `fold_in(…, 0xA77AC7)` key.
- For a **degenerate range** (`low == high`, i.e. every scalar config): old `uniform([v, v))` returns exactly `v`, and new `randint([v, v+1))` returns exactly `v`. Same integer value at runtime. Only the stored dtype changes (float32 → int32), which does not change any comparison result. **Every scalar `detection_range` config and the `[2,2]` attack_range config are behaviourally byte-identical after the fix.**

### The fractional-bound decision (closes the KNOWN_BUGS row)

The [[KNOWN_BUGS]] row is specifically about fractional bounds silently doing nothing. Grep confirms **no current config uses a fractional bound** on either field (all `detection_range` scalars are whole numbers; `attack_range` only ever `[2,2]` or `[2,4]`). Rather than silently truncating a future fractional bound (the existing `int()` behaviour of `move_interval`), the plan **rejects a non-whole-number bound with a `ValueError`** for these two integer-distance fields — loud failure, consistent with the project's no-silent-footgun / no-fallback-defaults philosophy. This is what permanently retires the KNOWN_BUGS gotcha. (Optionally back-porting the same guard to `move_interval`/`attack_delay` is noted as out-of-scope to avoid scope creep.)

---

## Implementation Plan

### Design

Move `detection_range` and `attack_range` from float-uniform to integer-inclusive sampling, mirroring `move_interval`/`attack_delay`:

1. **Config loader** — parse both as integers with a `low <= high` guard and a whole-number guard; store `int32` low/high arrays; remove `detection_range` from the shared float loop and `DISTRIBUTIONAL_FIELDS`.
2. **EnvParams** — update the dtype comments (arrays become int).
3. **Reset sampling** — swap the two `jax.random.uniform` calls for `jax.random.randint(lo, hi+1)`; make the `N == 0` else-branch zeros `int32`.
4. **Config revert** — `06-jump_range_2to3.yaml`'s `attack_range: [2,4]` (old workaround for {2,3}) must become `[2,3]`.
5. **Regression test** — new tests proving `[2,3]`→{2,3}, jump gate fires at distance 3, `[1,7]`→{1..7} incl. 7, and scalar configs unchanged.

Downstream comparisons need no change: `dist <= hunt_detect` and `dist <= attack_range_s` become int-vs-int; `hunt_detect * hunt_lose_interest` becomes int×float→float. The only non-env consumer, [`src/behavior/accumulators.py:531,538`](../../../../src/behavior/accumulators.py), already wraps the value in `float(...)` for WandB logging (`Episode/sampled_detect_<tag>`) — int32 input logs fine, no change.

### Scope decision

The plan is split so the user can choose blast radius:

- **Part 1 — `attack_range` only.** Smallest change. Fixes the jump/pounce field. Reverts one config (`06-jump_range_2to3.yaml`). No change to any detection behaviour.
- **Part 2 — add `detection_range`.** Applies the identical fix to the detection field. **This shifts behaviour for every config that writes `detection_range` as a genuine range** — the per-distance HUNT-trigger probability changes, and the top of the range (e.g. distance 7 for `[1,7]`) becomes reachable for the first time. Scalar `detection_range` configs are unaffected.

**Recommendation: do BOTH (Part 1 + Part 2).** The two fields share the identical `dist <= float` footgun; fixing one and leaving the other is a confusing half-measure that leaves a live footgun in a more widely-used field. The `detection_range` shift is real but small (see [affected configs](#affected-configs)), and — critically — **the code change affects FUTURE launches only.** Runs currently training already loaded their `EnvParams` at launch and are not disturbed; only the next relaunch of an affected config picks up the new semantics.

### File Changes

#### `src/environment/config_loader.py`

**(A) Remove `detection_range` from `DISTRIBUTIONAL_FIELDS`** (lines 111-117) — Part 2 only:
```python
# BEFORE:
DISTRIBUTIONAL_FIELDS = (
    "detection_range",
    "max_stamina",
    "stamina_recovery_rate",
    "hunt_stamina_threshold",
    "lose_interest_multiplier",
)

# AFTER:
# detection_range moved to inclusive-integer sampling (see INCLUSIVE_INTEGER_RANGE_SAMPLING.md);
# only genuinely-continuous fields remain here.
DISTRIBUTIONAL_FIELDS = (
    "max_stamina",
    "stamina_recovery_rate",
    "hunt_stamina_threshold",
    "lose_interest_multiplier",
)
```
Audit the docstring at [`config_loader.py:363`](../../../../src/environment/config_loader.py) ("`detection_range: [lo, hi]` (and four siblings in DISTRIBUTIONAL_FIELDS)") and update it to reflect that `detection_range` is now inclusive-integer, not float-uniform.

**(B) `attack_range` integer parse with whole-number guard** (lines 686-694) — Part 1:
```python
# BEFORE:
attack_range_low_list = []
attack_range_high_list = []
for i, e in enumerate(entries):
    lo, hi = _parse_distributional(
        e['dist_source'], 'attack_range', mandatory=False,
        entity_label=e['tag_label'], idx=i
    )
    attack_range_low_list.append(lo)
    attack_range_high_list.append(hi)

# AFTER:
# attack_range: scalar OR [lo, hi] INCLUSIVE-INTEGER Manhattan-distance jump range.
# Compared against integer grid distance (core.py `dist <= attack_range_s`), so a
# fractional bound is meaningless and is rejected loudly (see
# INCLUSIVE_INTEGER_RANGE_SAMPLING.md). Missing -> [0, 0] (jump disabled).
attack_range_low_list = []
attack_range_high_list = []
for i, e in enumerate(entries):
    lo_f, hi_f = _parse_distributional(
        e['dist_source'], 'attack_range', mandatory=False,
        entity_label=e['tag_label'], idx=i
    )
    for _b in (lo_f, hi_f):
        if _b != int(_b):
            raise ValueError(
                f"Animal entity {e['tag_label']!r} (index {i}): 'attack_range' bound "
                f"{_b} is not a whole number; this field is compared against integer "
                f"grid distance and must be integer-valued."
            )
    attack_range_low_list.append(int(lo_f))
    attack_range_high_list.append(int(hi_f))
```

**(C) `detection_range` dedicated integer parse loop** — Part 2. Remove `("detection_range", "animal_detect")` from `dist_field_names` (lines 717-723):
```python
# BEFORE:
dist_field_names = (
    ("detection_range", "animal_detect"),
    ("max_stamina", "animal_max_stamina"),
    ("stamina_recovery_rate", "animal_recovery"),
    ("hunt_stamina_threshold", "animal_hunt_thresh"),
    ("lose_interest_multiplier", "animal_lose_interest"),
)

# AFTER:
dist_field_names = (
    ("max_stamina", "animal_max_stamina"),
    ("stamina_recovery_rate", "animal_recovery"),
    ("hunt_stamina_threshold", "animal_hunt_thresh"),
    ("lose_interest_multiplier", "animal_lose_interest"),
)
```
Then add a dedicated integer parse loop (place it near the other integer loops, e.g. just after the `attack_delay` loop at line 677), mirroring `move_interval`/`attack_delay` but honouring `mandatory=e['mandatory_dist']` (detection_range is required for `hunt` behaviour):
```python
# detection_range: scalar OR [lo, hi] INCLUSIVE-INTEGER HUNT-trigger Manhattan range.
# Mandatory for hunt entities; compared against integer grid distance, so integer-valued.
detect_range_list = []  # list of (lo_int, hi_int)
for i, e in enumerate(entries):
    lo_f, hi_f = _parse_distributional(
        e['dist_source'], 'detection_range',
        mandatory=e['mandatory_dist'],
        entity_label=e['tag_label'], idx=i
    )
    for _b in (lo_f, hi_f):
        if _b != int(_b):
            raise ValueError(
                f"Animal entity {e['tag_label']!r} (index {i}): 'detection_range' bound "
                f"{_b} is not a whole number; this field is compared against integer "
                f"grid distance and must be integer-valued."
            )
    lo, hi = int(lo_f), int(hi_f)
    if hi < lo:
        raise ValueError(
            f"Animal entity {e['tag_label']!r} (index {i}): 'detection_range' range "
            f"must satisfy low <= high; got [{lo}, {hi}]."
        )
    detect_range_list.append((lo, hi))
```
(The `hi < lo` guard is already enforced inside `_parse_distributional`; the explicit check above mirrors the `move_interval` loop and is belt-and-braces. `developer` may drop it if redundant — the `_parse_distributional` guard suffices.)

**(D) Build int32 arrays** — `attack_range` (lines 814-815, Part 1) and `detection_range` (lines 821-822, Part 2):
```python
# BEFORE (814-815):
animal_attack_range_low = jnp.array(attack_range_low_list, dtype=jnp.float32)
animal_attack_range_high = jnp.array(attack_range_high_list, dtype=jnp.float32)
# AFTER:
animal_attack_range_low = jnp.array(attack_range_low_list, dtype=jnp.int32)
animal_attack_range_high = jnp.array(attack_range_high_list, dtype=jnp.int32)

# BEFORE (821-822):
animal_detect_low = jnp.array(dist_lows['animal_detect'], dtype=jnp.float32)
animal_detect_high = jnp.array(dist_highs['animal_detect'], dtype=jnp.float32)
# AFTER:
animal_detect_low = jnp.array([lo for lo, hi in detect_range_list], dtype=jnp.int32)
animal_detect_high = jnp.array([hi for lo, hi in detect_range_list], dtype=jnp.int32)
```
Note: `has_attack_feature = any(hi > 0 …)` at line 817 works unchanged with int bounds.

#### `src/environment/state.py` (lines 127-128, 145-146)

Update the dtype comments to match (no logic change, but the contract comment must be truthful):
```python
# BEFORE (127-128):
    animal_detect_low: jnp.ndarray         # [N]
    animal_detect_high: jnp.ndarray        # [N]
# AFTER (Part 2):
    animal_detect_low: jnp.ndarray         # [N] int — inclusive-integer HUNT-trigger range, low bound
    animal_detect_high: jnp.ndarray        # [N] int — inclusive-integer HUNT-trigger range, high bound

# BEFORE (145-146):
    animal_attack_range_low: jnp.ndarray      # [N] float — jump-trigger Manhattan-distance range, low bound
    animal_attack_range_high: jnp.ndarray     # [N] float — jump-trigger Manhattan-distance range, high bound
# AFTER (Part 1):
    animal_attack_range_low: jnp.ndarray      # [N] int — inclusive-integer jump-trigger Manhattan range, low bound
    animal_attack_range_high: jnp.ndarray     # [N] int — inclusive-integer jump-trigger Manhattan range, high bound
```
Also update the "Per-episode uniform bounds" comment at line 126 to note detection_range is now integer-randint, not uniform (Part 2).

#### `src/environment/core.py`

**(E) `detection_range` sampling** (lines 1277-1279, Part 2):
```python
# BEFORE:
ep_keys = jax.random.split(animal_episode_key, 7)
animal_detect_sampled = jax.random.uniform(
    ep_keys[0], (N,), minval=params.animal_detect_low, maxval=jnp.maximum(params.animal_detect_high, params.animal_detect_low))
# AFTER:
ep_keys = jax.random.split(animal_episode_key, 7)  # split size UNCHANGED (parity)
# detection_range: inclusive-integer randint([lo, hi+1)) — mirrors move_interval/attack_delay.
animal_detect_sampled = jax.random.randint(
    ep_keys[0], (N,), params.animal_detect_low, params.animal_detect_high + 1)
```

**(F) `attack_range` sampling** (lines 1298-1301, Part 1):
```python
# BEFORE:
attack_range_key = jax.random.fold_in(animal_episode_key, 0xA77AC7)
animal_attack_range_sampled = jax.random.uniform(
    attack_range_key, (N,), minval=params.animal_attack_range_low,
    maxval=jnp.maximum(params.animal_attack_range_high, params.animal_attack_range_low))
# AFTER:
attack_range_key = jax.random.fold_in(animal_episode_key, 0xA77AC7)
# inclusive-integer randint([lo, hi+1)) — degenerate [s,s] still yields s (parity).
animal_attack_range_sampled = jax.random.randint(
    attack_range_key, (N,), params.animal_attack_range_low, params.animal_attack_range_high + 1)
```

**(G) `N == 0` else-branch dtype** (lines 1303, 1310):
```python
# BEFORE:
animal_detect_sampled        = jnp.zeros(0, dtype=jnp.float32)   # line 1303 — Part 2
...
animal_attack_range_sampled  = jnp.zeros(0, dtype=jnp.float32)   # line 1310 — Part 1
# AFTER:
animal_detect_sampled        = jnp.zeros(0, dtype=jnp.int32)     # Part 2
...
animal_attack_range_sampled  = jnp.zeros(0, dtype=jnp.int32)     # Part 1
```

#### `configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml` (line 38) — Part 1 (config revert)

```yaml
# BEFORE:
      attack_range: [2, 4]           # VARIANT: was [2,2] -> effective per-episode reach 2 OR 3 (see header)
# AFTER:
      attack_range: [2, 3]           # inclusive-integer {2,3} (was [2,4] float-workaround; see INCLUSIVE_INTEGER_RANGE_SAMPLING.md)
```
Also update the file's header comment block, which explains the old `[2,4]→{2,3}` workaround, so it no longer describes the removed float semantics. `developer` should read the header (top ~30 lines) and rewrite the rationale to "inclusive-integer `[2,3]` = {2,3}".

#### New test file: `tests/env/test_inclusive_integer_range_sampling.py`

Mirror [`tests/env/test_int_distributional_sampling.py`](../../../../tests/env/test_int_distributional_sampling.py) (same `_BASE_YAML` harness, same `_load` helper). Required cases:

1. **`attack_range: [2, 3]` → sampled ∈ {2, 3}, both occur** over ~300 resets (read `state.animal_attack_range_sampled[0]`).
2. **Jump gate fires at distance 3** — place agent exactly 3 Manhattan cells from a hunting predator whose sampled `attack_range` == 3, step, assert a jump is attempted / attack timer set (mirror the assertions in [`tests/env/test_predator_jump.py`](../../../../tests/env/test_predator_jump.py)). This is the case that FAILS on pre-fix code (float `[2,3]` never reaches 3) and PASSES after.
3. **`detection_range: [1, 7]` → sampled ∈ {1..7}, and 7 occurs** over ~300 resets (Part 2). Pre-fix, 7 never occurs — this asserts the top of the range is now reachable.
4. **Degenerate/scalar unchanged** — `attack_range: [2, 2]` always samples 2; `detection_range: 5` always samples 5 (byte-identical guard, mirror `test_backward_compat_scalar_byte_identical`).
5. **Whole-number guard** — a config with `attack_range: [2, 3.5]` (and `detection_range: [1, 6.5]` for Part 2) raises `ValueError` at `load_env_params`.

If Part 1 only is chosen, cases 3 is omitted and case 5's detection sub-case is dropped.

### Documentation & registry follow-ups (in the same change)

- **No `scripts/` files touched** → no `SCRIPTS_DEPENDENCY_MAP.md` update needed.
- **No config-system schema change** (the YAML surface `[lo, hi]` is unchanged; only its runtime meaning) → `CONFIG_GUIDE.md` / `02_config_schema.md` do not strictly require edits. However, if either doc documents `detection_range`/`attack_range` as "float uniform", `developer` should correct that line to "inclusive integer".
- **KNOWN_BUGS registry** — do NOT edit `KNOWN_BUGS.md` here (owned by `bug-curator`). After the fix lands, hand off to `bug-curator` to flip the "Fractional `attack_range` bound silently has no effect" row to fixed, linked to the fix commit.
- **Break point / semantic-change record** — this is a per-episode-sampling semantic change to a range field (parallel to the earlier reward-fix break point). Once landed, record a memory insight (`/wiki-write`) noting that `[lo,hi]` on `attack_range`/`detection_range` changed from float-uniform to inclusive-integer as of the fix commit, so future analyses comparing pre/post-fix runs know the boundary.

### Affected configs

Scalar `detection_range` configs (values `5`, `3`, `10`, `12`, `7`, `2`, `0`) are **unchanged** — they sample the same integer either way. Only genuine `[low, high]` forms shift. Full audit via `grep -rn "attack_range:\|detection_range:" configs/`.

**`attack_range` — all occurrences (2 total):**

| Config | Old | Old effective set | New (inclusive int) | Action |
|---|---|---|---|---|
| `basic/07-jump_attack_10x10.yaml` | `[2, 2]` | {2} | {2} | none (degenerate) |
| `basic05_variants/06-jump_range_2to3.yaml` | `[2, 4]` | {2, 3} | {2, 3, 4} | **MUST revert to `[2, 3]`** → {2, 3} |

**`detection_range` — range-form occurrences (all scalar forms unchanged, omitted):**

Active (relaunch picks up new semantics):

| Config | Value | Old reachable HUNT-trigger set | New |
|---|---|---|---|
| `basic/05-random_init_10x10.yaml` | `[1, 7]` | {1..6} (7 unreachable) | {1..7} |
| `basic/07-jump_attack_10x10.yaml` | `[1, 7]` | {1..6} | {1..7} |
| `basic05_variants/02-relentless_stamina.yaml` | `[1, 7]` | {1..6} | {1..7} |
| `basic05_variants/03-fast_move_interval.yaml` | `[1, 7]` | {1..6} | {1..7} |
| `basic05_variants/04-all_combined.yaml` | `[1, 7]` | {1..6} | {1..7} |
| `basic05_variants/06-jump_range_2to3.yaml` | `[1, 7]` | {1..6} | {1..7} |

Archive (behaviour-shift noted; no action — these are historical):

| Config | Value | Old | New |
|---|---|---|---|
| `archive/hypervigilance/03-sameProp_R3_predatorDistributional.yaml` | `[0, 10]` | {0..9} | {0..10} |
| `archive/hypervigilance/04-sameProp_R4_chasingRabbit.yaml` | `[0, 10]` | {0..9} | {0..10} |
| `archive/v2_smoke/02-entities-distributional.yaml` | `[0, 5]` | {0..4} | {0..5} |

More precisely, for `detection_range: [1,7]` the *entire* per-distance HUNT-trigger probability shifts (old `P(detect at dist d) = (7-d)/6`; new `= |{d..7}|/7`), with distance 7 going from **impossible** to possible. This is a real but modest change to a field used (as a scalar) across most predator configs — but again, only the six active **range-form** configs above are affected, and only on their next launch.

### Running-experiment implications

- **`rppo_basic07_jumpreach_n113`** (jump-reach run on node 113) used `attack_range: [2, 4]` (config `06-jump_range_2to3.yaml`). It already loaded its `EnvParams` at launch and is **not disturbed** mid-flight. After the fix, **relaunch it with `[2, 3]`** to get a clean inclusive {2, 3} jump reach (the old `[2,4]` under the new rule would give {2,3,4}).
- **The six basic / basic05 runs** using `detection_range: [1, 7]` are likewise unaffected while running. Their **next relaunch** will use the corrected reaches-7 semantics — flag this to whoever owns those experiments so the behaviour delta is expected, not surprising.

---

## Checkpoints

Verify **during** implementation:

- [x] After config-loader change, `load_env_params` on `06-jump_range_2to3.yaml` yields `animal_attack_range_low/high` dtype `int32` with values `[2 2 0 0]`/`[3 3 0 0]` (predator entity slots 0-1; unused wander/static slots 2-3 default to `[0,0]`).
- [x] `jax_reset` over ~50 seeds on `[2,3]` produces `animal_attack_range_sampled` values in {2, 3} with both present; on `[2,2]` always 2. (Ran 300 seeds — both `test_attack_range_inclusive_both_ends_occur` and `test_scalar_attack_range_byte_identical`.)
- [x] (Part 2) `detection_range: [1,7]` over ~300 seeds yields all of {1..7} including 7; scalar `detection_range: 5` always 5.
- [x] A fractional bound (`attack_range: [2, 3.5]`) raises `ValueError` at load. (Also verified `detection_range: [1, 6.5]`.)
- [x] Full env test suite green: `pytest tests/env/ -q` — `test_int_distributional_sampling.py`, `test_predator_jump.py`, `test_no_recompile.py` all pass (19/19). Full suite: 197 passed, 4 failed (all 4 are `test_unified_parity.py::observability_gates_S1-S4`, confirmed **pre-existing** and unrelated — see Implementation Report). No `test_per_episode_sampling.py` file exists in the repo (checkpoint text assumed a file that isn't present; the equivalent coverage lives in `test_int_distributional_sampling.py`).
- [x] New `tests/env/test_inclusive_integer_range_sampling.py` passes (7/7); confirmed case 2 (jump at distance 3), case 1 (`{2,3}` both occur), and case 3 (`detection_range` reaches 7) all FAIL when the `core.py` sampling lines are temporarily reverted to `jax.random.uniform` — proving these are genuine regression tests, not vacuously-true assertions.
- [x] Speed check: measured `jax_step` throughput (batched, jitted) on `06-jump_range_2to3.yaml`, 5000 steps × 64 batch, same seed/config. Pre-fix and post-fix both land in the 141k-161k env-steps/sec band across repeated runs — the before/after delta is smaller than run-to-run noise. Expected: the change touches only `jax_reset` (once per episode), not the `jax_step` hot loop. No regression.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-04

**Both Part 1 (`attack_range`) and Part 2 (`detection_range`) implemented**, per the plan's recommendation.

### Files changed

**`src/environment/core.py`**
- Sampling block (was ~1272-1310, now ~1272-1311): `animal_detect_sampled` swapped from `jax.random.uniform(ep_keys[0], minval=..., maxval=...)` to `jax.random.randint(ep_keys[0], lo, hi+1)` — key and split position (`ep_keys[0]` of the unchanged size-7 split) untouched. `animal_attack_range_sampled` swapped from `jax.random.uniform(attack_range_key, ...)` to `jax.random.randint(attack_range_key, lo, hi+1)` — independent `fold_in` key untouched. `N==0` else-branch zeros for both fields changed from `dtype=jnp.float32` to `dtype=jnp.int32`. Updated the block comment from "5 float + 2 integer" to "4 float + 3 integer" (detect_range moved from the float bucket to the integer bucket).

**`src/environment/config_loader.py`**
- `DISTRIBUTIONAL_FIELDS` (was 111-117): removed `"detection_range"`; now only the four genuinely-continuous fields.
- Docstring (was ~363-364): updated to state `detection_range` is inclusive-integer, not grouped with the four float siblings.
- `attack_range` parse loop (was ~686-694): added a whole-number guard (raises `ValueError` if either bound is fractional) and now stores `int(lo_f)`/`int(hi_f)`.
- New dedicated `detection_range` parse loop (added just before the `dist_field_names` block, in place of its former membership there): mirrors the `attack_range` loop — whole-number guard, `mandatory=e['mandatory_dist']`, `hi < lo` guard, builds `detect_range_list` of `(lo_int, hi_int)` tuples.
- `dist_field_names` (was 717-723): removed the `("detection_range", "animal_detect")` tuple.
- Array-building section (was ~814-815, ~821-822): `animal_attack_range_low/high` and `animal_detect_low/high` now built with `dtype=jnp.int32` (was `float32`); `animal_detect_low/high` now sourced from `detect_range_list` instead of `dist_lows['animal_detect']`/`dist_highs['animal_detect']`.
- **Extra (flagged, not in the plan's explicit line list but same variables/dtype-consistency requirement)**: the `N==0` zero-animal branch (was ~558-559, ~564-565) also built `animal_attack_range_low/high` and `animal_detect_low/high` as bare `jnp.zeros(0)` (implicit float32). Changed to `dtype=jnp.int32` for consistency with the non-zero-animal build path (same fields, same fix). This is a same-file, same-variable extension of the plan's dtype change, not a new file or new scope — flagging per protocol rather than silently expanding.

**`src/environment/state.py`**
- Dtype comments updated: `animal_detect_low/high` (was 127-128) now documented as `[N] int — inclusive-integer HUNT-trigger range`; `animal_attack_range_low/high` (was 145-146) now documented as `[N] int — inclusive-integer jump-trigger Manhattan range` (previously said `float`). The "Per-episode uniform bounds" header comment (line 126) updated to note `detection_range` is now integer-randint, not uniform, for the four true float siblings.

**`configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml`**
- `attack_range: [2, 4]` → `[2, 3]` (line ~38), reverting the old float-workaround now that `[2,3]` correctly yields the inclusive set `{2, 3}`.
- Header comment block (lines 1-13) rewritten to describe the new inclusive-integer semantics instead of the retired float-workaround rationale.

**`tests/env/test_inclusive_integer_range_sampling.py`** (new file, 7 tests)
1. `test_attack_range_inclusive_both_ends_occur` — `attack_range: [2,3]`, 300 seeds, both 2 and 3 occur.
2. `test_jump_gate_fires_at_distance_3` — dedicated fixed-geometry config (agent at 0-indexed `[0,0]`, predator forced to spawn at `[3,0]`, Manhattan distance exactly 3); loops 60 seeds, and whenever the per-episode sample equals 3, asserts the jump fires (predator lands on agent, `hit_predator=True`); asserts at least one such seed was found across 60 draws.
3. `test_detection_range_reaches_top_of_range` — `detection_range: [1,7]`, 300 seeds, all of `{1..7}` including 7 occur.
4. `test_scalar_attack_range_byte_identical` / `test_scalar_detection_range_byte_identical` — scalar `attack_range: 2` / `detection_range: 5` always sample exactly 2 / 5 across 50 seeds.
5. `test_fractional_attack_range_bound_raises` / `test_fractional_detection_range_bound_raises` — `attack_range: [2, 3.5]` and `detection_range: [1, 6.5]` each raise `ValueError` at `load_env_params`.

### Fail-before / pass-after evidence

Temporarily reverted only the two `core.py` sampling calls (`jax.random.randint` → `jax.random.uniform`, restoring the exact pre-fix logic) via a saved patch, re-ran the new test file, then re-applied the patch:

- **Pre-fix** (`git apply -R` of the fix patch): `pytest tests/env/test_inclusive_integer_range_sampling.py -q` → **3 failed, 4 passed**. Failures were exactly the three cases that depend on reaching the top of an inclusive range:
  - `test_attack_range_inclusive_both_ends_occur`: `AssertionError: Expected both {2, 3} to occur over 300 draws; got {2}.`
  - `test_jump_gate_fires_at_distance_3`: `AssertionError: Sampled attack_range never equalled 3 across 60 seeds -- the top of the inclusive range [2,3] is unreachable (the pre-fix float-uniform bug).`
  - `test_detection_range_reaches_top_of_range`: `AssertionError: Expected all of {1..7} to occur over 300 draws; got [1, 2, 3, 4, 5, 6].`
  - The scalar-byte-identical and fractional-bound-guard tests still passed pre-fix (as expected — those code paths are guard logic added fresh, not the sampling call).
- **Post-fix** (patch re-applied): `pytest tests/env/test_inclusive_integer_range_sampling.py -q` → **7 passed**.

This confirms the regression tests are genuine (not vacuously true) and the fix resolves exactly the documented footgun.

### Full test suite

`pytest tests/env/ -q` → **197 passed, 4 failed, 492 skipped** (784s). The 4 failures are all `test_unified_parity.py::test_parity[configs__verification__observability_gates_S{1,2,3,4}]`, and are **pre-existing, unrelated to this change**:
- The failure is an `agent_pos` mismatch at step 0 (`ACTUAL: [4,4]` vs fixture's `DESIRED: [2,2]`) — this is upstream of any animal/predator sampling; agent position comes straight from `params.start_pos`/`random_start_pos`, never touched by this fix.
- Root cause: commit `84014e4` ("observability gates use fixed start pos, Finding G2"), landed earlier the same day, changed `observability_gates_S1-4.yaml`'s `start_pos` from `[2,2]`-equivalent to `[5,5]` (0-indexed `[4,4]`) *after* the `.npz` parity fixtures were captured with the old start position — a stale-fixture situation unrelated to `attack_range`/`detection_range`.
- **Verified directly**: temporarily reverted only the 3 fixed `src/` files back to pre-fix (via the same saved patch) and re-ran these 2 of the 4 parity tests — they failed identically (`ACTUAL: [4,4]` vs `DESIRED: [2,2]`), confirming the failure exists independent of this fix. Re-applied the fix afterward (confirmed clean diff match against the committed state).
- Per the task's explicit instruction, fixtures were **not** regenerated.

Also separately ran `tests/env/test_predator_jump.py tests/env/test_int_distributional_sampling.py tests/env/test_no_recompile.py` (the three named checkpoint files) — **19/19 passed**. (`test_per_episode_sampling.py` named in the plan's checkpoint text does not exist in the repo; no action taken beyond noting it here.)

### Config-load sanity (basic ladder + running-run configs)

Batch-loaded every config under `configs/environment/experiment/`, `configs/verification/`, `configs/continual/` through `load_env_config` + `load_env_params`. All of the **basic ladder** (`basic/01-07`), **`basic05_variants/01-06`** (including the reverted `06-jump_range_2to3.yaml`), **`basic_curriculum/01-04`**, and **`verification/observability_gates_S1-4`** loaded without error. The only failures found were pre-existing and structurally unrelated to `attack_range`/`detection_range`: (a) `configs/environment/experiment/archive/**` (legacy archived configs missing `sensory.injury_observable`) and (b) `configs/continual/*_schedule*.yaml` / other continual-stage fragment files missing `environment.resources` — both are fragment/legacy configs not meant to be loaded standalone via `load_env_config` (they compose through a different mechanism), and both fail on keys unrelated to this fix.

### Speed check

`jax_step` throughput (jitted, `vmap` batch=64, 5000 steps, same seed) on `configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml`:
- Post-fix: 144,195 env-steps/sec (one run); repeated runs on the identical post-fix code ranged 140,963-160,380 env-steps/sec.
- Pre-fix (patch-reverted): 160,861 env-steps/sec (one run).
- **Conclusion: no regression.** The pre/post delta (≈10%) is smaller than the run-to-run noise band measured on the *same* post-fix code across 3 repeats (140,963-160,380, ≈13% spread) — consistent with the change touching only `jax_reset` (once per episode) and leaving the `jax_step` hot loop byte-identical. Command: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python tmp/20260704_speed_check_inclusive_range.py` (script left in `tmp/` per project convention, not committed).

### Deviations from the plan

1. **N==0 zero-animal branch dtype** (`config_loader.py` ~558-559, ~564-565): the plan's File Changes section did not explicitly list this branch, but it builds the same `animal_attack_range_low/high`/`animal_detect_low/high` variables as bare `jnp.zeros(0)` (implicit float32). Changed to explicit `dtype=jnp.int32` for consistency with the non-zero build path — same fields, same fix, same file, no new scope. Flagging here per the "discover but don't silently expand" protocol.
2. **Checkpoint text referenced `test_per_episode_sampling.py`**, which does not exist in the repo; ran the three files that do exist (`test_int_distributional_sampling.py`, `test_predator_jump.py`, `test_no_recompile.py`) instead.
3. No other deviations. Both Part 1 and Part 2 implemented as specified; the whole-number/fractional-bound guard was implemented as designed in the plan (belt-and-braces `hi < lo` check kept, matching the plan's "developer may drop it if redundant... suffices" note left it in for parity with the `attack_range` loop).

### Commit

`7ff8d1f` — `fix(env): 🐛 inclusive-integer sampling for attack_range/detection_range ([2,3] means {2,3})` — includes `src/environment/core.py`, `src/environment/config_loader.py`, `src/environment/state.py`, `configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml`, `tests/env/test_inclusive_integer_range_sampling.py`. Not pushed.

### Follow-ups (not this developer's scope)

- **KNOWN_BUGS.md**: per instructions, not edited here. Hand off to `bug-curator` to flip the "Fractional `attack_range` bound silently has no effect" row to fixed, linked to commit `7ff8d1f`.
- **`/wiki-write`**: the plan recommends a memory insight recording the semantic-change break point (pre/post-fix `attack_range`/`detection_range` behaviour) for future run-comparison analyses — not done in this session; flagging for `senior-developer`/user follow-up.
- **Running-experiment relaunch**: per the plan's "Running-experiment implications" section, `rppo_basic07_jumpreach_n113` (using the now-reverted `06-jump_range_2to3.yaml`) and the six `detection_range: [1,7]` configs are unaffected while currently running (params loaded at launch) but should be relaunched with the corrected config to pick up the new semantics — this is an experiment-ownership action, not a code action, so left for the user/`senior-developer` to coordinate.

## Verification Report

> **Verified by**: [senior-developer]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]
