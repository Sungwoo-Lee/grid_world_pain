---
title: "Fix plan WP-E: eval-output data corruption — stats-CSV column shift (H8) + offline interrupted-feeding rate pinned to 0 (H9)"
topic: issues
status: active
created: 2026-07-06
last_updated: 2026-07-06
---

# Fix Plan WP-E — Eval-Output Correctness: H8 (stats-CSV column shift) + H9 (offline M1 rate always 0)

> **Status**: PLANNED (user-approved; ready for `developer`)
> **Opened**: 2026-07-06
> **Related**: [[06_evaluation_path]] (H8 source finding), [[07_behavior_measures]] (H9 source finding), [[00_combined_diagnosis]] §2 rows H8/H9

---

## Context

Two independent bugs corrupt the **numbers that current studies read from evaluation
outputs** — neither touches training; both silently distort analysis.

**Bug 1 (H8 — mislabeled sensor columns in the eval stats CSV).** During evaluation the
project writes a per-step CSV with one column per sensor reading (satiation, nociception,
smell, etc.). The code that writes the column *names* forgot one sensor — the body-pain
signal ("Interoceptive Nociception"), which is switched on by default — while the *values*
are written in the sensor's true order. Result: from that sensor onward, every value sits
under the previous sensor's name. Anyone reading a column by name (e.g. the noise-diagnostics
tool) gets a different sensor's numbers with no error. Planning verification against real run
CSVs additionally found the corruption is **worse than the diagnosis stated** on the default
environment: when the world contains obstacles, each data row is one field *longer* than the
header, so a pandas read misassigns *every* column, not just the sensor block (details in
Analysis below).

**Bug 2 (H9 — "interrupted feeding" rate structurally always zero).** The offline replay
that recomputes behavior measures from saved evaluation episodes — the path current studies
consume via `online_replay.json` — checks "has the agent avoided eating for a full window?"
*before* updating its has-the-agent-eaten counter for the current step. The check can
therefore never succeed: the reported interrupted-feeding rate is 0.0 for every agent, no
matter how it behaves. Any past conclusion of the form "no feeding interruption in condition
X" drawn from this file is unsupported until recomputed.

This plan fixes both, adds regression tests that fail on the current code, and documents
exactly what to do with already-written output files (they are largely recoverable — see the
consumer analysis).

## Scope fence

**In scope:** exactly the two bugs above (combined-diagnosis rows H8 and H9), their
regression tests, and the one direct name-based CSV consumer.

**Out of scope (do NOT touch, even though they live in the same files):**
- The other eval findings from [[06_evaluation_path]]: parallel-eval first-wave survival
  undercount (Finding 2), stochastic-mode PRNG key reuse (Finding 3), dead
  `eval_obs_noise` / `eval_max_steps` config keys (Finding 4), `trajectory_story` decode
  bugs (Findings 5–6), `render_recordings` skip/concat interaction (Finding 7).
- The offline-vs-online **denominator-timing divergence** ([[07_behavior_measures]]
  Finding 4, Med): the offline replay counts the M1 denominator at window-resolution time
  while the fixed online accumulator counts it at event-record time. **This stays open.**
  The H9 fix below deliberately leaves the offline denominator where it is (resolution
  time) and only repairs the numerator condition ordering. Do not "helpfully" align the
  denominator — that is a separate decision tied to the undecided episode-end rule.
- The offline eat-under-threat ratio explosion ([[07_behavior_measures]] Finding 5).
- Entity column names (`obs_entity_*` etc.) stay as they are — renaming would break
  existing name-based readers of those columns.

No new config keys. No schema changes.

---

# Issue 1 (H8) — Eval stats CSV: sensor columns shifted / rows misaligned

## Analysis

### Root cause

`src/utils/evaluation_core.py` builds the stats-CSV sensor headers by iterating
`get_observation_breakdown(params)` through an `elif` chain — once for `obs_*` headers
(lines 204–222) and once, duplicated, for `true_*` headers (lines 225–243). The chain
handles `Satiation`, `Nutrition`, `Injury`, `Extero Nociception`, `Olfaction`, `Collision`,
`Location`, `Visual`, `Proprioception` — and silently falls through for
**`"Interoceptive Nociception"`**, which `get_observation_breakdown`
(`src/environment/sensor.py:362-364`) emits between Satiation and Extero Nociception
whenever `interoceptive_nociception_enabled` is true (default **true**,
`configs/environment/default.yaml:217`). The row values, by contrast, are written
positionally from the full observation vector (`_write_episode_stats`,
`evaluation_core.py:109-117`). Header list: N−1 names. Value vector: N values.

### Second defect found during planning: the `obs_entity_` prefix collision

`_write_episode_stats` counts how many obs values to write via
`evaluation_core.py:74`:

```python
obs_header_indices = [i for i, h in enumerate(stat_headers) if h.startswith("obs_")]
```

Obstacle-entity headers are named `obs_entity_{i}_r` / `obs_entity_{i}_c`
(`evaluation_core.py:262-264`) — they **also** match `startswith("obs_")`. So on any env
with obstacle entities, `num_obs_headers` overshoots the sensor-column count and the
`min(num_obs_headers, len(obs_vec))` clamp at line 110 writes the **full** observation
vector. Pre-bug (headers complete) the clamp made this benign; combined with the H8
omission it changes the corruption shape entirely.

### The two corruption modes (ground-truthed against real run CSVs, 2026-07-06)

Verified by reading headers + first data row of real eval CSVs the system produced:

| Mode | Env condition | Real file checked | Header len | Row len | Corruption |
|---|---|---|---|---|---|
| **A** | no obstacle entities | `results/JAX_RecurrentPPO/20260627-210635_rppo_nmn_film_g4_curric_longL4_s42/stats/100017/` | 83 | 83 | Pure mislabel: from the intero-noc slot onward every sensor value sits under the previous sensor's name (`obs_noc` holds intero-noc, `obs_olf_0` holds extero-noc, …); the **last element of the obs vector and the last element of the true vector are dropped** (never written). |
| **B** | obstacle entities present (**includes the default env**, up to 12 rocks) | `results/JAX_RecurrentPPO/20260703-154633_rppo_basic05v02_relentstam_n108/stats/4200006/` | 171 | **172** | The prefix collision writes the **full** obs vector (nothing dropped from obs) under a header that is one name short → every column after the obs block shifts, and the data row is one field longer than the header. `pandas.read_csv` then treats the first field as an index, misassigning **every** column name in the file. The true block is still written one element short (no `true_`-prefixed entity headers exist, so its count is exact at N−1). |

Note the diagnosis in [[06_evaluation_path]] Finding 1 stated "the CSV stays structurally
valid (column counts match)" — that is only true for mode A. A refinement pointer is
appended to that doc as part of this plan.

Videos are unaffected (`build_sensory_viz`, `sensor.py:432-441`, handles the sensor
correctly) — only the CSVs are wrong.

## Consumer analysis (required deliverable)

Who reads these CSV columns (grep over `src/`, `scripts/`, `docs/`, tests for the column
names and the `*ep_stats.csv` filename):

| Consumer | How it reads | Impact | Action in this plan |
|---|---|---|---|
| `scripts/verification/analyze_noise_diagnostics.py` (the noise-diagnostics tool named in the diagnosis; currently listed ORPHAN in [SCRIPTS_DEPENDENCY_MAP](../../../../environment/SCRIPTS_DEPENDENCY_MAP.md) but it is the canonical per-modality noise checker) | By name: `MODALITY_PAIRS` (`obs_noc`↔`true_noc`, `obs_intero_*`, `obs_loc_*`) + dynamic `obs_olf_*` / `obs_coll_*` / `obs_vis_*` / `obs_prop_*` prefixes (lines 18–70) | Mode A: internally consistent but mislabeled per-modality stats (the "Extero Nociception" row is really the interoceptive channel; olfaction σ blends nociception in). Mode B: **all** columns misassigned via the pandas implicit-index shift — every number garbage. | Update it to also decode the new `obs_intero_nociception` / `true_intero_nociception` pair (File Changes below), so the fixed column is not silently unanalyzed. |
| `src/environment/renderer.py` sensory panel / videos | Does **not** read the CSV (reads recordings + `build_sensory_viz`) | none | none |
| `scripts/eval/trajectory_story.py`, `scripts/eval/render_recordings.py` | Read `.rec.gz` recordings, not stats CSVs | none | none |
| Ad-hoc pandas analyses (the diagnosis flags "the `action_scatter` pipeline if it ever touches obs columns" — no `action_scatter` file exists in `src/`/`scripts/` today; no notebook or experiment doc was found reading `obs_*` sensor columns by name) | by name, hypothetically | future risk only | covered by the structural fix |
| Docs enumerating column layout | `docs/develop/archive/EVALUATION_RECORDING_STATS.md` (archived; not retroactively rewritten), `docs/environment/09_sensors_and_observation.md` (describes the observation layout, not the CSV) | reader confusion only | none (archive stays) |

### Verdict on EXISTING already-written CSVs: **re-derivable by column remap — no regeneration needed**, minus one (mode B) or two (mode A) unrecoverable trailing values

All written values are correct and in true observation order; only the *names* (and, mode B,
the header/row width) are wrong. The documented remap, per file:

1. **Detect the mode**: compare `len(first_data_row)` vs `len(header)` (raw `csv` read).
   Equal → mode A. Row = header+1 → mode B. (Row = header **and** the header already
   contains `obs_intero_nociception` → file was written post-fix; no remap.)
2. **Recover the run's true sensor layout** from the run's own saved `config.yaml`
   (next to its checkpoints) → `load_env_params` → `get_observation_breakdown(params)` →
   the corrected name list via the new `_sensor_stat_columns` helper (below). Do not
   re-derive from the current repo configs — use the run's saved config.
3. **Mode A remap** (pure positional rename): let `correct_obs` be the corrected obs-name
   list (length N, includes `obs_intero_nociception`). Rename the file's obs sensor columns
   positionally: the column currently named by old header index *i* is really
   `correct_obs[i]`, for the N−1 written columns. Concretely for the default layout:
   `obs_noc` → `obs_intero_nociception`, `obs_olf_0` → `obs_noc`,
   `obs_olf_k` → `obs_olf_{k-1}` (k=1..7), first collision column → `obs_olf_7`, and so on
   down the line. Same positional rename for the `true_*` block. **Unrecoverable:** the
   final element of the obs vector AND the final element of the true vector (the last
   modality in that run's breakdown order — for current runs the last Visual component;
   `obs_loc_c`/`true_loc_c` if the run had the location sensor enabled). Those two scalars
   were never written.
4. **Mode B remap** (header rewrite): build a corrected full header of length old+1 by
   (a) inserting `obs_intero_nociception` after `obs_intero_satiation` and (b) positionally
   renaming the `true_*` block as in mode A (its N−1 written columns take
   `correct_true[:N-1]`). Re-read the file with that header (e.g.
   `pd.read_csv(f, header=None, skiprows=1, names=corrected_header)`). All obs values —
   including interoceptive nociception — are then correctly named; entity and tail columns
   fall back into alignment. **Unrecoverable:** only the final element of the `true_*`
   vector.
5. **Never trust a plain `pd.read_csv` of an unremapped mode-B file** — the implicit-index
   shift silently misassigns every column, including `step`, positions, and rewards.

This remap recipe is the deliverable studies act on; a bulk-remap script is **not** in
scope (write one later only if a study actually needs old per-sensor CSV numbers — so far
the only regular consumer is the orphaned noise-diagnostics tool).

## Implementation Plan (H8)

### Design

Make header and values structurally inseparable: both must derive from the **same**
`get_observation_breakdown(params)` ordering, with unknown sensors failing loudly.

1. **One name source.** Add a module-level helper `_sensor_stat_columns(sensor_name, dim,
   params, prefix)` in `evaluation_core.py` that maps every breakdown key to its column
   names, validates `len(names) == dim`, and **raises `ValueError` on an unmapped sensor
   name**. Both the `obs_` and `true_` header blocks call it — a future sensor added to
   `get_observation_breakdown` without a mapping crashes the eval at header-build time
   instead of writing a shifted CSV (kills the silent encode/decode-drift class here for
   good).
2. **Extract the full header build** into `build_stat_headers(params, breakdown,
   record_true_obs)` (verbatim move of the fixed-column, entity, and tail parts) so the
   regression test exercises the exact production header path.
3. **Fail loud at write time.** In `_write_episode_stats`: exclude `obs_entity_` from the
   obs-header count, and replace the silent `min()` clamps with a strict equality check —
   header count must equal the vector length or `ValueError` (checked once per episode,
   before the row loop).

Implementation order for the fail-then-pass demonstration: **Stage 1** — pure refactor
(extract `build_stat_headers` + `_sensor_stat_columns` preserving the current buggy branch
list, i.e. still no Interoceptive Nociception, still `startswith("obs_")`), add the
regression test, run it → **must FAIL** (missing `obs_intero_nociception` column; row
length 1 greater than header on the default env). Record the failure output. **Stage 2** —
apply the fix (add the mapping + raise-on-unknown + predicate/equality changes), run again
→ must PASS.

### File Changes

#### 1. `src/utils/evaluation_core.py`

**(a) New module-level helper** (place above `evaluate_jax_checkpoint`, near line 143):

```python
def _sensor_stat_columns(sensor_name, dim, params, prefix):
    """Stats-CSV column names for one sensor modality.

    MUST have a branch for every key get_observation_breakdown() can emit, and
    must return exactly `dim` names. Raises on unknown sensors so a new modality
    can never silently shift the CSV again (silent encode/decode-drift class —
    see docs/develop/active/issues/diag_fable5_20260704/06_evaluation_path.md
    Finding 1).
    """
    if sensor_name == "Olfaction":
        names = [f"{prefix}olf_{i}" for i in range(dim)]
    elif sensor_name == "Extero Nociception":
        names = [f"{prefix}noc"]
    elif sensor_name == "Interoceptive Nociception":
        names = [f"{prefix}intero_nociception"]
    elif sensor_name in ("Satiation", "Nutrition", "Injury"):
        names = [f"{prefix}intero_{sensor_name.lower()}"]
    elif sensor_name == "Collision":
        coll_offsets = get_visual_offsets(params.sensor_range)
        names = [f"{prefix}coll_r{dr}c{dc}" for dr, dc in coll_offsets]
    elif sensor_name == "Location":
        names = [f"{prefix}loc_r", f"{prefix}loc_c"]
    elif sensor_name == "Visual":
        names = [f"{prefix}vis_{i}" for i in range(dim)]
    elif sensor_name == "Proprioception":
        names = [f"{prefix}prop_{i}" for i in range(dim)]
    else:
        raise ValueError(
            f"No stats-CSV column mapping for sensor {sensor_name!r} (dim={dim}). "
            f"Add a branch in _sensor_stat_columns when adding a sensor to "
            f"get_observation_breakdown()."
        )
    if len(names) != dim:
        raise ValueError(
            f"Stats-CSV column mapping for {sensor_name!r} produced {len(names)} "
            f"names for dim={dim}."
        )
    return names
```

**(b) New `build_stat_headers(params, breakdown, record_true_obs)`** — extract the entire
header construction currently inline at lines 200–267 into a module-level function; the
two sensor `elif` chains (204–222 obs, 226–243 true) are each replaced by:

```python
for sensor_name, dim in breakdown.items():
    stat_headers += _sensor_stat_columns(sensor_name, dim, params, "obs_")
# ... and, if record_true_obs:
for sensor_name, dim in breakdown.items():
    stat_headers += _sensor_stat_columns(sensor_name, dim, params, "true_")
```

The fixed columns (lines 200–203), entity columns (245–264), and tail (267) move verbatim.
Call site in `evaluate_jax_checkpoint` (the `if record_stats:` block, lines 198–267)
becomes `stat_headers = build_stat_headers(params, breakdown, record_true_obs)` after the
`os.makedirs`.

**(c) `_write_episode_stats` counting + fail-loud** (lines 74–79 and 109–117):

```python
# BEFORE (line 74):
obs_header_indices = [i for i, h in enumerate(stat_headers) if h.startswith("obs_")]
# AFTER:
obs_header_indices = [i for i, h in enumerate(stat_headers)
                      if h.startswith("obs_") and not h.startswith("obs_entity_")]
```

and, once before the per-step loop (after `batched_obs` / `batched_true_obs` are built,
~line 79):

```python
if num_obs_headers != batched_obs.shape[-1]:
    raise ValueError(
        f"Stats-CSV drift: {num_obs_headers} obs_* headers but observation dim "
        f"{batched_obs.shape[-1]} — header names and observation layout no longer "
        f"derive from the same breakdown.")
if batched_true_obs is not None and num_true_obs_headers != batched_true_obs.shape[-1]:
    raise ValueError(
        f"Stats-CSV drift: {num_true_obs_headers} true_* headers but true-obs dim "
        f"{batched_true_obs.shape[-1]}.")
```

then the write loops at 110 and 116 drop the `min()`:
`for i in range(num_obs_headers): row.append(float(obs_vec[i]))` (same for true).

Check both header-consuming paths: the parallel eval path shares the same `stat_headers`
list built in `evaluate_jax_checkpoint`, so no second construction site exists — confirm
with a grep for `stat_headers` before finishing.

#### 2. `scripts/verification/analyze_noise_diagnostics.py` (lines 18–31, 96–100)

Consumer update so the newly-correct column is decoded:

```python
MODALITY_PAIRS = [
    ("Injury",                  ["obs_intero_injury"],        ["true_intero_injury"]),
    ("Nutrition",               ["obs_intero_nutrition"],     ["true_intero_nutrition"]),
    ("Satiation",               ["obs_intero_satiation"],     ["true_intero_satiation"]),
    ("Interoceptive Nociception", ["obs_intero_nociception"], ["true_intero_nociception"]),
    ("Extero Nociception",      ["obs_noc"],                  ["true_noc"]),
    ("Location",                ["obs_loc_r", "obs_loc_c"],   ["true_loc_r", "true_loc_c"]),
]
```

Add `"Interoceptive Nociception": 0.1` to `EXPECTED_SIGMA` (the configured sigma,
`configs/environment/default.yaml:287-292`; mode is `state_dependent` like Injury) and
include it in the wide-tolerance `OK (SD)` status branch alongside `"Injury"`
(line 97: `if name in ("Injury", "Interoceptive Nociception"):`). The script's absent-column
`continue` (line 77) keeps it backward-compatible with old/remapped files.

#### 3. `tests/scripts/test_eval_stats_csv_columns.py` (NEW — regression test, must fail pre-fix)

Follow the import pattern of `tests/scripts/test_eval_rollout_stage_config.py`
(`sys.path.insert` of the repo root). Sketch:

1. Build `env_cfg = get_default_config()` (+ merge `configs/train/default.yaml`,
   `configs/evaluation/default.yaml` as the existing eval tests do) → `params =
   load_env_params(env_cfg)`. The default env has `interoceptive_nociception_enabled: true`
   **and** obstacle entities, so both the omission and the `obs_entity_` prefix collision
   are exercised.
2. `breakdown = get_observation_breakdown(params)`; assert `"Interoceptive Nociception"
   in breakdown` (guards the fixture, not the fix).
3. `stat_headers = build_stat_headers(params, breakdown, record_true_obs=True)`.
4. Plant sentinel vectors: `obs = np.arange(obs_dim, dtype=np.float32)`,
   `true = obs + 1000.0` where `obs_dim = sum(breakdown.values())`.
5. Build one synthetic step: a state dict with the keys `_write_episode_stats` reads
   (`agent_pos`, `satiation`, `nutrition`, `injury_level`, `rest_streak`, `res_pos`,
   `res_active`, `animal_pos`, `obs_pos` — take them from a real `jax_reset(key, params)`
   state), `ep_jax_infos=[{}]` (the writer uses `.get(ik, 0)`), `ep_actions=[-1]`,
   `ep_rewards=[0.0]`, `ep_obs=[obs]`, `ep_true_obs=[true]`. Call
   `_write_episode_stats(str(tmp_path), 1, ...)`.
6. Read back and assert **by column name**:
   - raw `csv` check: `len(data_row) == len(header)` (fails pre-fix, mode B: 172 vs 171);
   - compute per-modality offsets by walking `breakdown` in order; for **every** modality
     assert `df[first_column_of_modality].iloc[0] == float(offset)` and
     `df[last_column_of_modality].iloc[0] == float(offset + dim - 1)`, with the expected
     column names hardcoded for the load-bearing ones:
     `df["obs_intero_nociception"]` == the planted intero-noc slot value (fails pre-fix:
     column absent → KeyError), `df["obs_noc"]` == the planted **extero**-noc slot value
     (fails pre-fix: holds intero-noc), `df["obs_olf_0"]` == the first olfaction slot;
   - same three assertions on the `true_*` block against `+1000` sentinels;
   - the **final** obs column of the last modality is present and equals
     `float(obs_dim - 1)` (fails pre-fix, mode A behavior: last element dropped).

#### 4. `docs/environment/SCRIPTS_DEPENDENCY_MAP.md`

Per its Maintenance Contract (callers of `scripts/` files change): add the two new test
files as callers — `tests/scripts/test_eval_rollout_online_replay.py` →
`scripts/eval/eval_rollout.py` row; note under `scripts/verification/analyze_noise_diagnostics.py`
that it gained the intero-noc modality pair (it remains an orphan w.r.t. runtime callers).

## Checkpoints (H8)

- [x] Stage 1 (pure refactor + test): regression test **FAILS** on pre-fix logic — recorded
      in the Implementation Report (row width 154 vs header 153 on the test env; sentinel
      shift `obs_intero_satiation` expected 0.0 got 1.0; no raise on unknown sensor).
- [x] Stage 2 (fix): same tests **PASS** (3/3).
- [x] `_sensor_stat_columns("Bogus Sensor", 1, params, "obs_")` raises `ValueError`
      (`test_unknown_sensor_raises`).
- [x] Grep confirms no other site builds `obs_*`/`true_*` stats headers — only hits are
      the name-based *reader* `analyze_noise_diagnostics.py`; `stat_headers` is built
      solely in `evaluation_core.py` (shared by single + parallel eval paths).
- [x] Real eval smoke (`evaluate_jax_checkpoint`, default env, `record_stats=True`,
      model=None, 1 episode): header width 155 == row width 155; header contains
      `obs_intero_nociception` and `true_intero_nociception` (`tmp/20260706_1910_h8_eval_smoke.py`).

---

# Issue 2 (H9) — Offline replay: interrupted-feeding rate structurally always 0.0

## Analysis

`_compute_online_replay` (`scripts/eval/eval_rollout.py:296-394`) recomputes the M1
(interrupted-feeding), M2 (bush-dive), and M5 (eat-under-threat) measures from the saved
per-episode arrays and writes them to `online_replay.json` — the file the current eval
protocol's studies consume (round-25/26 hypervigilance designs, the discrimination-measures
grounding doc).

The M1 block (lines 341–353) resolves a pending candidate **before** updating
`steps_since_eat` for the current step:

```python
# M1 — age FIRST (age-first ordering matches train.py)   <- comment wrong in this detail
if cand_age >= 0:
    cand_age += 1
    if cand_age >= K:
        totals[f"m1_candidates_{prefix}"] += 1
        if steps_since_eat >= K:            # reads the value as of step t-1
            totals[f"m1_interrupted_{prefix}"] += 1
        cand_age = -1

steps_since_eat = 0 if ate[t] else steps_since_eat + 1   # updated AFTER resolution
```

A candidate is recorded at an eat step, where `steps_since_eat` becomes 0. At the
resolution step (age = K, i.e. K steps later) the value read is the **pre-update** one —
at most K−1 — so `steps_since_eat >= K` is unsatisfiable. `interrupted_feeding_rate_*` is
0.0 whenever any candidates exist (NaN otherwise), regardless of behavior.

The fixed online accumulator (`3e1e53e`, `src/behavior/accumulators.py`) orders the same
computation as: **(1)** update `m1_steps_since_eat` (`accumulators.py:211`), **(2)** age
and resolve pending candidates (`:215-232`), **(3)** record new candidates (`:234-257`).
With that ordering, "interrupted" means: the agent did not eat at any of the K steps
following the candidate eat (an eat at the resolution step itself resets the counter to 0
*before* the check, so it counts as not-interrupted). The offline replay must carry the
identical K-window meaning.

**What can NOT change here** (fenced): the offline denominator
(`m1_candidates_{prefix} += 1` inside the resolve branch, line 345) counts at resolution
time; online counts at record time (`accumulators.py:242`). That divergence is
[[07_behavior_measures]] Finding 4 (Med) and **stays open** — it interacts with the
undecided episode-end rule for in-progress events and is explicitly out of scope.

## Implementation Plan (H9)

### Design

Reorder the loop body to the online accumulator's ordering: update `steps_since_eat`
first, then age/resolve, then record. Fix the misleading comment. Nothing else in the
function changes (M5 block stays first; M2 block untouched; rate derivation untouched).

### File Changes

#### 5. `scripts/eval/eval_rollout.py` (`_compute_online_replay`, lines 341–353)

```python
# BEFORE:
                # M1 — age FIRST (age-first ordering matches train.py)
                if cand_age >= 0:
                    cand_age += 1
                    if cand_age >= K:
                        totals[f"m1_candidates_{prefix}"] += 1
                        if steps_since_eat >= K:
                            totals[f"m1_interrupted_{prefix}"] += 1
                        cand_age = -1

                steps_since_eat = 0 if ate[t] else steps_since_eat + 1

                if ate[t] and in_R:
                    cand_age = 0  # record new candidate

# AFTER:
                # M1 — update steps_since_eat FIRST, then age/resolve, then record:
                # same ordering as the online accumulator (accumulators.py:211 update,
                # :215-232 resolve, :234-257 record; fix 3e1e53e), so "interrupted"
                # means: no eat at any of the K steps after the candidate eat.
                # NOTE: the denominator below still counts at RESOLUTION time, unlike
                # the online record-time counting — known open divergence, see
                # diag_fable5_20260704/07_behavior_measures.md Finding 4. Do not
                # change it here.
                steps_since_eat = 0 if ate[t] else steps_since_eat + 1

                if cand_age >= 0:
                    cand_age += 1
                    if cand_age >= K:
                        totals[f"m1_candidates_{prefix}"] += 1
                        if steps_since_eat >= K:
                            totals[f"m1_interrupted_{prefix}"] += 1
                        cand_age = -1

                if ate[t] and in_R:
                    cand_age = 0  # record new candidate
```

Semantics check the developer should replay mentally (also encoded in the test): candidate
recorded at eat step t₀ (`steps_since_eat` = 0 that step); resolution at t₀+K, where
`steps_since_eat` has just been updated — K if no eats in (t₀, t₀+K], < K (or 0) otherwise.
Identical to online.

#### 6. `tests/scripts/test_eval_rollout_online_replay.py` (NEW — regression test, must fail pre-fix)

Import pattern from `tests/scripts/test_eval_rollout_stage_config.py`
(`sys.path.insert(0, REPO)` + `sys.path.insert(0, REPO/scripts/eval)`;
`import eval_rollout as er`). No env, no JAX model — `_compute_online_replay` is a pure
function of episode dicts + a bm-config namespace:

```python
from types import SimpleNamespace
import numpy as np

def _episode(T, eat_steps, dist=1.0):
    return {
        "length": T,
        "dist_per_predator": np.full((T, 1), dist, dtype=np.float32),  # always in radius
        "dist_per_neutral":  np.zeros((T, 0), dtype=np.float32),       # no neutrals
        "ate_food":     np.isin(np.arange(T), eat_steps),
        "agent_in_bush": np.zeros(T, dtype=bool),
    }

def test_interrupted_feeding_fires():
    bm_cfg = SimpleNamespace(cue_radius=3.0, obs_window=5)
    # Agent eats once under threat at t=2, threat stays within radius, never eats again
    # -> the K=5 window after the eat contains no eat -> interrupted.
    res = er._compute_online_replay([_episode(T=20, eat_steps=[2])], bm_cfg)
    assert res["m1_candidates_predator"] == 1
    assert res["m1_interrupted_predator"] == 1          # pre-fix: 0 -> FAILS
    assert res["interrupted_feeding_rate_predator"] == 1.0  # pre-fix: 0.0 -> FAILS

def test_not_interrupted_when_agent_eats_within_window():
    bm_cfg = SimpleNamespace(cue_radius=3.0, obs_window=5)
    # Eats at t=2 and again at t=5 (inside the 5-step window) -> not interrupted.
    res = er._compute_online_replay([_episode(T=20, eat_steps=[2, 5])], bm_cfg)
    assert res["interrupted_feeding_rate_predator"] == 0.0  # guards against over-fixing
```

(The second eat at t=5 is itself under threat, so it records a fresh candidate — assert
`m1_candidates_predator == 2` there if the developer wants the extra precision; both its
window outcomes are not-interrupted... no: the second candidate's window (t=6..10) has no
eat → it resolves interrupted. Developer: either move the second eat out of radius is not
possible here (dist constant), so instead assert `m1_interrupted_predator == 1` and
`rate == 0.5` for the two-eat episode — the first candidate not-interrupted, the second
interrupted. Pre-fix this also reads 0.0, so it still fails pre-fix and doubles as the
overwrite-free multi-candidate check. Pick exact expected numbers by hand-tracing before
writing the assertions; the plan's headline test is `test_interrupted_feeding_fires`.)

## Checkpoints (H9)

- [x] Positive tests FAIL pre-fix (`m1_interrupted_predator` 0 instead of 1 → rate 0.0)
      — recorded in the Implementation Report. (Negative control passes pre- and
      post-fix by design — it guards against over-fixing, not the bug itself.)
- [x] All 3 PASS post-fix.
- [x] Hand-trace parity: `eat_steps=[2]` → candidates=1, interrupted=1, rate=1.0,
      matching the online accumulator's documented behavior. (Note: the plan
      parenthetical's two-eat guess of candidates=2/rate=0.5 does not hold for the
      fixed offline code — the single-slot tracker overwrites the unresolved first
      candidate and the offline resolution-time denominator never counts it; the
      hand-traced expectation candidates=1/interrupted=1/rate=1.0 is asserted in
      `test_second_eat_under_threat_overwrites_candidate`, still failing pre-fix.)
- [x] Diff touches ONLY the M1 ordering + comments — M2/M5/rate-derivation lines
      byte-identical (`git diff scripts/eval/eval_rollout.py` inspected; the other
      hunks in that file's working diff belong to the parallel batched-rollout WP).
- [x] Denominator counting still at resolution time (fenced divergence intact —
      fence-note comment added at the site).

### What studies do with old `online_replay.json` files (remediation note)

Old files' `interrupted_feeding_rate_*` values are invalid (structurally 0.0/NaN); their
M2/M5 entries are unaffected by H9. **Regeneration is cheap and does not require re-running
the policy**: `eval_rollout` saves the per-episode step arrays as `episodes/<idx>.npz` in
the same output directory, and `_compute_online_replay` is a pure function of those arrays
— a 10-line ad-hoc script (load npz's, call the fixed function, rewrite
`online_replay.json`) recovers correct rates for any past eval. Writing that script is not
in this plan's scope; the analyzer can do it per-study when needed.

---

## Test plan (whole work package)

Interpreter: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`.

1. New tests, pre-fix (H8 stage 1 / H9 before reorder): both new test files FAIL for the
   asserted reasons. Record outputs.
2. New tests, post-fix: PASS.
   ```
   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest \
     tests/scripts/test_eval_stats_csv_columns.py \
     tests/scripts/test_eval_rollout_online_replay.py -v
   ```
3. Full suite: `... -m pytest tests/ -x -q` — **known-red baseline (pre-existing, not
   blockers, must not grow)**: 4× A1 parity + 3× stale-config (`b093023`) + 1× dreamer_srl
   offline-WM smoke. Any NEW failure beyond these eight is a blocker.
4. Eval smoke (H8 checkpoint 5) — one tiny `evaluate_jax_checkpoint`-path run with
   `record_stats=True` on a default-env config; inspect the written CSV header/row widths.

**Speed check waiver:** neither change touches a training path (eval-only CSV header build
+ a pure post-hoc replay reorder); per-step eval cost is unchanged (same number of column
writes). No training speed benchmark required; the developer should state this waiver in
the Implementation Report rather than silently skip.

## File Changes summary

| # | File | Change |
|---|---|---|
| 1 | `src/utils/evaluation_core.py` | H8: `_sensor_stat_columns` helper (raise on unknown) + `build_stat_headers` extraction + `obs_entity_` predicate fix + strict header/vector equality (fail loud) |
| 2 | `scripts/verification/analyze_noise_diagnostics.py` | H8 consumer: decode `obs_intero_nociception`/`true_intero_nociception`; `EXPECTED_SIGMA` 0.1; SD-status branch |
| 3 | `tests/scripts/test_eval_stats_csv_columns.py` | NEW H8 regression test (fails pre-fix) |
| 4 | `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | Maintenance Contract: new test callers of `scripts/eval/eval_rollout.py`; note on analyze_noise_diagnostics update |
| 5 | `scripts/eval/eval_rollout.py` | H9: reorder `steps_since_eat` update before M1 resolve in `_compute_online_replay`; fix comment; fence-note the denominator divergence |
| 6 | `tests/scripts/test_eval_rollout_online_replay.py` | NEW H9 regression test (fails pre-fix) |

New config keys: none.

## Handoffs after implementation

- `senior-developer`: Verification Protocol over this plan.
- `bug-curator`: update the Known Bugs registry rows for H8 and H9 (recorded as open rows
  in the 2026-07-04 registry sweep) to fixed, linking the fix commit; the offline/online
  denominator-timing Med row **stays open**.
- Studies/`experiment-analyzer`: apply the old-file guidance above — CSV column remap
  (H8, both modes documented) and `online_replay.json` regeneration from `episodes/*.npz`
  (H9) — before reusing any pre-fix eval outputs.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-06

### Summary (file-by-file)

| File | What was done |
|---|---|
| `src/utils/evaluation_core.py` | Stage 1: extracted `build_stat_headers(params, breakdown, record_true_obs)` (fixed columns + sensor blocks + entity columns + tail, verbatim) and `_sensor_stat_columns(sensor_name, dim, params, prefix)` from the inline `elif` chains; call site in `evaluate_jax_checkpoint` reduced to one line. Stage 2: added the `"Interoceptive Nociception"` → `{prefix}intero_nociception` branch, raise-`ValueError`-on-unmapped-sensor + `len(names) == dim` validation; in `_write_episode_stats` the obs count predicate now excludes `obs_entity_` headers, the silent `min()` clamps are removed, and two strict header-count == vector-dim `ValueError` checks run once per episode before the row loop. |
| `scripts/verification/analyze_noise_diagnostics.py` | Consumer update: `MODALITY_PAIRS` gained `("Interoceptive Nociception", ["obs_intero_nociception"], ["true_intero_nociception"])`; `EXPECTED_SIGMA["Interoceptive Nociception"] = 0.1` (verified against `configs/environment/default.yaml` — sigma 0.1, mode `state_dependent`); the wide-tolerance `OK (SD)` status branch now covers both `Injury` and `Interoceptive Nociception`. Absent-column `continue` keeps old files readable. |
| `tests/scripts/test_eval_stats_csv_columns.py` | NEW H8 regression test (3 tests): default env (intero-noc on + obstacle entities → both defects exercised), sentinel obs/true vectors (`arange` / `arange+1000`) written through the real `build_stat_headers` + `_write_episode_stats`, read back by column name for every modality (first + last column per modality, hardcoded checks on `obs_intero_nociception` / `obs_noc` / `obs_olf_0` and `true_*` twins, final-obs-element presence); raw-csv row-width == header-width check; `ValueError` on unmapped sensor. |
| `scripts/eval/eval_rollout.py` | H9: `_compute_online_replay` M1 block reordered to update `steps_since_eat` BEFORE age/resolve, then record — the online accumulator's ordering (`accumulators.py:211/:215-232/:234-257`, fix `3e1e53e`). Misleading "age FIRST … matches train.py" comment replaced; fence-note added stating the resolution-time denominator divergence ([[07_behavior_measures]] Finding 4) is deliberately NOT changed. M2/M5/rate-derivation untouched (diff-verified). |
| `tests/scripts/test_eval_rollout_online_replay.py` | NEW H9 regression test (3 tests): eat-under-threat then no eat for K → candidates=1/interrupted=1/rate=1.0; negative control (threat leaves radius, second eat inside window resets counter without recording → rate 0.0); two eats under sustained threat → overwrite semantics (candidates=1/interrupted=1/rate=1.0, see deviation note below). |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | Maintenance Contract: §1c new row for the bare-import test callers of `scripts/eval/eval_rollout.py` (`test_eval_rollout_online_replay.py` + the pre-existing, previously unlisted `test_eval_rollout_stage_config.py`); §3 roll-up row for `eval_rollout.py` → `TOOL+TEST`; §3 note that `analyze_noise_diagnostics.py` gained the intero-noc pair (still ORPHAN w.r.t. runtime callers). |

### Red → green evidence

**H8 stage 1 (refactor preserving buggy logic; recorded pre-fix run):**

```
FAILED test_eval_stats_csv_columns.py::test_row_width_matches_header
       - AssertionError: stats-CSV row width 154 != header width 153
FAILED test_eval_stats_csv_columns.py::test_sensor_columns_read_back_by_name
       - AssertionError: obs_intero_satiation: expected sentinel 0.0, got 1.0
         (pandas implicit-index shift, mode B — every column misassigned)
FAILED test_eval_stats_csv_columns.py::test_unknown_sensor_raises
       - Failed: DID NOT RAISE <class 'ValueError'>
```

(154 vs 153 rather than the plan's illustrative 172 vs 171 because the test env is the
bare default env, not the ladder run's env — same mode-B off-by-one shape.)

**H9 pre-fix (reorder not yet applied):**

```
FAILED test_eval_rollout_online_replay.py::test_interrupted_feeding_fires
       - assert 0 == 1   (m1_interrupted_predator; rate 0.0)
FAILED test_eval_rollout_online_replay.py::test_second_eat_under_threat_overwrites_candidate
       - assert 0 == 1   (m1_interrupted_predator; rate 0.0)
PASSED test_eval_rollout_online_replay.py::test_not_interrupted_when_agent_eats_within_window
       (negative control — passes pre- AND post-fix by design; guards over-fixing)
```

**Post-fix:** all 6 new tests pass (`tests/scripts/test_eval_stats_csv_columns.py` 3/3,
`tests/scripts/test_eval_rollout_online_replay.py` 3/3). Eval smoke (checkpoint 5):
real `evaluate_jax_checkpoint` run, default env, `record_stats=True` → header width 155
== row width 155, `obs_intero_nociception` + `true_intero_nociception` present
(script kept at `tmp/20260706_1910_h8_eval_smoke.py`).

### Full suite

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/ -q \
  --ignore=tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py
8 failed, 398 passed, 501 skipped in 21m16s
```

The 8 failures are exactly the known-red baseline: 4× A1 parity gates
(`test_unified_parity` observability_gates S1–S4), 3× stale-config `b093023`
(`test_inactive_animal_offgrid` ×1, `test_truncation_not_death` ×2 — FileNotFoundError),
1× dreamer_srl offline-WM smoke (`test_dreamer_srl_offline_wm_test`). **Zero new failures.**

`--ignore` note: `tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py` is a
parallel work package's in-flight test currently sitting in the tree; it fails at
COLLECTION (`ImportError: cannot import name '_reset_terminal_step_data' from
src.algorithms.dreamer_srl.dreamer_srl_main`), which aborts the entire pytest run.
Not part of the known-red baseline and not caused by this WP — its own developer's
uncommitted changes are mid-flight. Flagged for `senior-developer` awareness during
verification.

### Speed check

**Waived per the plan's explicit waiver**: neither change touches a training path —
H8 is an eval-only CSV header build (same number of per-step column writes; the new
equality checks run once per episode, not per step), H9 is a pure post-hoc replay
reorder over saved arrays. No training speed benchmark run.

### Deviations

1. **H9 two-eat test expectations** (plan-anticipated, not a design change): the plan's
   parenthetical suggested `candidates=2, interrupted=1, rate=0.5` for the
   `eat_steps=[2, 5]` always-in-radius episode, but instructed "pick exact expected
   numbers by hand-tracing before writing the assertions". The hand-trace against the
   fixed code shows the single-slot candidate tracker overwrites the unresolved t=2
   candidate at t=5, and — because the offline denominator counts at RESOLUTION time
   (the fenced Finding-4 divergence, untouched) — the overwritten candidate is never
   counted: actual fixed-code result is `candidates=1, interrupted=1, rate=1.0`
   (asserted in `test_second_eat_under_threat_overwrites_candidate`; still fails
   pre-fix, so it keeps its regression value). The plan's suggested negative control
   (second eat inside the window ⇒ rate 0.0) is impossible with a constant in-radius
   distance — any in-radius eat records a fresh candidate — so the negative control
   instead moves the threat OUT of radius after the candidate eat; the second eat then
   resets the counter without recording, and the candidate resolves not-interrupted
   (rate 0.0). Headline test `test_interrupted_feeding_fires` is exactly as planned.
2. **Full-suite `-x` dropped**: the plan's command used `-x`, which would stop at the
   first known-red failure; run without `-x` (plus the `--ignore` above) so the failure
   set could be compared against the 8-row baseline.

Nothing else deviates: no new config keys, scope fence respected (offline denominator
timing untouched; no other eval findings touched; entity column names unchanged;
parallel packages' uncommitted changes left alone).

### Blockers / follow-ups

- None blocking. Handoffs as planned: `senior-developer` verification;
  `bug-curator` registry rows H8/H9 → fixed after commit; studies apply the CSV
  remap (mode A/B) + `online_replay.json` regeneration guidance before reusing
  pre-fix eval outputs.

> Implemented by: developer

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-06

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/utils/evaluation_core.py` | `_sensor_stat_columns` + `build_stat_headers` + `obs_entity_` exclusion + strict equality checks | ✅ | Diff matches the plan's snippets essentially verbatim. Helper raises on unmapped sensor and on `len(names) != dim`; intero-noc branch present; `min()` clamps removed; the two `ValueError` drift checks run once per episode before the row loop. Grep confirms `stat_headers` is built at exactly one site (line 298) and threaded through both the single and parallel eval paths — no second construction site. |
| `scripts/eval/eval_rollout.py` (M1 hunk only) | H9 reorder: `steps_since_eat` update before age/resolve, then record; comment fix + fence note | ✅ | Independently checked against the committed online accumulator (`git show HEAD:src/behavior/accumulators.py` — update at :211, resolve `steps_since_eat >= bm_K` at :215-232, record at :234-257): ordering and K-window meaning now identical. Denominator increment stays inside the resolve branch (resolution time) — the fenced Finding-4 divergence is intact and the fence-note comment marks it. M2/M5/rate derivation untouched. **Note**: this file's working diff is shared with the parallel batched-rollout WP (`_run_episodes_batched`, `--batched`, `rollout_mode` manifest key) — those hunks belong to [[EVAL_ROLLOUT_BATCHING_PERF]] (plan doc present in tree, own parity script), not to WP-E. Commit hygiene: WP-E's M1 hunk cannot be committed independently of that WP with plain `git add` on this file — use `git add -p` or coordinate the commits. |
| `scripts/verification/analyze_noise_diagnostics.py` | Intero-noc modality pair + `EXPECTED_SIGMA` 0.1 + `OK (SD)` branch | ✅ | Matches plan. Sigma 0.1 / `state_dependent` consistent with `configs/environment/default.yaml`; absent-column `continue` preserved so old/remapped files stay readable. |
| `tests/scripts/test_eval_stats_csv_columns.py` (NEW) | H8 regression test, 3 tests | ✅ | Exercises the real production path (`build_stat_headers` + `_write_episode_stats`) on the default env (intero-noc on, `obs_blocking.shape[0] > 0` fixture-guarded → both defects exercised), sentinel read-back by column name for **every** modality (first+last per modality) plus the hardcoded load-bearing names, raw-csv width check, unknown-sensor raise. Pre-fix red evidence in the Implementation Report is behavioral (154 vs 153 row width; misassigned sentinel; DID-NOT-RAISE). |
| `tests/scripts/test_eval_rollout_online_replay.py` (NEW) | H9 regression test, 3 tests | ✅ | Headline test exactly as planned. **Pre-fix red independently reproduced by the verifier**: extracted `HEAD:scripts/eval/eval_rollout.py` and ran `_compute_online_replay` on the headline episode → candidates=1, interrupted=0, rate=0.0 (fixed code: 1/1/1.0). The developer's documented deviation (two-eat expectations candidates=1/interrupted=1/rate=1.0; negative control moved out-of-radius) was hand-traced independently and is **correct**: the plan's parenthetical guess of candidates=2/rate=0.5 is impossible under the single-slot tracker + resolution-time denominator, and the adjustment *documents* the fenced denominator behavior rather than changing it — no fenced-out fix smuggled in. Negative-control trace (threat leaves radius; eat at t=5 resets counter without recording; resolves not-interrupted) also verified by hand. |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | Maintenance Contract update | ✅ | §1c bare-import row (both new + pre-existing eval_rollout test callers), §3 `TOOL+TEST` roll-up, §3 ORPHAN note for `analyze_noise_diagnostics.py`. |
| `06_evaluation_path.md` / `07_behavior_measures.md` | Fix-plan pointers + Finding-1 mode-B refinement | ✅ | Appended sections (senior-developer, planning phase) cross-link both diagnosis findings to this plan; Finding 4 explicitly marked still-open. |

**Independent re-runs (verifier, 2026-07-06):**
- `pytest tests/scripts/test_eval_stats_csv_columns.py tests/scripts/test_eval_rollout_online_replay.py -v` → **6/6 PASS** (11.3 s).
- H8 eval smoke (`tmp/20260706_1910_h8_eval_smoke.py`, real `evaluate_jax_checkpoint`, default env, `record_stats=True`) → header width 155 == row width 155, `obs_intero_nociception` + `true_intero_nociception` present. SMOKE OK.
- Targeted blast radius `pytest tests/scripts/ -q` → 18 passed, 1 failed = the known-red dreamer_srl offline-WM smoke (baseline row, pre-existing). Zero new failures in this WP's blast radius; the developer's full-suite run (8 failed = exact 8-row baseline, `--ignore` for the parallel WP's mid-flight collection error) accepted as recorded.

**Scope fence verdict**: held. Offline M1 denominator still counts at resolution time (Finding 4 open, fence-note in code); Findings 1/2/5 of 07 and Findings 2–7 of 06 untouched; entity column names unchanged; no new config keys; no schema changes.

**Speed check verdict**: ✅ no regression — plan's explicit waiver valid (eval-only header build, same per-step column-write count; new drift checks are once-per-episode; H9 is a pure post-hoc reorder over saved arrays). Developer stated the waiver in the Implementation Report as required.

**Out-of-scope files in the working tree** (`train.py`, `src/behavior/accumulators.py`, `src/models/dreamer_v3_trainer.py`, dreamer_srl tests, batched-rollout hunks in `eval_rollout.py`, `train_command-agent.sh`): all attributable to the three named parallel in-flight packages + the batched-rollout WP — not WP-E scope creep.

**Conclusion**: **PASS** — all six planned file changes implemented as specified, red→green evidence behavioral and independently reproduced, scope fence intact, tests green on re-run, consumer analysis + old-CSV remap recipe delivered in this doc and linked from the diagnosis docs. Only handling note: coordinate the shared-file commit of `eval_rollout.py` with the batched-rollout WP (`git add -p`).

Verified by: senior-developer
