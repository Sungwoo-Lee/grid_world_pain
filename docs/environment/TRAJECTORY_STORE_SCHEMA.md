# Trajectory Store — permanent schema contract

## What this store is, and what one row means

When we finish training an agent, we can put it **back into the exact world it was
trained in** and let it live there for a very large number of episodes — a million per
training run is the target. This document describes the file format we write those
episodes into, and it is the contract every future analysis reads against.

The thing that makes the store worth building is that our training worlds are
**re-randomised at the start of every episode**. Each episode secretly re-rolls how many
predators exist, how far each one can see, how fast it moves, how long it waits before
attacking, how many bushes there are, and more. Those secret rolls have never been
written down anywhere. This store records them next to the behaviour, so a question like
*"how much of the change in bush-hiding time is explained by predator sight range versus
by the healing rate?"* becomes a table lookup instead of a guess.

Two files per shard, two kinds of row:

- a **step row** — one per (episode, time index `t`): where the agent was, what it saw,
  what it did, what happened to it;
- an **episode row** — one per episode: how long it survived, how it ended, and the
  complete set of environment dice-rolls for that episode.

They join on `episode_seed`.

**The one hard requirement.** One reader must work for every training run, forever.
Every environment writes exactly the same column names in exactly the same order with
exactly the same Arrow types. Only the *widths* of the per-entity array columns change
from world to world, and those widths are recorded in the store's manifest. No sensor
toggle, no evaluation flag and no config option may ever add or remove a column.

Plan and rationale: [[TRAJECTORY_COLLECTION_PIPELINE]]
(`docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md`).
Code: `src/utils/trajectory_store.py` (schema, writer, reader),
`scripts/eval/traj_collect/` (collector, worker, driver).

<!-- BEGIN GENERATED: schema_version — do not edit by hand; run scripts/eval/traj_collect/gen_schema_doc.py -->

`SCHEMA_VERSION = 1` — 36 step columns, 23 episode columns.

<!-- END GENERATED: schema_version -->

---

## 1. The row convention — ARRIVAL

There is exactly one sentence to remember:

> Row `t` holds **(a)** the environment state **at time `t`**, and **(b)** the action,
> reward, and transition outputs of the step that **arrived at** time `t`.

Row `t = 0` is the **reset state**, with `action = -1`, `reward = 0.0`, and every
transition-output field at its zero value. An episode of length `T` therefore has
**`T + 1` rows**.

The observation columns in row `t` are the observation of **state `t`** — that is, the
observation the policy consumed when choosing the action recorded in row `t + 1`. (This
is structural, not an argument: the collector carries the observation forward through
the scan, so the array stored in row `t` is literally the array the policy read.)

### Worked three-step example

An episode that lasts `T = 3` steps and ends by injury:

| `t` | `action` | `reward` | `agent_row`,`agent_col` | `damage` | `terminated` | `termination_reason` | meaning |
|---:|---:|---:|---|---:|:---:|---:|---|
| 0 | −1 | 0.0 | reset position | 0.0 | false | 0 | the world as it was created |
| 1 | 2 | −0.01 | after step 1 | 0.0 | false | 0 | action 2 was taken from row 0's state and landed here |
| 2 | 2 | −0.02 | after step 2 | 4.3 | false | 0 | the agent was hit on the way into this row |
| 3 | 0 | −0.90 | after step 3 | 6.1 | **true** | **4** | the step that arrived here ended the episode |

Read a column by asking which half of the sentence it belongs to: `agent_row` is *state
at t*; `action` and `damage` are *the step that arrived at t*.

---

## 2. Path scheme and the anti-overwrite guarantee

```
<store_root>/<run_tag>/<ckpt_step>/<env_fp>/
    _manifest.json
    episodes_00000.parquet      # one row per episode, block 0
    steps_00000.parquet         # T+1 rows per episode, block 0
    episodes_00001.parquet
    steps_00001.parquet
    ...
```

- `run_tag` — the training run directory name, e.g. `20260816-152742_rppo_restpremNH_a10_n112`.
- `ckpt_step` — the checkpoint step, as an integer string.
- `env_fp` — the first 10 hex characters of the SHA-256 of the canonicalised resolved
  environment config (`yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)`).

Three independent layers stop a store being silently overwritten — a real incident on
2026-07-04, where re-running one checkpoint under a *different* environment config and
the same output root overwrote episode-for-episode with no warning:

1. **Structural.** A different environment config produces a different `env_fp` and
   therefore a different directory. Two configs can no longer collide on one path.
2. **Manifest guard.** On resume the collector recompares `schema_version`, `env_fp`,
   `seed_base`, `n_episodes`, `shard_episodes`, `obs_precision`, `dims`, `max_steps`,
   `checkpoint_path`, `device` and `restore_check`. **Any mismatch is a hard `ValueError`
   and nothing is written.**
   (A git-SHA difference is a loud warning, not a failure — code legitimately moves
   between collection sessions.)
3. **Atomic shards.** Each shard is written to `<name>.parquet.tmp`, fsynced, then
   `os.replace`d, and the directory is fsynced. A shard on disk is always complete.

**Resume** is stateless: a block is complete iff both of its shards exist. Block `b`
covers episodes `[shard_episodes·b, shard_episodes·(b+1))` and episode `i` uses seed
`seed_base + i`, so a redone block is bit-identical to what a dead process would have
produced and a partial loss is recoverable by re-running only the missing shards.

### `_manifest.json`

Written **first**, before any shard, and never rewritten — it carries the resolved
config, `seed_base` and `obs_precision`, so it is the one file whose loss is not
recoverable by re-running shards. Fields:

| Field | Meaning |
|---|---|
| `schema_version` | readers hard-fail on an unknown value |
| `env_fp`, `resolved_env_config` | the fingerprint and the full resolved training config |
| `run_path`, `run_dir_name`, `train_config_mtime` | provenance of the training run |
| `checkpoint_path`, `ckpt_step` | which checkpoint was rolled out |
| `dims` | `{A, R, B, V, VV, D}` — the widths of every array column |
| `max_steps`, `action_dim`, `policy_mode` | rollout parameters (`policy_mode` is always `deterministic_argmax`) |
| `seed_base`, `n_episodes`, `shard_episodes`, `batch_size`, `device` | the episode population and how it was produced; **every resolved value is recorded, defaulted or not** |
| `obs_precision` | `float16` or `float32` — see §5 |
| `scene_format`, `scene_ambiguous` | which scene block the loader used, and whether the ambiguity guard was overridden (§6) |
| `restore_check` | `"strict"` (the checkpoint's own structure was compared against the rebuilt model) or `"weak_allowed"` (the operator passed `--allow-weak-restore-check`, so that comparison may have been skipped — treat the agent's architecture as unverified) |
| `collection_git_sha`, `collected_at` | provenance of the *collection* |
| `training_git_sha`, `training_git_dirty`, `training_started_utc` | provenance of the *training run*, copied from `<run_dir>/models/provenance.json` (written by `train.py` at startup since 2026-08-20). **`null` and `"unknown"` mean different things**: `null` = the run predates the stamp, so no file existed — this is the whole pre-2026-08-20 `results/` corpus and is not an error; `"unknown"` = the run WAS stamped but `git` could not be read on that node at that moment. `training_git_dirty: true` means uncommitted edits to tracked files were present at training start, so `training_git_sha` does **not** fully describe the code that ran — treat it as an upper bound, not an identity |
| `animal_tags`, `animal_classes`, `animal_behaviours`, `resource_names`, `obstacle_names` | human labels, so array index `i` means something |
| `observation_breakdown` | `{sensor name: dimension}` for the `D`-wide observation columns |
| `animal_damage`, `animal_is_damaging`, `obs_hides_agent`, `obs_blocking`, `res_type`, `res_damage`, `obs_damage` | static per-entity parameters that are join partners for the draws |
| `animal_detect_low/high`, `animal_move_int_low/high`, `animal_attack_delay_low/high`, `animal_attack_range_low/high`, the four float-trait bounds, `*_count_low/high` | the per-episode **sampling bounds** — the join partner that turns a realised draw into a position within its own range |
| `animal_property_std`, `res_property_std`, `obs_property_std` and their visual siblings | per-element jitter, which governs whether a property column should vary at all |

---

## 3. The fixed key list

Widths below are dimension expressions evaluated against the manifest's `dims`:
`A` = animal slots, `R` = resource slots, `B` = obstacle slots, `V` = olfactory property
width, `VV` = visual property width, `D` = observation dimension.

### 3.1 Per-step record — `steps_NNNNN.parquet`

One row per `(episode, t)`, `t ∈ [0, T]`, sorted by `(episode_seed, t)`.

<!-- BEGIN GENERATED: step_columns — do not edit by hand; run scripts/eval/traj_collect/gen_schema_doc.py -->

| # | Column | Arrow type | Row-convention timing | Source |
|---:|---|---|---|---|
| 1 | `episode_seed` | `int64` | episode key | the seed whose PRNGKey produced the episode |
| 2 | `t` | `int16` | index, 0…T | — |
| 3 | `action` | `int8` | arriving (-1 at t=0) | argmax(logits) |
| 4 | `reward` | `float32` | arriving (0.0 at t=0) | jax_step return |
| 5 | `agent_row` | `int16` | state at t | state.agent_pos[0] |
| 6 | `agent_col` | `int16` | state at t | state.agent_pos[1] |
| 7 | `satiation` | `float32` | state at t | state.satiation |
| 8 | `nutrition` | `float32` | state at t | state.nutrition |
| 9 | `injury_level` | `float32` | state at t | state.injury_level |
| 10 | `rest_streak` | `int16` | state at t | state.rest_streak |
| 11 | `last_collision_noc` | `float32` | state at t | state.last_collision_noc |
| 12 | `terminated` | `bool` | state at t | state.terminated |
| 13 | `damage` | `float32` | arriving (0.0 at t=0) | info['damage'] |
| 14 | `ate_food` | `bool` | arriving (False at t=0) | info['ate_food'] |
| 15 | `rested` | `bool` | arriving (False at t=0) | info['rested'] |
| 16 | `hit_predator` | `bool` | arriving (False at t=0) | info['hit_predator'] |
| 17 | `hit_neutral` | `bool` | arriving (False at t=0) | info['hit_neutral'] |
| 18 | `hit_hiding_predator` | `bool` | arriving (False at t=0) | info['hit_hiding_predator'] |
| 19 | `event_collided` | `bool` | arriving (False at t=0) | info['event_collided'] |
| 20 | `agent_in_bush` | `bool` | state at t | info['agent_in_bush'] for t>=1; recomputed at reset for t=0 (plan §D8) |
| 21 | `termination_reason` | `int8` | arriving (0 except final row) | info['termination_reason']; 0=active, 1=max_steps, 2=starvation, 3=overeating, 4=injury |
| 22 | `animal_row` | `list<int16>[A]` | state at t | state.animal_pos[:,0] |
| 23 | `animal_col` | `list<int16>[A]` | state at t | state.animal_pos[:,1] |
| 24 | `animal_state` | `list<int8>[A]` | state at t | state.animal_state — 0=PATROL, 1=HUNT, 2=RETURN |
| 25 | `animal_stamina` | `list<float32>[A]` | state at t | state.animal_stamina |
| 26 | `animal_move_timer` | `list<int16>[A]` | state at t | state.animal_move_timer |
| 27 | `animal_attack_timer` | `list<int16>[A]` | state at t | state.animal_attack_timer |
| 28 | `res_row` | `list<int16>[R]` | state at t | state.res_pos[:,0] |
| 29 | `res_col` | `list<int16>[R]` | state at t | state.res_pos[:,1] |
| 30 | `res_active` | `list<bool>[R]` | state at t | state.res_active |
| 31 | `res_cons_count` | `list<int16>[R]` | state at t | state.res_cons_count |
| 32 | `res_reg_timer` | `list<int16>[R]` | state at t | state.res_reg_timer |
| 33 | `obs_row` | `list<int16>[B]` | state at t | state.obs_pos[:,0] — constant within an episode; deliberately kept per-step (plan §D4.1) |
| 34 | `obs_col` | `list<int16>[B]` | state at t | state.obs_pos[:,1] |
| 35 | `obs_noised` | `list<float16 \| float32>[D]` | state at t | get_observation(state, params) — what the policy received |
| 36 | `obs_true` | `list<float16 \| float32>[D]` | state at t | get_observation(state, params, apply_noise=False) — noise-free ground truth |

<!-- END GENERATED: step_columns -->

**Deliberately NOT recorded** — with reasons, so nobody re-adds them by accident:

| Field | Why not |
|---|---|
| `nociception` | always exactly `0.0`; the key does not exist in the env `info` dict |
| `dist_per_predator` / `dist_per_neutral` / `dist_to_*` | recomputable exactly from the recorded positions; the env's own versions also carry a one-step staleness bug (computed from pre-move animal positions against the post-move agent position) |
| `reward_homeostatic` / `reward_extrinsic` / `drive_hunger` / `drive_injury` / `metabolic_drain` | reconstructable from the recorded body-state columns plus the manifest's config |
| `injury_buffer`, `nociception_history_buffer` | pure functions of the recorded `injury_level` history and the config's kernel parameters |
| `state.key` | large, and the episode is fully determined by `episode_seed` |

### 3.2 Per-episode record — `episodes_NNNNN.parquet`

One row per episode. Columns 7–23 are the complete independent-variable side of the
analysis: everything the environment secretly re-rolled at the start of this episode.
Joined against the manifest's sampling bounds, each realised draw can be expressed as a
position within its own range.

<!-- BEGIN GENERATED: episode_columns — do not edit by hand; run scripts/eval/traj_collect/gen_schema_doc.py -->

| # | Column | Arrow type | Meaning |
|---:|---|---|---|
| 1 | `episode_seed` | `int64` | join key to `steps`; seed = seed_base + episode_index |
| 2 | `episode_index` | `int64` | 0 … n_episodes-1 |
| 3 | `block_id` | `int32` | shard block this episode belongs to |
| 4 | `length` | `int32` | T (environment steps; the step record has T+1 rows) |
| 5 | `termination_reason` | `int8` | terminal code (never 0) |
| 6 | `reward_sum` | `float32` | sum of `reward` over the episode — data, not the evaluation metric; survival steps (`length`) is the metric |
| 7 | `animal_active` | `list<bool>[A]` | realised draw — which animal slots exist this episode |
| 8 | `animal_detect_sampled` | `list<int32>[A]` | realised draw — HUNT-trigger sight range |
| 9 | `animal_max_stamina_sampled` | `list<float32>[A]` | realised draw |
| 10 | `animal_recovery_sampled` | `list<float32>[A]` | realised draw |
| 11 | `animal_hunt_thresh_sampled` | `list<float32>[A]` | realised draw |
| 12 | `animal_lose_interest_sampled` | `list<float32>[A]` | realised draw |
| 13 | `animal_move_int_sampled` | `list<int32>[A]` | realised draw — move interval (lower = faster) |
| 14 | `animal_attack_delay_sampled` | `list<int32>[A]` | realised draw |
| 15 | `animal_attack_range_sampled` | `list<int32>[A]` | realised draw — jump/pounce range; 0 = disabled |
| 16 | `animal_property_sampled` | `list<float32>[A*V]` | realised draw, flattened row-major [A, V] |
| 17 | `animal_visual_property_sampled` | `list<float32>[A*VV]` | realised draw, flattened row-major [A, VV] |
| 18 | `res_allocated` | `list<bool>[R]` | realised draw — immutable per-episode resource allocation mask |
| 19 | `res_property_sampled_init` | `list<float32>[R*V]` | realised draw AT RESET, flattened [R, V] — re-drawn on regeneration; see the schema doc's caveat |
| 20 | `res_visual_property_sampled_init` | `list<float32>[R*VV]` | realised draw AT RESET, flattened [R, VV] — re-drawn on regeneration; see the schema doc's caveat |
| 21 | `obs_active` | `list<bool>[B]` | realised draw — which obstacle slots exist this episode |
| 22 | `obs_property_sampled` | `list<float32>[B*V]` | realised draw, flattened [B, V] |
| 23 | `obs_visual_property_sampled` | `list<float32>[B*VV]` | realised draw, flattened [B, VV] |

<!-- END GENERATED: episode_columns -->

**Flattening.** `[A, V]`-shaped draws are stored flattened row-major into a single list
column so the column count is fixed and independent of `V`. `(A, V)` are in the manifest;
`TrajectoryStore.reshape` does the reshape for you.

**All per-episode float columns are `float32` regardless of `obs_precision`.** They are
the independent variables of every future analysis, they cost ~1.8 KB per episode against
~27–44 KB of step data, and reducing their precision would make a join partner lossy for
no measurable saving.

### 3.3 Why the array columns are `list<T>` and not `fixed_size_list<T>[w]`

The plan specified Arrow's `fixed_size_list<T>[w]`. **Parquet cannot round-trip a
`fixed_size_list` of width ZERO** — pyarrow 24.0.0 writes such a column and reads it back
as `[[None], [None], …]`, which is silently wrong data rather than merely a different
type. Zero-width columns are not hypothetical: 18 of the 334 saved configs in the results
tree have no animals at all (`A = 0`).

So every array column uses the variable-size `list<T>`, with the constant width enforced
by the **writer** (offsets are built as a ramp of the manifest width) and re-checked by
`validate_store_shapes`. This is strictly *more* invariant than the plan asked for: with
`fixed_size_list` the column **type** differs between environments
(`fixed_size_list<int16>[4]` vs `[22]`), whereas `list<int16>` is byte-identical for every
environment. Measured cost: 823 B vs 835 B for a 20,000-row × 22-wide `int16` column under
zstd — the variable-size form is marginally *smaller*, because Parquet has no fixed-size
list physical type either and encodes both as a repeated group.

---

## 4. Reading the store

```python
from src.utils.trajectory_store import open_store
import numpy as np

store = open_store("results/trajectories/<run_tag>/<ckpt_step>/<env_fp>")
store.manifest["obs_precision"]        # 'float16' or 'float32' — always check
store.dims                             # {'A': 4, 'R': 4, 'B': 22, 'V': 5, 'VV': 8, 'D': 27}
```

**Load the episode table (the independent variables).**

```python
ep = store.episodes(columns=["episode_seed", "length", "termination_reason",
                             "animal_active", "animal_detect_sampled"])
lengths = np.asarray(ep.column("length"))                       # survival steps — THE metric
detect  = store.to_2d(ep, "animal_detect_sampled")              # (n_episodes, A)
active  = store.to_2d(ep, "animal_active")                      # (n_episodes, A) bool
```

**Select "all predators".** Array index `i` is meaningless without a label; the manifest
carries them, and `select_animals` mirrors `src/environment/state.py::select_by_class`.

```python
pred = store.select_animals("predator")            # (A,) bool mask
mean_pred_sight = detect[:, pred][active[:, pred]].mean()
```

**Reshape a flattened `[A, V]` draw.**

```python
props = store.to_2d(ep, "animal_property_sampled")             # (n_episodes, A*V)
props = store.reshape("animal_property_sampled", props)        # (n_episodes, A, V)
```

**Join steps to episodes on `episode_seed`.**

```python
st = store.steps(columns=["episode_seed", "t", "agent_in_bush", "injury_level"])
import pyarrow as pa
joined = st.join(ep, keys="episode_seed")                       # pyarrow table join
```

**Per-episode bush-occupancy fraction, in `[0, 1]`.** This is the *fraction of the
episode's steps on which the agent stood in a concealing bush* — a dimensionless number
between 0 and 1, **not** a count of steps and **not** a "dwell time". See the
comparability warning in §6.

```python
import pandas as pd
df = st.select(["episode_seed", "agent_in_bush"]).to_pandas()
bush_fraction = df.groupby("episode_seed")["agent_in_bush"].mean()   # in [0, 1]
```

**Verify a manifest with bare pyarrow, importing nothing from `trajectory_store`.**
Useful when you want a read that does not share any code with the writer:

```python
import pyarrow.parquet as pq, json, glob
schema = pq.read_schema(sorted(glob.glob(f"{store_dir}/steps_*.parquet"))[0])
print(schema.names)     # compare against the generated table in §3.1
```

---

## 5. When cross-run pairing holds

Because an episode is a pure function of `jax.random.PRNGKey(seed)`, two runs collected
with the same `seed_base` can face the **same** environment draw at episode `i` — the same
number of predators, the same sight ranges, the same bush layout — which makes comparisons
**paired** and is a large gain in statistical power. But this holds only under a precise
condition, and an analyst cannot infer it from the data:

> Two runs' episode `i` face the same environment draw **iff** they share a `seed_base`
> **and** the environment parameters consumed by `jax_reset` are identical between them —
> the entity slot counts (`count_low` / `count_high` for resources, entities and obstacles)
> and every per-episode sampling bound (`animal_detect_low/high`, `animal_move_int_low/high`,
> `animal_attack_delay_low/high`, `animal_attack_range_low/high`, the four float-uniform
> trait bounds, the property `std` arrays, and the spawn areas). Changing any of these
> changes what the same key draws.
>
> Changes that do **not** break pairing: anything the reset sampler does not read — agent
> architecture, learning rates, training length, reward shaping, `max_steps`, and any
> body-dynamics parameter (including the healing rate) that is applied during stepping
> rather than at reset.
>
> **How to check, rather than assume**: `env_fp` being equal is *sufficient* for pairing.
> It is stricter than necessary — it also changes on parameters that do not affect the
> draw — so when fingerprints differ, compare the `seed_base` and the sampling-bound block
> recorded in the two manifests before claiming or denying pairing.
> **Never assume pairing from run labels.**

**Pairing is exact only per-device and per-compilation.** Two stores collected with the
same `seed_base` on *different* devices are paired in distribution but not necessarily
bit-for-bit, because **the environment's own reset is not bit-reproducible across
compilations**: two runs of `jax_reset` on the same seed can differ by one float32 ULP
(≤ 5.96e-08) on `animal_property_sampled`. The cause is compiler-level arithmetic
reordering — XLA may emit `mean + std · noise` as a separate multiply and add, or fuse it
into a single fused multiply-add, which carries more intermediate precision — and a
different backend can make that choice differently. It is a property of the environment,
not of this pipeline. Full evidence chain, including the refutation of the initial
"batching did it" explanation:
[`docs/llm_wiki/entries/env_entities/20260820_1606_reset_ulp_divergence_is_compiler_fusion.md`](../llm_wiki/entries/env_entities/20260820_1606_reset_ulp_divergence_is_compiler_fusion.md).

`device` is therefore a manifest-guarded field — one store cannot mix CPU-collected and
GPU-collected blocks — and a cross-store comparison should check `device` alongside
`env_fp` and `seed_base` before asserting exact equality of any float draw. **Integer and
boolean draws (slot counts, sight ranges, move intervals, activation masks) are exact
everywhere**, so any analysis keyed on those is unaffected.

---

## 6. Known caveats — read before analysing

### Resource properties are an episode-level approximation

`res_property_sampled_init` and `res_visual_property_sampled_init` are the draws **at
reset**. The environment **re-draws them whenever a resource regenerates**
(`src/environment/core.py:808-809`). An analysis that treats food properties as constant
within an episode is therefore making an approximation.

*Where it breaks*: any episode in which a resource was consumed and regenerated —
detectable per step from `res_cons_count` incrementing and `res_active` toggling
`False → True`. The approximation is **exact** for the window before the first
regeneration and degrades with the number of regenerations, so it is worst in long
episodes with a short `res_reg_delay` and best in short ones. Analyses that condition on
resource property should either restrict to the pre-first-regeneration window or report
the regeneration count as a covariate.

### `agent_in_bush` is not comparable to the existing `bush_dwell`

This store records the environment's **own** definition (`core.py:793-797`): the agent is
hidden iff it stands on **any active obstacle slot flagged as concealing**, over all slots.

The existing dwell-sweep measure `bush_dwell`
(`scripts/behavior_measures/avoidance_stats_heatmap.py:77-79`) hardcodes obstacle **slot 0**
and ignores both `obs_hides_agent` and `obs_active`. Because bush count **varies per
episode** in a training environment, that measure is systematically wrong there. Numbers
produced from this store are therefore **not directly comparable** to existing dwell-sweep
numbers. This is documented, not fixed — changing the dwell-sweep pipeline is out of scope.

### Observation precision is a lossy, recorded choice

Read `obs_precision` from the manifest before doing anything numerical with the
observation columns. Compression (zstd) is lossless and exact; `float32 → float16` is
**not**.

| | `float32` | `float16` |
|---|---|---|
| Decimal digits retained | ~7.2 | **~3.3** |
| Worst-case absolute error on values in `[0, 1]` | ~6e-08 | **2.44e-04** |
| Round-trip bit-identical | yes | **no** |
| Overflow threshold | ~3.4e38 | **65,504** |

> **The recommended value is `float32`, decided 2026-08-20 by measurement.** The plan
> originally adopted `float16` on a synthetic benchmark predicting a 48.6 % saving on the
> observation block and ~38 % store-wide, against a **20 % adoption threshold registered
> before any measurement was taken**. Measured on a real paired collection — the *same*
> 5,000 episodes of the reference training run written twice, once at each precision — the
> saving is **19.3 % on the observation block and 12.8 % store-wide**. Below the threshold,
> so the threshold decides.
>
> The prediction missed by ~7× because this observation vector is mostly **not continuous
> data**: of its 27 channels, 5 are identically zero and 16 more take only the values
> {0, 1, 2} (collision, proprioception and visual are one-hot indicators stored as floats).
> Only 6 olfactory and interoceptive channels carry genuinely continuous values, so zstd
> already compresses the block to **0.49 bytes per value at `float32`** — against the
> 3.26 B/value the synthetic benchmark measured — and half precision has little redundancy
> left to remove. At ~8 GB per run the saving is ~1 GB against a **lossless** store.
>
> **`float16` remains fully supported and is a sound *accuracy* choice** if you want the
> space back — see the measured error figures immediately below. `obs_precision` is
> **mandatory with no default** in both the CLI and the spec, so nothing silently picks
> either value.
>
> **One consequence of `float32`, stated so it is not discovered later:** the collection
> guard returns straight after its finiteness check, so **both magnitude clauses go
> dormant**. That is correct — `float32` cannot misrepresent these magnitudes, so there is
> nothing for them to catch — but it means a `float32` store carries **no recorded evidence
> about observation range**, and concluding later that a run "could have been `float16`" is
> not supported by anything the store contains.

The `2.44e-04` figure above is the worst case **on values in `[0, 1]` only**, and this
store's observations are not all in `[0, 1]`: measured over 1,028,932 real step rows,
**olfaction channels reach 6.95**, where float16's half-ulp is `1.95e-03`. That measured
worst case across the whole store is still **~100× below olfaction's own injected σ of
0.20**, so the accuracy argument for half precision holds even though the size argument
did not — the quantisation error is far below the noise the environment adds on purpose
(per-modality σ: 0.20 olfaction/visual, 0.10 satiation/nociception, 0.05 proprioception,
0.01 collision/location).

**The real hazard is range, not precision, and it is a hard runtime guard.** `float16`
overflows to `inf` above 65,504, and its absolute error grows with magnitude
(half-ulp = `2⁻¹¹·|x|`). The collector checks, on every chunk and **before any shard is
written**, that observations are finite, that `max|obs| ≤ 1e4` (the outer backstop), and
that the float16 round-trip **absolute** error is `≤ 1e-2` (the tight clause — it fires
once any channel's magnitude reaches ~32, about 4.6× the largest magnitude the reference
config actually produces). A violation raises with the offending **observation index** and
the **sensor name** it belongs to, and because shards are written by atomic rename,
nothing is left on disk. So an out-of-range channel cannot be sitting in a store
unnoticed.

The criterion is **absolute, not relative**, and that is a deliberate correction to the
plan: a relative ceiling fires on legitimate near-zero readings. Measured live — an
interoceptive-nociception value of `1.34e-06` is stored as `1.37e-06`, a 2.2 % *relative*
error but a `2.9e-08` *absolute* one, roughly 300,000× below that channel's own injected
noise. A relative criterion is the wrong instrument near zero, and as originally specified
it made `float16` collection impossible.

### `termination_reason` has two documented latent quirks

The schema table says the episode-level `termination_reason` is "never 0", and the
collector **hard-fails** rather than writing a zero, so no store can contain one. But the
promise rests on environment behaviour that has two known, currently-dormant exceptions —
**no live config triggers either**, and both are recorded upstream as latent findings:

- **A body system switched off can produce reason 0 on a real death.** In an
  injury-disabled config (none exists today), an instant predator kill leaves the
  termination-reason code below the trainer's "real death" threshold, so the episode ends
  while the reason still reads `0` = active. The collector refuses such a chunk rather
  than recording it; if you hit that refusal, the environment config is the thing to look
  at, not the collector.
- **`overeating_death` stamps reason 3 on non-terminal steps.** With that setting on (no
  live config turns it on), over-eating never actually ends the episode, but many ordinary
  **non-terminal** rows get the "died from overeating" code. The per-step
  `termination_reason` column would then be misleading on rows where `terminated` is
  `False`. **Read the per-step column jointly with `terminated`, never alone.**

Upstream detail: `src/environment/core.py:117-126,704-712`.

### `animal_damage` is per-step, not per-episode

The damage an animal deals is re-drawn **every step** (`core.py:622-623`), so it is not a
per-episode quantity and is not in the episode record. Only its **bounds** are, in the
manifest's `animal_damage` field.

### Applicability boundary — which runs this store can be trusted for

Runs trained **after 2026-07-23** (commit `828b77e`) are unambiguous: the trainer and
today's config loader resolve the scene identically. Runs from **before** that date may
not be. `src/environment/config_loader.py:428-435` gives a non-empty legacy
`predators:`/`neutral_animals:` block precedence over a modern `entities:` block, and
`train.py` resolved the same file the *other* way before `828b77e` — so for a run whose
saved config carries both formats, reloading rebuilds the scene the trainer **discarded**,
and nothing in the run directory records which branch was taken.

The collector therefore **refuses** any run whose saved config carries both blocks. It does
not attempt to pick one, because there is no correct scene to reconstruct — only a choice.
`--allow-ambiguous-scene` overrides this for someone who has independently established
which scene is correct, and stamps `scene_ambiguous: true` into the manifest so every
downstream reader inherits the caveat. Measured against the results tree as of 2026-08-19,
exactly **12 of 334** saved configs are affected, all dated 2026-05-29 to 2026-06-11.

**Check `scene_format` and `scene_ambiguous` in the manifest before trusting a store built
from an older run.**

### Code drift between training and collection — recorded only for runs trained after 2026-08-20

This store records reset-time state under whatever environment code existed **at
collection time**.

Since 2026-08-20 `train.py` writes `<run_dir>/models/provenance.json` at startup (git sha,
short sha, branch, dirty flag, start time, Python version, argv), and this collector copies
it into the manifest as `training_git_sha` / `training_git_dirty` / `training_started_utc`.
For such a run, the pair (`training_git_sha`, `collection_git_sha`) **names both ends of
the drift** — subject to `training_git_dirty`, which if `true` says uncommitted edits were
present and the training sha is only an upper bound on what actually ran.

For every run trained **before** that date those three fields are `null` — no file existed,
which is not an error and is deliberately distinguishable from the `"unknown"` a stamped
run records when `git` could not be read. For those runs the training-time code version is
unrecorded anywhere in the project: use `collection_git_sha` together with `run_dir_name`
and `train_config_mtime` to **bound** what may have changed in between, rather than
assuming nothing did.

As of 2026-08-20 the reset-parity gate (`tests/env/test_unified_parity.py`) is **fully
green (34/34 executed scenarios)**. It had four failures — all
`configs/verification/observability_gates_S1`–`S4` — which were fully explained by stale
fixtures (generated at `3d20aab`, 2026-05-28) predating a deliberate start-position change
(`84014e4`, 2026-07-04, which set `random_start_pos: false`); the environment reset code
did not drift. The four fixtures were regenerated on 2026-08-20 as a precondition of this
pipeline.

---

## 7. Why the schema looks like this

This section exists to stop the next reader "optimising" the schema. Every number below is
measured.

- **Only float columns cost real bytes.** A genuinely constant integer column measures
  **211× smaller than raw binary** under Parquet + zstd. Integers, booleans, positions,
  timers, counters, the episode key and the step index are collapsed by Parquet's
  encodings to a small fraction of their raw size.
- **Genuinely continuous floats barely compress — but most of this observation vector is
  not continuous.** On a synthetic autocorrelated block: raw `float32` 4.00 B/value;
  Parquet `float32` + zstd 3.26 B/value (1.23×); Parquet `float16` + zstd 1.68 B/value
  (2.38×). **Measured on the real thing, the numbers are completely different**: 0.49
  B/value at `float32` and 0.40 B/value at `float16`, because 21 of the 27 observation
  channels are constants or {0,1,2} indicators. Whole-store bytes per step row: **39.6 at
  `float32`, 34.5 at `float16`** — against the plan's predicted 226 and 140. Extrapolated,
  one 10⁶-episode run is **~8.2 GB / ~7.1 GB**, not the predicted 45 GB / 28 GB.
  *The lesson: benchmark the actual payload, not a plausible-looking stand-in for it.*
- **Hoisting static entity positions to the episode record was measured at a 0.75 % net
  saving and rejected.** Obstacles provably never move within an episode, and an earlier
  draft hoisted their positions out of the step record claiming a 31 % saving. That figure
  was a fraction of *raw uncompressed* bytes, which is not a valid basis for sizing a
  compressed columnar store. Measured properly: 0.08 MB kept per-step vs 0.02 MB hoisted,
  against a 17.37 MB observation block — well under 1 % of the store, in exchange for a
  join at read time on the single most common question this store will be asked, plus a
  permanent regression test asserting that obstacles never move. Deleting the optimisation
  deleted the assumption; if moving obstacles are ever added, this store records them
  correctly with no schema change.
- **A compression ratio is meaningless without its denominator.** Mixing CSV-denominated
  ratios (a `bool` costs 1 byte raw but 5 characters as `"True,"`) into raw-binary byte
  counts produced two *wrong* answers during this plan's drafting, in opposite directions.
  Never quote a ratio without naming what it is measured against.
- **File count, not bytes, is what this filesystem punishes.** `du -sh results/` on this
  NAS times out after two minutes — merely *walking* the tree exceeds two minutes. One
  file per episode would be 10⁶ files per run; the sharded layout is 400. Do not reduce
  `shard_episodes` below ~1000 at production scale.

---

## 8. Maintenance Contract

**Any change to the fixed key set, any dtype change, and any row-convention change MUST
bump `SCHEMA_VERSION` in `src/utils/trajectory_store.py` and update this document in the
same commit.** Readers hard-fail on an unknown `SCHEMA_VERSION`.

The two column tables in §3 and the version line near the top are **generated from
`STEP_COLUMNS` / `EPISODE_COLUMNS`** by
`scripts/eval/traj_collect/gen_schema_doc.py` and live between marker comments. Do not
hand-edit them. `tests/test_trajectory_collection.py::test_schema_doc_matches_code`
asserts the committed doc matches freshly-generated output, so a code change that skips
the regeneration fails the test suite rather than silently rotting the doc.
