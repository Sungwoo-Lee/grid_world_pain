---
title: "Trajectory Collection Pipeline — large-scale in-training-environment rollout store"
topic: behavior
status: active
created: 2026-08-19
last_updated: 2026-08-19
phase: null
aliases: [trajectory_collection_pipeline]
---

# Trajectory Collection Pipeline

> **Status**: PLANNED
> **Opened**: 2026-08-19
> **Related**: [[EVAL_ROLLOUT_BATCHING_PERF]] (the batched-rollout kernel this reuses the correctness argument from), [[PER_EPISODE_ENV_VARIANCE]] (the per-episode random draws this pipeline records), [[BEHAVIOR_ANALYSIS]] (the existing probe-based behaviour measurement this complements)

---

## Context

The project keeps changing the **training** environment — making the predator faster, changing how quickly the agent heals, adding or removing hiding bushes — and wants to know what those changes actually do to the agent's behaviour. Today the only way we measure that is to take the trained agent out of its own world and drop it into a small set of hand-built **test** worlds (the "behaviour probes" / dwell sweep: twelve fixed scenarios, thirty episodes each). Several training-environment changes have come back from those probes looking essentially identical, which tells us the probes are too narrow to explain what changed.

This plan builds the missing tool: after training finishes, put the trained agent **back into the exact world it was trained in**, run it for a very large number of episodes (target: one million per training run), and write down everything that happened in a form that can be sliced later. The key thing that makes this worth doing is that our training worlds are **randomised every episode** — each episode secretly re-rolls how many predators exist, how far each one can see, how fast it moves, how hard it hits, how many bushes there are, and so on. Those secret rolls have never been written down. If we record them alongside the behaviour, then questions like *"how much of the change in bush-hiding time is explained by predator sight range versus by the healing rate?"* become a straightforward table lookup instead of a guess.

Two things this document is **not**. It is not an analysis of any particular set of runs — it is a **general tool** that must work on any training run, including runs that do not exist yet. And it does not build the analysis: the deliverable is the collection pipeline plus a documented, permanently-readable data store. The actual questions get asked afterwards, ad hoc.

The hard requirement that shapes everything below: **one reader must work for every run, forever.** Every environment writes exactly the same named columns in exactly the same order; only the *lengths* of the per-entity array columns change from world to world. This rules out both the current per-step CSV (whose column list changes shape whenever a sensor is toggled) and any per-run bespoke schema.

---

## Analysis

### A1. Why the existing per-step CSV cannot be reused

`build_stat_headers` (`src/utils/evaluation_core.py:198`) generates a column list that varies on four independent axes:

| Axis | Mechanism | Effect |
|---|---|---|
| Entity slot counts | sum of `count_high` over resources / entities / obstacles | column **count** changes |
| Which sensors are on | `get_observation_breakdown` (`src/environment/sensor.py:350-390`) | column **count and ORDER** change — `Injury` / `Nutrition` insert at the front, `Location` appends at the back |
| Per-sensor dimensions | `vector_size`, `visual_vector_size`, `sensor_range` | column count changes |
| Eval-time flag | `testing.record_true_observations` (`evaluation_core.py:293-295`) | whole column groups appear/disappear |

Measured consequence: two probe configs differing only by one predator produce **50 vs 48** columns; two "basic" configs produce **58 vs 146**. A reader written against one is silently wrong against the other.

Separately, the fast rollout path and the detailed record are **disjoint**: `scripts/eval/eval_rollout.py` never imports `evaluation_core` and never writes `ep_stats.csv`. The only writers are `_run_single_env_eval` (`evaluation_core.py:550-553`) and `_run_parallel_env_eval` (`:675-679`), reachable only through `evaluate_jax_checkpoint` (`:242`), which is called from `train.py:2474` and root `evaluation.py:429`. `_run_parallel_env_eval` performs a host round-trip per environment per step (`:653-654`, `:666`), making it roughly two orders of magnitude slower than the batched path. So there is no existing artifact to extend — this is a new writer.

### A2. The per-episode random draws are already on state and are trivially readable

`jax_reset` samples them and stores them on `EnvState` (`src/environment/core.py:1300-1362`; field declarations `src/environment/state.py:53-74`). `jax_step` copies every one of them through unchanged (`core.py:818-828`), so they are genuinely constant within an episode and can be read once at reset.

A dead-code reader already exists at `src/behavior/accumulators.py:546` (`build_episode_log_dict`) — **zero callers**, covers only 5 of the 8 scalar draws, and omits the activation masks entirely. It is not usable as-is. (Noting it here as pre-existing dead code; this plan does not delete it.)

Three correctness notes that the schema below depends on:

1. **`animal_damage` is NOT a per-episode quantity.** It is redrawn every step (`core.py:622-623`). Only its *bounds* are well-defined at episode level, so the bounds go in the run manifest, not in the per-episode record.
2. **`res_property_sampled` and `res_visual_property_sampled` are NOT constant within an episode** — they are re-drawn on resource regeneration (`core.py:808-809`, `res_property_sampled_after_reg`). See the caveat in §D4.
3. **`res_active` mutates during the episode** (consumption / regeneration), whereas `res_allocated` is the immutable per-episode allocation mask set once at reset (`core.py:1325`). Therefore `res_allocated` belongs in the per-episode record and `res_active` in the per-step record. `animal_active` and `obs_active` are genuinely constant (`core.py:827-828`) and belong at episode level.

### A3. The batched rollout will not survive this scale unchanged

`_run_episodes_batched` (`eval_rollout.py:356`) vmaps over episodes and `lax.scan`s over time inside `nnx.jit` (`:430`). Problems at the target scale:

- **No chunking, no cap.** Batch size is exactly `--eval-n-episodes` (`:392`) and the entire scan result is pulled to host in one shot (`:443`). At a measured marginal host cost of ~2.1 KB per env-step, an unchunked 10⁵-episode batch at 500 steps would need ~105 GB of host RAM.
- **Two O(n) Python bottlenecks.** The PRNG parity guard runs one *unbatched* `jax_reset` plus a host sync **per seed** (`:405-415`), and per-episode result slicing is a Python loop (`:456-475`). Both are free at 30 episodes and fatal at 10⁶.
- **Never validated above 30 episodes.** Every file in `configs/eval_sweeps/` sets `episodes: 30`. Everything above that is unproven.

### A4. ~62 % of scan compute is dead

`lax.scan` runs `length=max_steps` unconditionally (`eval_rollout.py:351`); truncation happens afterwards on the host (`:448-452`). In the real training environment mean episode length is **192** against `max_steps=500`, so 61.6 % of scanned steps are past the end of their episode. This was invisible in the 100-step probes the pipeline was tuned on. §D6 gives the decision and its cost.

### A5. The JIT gotcha that silently corrupts results

After `nnx.update(model, restored_tree)`, calling the model **eagerly** reads a stale view of the restored parameters and every trajectory silently diverges from step 0 onward. `lax.scan` alone does not fix it — it compiles the outer loop but does not perform nnx's graphdef/state split. The full root-cause note is at `eval_rollout.py:293-311`. **Any new rollout kernel must be entered through `nnx.jit`.** This is a checkpoint below, not a footnote.

### A6. The output-path collision hazard (a real past incident)

`out_dir` is derived only from the checkpoint path (`eval_rollout.py:1012-1014`: `out_root / run_tag / ckpt_path.name`), so re-running the same checkpoint under a *different* environment config and the same output root overwrites episode-for-episode with no warning. This has already caused a real data-contamination incident, recorded in `docs/diary/2026-07-04.md`. The new store's path scheme must make this structurally impossible (§D3).

### A7. Defects the new pipeline must not inherit

| Defect | Location | Consequence | Handling here |
|---|---|---|---|
| `nociception` is always exactly `0.0` | `eval_rollout.py:468-471` — no such key exists in the env `info` dict | a column of zeros masquerading as data | **not recorded**; the real interoceptive nociception is already inside the observation vector |
| `bush_dwell` hardcodes obstacle slot 0 and ignores `obs_hides_agent` / `obs_active` | `scripts/behavior_measures/avoidance_stats_heatmap.py:77-79` | wrong whenever bush count varies per episode — i.e. always, in training environments | record the env's own `agent_in_bush` (`core.py:793-797`) directly; see the comparability warning in §D8 |
| npz mixes pre-step and post-step quantities in one row | `agent_pos` is pre-step, `info` fields are post-step | silent off-by-one in any joint analysis | single documented row convention, §D2 |
| `ep_stats.csv` and the npz use opposite row conventions | — | the two files are off by one relative to each other | single store, single convention |

### A8. Measured throughput and memory (live probe)

Real rest-premium checkpoint, rolled out in its own training environment, CPU, one thread:

| Quantity | Measured |
|---|---|
| Episodes/s per process | ~14.3 |
| Scanned env-steps/s per process | ~7,150 |
| Peak RSS per process | ~1.2 GB |
| Fixed per-process startup (JAX import + model build + checkpoint restore) | 15–17 s, independent of episode count |

### A9. Representative environment dimensions

Taken from a real saved training config (`results/JAX_RecurrentPPO/20260816-152742_rppo_restpremNH_a10_n112/models/config.yaml`), loaded through `load_env_params` and `get_observation_breakdown`:

| Symbol | Meaning | Value |
|---|---|---|
| `A` | animal slots (`sum count_high` over entities: 2 predator + 2 neutral) | 4 |
| `R` | resource slots (food, `count_high=4`) | 4 |
| `B` | obstacle slots (bush `count_high=10` + rock `count_high=12`) | 22 |
| `V` | `sensory.vector_size` (olfactory property width) | 5 |
| `VV` | `sensory.visual_vector_size` | 8 |
| `D` | observation dimension (Satiation 1, Intero-Nociception 1, Extero-Nociception 1, Olfaction 5, Collision 5, Proprioception 6, Visual 8) | 27 |
| `max_steps` | episode step limit | 500 |
| mean `T` | measured mean episode length | 192 |

All size arithmetic below uses these numbers.

### A10. The saved config is the correct source of truth

`train.py:881-883` writes the **fully resolved** config to `<run>/models/config.yaml` — no `extends:` survives. It carries the per-episode sampling *bounds* (`detection_range: [1,7]`, `count_low` / `count_high`, and so on), which are the join partner for the realised draws recorded per episode. Loading the environment from this file rather than from `configs/` is what makes the pipeline honest about which world the agent actually trained in, per the project's "verify actual state, not a re-derivation" rule.

---

## Implementation Plan

### Design

#### D1. Fixed schema, variable slot lengths

One schema, `SCHEMA_VERSION = 1`, defined in exactly one place. Every column name and its position in the schema are **identical for every environment**. What varies is the *length* of the fixed-size list columns, which is a property of the environment's slot counts `(A, R, B, V, VV, D)` recorded in the store manifest. This is the `.rec.gz` model — a fixed record whose per-entity arrays follow the environment's slot count — expressed in Parquet's `fixed_size_list` type.

Two consequences worth stating plainly:

- A column whose slot count is zero in some environment (e.g. `A = 0`, a world with no animals) is present as a **zero-length list**, not absent. Readers never branch.
- No sensor toggle, no eval-time flag, and no config option may ever add or remove a column. Observation columns are one `fixed_size_list<float16>[D]` each, not `D` separate scalar columns — this is precisely what makes the sensor-ordering problem in §A1 disappear.

`pyarrow 24.0.0` and `pandas 3.0.0` are already installed in the `grid_world_pain` env; no new dependency.

#### D2. Row convention — ARRIVAL (single, documented, testable)

Row `t` of an episode holds:

- **(a)** the environment state **at time `t`**, and
- **(b)** the action, reward, and transition outputs of the step that **arrived at** time `t`.

Row `t = 0` is the **reset state**, with `action = -1`, `reward = 0.0`, and all transition-output fields at their zero value. An episode of length `T` therefore has **`T + 1` rows**.

This eliminates the pre/post mixing of §A7 entirely: every field in a row is either "the state at time t" or "the step that produced the state at time t". There is exactly one sentence to remember.

The observation columns in row `t` are the observation **of state `t`** — that is, the observation the policy consumed when choosing the action recorded in row `t + 1`. Stated once here, restated in the schema table, and restated in the schema doc.

Cost of the extra row: `1/193 ≈ 0.5 %`. Benefit: no off-by-one is possible, and the convention is directly falsifiable (verification V3).

The existing `EpisodeRecorder` already writes an initial snapshot with `action_idx=-1, reward=0.0` (`eval_rollout.py:480-494`), so this convention is consistent with the recording format the project already uses.

#### D3. Store layout and the anti-overwrite guarantee

```
<store_root>/<run_tag>/<ckpt_step>/<env_fp>/
    _manifest.json
    episodes_00000.parquet      # one row per episode, block 0
    episodes_00001.parquet
    ...
    steps_00000.parquet         # T+1 rows per episode, block 0
    steps_00001.parquet
    ...
```

- `run_tag` — the training run directory name, e.g. `20260816-152742_rppo_restpremNH_a10_n112`.
- `ckpt_step` — the checkpoint step, as an integer string.
- `env_fp` — **first 10 hex characters of the SHA-256 of the canonicalised resolved environment config** (`yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)` of the dict loaded from `<run>/models/config.yaml`).

Three layers defeat the §A6 hazard:

1. **Structural.** A different environment config produces a different `env_fp` and therefore a different directory. Two configs can no longer collide on one path.
2. **Manifest guard.** On any resume, the collector recomputes `env_fp`, the git SHA, `SCHEMA_VERSION`, `seed_base`, and `n_episodes`, and compares them to `_manifest.json`. **Any mismatch is a hard `ValueError` and nothing is written.** (Git SHA mismatch is a warning, not a failure — code can legitimately move between collection sessions — but it is recorded per shard so provenance is never lost.)
3. **Atomic shards.** Each shard is written to `<name>.parquet.tmp` and then `os.replace`d. A shard on disk is always complete. There is no partial-file state for a reader to trip over.

`_manifest.json` records: `schema_version`, `env_fp`, the full resolved env config, `run_path`, `checkpoint_path`, `ckpt_step`, slot counts `(A, R, B, V, VV, D)`, `max_steps`, `action_dim`, `seed_base`, `n_episodes`, `shard_episodes`, `batch_size`, `device`, `policy_mode` (always `"deterministic_argmax"`), **`obs_precision`** (§D12 — manifest-guarded, so a `float16` store can never be resumed as `float32`), `git_sha`, the per-entity **names and classes** (`animal_classes`, `animal_behaviours`, resource / obstacle names) so array index `i` can be given a human label, and the static per-entity parameter arrays needed as join partners: `animal_damage` bounds, `obs_hides_agent`, `res_type`, `animal_is_damaging`, and the per-episode sampling bounds (`animal_detect_low/high`, `count_low/high`, …).

Storage root: `results/trajectories/`. This is gitignored data on the NAS — the standing git-safety rule applies (never `git clean -x` / `-X` / `-fdx`).

#### D4. The FIXED KEY LIST

`E` denotes the episode's length `T`; list lengths are `A`, `R`, `B`, `V`, `VV`, `D` from the manifest.

##### D4.1 Per-step record — `steps_NNNNN.parquet`

One row per `(episode, t)`, `t ∈ [0, T]`. Sorted by `(episode_seed, t)`. Columns in this exact order:

| # | Column | Arrow type | Row-convention timing | Source |
|---:|---|---|---|---|
| 1 | `episode_seed` | `int64` | episode key | the seed whose `PRNGKey` produced the episode |
| 2 | `t` | `int16` | index, `0…T` | — |
| 3 | `action` | `int8` | **arriving** (`-1` at `t=0`) | `argmax(logits)` |
| 4 | `reward` | `float32` | **arriving** (`0.0` at `t=0`) | `jax_step` return |
| 5 | `agent_row` | `int16` | state at `t` | `state.agent_pos[0]` |
| 6 | `agent_col` | `int16` | state at `t` | `state.agent_pos[1]` |
| 7 | `satiation` | `float32` | state at `t` | `state.satiation` |
| 8 | `nutrition` | `float32` | state at `t` | `state.nutrition` |
| 9 | `injury_level` | `float32` | state at `t` | `state.injury_level` |
| 10 | `rest_streak` | `int16` | state at `t` | `state.rest_streak` |
| 11 | `last_collision_noc` | `float32` | state at `t` | `state.last_collision_noc` |
| 12 | `terminated` | `bool` | state at `t` | `state.terminated` |
| 13 | `damage` | `float32` | **arriving** (`0.0` at `t=0`) | `info['damage']` — see **Open Question 1** |
| 14 | `ate_food` | `bool` | **arriving** (`False` at `t=0`) | `info['ate_food']` |
| 15 | `rested` | `bool` | **arriving** | `info['rested']` |
| 16 | `hit_predator` | `bool` | **arriving** | `info['hit_predator']` |
| 17 | `hit_neutral` | `bool` | **arriving** | `info['hit_neutral']` |
| 18 | `hit_hiding_predator` | `bool` | **arriving** | `info['hit_hiding_predator']` |
| 19 | `event_collided` | `bool` | **arriving** | `info['event_collided']` |
| 20 | `agent_in_bush` | `bool` | state at `t` | `info['agent_in_bush']` for `t≥1`; recomputed at reset for `t=0` (§D8) |
| 21 | `termination_reason` | `int8` | **arriving** (`0` except final row) | `info['termination_reason']`; codes `0`=active, `1`=max_steps, `2`=starvation, `3`=overeating, `4`=injury (`core.py:702-712`) |
| 22 | `animal_row` | `fixed_size_list<int16>[A]` | state at `t` | `state.animal_pos[:,0]` |
| 23 | `animal_col` | `fixed_size_list<int16>[A]` | state at `t` | `state.animal_pos[:,1]` |
| 24 | `animal_state` | `fixed_size_list<int8>[A]` | state at `t` | `state.animal_state` — `0`=PATROL, `1`=HUNT, `2`=RETURN |
| 25 | `animal_stamina` | `fixed_size_list<float32>[A]` | state at `t` | `state.animal_stamina` |
| 26 | `animal_move_timer` | `fixed_size_list<int16>[A]` | state at `t` | `state.animal_move_timer` |
| 27 | `animal_attack_timer` | `fixed_size_list<int16>[A]` | state at `t` | `state.animal_attack_timer` |
| 28 | `res_row` | `fixed_size_list<int16>[R]` | state at `t` | `state.res_pos[:,0]` |
| 29 | `res_col` | `fixed_size_list<int16>[R]` | state at `t` | `state.res_pos[:,1]` |
| 30 | `res_active` | `fixed_size_list<bool>[R]` | state at `t` | `state.res_active` |
| 31 | `res_cons_count` | `fixed_size_list<int16>[R]` | state at `t` | `state.res_cons_count` |
| 32 | `res_reg_timer` | `fixed_size_list<int16>[R]` | state at `t` | `state.res_reg_timer` |
| 33 | `obs_row` | `fixed_size_list<int16>[B]` | state at `t` | `state.obs_pos[:,0]` — constant within an episode; deliberately kept per-step, see below |
| 34 | `obs_col` | `fixed_size_list<int16>[B]` | state at `t` | `state.obs_pos[:,1]` |
| 35 | `obs_noised` | `fixed_size_list<float16 \| float32>[D]` | state at `t` | `get_observation(state, params)` — what the policy received; element type set by the mandatory `obs_precision` key (§D12) |
| 36 | `obs_true` | `fixed_size_list<float16 \| float32>[D]` | state at `t` | `get_observation(state, params, apply_noise=False)` — noise-free ground truth; same element type |

**Obstacle and resource positions stay per-step, even though obstacles provably never move.** `obs_pos` is absent from the `state._replace(...)` call at `core.py:800-838`, so it is constant within an episode. An earlier draft of this plan hoisted it to the per-episode record and claimed a **31 % saving**. **That figure was wrong and the optimisation is deleted.** 31 % was a fraction of *raw uncompressed* bytes, and **raw-byte accounting is not valid for sizing a compressed columnar store** — Parquet's run-length encoding removes nearly all of that redundancy before any schema change gets a chance at it.

Measured directly, on a realistic layout (1,000 episodes × 192 steps × 22 obstacle slots, zstd, `fixed_size_list` columns matching the schema above):

| | size |
|---|---:|
| obstacle position columns, kept per-step | 0.08 MB |
| same, hoisted to per-episode | 0.02 MB |
| saving **on those columns** | 75 % |
| those columns as a share of total payload | **~0.5 %** (0.08 MB against a 17.37 MB `float16` observation block, §D11) |
| **net saving on the store** | **well under 1 %** |

Well under one percent — in exchange for schema complexity and a join at read time on every analysis wanting agent-versus-obstacle geometry, which is the single most common question this store will be asked. **Recorded here so nobody re-proposes it.**

Supporting column-level measurements on 200,000 real rows, showing why the redundancy was already gone before the optimisation was considered:

| Column shape | CSV | Parquet + zstd | Ratio **vs CSV** |
|---|---:|---:|---:|
| constant integer (an obstacle position that never moves) | 391 KB | **2 KB** | 213× |
| mostly-`False` boolean flag | 1,162 KB | **11 KB** | 104× |
| 6-value low-cardinality string | 911 KB | **71 KB** | 13× |

> **⚠️ Denominator warning — this table is CSV-denominated, not raw-binary-denominated.** CSV is a text encoding and is itself several times larger than packed binary (an `int16` costs 2 bytes raw but ~3–4 characters as text; a bool costs 1 byte raw but 5 characters as `"True,"`). **These ratios must never be divided into raw binary byte counts.** Doing so is what produced the wrong conclusion recorded in §D12. When sizing this store, use raw-binary-denominated figures only.
>
> The obstacle measurement above happens to survive the correction because it was also measured raw-denominated: 0.08 MB compressed against `192,000 rows × 88 B = 16.9 MB` raw is **211× vs raw binary**. Genuinely constant columns really are free. Varying low-cardinality columns are cheap but not free — see §D11's non-float estimate.

The general rule, and the one that should govern any future schema debate: *do not restructure the schema to remove redundancy the encoder already removes for free, and always check what a compression ratio is denominated in before using it.* Constant and near-constant integer / boolean columns are effectively free. **Float columns are the only ones that cost real bytes (§D11).**

The corollary is why this is a net simplification rather than a concession. The hoisting draft required a permanent regression test asserting that obstacles never move — a correctness assumption baked into the store — purely to protect a 0.75 % saving. **Deleting the optimisation deletes the assumption and the test with it.** If moving obstacles are ever added, this store records the movement correctly, with no schema change and no silent corruption.

**Not recorded, deliberately** — with reasons, so nobody re-adds them by accident:

| Field | Why not |
|---|---|
| `nociception` | always exactly `0.0`; the key does not exist in the env `info` dict (§A7) |
| `dist_per_predator` / `dist_per_neutral` / `dist_to_*` | recomputable exactly from the recorded agent and entity positions; the existing fields also carry a one-step staleness bug (they are computed from `state.animal_pos`, the *pre*-move positions, against `new_agent_pos`) |
| `reward_homeostatic` / `reward_extrinsic` / `drive_hunger` / `drive_injury` / `metabolic_drain` | reconstructable from the recorded body-state columns plus the manifest's config; `drive_*` are closed-form functions of `satiation` / `injury_level` (`core.py:725-726`) |
| `injury_buffer`, `nociception_history_buffer` | pure functions of the recorded `injury_level` history and the config's kernel parameters |
| `state.key` | large, and the episode is fully determined by `episode_seed` |

##### D4.2 Per-episode record — `episodes_NNNNN.parquet`

One row per episode. Columns in this exact order:

| # | Column | Arrow type | Meaning |
|---:|---|---|---|
| 1 | `episode_seed` | `int64` | join key to `steps`; `seed = seed_base + episode_index` |
| 2 | `episode_index` | `int64` | `0 … n_episodes-1` |
| 3 | `block_id` | `int32` | shard block this episode belongs to |
| 4 | `length` | `int32` | `T` (number of environment steps; the step record has `T+1` rows) |
| 5 | `termination_reason` | `int8` | terminal code (never `0`) |
| 6 | `reward_sum` | `float32` | sum of `reward` over the episode — **data, not the evaluation metric**; survival steps (`length`) is the metric |
| 7 | `animal_active` | `fixed_size_list<bool>[A]` | **realised draw** — which animal slots exist this episode |
| 8 | `animal_detect_sampled` | `fixed_size_list<int32>[A]` | **realised draw** — HUNT-trigger sight range |
| 9 | `animal_max_stamina_sampled` | `fixed_size_list<float32>[A]` | **realised draw** |
| 10 | `animal_recovery_sampled` | `fixed_size_list<float32>[A]` | **realised draw** |
| 11 | `animal_hunt_thresh_sampled` | `fixed_size_list<float32>[A]` | **realised draw** |
| 12 | `animal_lose_interest_sampled` | `fixed_size_list<float32>[A]` | **realised draw** |
| 13 | `animal_move_int_sampled` | `fixed_size_list<int32>[A]` | **realised draw** — move interval (lower = faster) |
| 14 | `animal_attack_delay_sampled` | `fixed_size_list<int32>[A]` | **realised draw** |
| 15 | `animal_attack_range_sampled` | `fixed_size_list<int32>[A]` | **realised draw** — jump/pounce range; `0` = disabled |
| 16 | `animal_property_sampled` | `fixed_size_list<float32>[A*V]` | **realised draw**, flattened row-major `[A, V]` |
| 17 | `animal_visual_property_sampled` | `fixed_size_list<float32>[A*VV]` | **realised draw**, flattened row-major `[A, VV]` |
| 18 | `res_allocated` | `fixed_size_list<bool>[R]` | **realised draw** — immutable per-episode resource allocation mask |
| 19 | `res_property_sampled_init` | `fixed_size_list<float32>[R*V]` | **realised draw at reset**, flattened `[R, V]` — see caveat below |
| 20 | `res_visual_property_sampled_init` | `fixed_size_list<float32>[R*VV]` | **realised draw at reset**, flattened `[R, VV]` — see caveat below |
| 21 | `obs_active` | `fixed_size_list<bool>[B]` | **realised draw** — which obstacle slots exist this episode |
| 22 | `obs_property_sampled` | `fixed_size_list<float32>[B*V]` | **realised draw**, flattened `[B, V]` |
| 23 | `obs_visual_property_sampled` | `fixed_size_list<float32>[B*VV]` | **realised draw**, flattened `[B, VV]` |

Rows 7–23 are the complete independent-variable side of the analysis: everything the environment secretly re-rolled at the start of this episode. Joined against the manifest's sampling bounds (§A10), each realised draw can be expressed as a position within its own range.

Obstacle **positions** are not here — they are per-step columns 33–34, for the reason given in §D4.1. Only the obstacle *draws* (which slots exist, and their sampled properties) live at episode level, matching how animals and resources are handled.

All per-episode float columns are `float32` regardless of `obs_precision`: they are the independent variables of every future analysis, they cost ~1.8 KB per episode against ~47 KB of step data, and reducing their precision would save nothing measurable while making a join partner lossy.

**Caveat on columns 19–20 (must be in the schema doc).** `res_property_sampled` and `res_visual_property_sampled` are re-drawn whenever a resource regenerates (`core.py:808-809`). The recorded values are the **reset draw only**. A regeneration event is detectable per step from `res_cons_count` incrementing and `res_active` toggling, so an analysis that conditions on resource property should either restrict to the pre-first-regeneration window or accept the approximation. Recording them per step would add `R*(V+VV) = 52` float values per step — a ~40 % increase on the 128-float-per-step observation block that dominates the store (§D11), since floats are the only columns that cost anything. Deferred; see **Open Question 2**.

**Flattening note.** `[A, V]`-shaped draws are stored flattened row-major into a single list column so that the column count is fixed and independent of `V`. `(A, V)` are in the manifest; the reader reshapes.

**Human labels.** Array index `i` is meaningless without a name. The manifest carries `animal_classes` / `animal_behaviours` (e.g. `("predator","predator","neutral","neutral")` and `("hunt","hunt","wander","wander")`), resource names, and obstacle names, so an analysis can select "all predators" or "all bushes" the way `select_by_class` (`src/environment/state.py:7`) does today.

#### D5. Chunking and the RAM arithmetic

The scan kernel emits, per environment-step, native-dtype device arrays (JAX has no `int8`/`float16` output here — downcast happens on host):

| Group | Bytes / env-step |
|---|---:|
| agent + body scalars + info scalars + flags | 53 |
| animals (`pos` 32, `state` 16, `stamina` 16, `move_timer` 16, `attack_timer` 16) | 96 |
| resources (`pos` 32, `active` 4, `cons_count` 16, `reg_timer` 16) | 68 |
| obstacles (`pos` `int32[22,2]`) | 176 |
| observations (`obs` 27×4, `obs_true` 27×4) | 216 |
| **Total** | **≈ 609** |

Scan output for a chunk of `B_c` episodes: `max_steps × B_c × 609 B = 304.5 KB × B_c`. The device buffer and its host copy coexist during transfer, so peak ≈ 2×.

| `B_c` | scan buffer | device + host peak | process peak RSS (base 1.2 GB) |
|---:|---:|---:|---:|
| 1024 | 312 MB | 624 MB | **≈ 1.8 GB** |
| 2048 | 623 MB | 1.25 GB | ≈ 2.5 GB |
| 4096 | 1.25 GB | 2.50 GB | ≈ 3.7 GB |
| 102400 (unchunked, illustrative) | 30 GB | 61 GB | OOM |

**Chosen: `batch_size = 1024`.** At 16 worker processes per node this is ~29 GB of node RAM, comfortable on any lab node. On CPU, "device memory" *is* host memory, so the 2× factor is real, not conservative. GPU changes the trade-off — see §D7.

**The obstacle-position block is emitted per step by the scan, not broadcast on host.** Because `obs_pos` is constant within an episode, the scan *could* emit it once from `states0` and let the host writer broadcast it across the episode's rows, saving 176 of 609 bytes per env-step of device→host traffic (29 %). **Rejected.** That reintroduces exactly the silent-corruption failure mode §D4.1 just removed: if moving obstacles are ever added, the writer would quietly record the reset position for every step. RAM at `batch_size = 1024` is 1.8 GB either way and the whole collection is a ~1.5 h job (§D6), so the saving buys nothing that matters and costs a correctness caveat. Emit it plainly.

Per chunk: `1024 × 500 = 512,000` scanned steps ÷ 7,150 steps/s ≈ **72 s**. One 5,000-episode shard block = 5 chunks ≈ 6 minutes.

**The two O(n) bottlenecks (§A3) are removed:**

- *PRNG parity guard.* The per-seed unbatched `jax_reset` loop with a host sync per seed (`eval_rollout.py:405-415`) is replaced by a **vectorised, once-per-chunk** guard: `ref = jax.lax.map(lambda k: jax_reset(params, k).key, keys)` compared against `states0.key`, one host sync per chunk. `lax.map` lowers to `lax.scan` — genuinely *not* `vmap` — so it remains a valid independent reference for exactly the invariant the original guard protected (that the reset keys are `vmap(jax_reset)` over `stack([PRNGKey(s)])` and **not** `ParallelEnv.reset`, whose internal `jax.random.split(key, num_envs)` derives a different key set). The full-strength version of this check runs offline as verification V1.
- *Per-episode slicing.* The Python `for i in range(num_envs)` loop (`:456-475`) is replaced by fully vectorised NumPy: compute `T = argmax(done_seq, axis=0) + 1` (already vectorised at `:452`), build a `(max_steps+1, B_c)` validity mask `t <= T`, and use it to flatten every field to `(sum(T+1),)` in one `[mask]` gather per column. No Python-level per-episode work.

#### D6. The 62 % dead-step waste — decision: **ACCEPT for v1**, quantified

**Options considered.**

*Chunk-level early exit* (break out of the scan when all lanes are done) is worthless here. With a chunk of 1024 episodes, the scan can only stop when the **longest** episode in the chunk finishes, and at these episode-length distributions essentially every chunk contains at least one episode that runs to `max_steps=500`. Expected saving: ≈ 0 %.

*Continuous batching / auto-reset refill* is the real fix: when a lane's episode ends inside the scan, reset that lane from `PRNGKey(next_seed)` (lane `l` consumes seeds `l, l+L, l+2L, …`), reset the recurrent hidden state via `h = where(done, h0, h_new)`, emit a per-lane `episode_slot` id per step, and regroup rows by `(lane, episode_slot)` on host, discarding the partial episodes still open when the scan window closes. This makes ~100 % of scanned steps productive.

**Cost of accepting the waste**, per training run of 10⁶ episodes:

| | scanned steps | core-hours @ 7,150 steps/s |
|---|---:|---:|
| Accept (scan `max_steps` unconditionally) | `10⁶ × 500 = 5.0×10⁸` | **19.4** |
| Auto-reset refill | `10⁶ × 193 = 1.93×10⁸` | 7.5 |
| **Difference** | | **11.9 core-hours per run** |

Across 10 runs that is 119 core-hours saved. On 8 nodes × 16 processes = 128 workers, the whole 10-run collection is **≈ 1.5 h wall-clock accepting the waste** versus ≈ 0.6 h with refill.

**Decision: accept.** A 2.6× compute factor on a job that already completes in under two hours of wall-clock does not justify adding an in-scan episode-boundary state machine to a pipeline whose entire value proposition is that its data is trustworthy. Auto-reset refill is specified above in enough detail to implement as a Phase 2 if the episode budget grows by an order of magnitude or the environment's `max_steps`-to-mean-length ratio worsens. Recorded here so the choice is visible rather than accidental.

#### D7. Parallelisation, and how the device flag interacts with it

Three levels:

| Level | Mechanism | Unit |
|---|---|---|
| L1 — intra-process | `vmap` over episodes inside one `nnx.jit` `lax.scan` | `batch_size` episodes per chunk (1024) |
| L2 — intra-node | `xargs -P npar` over **disjoint blocks** in a per-node worklist | 5,000-episode block |
| L3 — multi-node | LPT partition of `(run × block-range)` cells, launched via `run_command.py`, completion markers polled | node |

Blocks are disjoint and each block owns its own two output files, so **L2 and L3 need no locking whatsoever** — the atomic-rename write is the only coordination primitive.

L3 reuses the pattern proven by `scripts/eval/dwell_sweep/run_sweep.py`: LPT partitioning (`:238`), per-node worklist files (`:252`), launch via `run_command.py` (`:275`), and completion markers polled at `<scratch>/_run_markers/done_<node>` (`:283`). The worker script copies `sweep_worker.sh`'s CPU thread caps (`:29-31`) and per-node persistent XLA compile cache (`:38-40`), which gave a measured ~4× wall-time improvement there and matters more here (591 checkpoints' worth of compiles are not needed, but the ~7 s compile per process across ~160 processes is).

**`run_command.py` is not parallel-safe** — concurrent invocations race through a shared SSH control socket and can return one node's answer to every caller. The driver must launch nodes **strictly serially**, one `subprocess.run` at a time, exactly as `run_sweep.py:275` does.

**Device flag interaction** (`--device`, default `cpu`):

| | `--device cpu` (default) | `--device gpu` |
|---|---|---|
| env | `JAX_PLATFORMS=cpu`, `OMP/MKL/OPENBLAS/TF_*_THREADS=1`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1` | `JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `CUDA_VISIBLE_DEVICES=<idx>` |
| `npar` per node | `min(cores − 2, 16)` — throughput comes from L2 | **1 per GPU** — a second JAX process on one GPU contends for memory and is slower, not faster |
| `batch_size` | 1024 (host-RAM bound, §D5) | 8192 (GPU-memory bound; 8192 → ~1.8 GB scan buffer, fits an 11 GB 2080 Ti with headroom) |
| where the parallelism lives | L2 (many processes) | L1 (one big vmap batch) |

The two knobs move in opposite directions and **must be set together**; the driver derives `npar` and `batch_size` from `device` unless both are given explicitly in the spec, and writes the resolved values into `_manifest.json`. Before any GPU launch the lab GPU rules apply: consult `docs/environment/LAB_NODE_GPU_SPEC.md` for which GPU indices exist on the target node, and check live occupancy with `scripts/lab/gpu_status.py`. CPU collection can run on nodes that are busy training on GPU; GPU collection cannot.

#### D8. `agent_in_bush` — correct definition, and the comparability warning

The env's own definition (`core.py:793-797`) is:

```
agent_in_bush = any( all(obs_pos == new_agent_pos, axis=-1)
                     & (params.obs_hides_agent & state.obs_active) )
```

i.e. the agent is hidden iff it stands on **any active obstacle slot flagged as hiding**, over all slots. For `t ≥ 1` the pipeline records `info['agent_in_bush']` verbatim. For `t = 0` (the reset row) the env produces no `info`, so the collector applies the same expression to the reset state via a small helper defined in the new scan module (**not** in `src/environment/`, which this plan does not touch). Verification V4 asserts the helper and the env agree at every step.

> **Comparability warning — must appear in the schema doc and in any downstream analysis.**
> The existing dwell-sweep measure `bush_dwell` (`scripts/behavior_measures/avoidance_stats_heatmap.py:77-79`) hardcodes obstacle slot 0 and ignores both `obs_hides_agent` and `obs_active`. Because bush **count varies per episode** in training environments, that measure is systematically wrong there. Numbers produced from this store are therefore **not directly comparable** to existing dwell-sweep numbers. This plan does **not** change the dwell-sweep pipeline (out of scope); it records the correct quantity and flags the discrepancy.

#### D9. Checkpoint selection

Final checkpoint only, but the spec format accepts a list without redesign (`checkpoints: [final]` today; `checkpoints: [final, 30000000]` or `checkpoints: all` later).

**Numeric sort is mandatory.** The rPPO run inspected in §A9 has **591** checkpoint directories under `<run>/models/` with names like `100145`, `10100043`, `58900041`, `59100070`. These are not zero-padded, so a lexicographic maximum picks the wrong directory (a `9`-prefixed six-digit name sorts above `59100070`). Select with `max(int(d.name) for d in dirs if d.name.isdigit())`. This is a live bug waiting to happen and must be a checkpoint during implementation.

#### D10. Resume and failure recovery

- Episode index space is partitioned into contiguous blocks of `shard_episodes` (5,000). Block `b` covers `[5000b, 5000(b+1))`; episode `i` uses seed `seed_base + i`, where `seed_base` is this run's **effective** value (batch-level, or the run-level override — §D13). Both the block partition and the seed mapping are pure functions of `(seed_base, shard_episodes)` recorded in the manifest — no state, no counter file.
- A block is **complete** iff both `episodes_%05d.parquet` and `steps_%05d.parquet` exist (atomic rename guarantees each is whole). Resume = list complete blocks, skip them, work the rest.
- A node dying mid-block loses at most one block ≈ **6 minutes** of work, and leaves at most two `.tmp` files, which the collector deletes on startup for blocks it is about to redo.
- Because blocks are pure functions of the seed, a redone block is **bit-identical** to what the dead process would have produced. Resume can never produce a mixed population.
- The driver's final step validates the store: every expected block present, no duplicate `episode_seed` across all shards, and `steps.groupby(episode_seed).size() == length + 1` for every episode.

#### D11. Budget at the 10⁶-episodes-per-run target

##### The one fact that governs the whole budget

**Only float columns cost real bytes.** Integers, booleans, positions, timers, counters, the episode key, and the step index are collapsed by Parquet's encodings to a small fraction of their raw size — a genuinely constant column measures **211× smaller than raw binary** (§D4.1). Varying low-cardinality columns are cheap but not free.

**Floats barely compress, and smoothness does not help.** Measured on a realistic observation block — `192,000 rows × 54 values`, autocorrelated within episode, values in `[0,1]`, `fixed_size_list`, zstd — i.e. exactly the schema above:

| | size | bytes / value | vs raw |
|---|---:|---:|---:|
| raw `float32`, uncompressed | 41.47 MB | 4.00 | — |
| Parquet `float32` + zstd | 33.76 MB | **3.26** | 1.23× |
| Parquet `float16` + zstd | 17.37 MB | **1.68** | 2.38× |

**`float16` saves 48.6 % against `float32`** on this block (1.94× smaller). A pure-noise control gave 52.9 %, confirming that autocorrelation in real sensor data buys essentially nothing.

The mechanism is worth stating because it is counter-intuitive: `float32` gets its 1.23× almost entirely from its redundant exponent byte; `float16` has already had that redundancy removed by the cast and so compresses hardly at all. That erodes the 2.00× raw-size advantage of half precision — but only to 1.94×. It does not cancel it.

Consequence: since the user chose to record **both** the noised observation and the noise-free ground truth (54 float values per step, 27 dimensions each), **the observation block dominates the store and is the only lever that changes the bill.**

##### Per-step arithmetic

| Block | Content | `float32` obs | `float16` obs | Basis |
|---|---|---:|---:|---|
| Observations | `obs_noised[27]` + `obs_true[27]` = 54 values | **176.0** | **90.7** | measured: 3.26 / 1.68 B per value |
| Other floats | `reward` 1, body + `damage` 5, `animal_stamina` 4 = 10 `float32` values | 32.6 | 32.6 | 3.26 B per value |
| Non-float | `episode_seed`, `t`, `action`, agent, `rest_streak`, 8 bools, `termination_reason`, animals, resources, obstacles — 186 raw B | ~17 | ~17 | estimate, see note |
| **Compressed bytes / step row** | | **≈ 226** | **≈ 140** | |

*Non-float note (an estimate, not a measurement — C8 measures it).* Built bottom-up in raw-binary terms: obstacles 88 raw B at the measured 211× → 0.4 B; `episode_seed` is 193 identical consecutive values and `t` is a perfect ramp, both ~0 under RLE / delta encoding; 8 mostly-`False` bools ~0.4 B; `action` (6 values) ~0.4 B; the varying low-cardinality integer block (agent, animals, resources, timers, counters, 79 raw B) at a conservative ~4–8× → ~15 B. Total ~17 B. This is the one figure in the table not backed by a direct measurement, and it is small enough that a 2× error moves the per-run total by under 8 %.

| | `float32` observations | `float16` observations |
|---|---:|---:|
| Compressed bytes / step row | 226 | 140 |
| Rows per episode (`T + 1`) | 193 | 193 |
| Compressed KB / episode (steps) | 43.6 | 27.0 |
| Compressed KB / episode (episode row)¹ | 1.34 | 1.34 |
| **Per run (10⁶ episodes)** | **≈ 45 GB** | **≈ 28 GB** |
| **10 runs** | **≈ 450 GB** | **≈ 285 GB** |

¹ The episode row is 407 `float32` values (dominated by the obstacle property columns, 286 of them) plus 119 non-float raw bytes → ~1.34 KB compressed. Per-episode floats stay `float32` regardless of `obs_precision` (§D4.2). Where obstacle-property std is zero these columns are *constant across every episode* and compress far better; 1.34 KB is a ceiling.

**Store-level saving from half precision: ~38 %** (45 GB → 28 GB). This is lower than the 48.6 % measured on the observation block alone, because the per-episode record and the non-observation columns do not change. 38 % is the number the precision decision is actually made on, and it clears the pre-registered 20 % threshold comfortably (§D12).

##### How the two earlier estimates in this plan were both wrong

Recorded because the failure mode is reusable, not to assign blame — **both errors were denominator errors, in opposite directions, and both survived because a ratio was quoted without its denominator.**

| Claim | Error | Correct |
|---|---|---|
| "floats compress 1.9×" → `4 ÷ 1.9 = 2.1` compressed B/value → half precision saves ~2 % | The 1.9× was measured **against CSV text**, not raw binary (same denominator as the 213× / 104× / 13× column figures). Dividing a CSV-denominated ratio into a raw binary byte count is meaningless. | `float32` is **3.26** compressed B/value (1.23× vs raw) |
| "~113 GB (`float32`) vs ~23 GB (`float16`)" → half precision saves ~79 % | That benchmark cast **every** float column, including `float64` ones, down to `float16`, inflating the apparent gain. | the real saving is **48.6 %** on the observation block, **~38 %** store-wide |

The directional intuition that `float32`'s redundant exponent byte is what compresses was correct — `float32` gets 1.23× while `float16` gets essentially none — it simply erodes half precision's 2.00× raw advantage to 1.94× rather than cancelling it.

**Rule for this document and any successor: never quote a compression ratio without naming its denominator, and never mix CSV-denominated and raw-denominated figures in one calculation.**

##### Disk is not a constraint — do not scope around it

`/media/nas01` has **59 TB free** of 192 TB (70 % used). The largest figure above is ~450 GB for ten runs at full precision — **under 1 % of free space**. No design decision in this plan may be justified by saving disk, and no scope reduction may be proposed for disk reasons. Pre-flight `df -h /media/nas01` anyway, because the NAS is shared.

The two real constraints are:

1. **Write throughput** during collection (bounded by rollout compute, §D6, not by bytes).
2. **Full-corpus read time** for every future analysis. This is the standing cost: a store that is ~38 % smaller is scanned ~38 % faster, forever, by every question anyone asks it. This — not disk — is the argument that carries the precision decision in §D12.

##### Data-loss exposure and the protective rules

The store is **gitignored data on a NAS that does not support symlinks**, so the standard "keep data outside the repo and symlink it in" protection is unavailable. The `results/` tree has already been destroyed once, when an aggressive cleanup followed a failed merge, and recovery required re-training. At ~28 GB and ~19 core-hours per run, this store is expensive — though not catastrophic — to regenerate.

Rules that apply to `results/trajectories/`, unchanged from the project-wide git-safety policy:

- **Never** `git clean -x` / `-X` / `-fdx` / `-fdX` — the `-x`/`-X` flag deletes gitignored files. `git clean -fd` is safe; always preview with `git clean -fdn` first and surface the listed paths.
- **Never** force-checkout or force-switch branches without checking whether the destination branch tracks paths currently untracked locally.
- **Avoid** `git stash -u` followed by `git stash drop`.
- **Snapshot before any merge / rebase / branch switch / non-trivial git operation.** `git reset --hard` alone is safe for gitignored data; the danger is the `git clean -x` that often follows it.

One mitigation is already built in: because the store is content-addressed by block (§D10) and every block is a pure function of `(seed_base, shard_episodes)`, **a partial loss is recoverable by re-running only the missing shards.** Losing 20 of 200 shards costs 10 % of one run's compute, not the whole collection. Losing `_manifest.json` alone is *not* recoverable in the same way — it carries the resolved config, `seed_base`, and `obs_precision` — so the driver writes it first, before any shard, and never rewrites it.

##### File count, not bytes, is what this filesystem punishes

An independent, decisive datapoint: `du -sh results/` on this NAS **timed out after two minutes**. Merely *walking* the existing tree exceeds two minutes. The binding cost on this filesystem is **file count**, not total bytes.

This is a second, independent argument for the sharded layout, separate from size: the alternative of one file per episode would create **10⁶ files per run, 10⁷ across ten runs**, on a filesystem that already cannot walk its own results tree in two minutes. The chosen layout produces:

| | files per run | files, 10 runs |
|---|---:|---:|
| One file per episode (rejected) | 1,000,000 | 10,000,000 |
| **Sharded, 5,000 episodes/block** | **400** (200 blocks × 2) | **4,000** |

400 files per run is a directory listing that returns instantly. It also sets a floor on shard size: do not reduce `shard_episodes` below ~1,000 without re-checking this, because file count is the scarce resource here.

##### Runtime

19.4 core-hours per run (§D6) → 194 core-hours for 10 runs → **≈ 1.5 h wall on 8 nodes × 16 processes**. Plus a fixed 15–17 s JAX-startup cost per worker process; with one process per block that would be `200 × 16 s = 53 min` of pure startup per run, so **each worker process must handle multiple consecutive blocks** (a worklist line is a *block range*, not a single block), amortising startup to ~16 s per worker. This is the direct analogue of `run_sweep.py`'s per-checkpoint grouping and is required, not optional.

#### D12. Observation precision — an explicit lossy decision, not a compression setting

**These are two different things and the plan must not conflate them.**

| | What it does | Reversible? |
|---|---|---|
| Compression (zstd + Parquet encodings) | makes the same values occupy fewer bytes | **lossless and exact** — round-trip is bit-identical |
| `float32 → float16` | **discards information** | **irreversible** |

Storing observations at half precision is therefore a **scientific decision about acceptable measurement error**, not a storage optimisation, and it is recorded as one.

**The error bound, verified:**

| Property | `float32` | `float16` |
|---|---|---|
| Decimal digits retained | ~7.2 | **~3.3** |
| Worst-case absolute error on values in `[0, 1]` | ~6e-08 | **2.44e-04** |
| Round-trip bit-identical | yes | **no** |
| Overflow threshold | ~3.4e38 | **65,504** |

##### Decision: `float16`, because the quantisation error sits ~1000× below the signal's own noise floor

**Default: `obs_precision: float16`.** Two independent reasons, and the second is the stronger one.

*Size.* Half precision saves **48.6 %** on the observation block and **~38 %** store-wide (§D11), clearing the 20 % threshold this plan pre-registered before the measurement was taken. That is a permanent ~38 % reduction in the scan cost of every future analysis.

*The error is not merely "acceptable" — it is far below the resolution the data actually carries.* The environment deliberately injects perceptual noise, and the per-modality standard deviations in the representative training config (§A9) are:

| Modality | injected noise σ |
|---|---:|
| Olfaction | 0.20 |
| Visual | 0.20 |
| Satiation, interoceptive nociception, extero nociception | 0.10 |
| Proprioception | 0.05 |
| Collision, location | 0.01 |

The `float16` quantisation error is `2.44e-04`. Against the dominant sensor channels that is **~400–800× below the injected noise**, and against the quietest channel (σ = 0.01) still **~40× below**. For the *noised* observation the quantisation is therefore invisible beneath noise the environment itself added on purpose. For the *noise-free* ground-truth observation there is no injected noise to hide under, but `2.44e-04` remains far below any meaningful resolution of a sensor whose inputs are grid positions and bounded property vectors.

**Precision loss is a non-issue here, and the ratio above is why.** This is a recorded judgement with a stated basis, not an assumption.

##### The real hazard is RANGE, not precision — enforced as a runtime guard

`float16` fails badly rather than gracefully when values leave its range: it overflows to `inf` above **65,504**, and underflows to subnormals below **6.1e-05**. Most observation channels are bounded in `[0,1]`, but **not all are guaranteed to be** — the location sensor emits raw grid coordinates, and any future unbounded channel would be silently mangled. (Note that *relative* precision is constant at ~4.9e-04 across the whole normal range, so gradual precision loss with magnitude is not the risk; range is.)

**A store that silently clipped an out-of-range channel is worse than a store that is twice as large.** This is therefore promoted from a test case to a **hard runtime guard that fails the collection**, not a post-hoc check:

- **Where**: in the collector, after the scan and the host flatten, **before** the shard is written. Because shards are written by atomic rename (§D3), a raised guard leaves nothing on disk.
- **When**: every chunk, whenever `obs_precision == float16`. Skipped entirely for `float32` apart from the finiteness assertion.
- **What it asserts**, on the `float32` observations the scan already produced:
  1. `isfinite(obs_f16).all()` — catches overflow to `inf`.
  2. `max|obs_f32| ≤ OBS_ABS_MAX`, a stated module constant set to **1e4** — comfortably above any bounded sensor, comfortably below the 65,504 overflow cliff, so a later chunk cannot creep over the edge unnoticed.
  3. `max( |obs_f32 − float32(float16(obs_f32))| / maximum(|obs_f32|, 1e-6) ) ≤ 1e-3` — the round-trip relative error, which additionally catches subnormal underflow.
- **On failure**: raise with a message naming the offending **observation index**, its value, and — resolved through `get_observation_breakdown(params)` — the **sensor name** that index belongs to. The operator learns "dimension 22 (Visual) exceeded the safe range at 1.2e5", not "assertion failed".
- **Cost**: two reductions over the chunk's observation array (~1.4×10⁷ elements at `batch_size = 1024`), against a ~72 s scan. Negligible.

Verification V5 tests that this guard **actually fires** — see the Verification Plan — rather than merely testing fidelity on data that happens to be in range.

**Design (unchanged, and it is what makes this a one-line flip rather than a redesign):**

- **`obs_precision` is a mandatory spec key** — `float16` or `float32`, read through `_req(spec, 'obs_precision')`, **no fallback default** (project Configuration Protocol). A collection cannot be launched without someone stating which precision they chose, because the choice is lossy.
- The chosen value is written to `_manifest.json` and is the element type of step columns 35–36. **An analysis can always tell which precision it is reading.**
- `obs_precision` is a manifest-guarded field (§D3): resuming a `float16` store with `float32` is a hard `ValueError`, so no store can end up half one and half the other.
- The reader is precision-agnostic by construction. Switching precision is a **spec-file edit, not a code change**.

#### D13. Seed policy — shared by default, overridable per run

**Default: one `seed_base` shared by every run in a batch spec.** Because an episode is a pure function of `jax.random.PRNGKey(seed)`, run A's episode `i` and run B's episode `i` then face the **same** environment draw — the same number of predators, the same sight ranges, the same bush layout. Comparisons across runs become **paired**, which is a large gain in statistical power for exactly the question this pipeline exists to answer ("what did this training change do?") and costs nothing.

**Per-run override is permitted and sometimes required.** Pairing is a property of the environment, not of the seed. Where two runs' environments differ *structurally*, the same key does not produce the same draw, and assuming pairing would be worse than not having it — an illusory pairing invites an analyst to run a paired test on unpaired data. The spec therefore accepts a `seed_base` on any individual run entry, overriding the batch-level value.

**The precise condition under which pairing holds** — this must appear in the schema doc, stated exactly, because an analyst cannot infer it:

> Two runs' episode `i` face the same environment draw **iff** they share a `seed_base` **and** the environment parameters consumed by `jax_reset` are identical between them — the entity slot counts (`count_low` / `count_high` for resources, entities, and obstacles) and every per-episode sampling bound (`animal_detect_low/high`, `animal_move_int_low/high`, `animal_attack_delay_low/high`, `animal_attack_range_low/high`, the four float-uniform trait bounds, the property `std` arrays, and the spawn areas). Changing any of these changes what the same key draws.
>
> Changes that do **not** break pairing: anything the reset sampler does not read — agent architecture, learning rates, training length, reward shaping, `max_steps`, and any body-dynamics parameter (including the healing rate) that is applied during stepping rather than at reset.
>
> **How to check, rather than assume**: `env_fp` (the resolved-config fingerprint in each store's manifest, §D3) being equal is sufficient for pairing. It is stricter than necessary — it also changes on parameters that do not affect the draw — so when fingerprints differ, compare the `seed_base` and the sampling-bound block recorded in the two manifests before claiming or denying pairing. **Never assume pairing from run labels.**

Mechanics: the effective `seed_base` is resolved per run (run-level value if present, else the batch-level value), written into that run's `_manifest.json`, and **guarded** — resuming a store with a different `seed_base` is a hard `ValueError` (§D3, verification V7 case (b)), so a store can never contain a mixed episode population.

A concrete example of when the override is needed: comparing a training run whose world has up to 2 predators against one whose world has up to 4. The animal slot count differs, so `jax_reset` consumes the key differently and episode `i` is not a matched pair. Give the second run its own `seed_base` and analyse the two as independent samples.

### File Changes

#### NEW — `src/utils/trajectory_store.py`

The single source of truth for the schema, plus the writer and the reader. Nothing else in the codebase may define these column names.

```python
SCHEMA_VERSION = 1

STEP_COLUMNS = [...]      # ordered list of (name, arrow_type_factory, doc) — §D4.1, exact order
EPISODE_COLUMNS = [...]   # ordered list — §D4.2, exact order

def build_step_schema(dims, obs_precision) -> pa.Schema: ...   # dims = (A, R, B, V, VV, D)
def build_episode_schema(dims) -> pa.Schema: ...   # per-episode floats are always float32 (§D4.2)

def env_fingerprint(resolved_cfg_dict) -> str:     # sha256(yaml.safe_dump(sort_keys=True))[:10]
def write_manifest(store_dir, manifest: dict) -> None:
def read_manifest(store_dir) -> dict:
def assert_manifest_compatible(store_dir, expected: dict) -> None:   # hard ValueError on mismatch

def write_shard_atomic(path: Path, table: pa.Table) -> None:         # .tmp + os.replace
def completed_blocks(store_dir) -> set[int]:

# Reader — the API every future analysis uses. Must work for every run, forever.
def open_store(store_dir) -> TrajectoryStore:      # validates SCHEMA_VERSION
class TrajectoryStore:
    manifest: dict
    def episodes(self, columns=None, filter=None): ...   # pyarrow.dataset over episodes_*.parquet
    def steps(self, columns=None, filter=None): ...      # pyarrow.dataset over steps_*.parquet
    def reshape(self, col, arr): ...                     # flattened [A,V] -> (n, A, V)
    def entity_labels(self): ...                         # index -> (name, class, behaviour)
```

Mandatory-key discipline: `read_manifest` and `assert_manifest_compatible` access every field via a `_req(d, key)` helper that raises `ValueError` on absence — no `.get(..., default)` anywhere in this module.

#### NEW — `scripts/eval/traj_collect/traj_scan.py`

The `nnx.jit`-wrapped batched scan kernel emitting exactly the fixed key set. Deliberately **separate** from `eval_rollout.py`'s `_rollout_scan_jit` so the dwell-sweep pipeline is untouched (out of scope). Contains:

- `_rollout_scan(model, params, states0, h0, max_steps)` — the scan body, **entered only via `nnx.jit`** (§A5).
- `_agent_in_bush(state, params)` — the reset-row helper, transcribing `core.py:793-797` exactly; lives here, **not** in `src/environment/`.
- `_reset_row(states0, params)` — builds the `t = 0` row group (§D2).
- `_prng_parity_guard(params, keys, states0)` — the `lax.map`-based, one-sync-per-chunk guard replacing `eval_rollout.py:405-415`.
- `_flatten_to_rows(scan_out, reset_rows, T)` — fully vectorised NumPy flatten + downcast to the schema dtypes, replacing the Python loop at `eval_rollout.py:456-475`.
- `OBS_ABS_MAX = 1e4` and `assert_obs_representable(obs_f32, obs_precision, params)` — the **hard runtime range guard** of §D12. Called once per chunk, after the flatten and **before** the shard write. Raises `ValueError` naming the offending observation index, its value, and the sensor it belongs to (resolved via `get_observation_breakdown(params)`). A raise leaves no shard on disk, because writes are atomic-rename. Do **not** downgrade this to a warning and do **not** move it after the write.

#### NEW — `scripts/eval/traj_collect/collect_trajectories.py`

Single-run, single-process collector. Flags:

`--run` (required, path to the training run dir) · `--checkpoint` (`final` or an explicit step; numeric-max selection per §D9) · `--out-root` · `--episodes` · `--seed-base` · `--blocks` (`lo:hi` block range for this worker) · `--batch-size` · `--shard-episodes` · `--obs-precision {float16,float32}` (**required, no default** — §D12) · `--device {cpu,gpu}` · `--quiet`

Flow: resolve checkpoint → load env from `<run>/models/config.yaml` (never from `configs/`) → compute `env_fp` → create-or-validate the store manifest → for each incomplete block in range: for each chunk of `batch_size`: `vmap(jax_reset)` → PRNG parity guard → `nnx.jit` scan → vectorised flatten → **`assert_obs_representable` range guard (§D12)** → accumulate → write both shards atomically. Policy is deterministic argmax, always.

Checkpoint-loading and rollout are kept behind two seams — `load_policy(agent_type, ckpt) -> (model, initial_state_fn)` and `policy_step(model, obs, h) -> (action, h)` — so Dreamer-SRL can be added later. **rPPO is the only supported and tested algorithm in this change**; `agent_type != "rppo"` raises `NotImplementedError` with a pointer to this section. No Dreamer support is claimed.

#### NEW — `scripts/eval/traj_collect/collect_worker.sh`

Per-node worker, modelled on `scripts/eval/dwell_sweep/sweep_worker.sh`. Carries the CPU thread caps (`sweep_worker.sh:29-31`), the per-node persistent XLA compile cache (`:38-40`), `xargs -P npar` over worklist lines, and `_run_markers/{done,fail,prog}_<node>`. Each worklist line is `RUN|CKPT|OUT_ROOT|SEED_BASE|BLOCK_LO:BLOCK_HI|DEVICE|BATCH_SIZE` — a **block range**, so JAX startup is amortised (§D11).

#### NEW — `scripts/eval/traj_collect/run_collection.py`

Multi-node driver. Reads the spec YAML, expands `(run × checkpoint × block-range)` cells, LPT-partitions across nodes, writes per-node worklists, launches **serially** via `run_command.py`, polls `done_<node>` markers, then runs the store validation of §D10 and prints a per-run summary (episodes collected, shards, bytes on disk, realised compression ratio).

#### NEW — `configs/trajectory_collection/example.yaml`

Spec template, mirroring `configs/eval_sweeps/*.yaml`:

```yaml
name: example
algo: rppo                 # MANDATORY — only 'rppo' is supported in this change
out_root: results/trajectories   # MANDATORY
episodes: 1000000          # MANDATORY — per run
seed_base: 1000000         # MANDATORY — shared by every run below so comparisons are PAIRED (§D13);
                           #             defines the episode population; never defaulted
checkpoints: [final]       # MANDATORY — list form, accepts explicit steps later
nodes: [101, 103, 104, 105]  # MANDATORY
obs_precision: float16     # MANDATORY — 'float16' or 'float32'; a LOSSY choice, never defaulted (§D12).
                           #             float16 saves ~38% store-wide; its 2.44e-04 error sits ~400x
                           #             below the environment's own injected sensor noise.
device: cpu                # optional, default 'cpu'
npar: 16                   # optional, default derived from device (§D7)
batch_size: 1024           # optional, default derived from device (§D7)
shard_episodes: 5000       # optional, default 5000 — do NOT go below ~1000 (file-count floor, §D11)
runs:                      # MANDATORY
  # Runs sharing the batch-level seed_base are PAIRED with each other (§D13).
  - {label: a01, path: results/JAX_RecurrentPPO/20260816-151827_rppo_restpremNH_a01_n106}
  - {label: a02, path: results/JAX_RecurrentPPO/20260816-151930_rppo_restpremNH_a02_n106}
  # Per-run override: this run's world has a different predator slot count, so the same
  # key does not produce the same draw and pairing would be illusory. Independent sample.
  - {label: b01, path: results/JAX_RecurrentPPO/20260816-152028_rppo_other_n107,
     seed_base: 5000000}
```

**No fallback defaults for scientific parameters.** `algo`, `out_root`, `episodes`, `seed_base`, `checkpoints`, `nodes`, `obs_precision`, `runs` are read through a `_req(spec, key)` helper that raises `ValueError` on absence. `obs_precision` is mandatory specifically because it is **lossy** (§D12) — a collection must not be launchable without someone stating the measurement precision they accepted. A run-level `seed_base` is the **only** permitted per-run override (§D13); every other key is batch-level. The four operational keys have documented defaults, and **every resolved value — defaulted or not, including the per-run effective `seed_base` — is written into `_manifest.json`**, so the value actually used is never in doubt.

**No environment-config schema change.** This pipeline adds no keys to `configs/` env YAMLs and does not touch `config_loader.py`, `state.py` `EnvParams`, or the config system. Therefore `docs/environment/CONFIG_GUIDE.md`, `docs/environment/02_config_schema.md`, and `docs/environment/CONFIG_CRITICAL_SETTINGS.md` require **no** update, and no critical-settings change-log entry is due. (Stated explicitly so the verifier can confirm the omission is deliberate.)

#### NEW — `docs/environment/TRAJECTORY_STORE_SCHEMA.md`

**A first-class deliverable, not an afterthought.** The permanent contract for everyone who ever reads this store. Contents:

1. Plain-language entry point: what the store is, what one row means, what question it answers.
2. The row convention (§D2) stated in one sentence, with a worked three-step example table.
3. The complete fixed key list (§D4.1, §D4.2) with dtype, shape, timing, and source, verbatim.
4. Path scheme and manifest schema (§D3), including how `env_fp` prevents overwrites.
5. Worked reader snippets: load episodes, join steps to episodes on `episode_seed`, reshape a flattened `[A, V]` draw, select "all predators", compute per-episode bush-dwell fraction.
6. **When cross-run pairing holds** — the §D13 condition reproduced verbatim, including the "check `env_fp` and the sampling bounds, never assume from run labels" instruction. This is the one thing an analyst is most likely to get wrong and cannot infer from the data.
7. Caveats, in a section titled **Known caveats — read before analysing**:
   - **Resource properties are an episode-level approximation.** `res_property_sampled_init` is the draw *at reset*; the environment re-draws it on every regeneration (`core.py:808-809`). An analysis treating food properties as constant within an episode is making an approximation. **Where it breaks**: any episode in which a resource was consumed and regenerated — detectable per step from `res_cons_count` incrementing and `res_active` toggling `False → True`. The approximation is exact for the window before the first regeneration, and degrades with the number of regenerations, so it is worst in long episodes with a short `res_reg_delay` and best in short ones. Analyses that condition on resource property should either restrict to the pre-first-regeneration window or report the regeneration count as a covariate.
   - The `agent_in_bush` comparability warning (§D8).
   - **Observation precision**: how to read `obs_precision` from the manifest; the `2.44e-04` worst-case absolute error if it says `float16`, stated against the environment's own injected noise (σ = 0.01–0.20, so the quantisation is ~40–800× below it); and the fact that a collection-time range guard hard-fails rather than silently clipping, so an out-of-range channel cannot be sitting in the store unnoticed (§D12).
   - `animal_damage` is per-step, not per-episode (§A2) — only its bounds are in the manifest.
8. A short **"why the schema looks like this"** note carrying the measurements: constant integer / boolean columns are effectively free (211× vs raw binary), floats cost **3.26** compressed bytes per value at `float32` and **1.68** at `float16` regardless of smoothness, hoisting static entity positions was measured at a **0.75 %** net saving and rejected, and **a compression ratio is meaningless without its denominator** — mixing CSV-denominated and raw-denominated figures is what produced two wrong answers during this plan's drafting. This section exists to stop the next reader from "optimising" the schema.
9. **Maintenance Contract**: any change to the fixed key set, any dtype change, and any row-convention change **must** bump `SCHEMA_VERSION` in `src/utils/trajectory_store.py` and update this document in the same commit. Readers hard-fail on an unknown `SCHEMA_VERSION`.

#### NEW — `scripts/eval/traj_collect/README.md`

Operator doc: how to run one run locally, how to run a spec across nodes, how to resume, how to validate, the CPU/GPU flag interaction (§D7), and the `run_command.py` serial-launch constraint.

#### NEW — `tests/test_trajectory_collection.py`

Pytest home for verifications V1, V3, V4, V5, V7 and V9 (see Verification Plan). Fast variants (small episode counts, tiny configs) so the suite stays runnable. V2, V6 and V8 are operator-run rather than pytest (they need a real checkpoint, a `SIGKILL`, and a lab node respectively).

#### MODIFIED — `docs/environment/SCRIPTS_DEPENDENCY_MAP.md`

**Required by that document's Maintenance Contract, in the same change.** Add entries for the four new files under `scripts/`:

| File | Called by | Calls |
|---|---|---|
| `scripts/eval/traj_collect/run_collection.py` | operator (CLI), spec YAMLs in `configs/trajectory_collection/` | `run_command.py`, `collect_worker.sh` |
| `scripts/eval/traj_collect/collect_worker.sh` | `run_collection.py` via `run_command.py` | `collect_trajectories.py` |
| `scripts/eval/traj_collect/collect_trajectories.py` | `collect_worker.sh`, operator (CLI), `tests/test_trajectory_collection.py` | `traj_scan.py`, `src/utils/trajectory_store.py`, `src/environment/*` |
| `scripts/eval/traj_collect/traj_scan.py` | `collect_trajectories.py`, tests | `src/environment/core.py`, `src/environment/sensor.py` |

Also note the `sys.path` repo-root depth for the new directory: `scripts/eval/traj_collect/` is **three** levels below the repo root (matching `scripts/eval/dwell_sweep/`, cf. `sweep_worker.sh:23`).

#### MODIFIED — `docs/develop/INDEX.md`

Regenerated via `python scripts/claude/regen_dev_index.py`. Never hand-edited.

#### NOT MODIFIED (explicit)

`scripts/eval/eval_rollout.py`, `scripts/eval/dwell_sweep/*`, `src/utils/evaluation_core.py`, `src/utils/eval_recording.py`, `scripts/behavior_measures/*`, and everything under `src/environment/`. The existing dwell-sweep pipeline is untouched by this change.

---

## Checkpoints

Verified by the implementing agent **during** implementation:

- [ ] **C1 — `nnx.jit` entry.** Assert the scan is never called eagerly: add a `RuntimeError` if `_rollout_scan` is invoked outside a trace, and confirm the first forward pass of the process goes through `nnx.jit` (§A5). Print the first 5 actions of seed 0 and compare to the legacy path before writing any shard.
- [ ] **C2 — Numeric checkpoint selection.** Print the resolved checkpoint directory for a run with 591 numerically-named dirs and confirm it is `59100070`, not a lexicographic winner (§D9).
- [ ] **C3 — Schema round-trip.** Write a 10-episode block, read it back with `open_store`, and assert column names + order + types match `build_step_schema(dims, obs_precision)` exactly. Run once per `obs_precision` value.
- [ ] **C4 — Row counts.** For 10 episodes assert `len(steps[seed]) == episodes[seed].length + 1`, `steps[t=0].action == -1`, `steps[t=0].reward == 0.0`.
- [ ] **C5 — Realised draws are constant within an episode.** Assert every episode-level draw read at reset equals the same field on the final state (they must be, per `core.py:818-828`) — a cheap in-loop guard that catches a wrong state being snapshotted.
- [ ] **C6 — Peak RSS.** Measure peak RSS of one worker at `batch_size=1024` and confirm ≤ 2.5 GB (§D5 predicts ~1.8 GB). Report the number.
- [ ] **C7 — Throughput.** Time one 5,000-episode block; report episodes/s and compare against the 14.3 eps/s baseline of §A8. Report as a before/after speed number in the Implementation Report.
- [ ] **C8 — Confirm the precision saving on real sensor data.** *(Confirmation, not a gate — the default is already `float16`, §D12.)* Write **the same 5,000 real episodes twice**, once with `obs_precision: float32` and once with `float16`, and report for each: on-disk bytes for the whole shard, on-disk bytes for the two observation columns alone, and compressed bytes per stored float value.
  - Expected from the synthetic benchmark (§D11): **3.26 B/value** at `float32`, **1.68 B/value** at `float16`, **48.6 %** saving on the observation block, **~38 %** store-wide. This checkpoint confirms those hold on real sensor data rather than on a synthetic autocorrelated walk.
  - **Escalate rather than silently proceed** if the store-wide saving comes in **below 20 %** — that would fall under the pre-registered adoption threshold and the default should revert to `float32`. Report the number either way; do not rationalise a near-miss.
  - Also report the realised non-float bytes/row against the ~17 B estimate of §D11, which is the one unmeasured figure in the budget.
  - Cost: ~12 minutes of compute. Do not substitute an estimate.
- [ ] **C8b — The `float16` range guard fires.** Confirm the runtime guard (§D12) raises, names the offending observation index, and resolves it to a sensor name — see V5. Confirm no shard is left on disk after the raise.
- [ ] **C9 — Resume is a no-op.** Run the same block range twice; assert the second run writes nothing and completes in under 30 s.
- [ ] **C10 — Zero-slot environment.** Collect 20 episodes from a config with `A = 0` (no animals) and confirm the animal columns are present as zero-length lists and the reader does not branch.

---

## Verification Plan

Every check below can **fail**, and none of them validates a code path using that same code path.

### V1 — Realised draws against an independent replay *(the primary correctness check)*

An episode is a pure function of `jax.random.PRNGKey(seed)`, so ground truth is freely available. Sample 200 recorded episodes at random. For each, take the recorded `episode_seed`, call `jax_reset(params, jax.random.PRNGKey(seed))` **unbatched, un-vmapped, outside any scan** — a genuinely different execution path from the batched collector — and assert **exact** equality (bitwise for ints/bools, `==` for floats, since it is the same computation) for all 19 episode-level draw columns.

**Fails on**: seed-to-row misassociation (the realistic failure mode — an off-by-one in block indexing or in the vectorised flatten), a wrong reset key derivation, or reading the draws from the wrong lane of the vmap batch.

### V2 — Trajectory parity against the legacy per-episode loop

For 30 seeds, run the existing legacy `_run_episode` path (Python `while` loop, unbatched `jax_step`, no vmap, no scan) on the same checkpoint and config. Assert the recorded `action` sequence and `(agent_row, agent_col)` sequence match **exactly** for all `T` steps, and `length` matches.

**Fails on**: vmap axis errors, the eager-nnx staleness bug of §A5, and any row-convention shift. This is the strongest possible statement that the fast path and the slow path describe the same agent.

### V3 — Row-convention self-consistency

For every sampled episode assert: `steps[0].t == 0`, `steps[0].action == -1`, `steps[0].reward == 0.0`; `len(steps) == length + 1`; `steps[t].termination_reason == 0` for all `t < length`; `steps[length].termination_reason == episodes.termination_reason != 0`; and `steps[length].terminated == True`.

**Fails on**: any off-by-one between the reset row and the scan rows, or a shard concatenated in the wrong order.

### V4 — `agent_in_bush` recomputed independently in NumPy

For every step row with `t ≥ 1`, recompute in pure NumPy from the **per-step** obstacle position columns (`obs_row`, `obs_col` — columns 33–34), the per-episode `obs_active` mask, and the manifest's `obs_hides_agent`:

```
expected = any( (obs_row == agent_row) & (obs_col == agent_col) & obs_hides_agent & obs_active )
```

and assert it equals the recorded `agent_in_bush`.

**Fails on**: accidentally inheriting the slot-0 hardcode of `avoidance_stats_heatmap.py:77-79`, a wrong `obs_active` mask, or a mis-transcribed reset-row helper. This is a NumPy reimplementation with no shared code with the JAX environment.

*(An earlier draft carried a further test asserting that obstacle positions never move, because the store depended on hoisting them to episode level. That dependency is gone — positions are recorded per step (§D4.1) — so the test has been dropped rather than kept as dead weight. If moving obstacles are ever added, this store records them correctly and V4 keeps passing.)*

### V5 — Observation precision: the range guard fires, and fidelity holds

Two halves. The second is the one that matters, because it tests the *guard* rather than the happy path.

**(a) Fidelity on in-range data.** Replay a sample of states, recompute `get_observation` at `float32`, compare against the store. When the manifest says `float16`, assert `max |obs_f32 − obs_stored| ≤ 2.44e-04` on values in `[0,1]`. When it says `float32`, assert a **bit-identical** round-trip — compression is lossless, so anything else is a writer bug.

**(b) The runtime guard actually raises (§D12).** Drive an out-of-range observation and assert the collection **fails** rather than silently clipping:

- Run with a config whose **location sensor is enabled on a large grid**, so a real channel carries large magnitudes. The representative config of §A9 has it off, so this must be a separate config.
- Additionally inject a synthetic out-of-range value (monkeypatch one observation element above `OBS_ABS_MAX`) to exercise the guard deterministically, independent of whether any real config currently exceeds it.
- Assert: the collector raises; the message names the **observation index** and the resolved **sensor name**; and **no shard file exists on disk** afterwards (the atomic-rename property, §D3).

**Fails on**: a missing or mis-thresholded guard, a guard that warns instead of raising, a guard placed after the shard write, an unhelpful error message, or a `float32` store that is not bit-exact (which would mean an unintended downcast in the writer). Part (b) is the check that stands between us and a store that silently mangled a channel — the failure mode that would be undetectable at read time and would invalidate every analysis touching that sensor.

### V6 — Resume and atomicity under a hard kill

Start a 3-block collection; `SIGKILL` the process partway through block 2; restart. Assert: (a) no `.parquet.tmp` files remain, (b) the final store contains exactly the expected episode count with **no duplicate `episode_seed`**, (c) `steps.groupby(episode_seed).size() == length + 1` holds for every episode, and (d) blocks 0 and 1 are byte-identical to a clean run.

**Fails on**: non-atomic writes, a resume that re-does or skips a block, or a seed mapping that depends on process state rather than on `(seed_base, shard_episodes)`.

### V7 — Overwrite-hazard guard *(direct regression test for the 2026-07-04 incident)*

Collect 100 episodes into a store. Then attempt to resume **the same store path** with (a) a mutated env config, (b) a different `seed_base`, (c) a bumped `SCHEMA_VERSION`, and (d) a different `obs_precision`. Assert each raises `ValueError` and that **no file in the store is modified** (compare mtimes and hashes before/after).

**Fails on**: a missing or weak manifest guard. Case (d) additionally guarantees no store can end up half `float16` and half `float32` — a silent mixed-precision corpus would be undetectable at read time without it. This is the check that the 2026-07-04 contamination cannot recur.

### V8 — Scale and speed

Time one 5,000-episode block on a lab node. Assert throughput ≥ 10 episodes/s/process (against the 14.3 eps/s baseline of §A8; ≥ 10 allows for the extra recording payload) and peak RSS ≤ 2.5 GB.

**Fails on**: the O(n) Python bottlenecks not actually being removed, or a chunking bug that materialises more than one chunk at a time. Per the project's speed-review rule, a > 15 % throughput regression against the baseline is a blocker unless explicitly accepted here — and it is **not** accepted here.

### V9 — Schema invariance across environments *(the hard requirement's direct test)*

Collect 200 episodes from **three** structurally different saved configs: (i) a probe config with one predator, (ii) a training config with four animal slots, (iii) a config with the location sensor enabled and different `V` / `VV`. Assert the Parquet column **names and order are byte-identical** across all three, that only the `fixed_size_list` widths differ, and that **one** reader function loads all three without branching.

**Fails on**: any conditional column creeping in — the exact failure mode that makes the existing CSV unusable (§A1).

---

## Out of Scope (explicit)

- **Any analysis layer.** No aggregation scripts, no plots, no measures, no regression of behaviour on environment factors. The deliverable stops at the documented store. Analyses are written ad hoc afterwards.
- **Dreamer-SRL support.** The checkpoint-loading and rollout seams are kept generic so Dreamer drops in later, but Dreamer is **not** implemented, **not** tested, and **not** claimed. `agent_type != "rppo"` raises.
- **Any change to the existing dwell-sweep pipeline** (`scripts/eval/dwell_sweep/*`, `scripts/eval/eval_rollout.py`, `scripts/behavior_measures/*`). The `bush_dwell` defect of §A7 is documented and avoided in the new store, not fixed in the old pipeline.
- **Reward decomposition and per-step distance columns.** Excluded by design, with per-field reasons in §D4.1.
- **Deleting the dead `build_episode_log_dict`** (`src/behavior/accumulators.py:546`). Noted as pre-existing dead code; removal is a separate change.

---

## Decisions Taken

All four questions this plan opened have been decided by the user. Recorded here so the reasoning survives; the design sections above already reflect them.

| # | Question | Decision | Where it lives |
|---:|---|---|---|
| 1 | Record per-step `damage`? | **Yes, include it.** It is not reward decomposition but the physical transition quantity that, with the healing rate, determines `injury_level` — without it, "got hit less" and "healed faster" are not separable, and healing rate is a named analysis target. ~4 B/step. | §D4.1 column 13 |
| 2 | Resource properties: reset draw only, or per step? | **Reset draw only (`_init`).** Do not pay the extra payload. The caveat is prominent in the schema doc, regeneration stays detectable per step from the consumption counters, and the doc states explicitly that treating food properties as an episode constant is an approximation and where it breaks. | §D4.2 cols 19–20; schema doc §7 |
| 3 | `seed_base` shared across runs, or per run? | **Shared by default, overridable per run.** Shared gives paired comparisons at zero cost; the override exists for runs whose environment differs structurally, where the same key does not produce the same draw and the pairing would be illusory rather than real. Effective value recorded per run in the manifest and guarded on resume. | §D13 |
| 4 | Store root? | **`results/trajectories/`.** Disk is not a constraint; the data-loss exposure and the protective rules are stated, along with the fact that the manifest-plus-resume design makes a partial loss recoverable by re-running only the missing shards. | §D11 "Data-loss exposure" |
| 5 | Observation precision — `float16` or `float32`? | **`float16`.** Settled by a compressed-vs-compressed measurement: 48.6 % saving on the observation block, ~38 % store-wide, clearing the 20 % threshold pre-registered before the measurement was taken. The `2.44e-04` quantisation error sits ~400–800× below the noise the environment already injects into the dominant sensor channels. The real hazard is **range**, not precision, and it is now a hard runtime guard rather than a post-hoc test. | §D12, §D11 |

## Open Questions

**None outstanding.** All five questions this plan opened have been decided, the last of them (precision) by direct measurement after both of the initial estimates turned out to be wrong in opposite directions — a history preserved in §D11 because the underlying trap (quoting a compression ratio without its denominator) is reusable.

Checkpoint C8 remains in the plan as a **confirmation on real sensor data** of a decision already taken on a synthetic benchmark, with an explicit escalation rule if it disagrees. It does not gate the start of implementation.

---

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the implementing agent. Must include:
     - Measured before/after throughput (C7) and peak RSS (C6) on the same node/config.
     - Realised Parquet compression ratio (C8).
     - Any deviation from the fixed key list in §D4, with justification.
     - The C8 paired precision measurement and which precision was adopted (§D12).
     - Which of V1-V9 were run and their outcomes. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]

---

## Feedback from plan-reviewer

> **Date**: 2026-08-19 · **Verdict**: **NOT READY** — two Critical findings, both cheap to fix. Full review: [[plan_trajectory_collection]] (`docs/reviews/plan_trajectory_collection.md`).

1. **🔴 F1 — training-world faithfulness is assumed, never verified.** §A10 declares the saved resolved config the source of truth, but the collector re-loads it through **today's** `load_env_params`, whose legacy-scene precedence (`config_loader.py:429-435`) can rebuild, for pre-`828b77e` runs whose dump carries both scene formats, the scene the trainer *discarded*. V1 and V2 both consume the same `params`, so both are circular with respect to config loading. Required: a hard-fail guard when a saved config contains both a non-empty legacy scene block and an `entities:` block; a documented applicability boundary (runs trained after 2026-07-23); run creation date recorded in the manifest.
2. **🔴 F2 — the red reset-parity gate is unacknowledged.** `tests/env/test_unified_parity.py` fails at step 0 on a clean tree (KNOWN_BUGS.md:73, twice-confirmed, unowned) — standing evidence that env reset behaviour drifted at least once. Triage it before the first production collection, and add the code-drift caveat (training-time git SHA is unrecorded) to the schema doc.
3. **🟡 F3–F6**: strict checkpoint-restore structural check (silent-unmodulated-agent + model-size-flag bugs); a doc↔code schema check plus one bare-pyarrow read (C3 is circular through the shared schema module); a named pilot→validate→scale sequence with V1/V3/V4 run on the *production* store; a whole-store bounds-and-variation check on the realised-draw columns.
4. **🟢 F7–F10**: §D7's GPU scan-buffer figure should be ~2.4 GB, not ~1.8 GB; §D4.1 col 13's "Open Question 1" pointer is stale (damage was decided); pre-state the V1 float-equality tolerance policy; state the explicit conda interpreter in `collect_worker.sh` and label the bush-dwell snippet's output a 0–1 fraction.

Known-bug hazards 2 (fixed-seed repetition), 3 (stale-data blends), and 4 (derived-measure smuggling) are genuinely closed by the design as written.

*Reviewed by: plan-reviewer*
