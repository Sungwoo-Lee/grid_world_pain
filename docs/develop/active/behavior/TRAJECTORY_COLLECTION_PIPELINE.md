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

`_manifest.json` records: `schema_version`, `env_fp`, the full resolved env config, `run_path`, `checkpoint_path`, `ckpt_step`, slot counts `(A, R, B, V, VV, D)`, `max_steps`, `action_dim`, `seed_base`, `n_episodes`, `shard_episodes`, `batch_size`, `device`, `policy_mode` (always `"deterministic_argmax"`), `obs_dtype`, `git_sha`, the per-entity **names and classes** (`animal_classes`, `animal_behaviours`, resource / obstacle names) so array index `i` can be given a human label, and the static per-entity parameter arrays needed as join partners: `animal_damage` bounds, `obs_hides_agent`, `res_type`, `animal_is_damaging`, and the per-episode sampling bounds (`animal_detect_low/high`, `count_low/high`, …).

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
| 33 | `obs_noised` | `fixed_size_list<float16>[D]` | state at `t` | `get_observation(state, params)` — what the policy received |
| 34 | `obs_true` | `fixed_size_list<float16>[D]` | state at `t` | `get_observation(state, params, apply_noise=False)` — ground truth |

**No obstacle columns appear per step.** Obstacle positions never change: `obs_pos` is absent from the `state._replace(...)` call at `core.py:800-838` and is therefore carried through every step unmodified. They are hoisted to the per-episode record. This assumption is load-bearing for a 31 % size saving and is guarded by a permanent regression test (verification V5).

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
| 22 | `obs_row` | `fixed_size_list<int16>[B]` | hoisted static obstacle positions |
| 23 | `obs_col` | `fixed_size_list<int16>[B]` | hoisted static obstacle positions |
| 24 | `obs_property_sampled` | `fixed_size_list<float32>[B*V]` | **realised draw**, flattened `[B, V]` |
| 25 | `obs_visual_property_sampled` | `fixed_size_list<float32>[B*VV]` | **realised draw**, flattened `[B, VV]` |

Rows 7–25 are the complete independent-variable side of the analysis: everything the environment secretly re-rolled at the start of this episode. Joined against the manifest's sampling bounds (§A10), each realised draw can be expressed as a position within its own range.

**Caveat on columns 19–20 (must be in the schema doc).** `res_property_sampled` and `res_visual_property_sampled` are re-drawn whenever a resource regenerates (`core.py:808-809`). The recorded values are the **reset draw only**. A regeneration event is detectable per step from `res_cons_count` incrementing and `res_active` toggling, so an analysis that conditions on resource property should either restrict to the pre-first-regeneration window or accept the approximation. Recording them per step would cost `R*V + R*VV` half-precision values per step, roughly **+16 %** payload; deferred, see **Open Question 2**.

**Flattening note.** `[A, V]`-shaped draws are stored flattened row-major into a single list column so that the column count is fixed and independent of `V`. `(A, V)` are in the manifest; the reader reshapes.

**Human labels.** Array index `i` is meaningless without a name. The manifest carries `animal_classes` / `animal_behaviours` (e.g. `("predator","predator","neutral","neutral")` and `("hunt","hunt","wander","wander")`), resource names, and obstacle names, so an analysis can select "all predators" or "all bushes" the way `select_by_class` (`src/environment/state.py:7`) does today.

#### D5. Chunking and the RAM arithmetic

The scan kernel emits, per environment-step, native-dtype device arrays (JAX has no `int8`/`float16` output here — downcast happens on host):

| Group | Bytes / env-step |
|---|---:|
| agent + body scalars + info scalars + flags | 53 |
| animals (`pos` 32, `state` 16, `stamina` 16, `move_timer` 16, `attack_timer` 16) | 96 |
| resources (`pos` 32, `active` 4, `cons_count` 16, `reg_timer` 16) | 68 |
| observations (`obs` 27×4, `obs_true` 27×4) | 216 |
| **Total** | **≈ 433** |

Scan output for a chunk of `B_c` episodes: `max_steps × B_c × 433 B = 216.5 KB × B_c`. The device buffer and its host copy coexist during transfer, so peak ≈ 2×.

| `B_c` | scan buffer | device + host peak | process peak RSS (base 1.2 GB) |
|---:|---:|---:|---:|
| 1024 | 222 MB | 444 MB | **≈ 1.7 GB** |
| 2048 | 443 MB | 886 MB | ≈ 2.2 GB |
| 4096 | 887 MB | 1.77 GB | ≈ 3.1 GB |
| 102400 (unchunked, illustrative) | 22 GB | 44 GB | OOM |

**Chosen: `batch_size = 1024`.** At 16 worker processes per node this is ~27 GB of node RAM, comfortable on any lab node. On CPU, "device memory" *is* host memory, so the 2× factor is real, not conservative. GPU changes the trade-off — see §D7.

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

- Episode index space is partitioned into contiguous blocks of `shard_episodes` (5,000). Block `b` covers `[5000b, 5000(b+1))`; episode `i` uses seed `seed_base + i`. Both the block partition and the seed mapping are pure functions of `(seed_base, shard_episodes)` recorded in the manifest — no state, no counter file.
- A block is **complete** iff both `episodes_%05d.parquet` and `steps_%05d.parquet` exist (atomic rename guarantees each is whole). Resume = list complete blocks, skip them, work the rest.
- A node dying mid-block loses at most one block ≈ **6 minutes** of work, and leaves at most two `.tmp` files, which the collector deletes on startup for blocks it is about to redo.
- Because blocks are pure functions of the seed, a redone block is **bit-identical** to what the dead process would have produced. Resume can never produce a mixed population.
- The driver's final step validates the store: every expected block present, no duplicate `episode_seed` across all shards, and `steps.groupby(episode_seed).size() == length + 1` for every episode.

#### D11. Budget at the 10⁶-episodes-per-run target

**Per-step raw payload** (post-downcast, the bytes actually handed to Parquet), using §A9 dimensions:

| Group | bytes/row |
|---|---:|
| `episode_seed` 8, `t` 2, `action` 1, `reward` 4 | 15 |
| agent 4, body (`satiation`/`nutrition`/`injury`) 12, `rest_streak` 2, `last_collision_noc` 4, `damage` 4 | 26 |
| 8 boolean flags + `termination_reason` | 9 |
| animals: row/col 16, state 4, stamina 16, timers 16 | 52 |
| resources: row/col 16, active 4, cons_count 8, reg_timer 8 | 36 |
| `obs_noised` `float16[27]` 54 + `obs_true` `float16[27]` 54 | 108 |
| **Total** | **246** |

| Quantity | Value |
|---|---|
| Rows per episode | `T + 1 ≈ 193` |
| Raw bytes per episode (steps) | `193 × 246 ≈ 47.5 KB` |
| Raw bytes per episode (episode row) | `≈ 1.84 KB` |
| **Raw per 10⁶ episodes** | **≈ 47.5 GB steps + 1.8 GB episodes ≈ 49 GB** |
| Parquet + zstd, expected 3–5× | **≈ 10–16 GB per run** |
| **10 runs** | **≈ 100–160 GB** |
| Shards per run | `10⁶ / 5000 = 200` blocks → 400 files |
| Per shard on disk | ≈ 60–80 MB |

The 3–5× compression estimate is well-founded for this data: eight boolean columns and `termination_reason` are near-constant (RLE collapses them to nearly nothing); positions are small integers with cardinality ≤ grid size (dictionary-encoded); proprioception occupies 6 of the 27 observation dimensions as a one-hot; and the per-episode obstacle property columns (columns 24–25, 1,144 raw bytes and by far the largest part of the episode row) are literally constant across all episodes whenever the obstacle property std is zero. Checkpoint C8 requires the developer to **measure** the realised ratio on a 5,000-episode block and record it, rather than trusting this estimate.

**Where the size optimisations came from** — both are already baked into the schema above:

| Optimisation | Mechanism | Saving |
|---|---|---|
| Hoist static obstacle data to the episode record | `obs_pos` never mutates (`core.py:800-838`); `obs_active` is constant per episode (`core.py:828`) | per-step payload would be `246 + 22×2×2 + 22 = 356 B`; hoisting saves **31 %** |
| Half-precision observations | `obs_noised` + `obs_true` at `float16` instead of `float32` | `246` vs `354 B`; saves **31 %** |
| Both together | | `47.5 KB` vs `≈ 88 KB` per episode — **46 %** |

Half-precision is safe for the sensors in §A9 (all values are normalised or small integers) but is **not safe unconditionally** — the `Location` sensor emits raw grid coordinates, and any future sensor emitting values above 65,504 would overflow to `inf`. Verification V6 is a real check, not a formality.

**Runtime**: 19.4 core-hours per run (§D6) → 194 core-hours for 10 runs → **≈ 1.5 h wall on 8 nodes × 16 processes**. Plus a fixed 15–17 s JAX-startup cost per worker process; with one process per block that would be `200 × 16 s = 53 min` of pure startup per run, so **each worker process must handle multiple consecutive blocks** (worklist line = a *block range*, not a single block), amortising startup to ~16 s per worker. This is the direct analogue of `run_sweep.py`'s per-checkpoint grouping and is required, not optional.

**Disk**: `/media/nas01` currently has **59 TB free** of 192 TB. 100–160 GB is not a constraint. Pre-flight `df -h /media/nas01` anyway.

### File Changes

#### NEW — `src/utils/trajectory_store.py`

The single source of truth for the schema, plus the writer and the reader. Nothing else in the codebase may define these column names.

```python
SCHEMA_VERSION = 1

STEP_COLUMNS = [...]      # ordered list of (name, arrow_type_factory, doc) — §D4.1, exact order
EPISODE_COLUMNS = [...]   # ordered list — §D4.2, exact order

def build_step_schema(dims) -> pa.Schema: ...      # dims = (A, R, B, V, VV, D)
def build_episode_schema(dims) -> pa.Schema: ...

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

#### NEW — `scripts/eval/traj_collect/collect_trajectories.py`

Single-run, single-process collector. Flags:

`--run` (required, path to the training run dir) · `--checkpoint` (`final` or an explicit step; numeric-max selection per §D9) · `--out-root` · `--episodes` · `--seed-base` · `--blocks` (`lo:hi` block range for this worker) · `--batch-size` · `--shard-episodes` · `--device {cpu,gpu}` · `--quiet`

Flow: resolve checkpoint → load env from `<run>/models/config.yaml` (never from `configs/`) → compute `env_fp` → create-or-validate the store manifest → for each incomplete block in range: for each chunk of `batch_size`: `vmap(jax_reset)` → parity guard → `nnx.jit` scan → vectorised flatten → accumulate → write both shards atomically. Policy is deterministic argmax, always.

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
seed_base: 1000000         # MANDATORY — defines the episode population; never defaulted
checkpoints: [final]       # MANDATORY — list form, accepts explicit steps later
nodes: [101, 103, 104, 105]  # MANDATORY
device: cpu                # optional, default 'cpu'
npar: 16                   # optional, default derived from device (§D7)
batch_size: 1024           # optional, default derived from device (§D7)
shard_episodes: 5000       # optional, default 5000
runs:                      # MANDATORY
  - {label: a01, path: results/JAX_RecurrentPPO/20260816-151827_rppo_restpremNH_a01_n106}
```

**No fallback defaults for scientific parameters.** `algo`, `out_root`, `episodes`, `seed_base`, `checkpoints`, `nodes`, `runs` are read through a `_req(spec, key)` helper that raises `ValueError` on absence. The five operational keys have documented defaults, and **every resolved value — defaulted or not — is written into `_manifest.json`**, so the value actually used is never in doubt.

**No environment-config schema change.** This pipeline adds no keys to `configs/` env YAMLs and does not touch `config_loader.py`, `state.py` `EnvParams`, or the config system. Therefore `docs/environment/CONFIG_GUIDE.md`, `docs/environment/02_config_schema.md`, and `docs/environment/CONFIG_CRITICAL_SETTINGS.md` require **no** update, and no critical-settings change-log entry is due. (Stated explicitly so the verifier can confirm the omission is deliberate.)

#### NEW — `docs/environment/TRAJECTORY_STORE_SCHEMA.md`

**A first-class deliverable, not an afterthought.** The permanent contract for everyone who ever reads this store. Contents:

1. Plain-language entry point: what the store is, what one row means, what question it answers.
2. The row convention (§D2) stated in one sentence, with a worked three-step example table.
3. The complete fixed key list (§D4.1, §D4.2) with dtype, shape, timing, and source, verbatim.
4. Path scheme and manifest schema (§D3), including how `env_fp` prevents overwrites.
5. Worked reader snippets: load episodes, join steps to episodes on `episode_seed`, reshape a flattened `[A, V]` draw, select "all predators", compute per-episode bush-dwell fraction.
6. Caveats, in a section titled **Known caveats — read before analysing**: the `res_property` regeneration re-draw (§D4.2), the `agent_in_bush` comparability warning (§D8), half-precision observation error bounds (V6), and `animal_damage` being per-step rather than per-episode (§A2).
7. **Maintenance Contract**: any change to the fixed key set, any dtype change, and any row-convention change **must** bump `SCHEMA_VERSION` in `src/utils/trajectory_store.py` and update this document in the same commit. Readers hard-fail on an unknown `SCHEMA_VERSION`.

#### NEW — `scripts/eval/traj_collect/README.md`

Operator doc: how to run one run locally, how to run a spec across nodes, how to resume, how to validate, the CPU/GPU flag interaction (§D7), and the `run_command.py` serial-launch constraint.

#### NEW — `tests/test_trajectory_collection.py`

Pytest home for verifications V1, V3, V4, V5, V6, V8, V10 (see Verification Plan). Fast variants (small episode counts, tiny configs) so the suite stays runnable.

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
- [ ] **C3 — Schema round-trip.** Write a 10-episode block, read it back with `open_store`, and assert column names + order + types match `build_step_schema(dims)` exactly.
- [ ] **C4 — Row counts.** For 10 episodes assert `len(steps[seed]) == episodes[seed].length + 1`, `steps[t=0].action == -1`, `steps[t=0].reward == 0.0`.
- [ ] **C5 — Realised draws are constant within an episode.** Assert every episode-level draw read at reset equals the same field on the final state (they must be, per `core.py:818-828`) — a cheap in-loop guard that catches a wrong state being snapshotted.
- [ ] **C6 — Peak RSS.** Measure peak RSS of one worker at `batch_size=1024` and confirm ≤ 2.5 GB (§D5 predicts ~1.7 GB). Report the number.
- [ ] **C7 — Throughput.** Time one 5,000-episode block; report episodes/s and compare against the 14.3 eps/s baseline of §A8. Report as a before/after speed number in the Implementation Report.
- [ ] **C8 — Realised compression ratio.** Report raw payload bytes vs on-disk Parquet bytes for one shard, and extrapolate to 10⁶ episodes. If the ratio is worse than 2.5×, stop and report before collecting at scale (§D11).
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

For every step row with `t ≥ 1`, recompute in pure NumPy from the recorded per-episode obstacle positions, `obs_active`, and the manifest's `obs_hides_agent`:

```
expected = any( (obs_row == agent_row) & (obs_col == agent_col) & obs_hides_agent & obs_active )
```

and assert it equals the recorded `agent_in_bush`.

**Fails on**: accidentally inheriting the slot-0 hardcode of `avoidance_stats_heatmap.py:77-79`, a wrong `obs_active` hoist, or obstacles that actually do move (which would also trip V5). This is a NumPy reimplementation with no shared code with the JAX environment.

### V5 — The static-obstacle assumption *(permanent regression test)*

For 20 seeds across **three structurally different** configs, run a direct env loop recording `state.obs_pos` at **every** step, and assert it is identical to the reset value throughout.

**Fails on**: anyone adding moving obstacles in the future — which would silently corrupt every store written under the hoisting optimisation (§D11). This test is the tripwire that makes a 31 % size saving safe to depend on. It must be permanent, not a one-off.

### V6 — Half-precision observation fidelity

Replay a sample of states and recompute `get_observation` at `float32`. Assert `max |obs_f32 − obs_f16| / (|obs_f32| + 1e-6) < 1e-3` and, separately, `isfinite(obs_f16).all()`.

**Fails on**: any sensor emitting values outside `float16`'s useful range — in particular the `Location` sensor on a large grid, or any future sensor emitting magnitudes above 65,504, which would silently become `inf`. Run this on at least one config with the location sensor **enabled**, since the representative config of §A9 has it off.

### V7 — Resume and atomicity under a hard kill

Start a 3-block collection; `SIGKILL` the process partway through block 2; restart. Assert: (a) no `.parquet.tmp` files remain, (b) the final store contains exactly the expected episode count with **no duplicate `episode_seed`**, (c) `steps.groupby(episode_seed).size() == length + 1` holds for every episode, and (d) blocks 0 and 1 are byte-identical to a clean run.

**Fails on**: non-atomic writes, a resume that re-does or skips a block, or a seed mapping that depends on process state rather than on `(seed_base, shard_episodes)`.

### V8 — Overwrite-hazard guard *(direct regression test for the 2026-07-04 incident)*

Collect 100 episodes into a store. Then attempt to resume **the same store path** with (a) a mutated env config, (b) a different `seed_base`, and (c) a bumped `SCHEMA_VERSION`. Assert each raises `ValueError` and that **no file in the store is modified** (compare mtimes and hashes before/after).

**Fails on**: a missing or weak manifest guard. This is the check that the 2026-07-04 contamination cannot recur.

### V9 — Scale and speed

Time one 5,000-episode block on a lab node. Assert throughput ≥ 10 episodes/s/process (against the 14.3 eps/s baseline of §A8; ≥ 10 allows for the extra recording payload) and peak RSS ≤ 2.5 GB.

**Fails on**: the O(n) Python bottlenecks not actually being removed, or a chunking bug that materialises more than one chunk at a time. Per the project's speed-review rule, a > 15 % throughput regression against the baseline is a blocker unless explicitly accepted here — and it is **not** accepted here.

### V10 — Schema invariance across environments *(the hard requirement's direct test)*

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

## Open Questions

1. **`damage` (per-step column 13).** This goes one field beyond the two additions selected. It is not reward decomposition — it is the physical per-step damage that, together with the healing rate, determines `injury_level`. Without it, "the agent got hit less" and "the agent healed faster" are not separable, and the healing-rate parameter is one of the named analysis targets. Cost: 4 bytes/step, ~1.6 % of payload. **Recommend keeping.** To drop it, delete row 13 from §D4.1 and from `STEP_COLUMNS` — a one-line change.
2. **Resource property re-draw on regeneration (§D4.2 columns 19–20).** The plan records the reset draw only. Recording per step costs ~+16 % payload. Acceptable as-is, or upgrade before collecting at scale?
3. **`seed_base` policy across runs.** Should every run share one `seed_base` (so run A's episode *i* and run B's episode *i* face the **same** environment draw — a paired comparison, much higher statistical power for "what did this training change do?"), or should each run get a distinct base (independent samples)? A shared base is strictly more informative and costs nothing. **Recommend a single project-wide `seed_base` for any set of runs intended to be compared**, recorded in the spec. Needs a decision before the first collection, because it cannot be changed afterwards without recollecting.
4. **Store root placement.** `results/trajectories/` on the NAS. 100–160 GB against 59 TB free — no constraint, but it is gitignored data subject to the project's git-safety rules. Confirm this root, or name another.

---

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the implementing agent. Must include:
     - Measured before/after throughput (C7) and peak RSS (C6) on the same node/config.
     - Realised Parquet compression ratio (C8).
     - Any deviation from the fixed key list in §D4, with justification.
     - Which of V1-V10 were run and their outcomes. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: [one-line summary]
