---
title: "Trajectory Collection Pipeline — large-scale in-training-environment rollout store"
topic: behavior
status: active
created: 2026-08-19
last_updated: 2026-08-20
phase: null
aliases: [trajectory_collection_pipeline]
---

# Trajectory Collection Pipeline

> **Status**: IMPLEMENTED (2026-08-20) — uncommitted, awaiting verification
> **Opened**: 2026-08-19
> **Review**: [[plan_trajectory_collection]] (`docs/reviews/plan_trajectory_collection.md`) — NOT READY (2 Critical). See **Review Response** below for the disposition of every finding.
> **Schema contract**: [`docs/environment/TRAJECTORY_STORE_SCHEMA.md`](../../../environment/TRAJECTORY_STORE_SCHEMA.md)
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

### A10. The saved config is the correct source of truth — but reloading it is not free of assumptions

`train.py:881-883` writes the **fully resolved** config to `<run>/models/config.yaml` — no `extends:` survives. It carries the per-episode sampling *bounds* (`detection_range: [1,7]`, `count_low` / `count_high`, and so on), which are the join partner for the realised draws recorded per episode. Loading the environment from this file rather than from `configs/` is what makes the pipeline honest about which world the agent actually trained in, per the project's "verify actual state, not a re-derivation" rule.

That is necessary but **not sufficient**. Reading the right file still means re-running it through *today's* loader, and the loader's behaviour has changed since some of these files were written. §A11 is the consequence.

### A11. The faithfulness hazard — reloading a saved config can rebuild the world the trainer discarded

**This is the assumption the entire tool rests on, and it is the one assumption no check inside the pipeline can test.** Stated plainly: the collector rebuilds the environment by feeding the run's saved config to `load_env_params`. If that rebuild does not reproduce the world the agent actually trained in, the pipeline records a million perfectly-formed episodes of the agent in the wrong world, fingerprints that wrong world in the manifest as if it were the truth, and produces no internal signal that anything is amiss.

**The mechanism is real and already documented.** `src/environment/config_loader.py:428-435` gives a non-empty legacy scene block precedence over the modern one:

```python
has_entities = config.get('environment.entities') is not None
has_legacy   = bool(config.get('environment.predators')) or \
               bool(config.get('environment.neutral_animals'))

if has_entities and not has_legacy:
    # modern `entities:` path
```

So when a saved config carries **both** formats, today's loader takes the legacy branch. Before commit `828b77e` (2026-07-23, "fix(config): legacy-scene precedence"), `train.py` resolved the same file the *other* way — it merged in the base config's newer scene and silently discarded the older-format list (KNOWN_BUGS registry, "Config layer trained a different scene than it evaluated"). For such a run, **reloading rebuilds the scene the trainer threw away.** The animals differ, the slot count differs, and every recorded episode is of a different world.

**Why the plan's own checks cannot catch it — the circularity.** V1 (realised draws vs. independent replay) and V2 (trajectory parity vs. the legacy loop) both consume the *same* `params` object the collector built. Both would pass, in full agreement, describing the wrong world consistently. A check that shares the suspected-broken step with the thing it validates returns the assumption as proof. Neither check is weak; they are simply blind to this class of error by construction.

**Measured scope — the boundary is a fact, not an assumption.** Scanning every saved config in the results tree (334 configs, runs from 2026-04-20 to 2026-08-16):

| Saved-config scene shape | Count | Reload risk |
|---|---:|---|
| `entities:` only | 153 | none — unambiguous |
| legacy `predators:` / `neutral_animals:` only | 151 | none — unambiguous, both old and new loaders take the legacy path |
| **both formats present** | **12** | **ambiguous — precedence decides, and it changed** |
| neither (no animals in the world) | 18 | none |

All 12 ambiguous runs date from **2026-05-29 to 2026-06-11**, comfortably before the 2026-07-23 fix; ten of them are throwaway `logcheck_*` runs. The full list, which doubles as the test corpus for the guard below:

```
20260529-212737_recurrent_ppo_04-sameProp_R4_chasingRabbit_s42
20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage_s42
20260611-150754_logcheck_ckpt100k_log200      20260611-151553_logcheck_ckpt250k_log500
20260611-150754_logcheck_ckpt250k_log500      20260611-152445_logcheck_ckpt100k_log200_2M
20260611-151157_logcheck_ckpt100k_log200      20260611-152456_logcheck_ckpt200k_log500_2M
20260611-151256_logcheck_ckpt250k_log500      20260611-152707_logcheck_ckpt200k_log500_2M
20260611-182431_logcheck_ckpt200k_log300_2M   20260611-182449_logcheck_ckpt200k_log400_2M
```

No run after 2026-06-11 carries both formats, so the hazard is **historical, not live** — but it is not self-announcing, and `train.py` records no training-time git SHA, so nothing in a run directory says which loader built its world.

**The fix — hard-fail, do not guess (§D14).** The coexistence of both blocks is precisely the fingerprint of an ambiguous dump, so the collector detects exactly that and refuses to run. This is a *detector for the ambiguity*, not a repair of the config: nothing in the run directory records which branch the trainer took, so there is no correct scene to reconstruct — only a choice, which is why the answer is to stop rather than pick. Design in §D14, guard test in V10.

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
| 13 | `damage` | `float32` | **arriving** (`0.0` at `t=0`) | `info['damage']` — included by decision, see **Decisions Taken #1** |
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
| `batch_size` | 1024 (host-RAM bound, §D5) | 8192 (GPU-memory bound; 8192 × 500 × 609 B ≈ **2.4 GB** scan buffer, still fitting an 11 GB 2080 Ti with headroom) |
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

#### D14. Scene-ambiguity guard, and the applicability boundary

Addresses the §A11 hazard. Three parts: refuse ambiguous runs, record what was actually used, state where the tool applies.

**(a) Hard-fail on scene-format coexistence.** Immediately after loading `<run>/models/config.yaml` and **before** building any params, the collector checks:

```
has_entities = bool(cfg['environment'].get('entities'))
has_legacy   = bool(cfg['environment'].get('predators')) or \
               bool(cfg['environment'].get('neutral_animals'))
if has_entities and has_legacy:  -> raise ValueError
```

The error must name the run, say that the saved config carries both a modern `entities:` block and a legacy `predators:` / `neutral_animals:` block, explain that the trainer's scene precedence changed at commit `828b77e` (2026-07-23) so it cannot be determined which scene this run actually trained on, and point at §A11. **It must not attempt to pick one.**

An escape hatch exists for someone who has independently established which scene is correct: `--allow-ambiguous-scene`, which downgrades the failure to a loud warning and sets `scene_ambiguous: true` in the manifest so every downstream reader inherits the caveat. Absent that flag, the collection stops.

**(b) Record scene provenance in `_manifest.json`.** A creation date alone is a weak proxy; record the facts directly:

| Manifest field | Value | Why |
|---|---|---|
| `scene_format` | `"entities"` \| `"legacy"` \| `"none"` | the branch the loader actually took — an audit reads it instead of re-deriving it |
| `scene_ambiguous` | bool | true only when `--allow-ambiguous-scene` overrode the guard |
| `run_dir_name` | e.g. `20260816-152742_rppo_…` | carries the run's creation date as the provenance proxy |
| `train_config_mtime` | ISO timestamp of `<run>/models/config.yaml` | independent of directory naming convention |
| `collection_git_sha` | git SHA at collection time | **note the asymmetry** — see §D15 |

**(c) Applicability boundary — state it, do not bury it.** In this plan, in the collector's `--help`, and in the schema doc:

> **Runs trained after 2026-07-23 (commit `828b77e`) are unambiguous** — the trainer and today's loader resolve the scene identically. **Runs before that date may not be**, and any run whose saved config carries both scene formats is not reconstructable with confidence and is refused by default. Measured against the current results tree, exactly 12 of 334 saved configs are affected, all dated 2026-05-29 to 2026-06-11 (§A11).

This boundary is what makes the "training-run agnostic" claim honest. The tool works on any run it accepts, and it refuses the runs it cannot faithfully reconstruct, rather than silently doing its best.

#### D15. Reset-time state is covered by a green gate — and code drift is unrecorded

Two things here: a **positive assurance** that was missing from earlier drafts, and a **residual gap** that survives it.

##### The reset-parity gate is essentially green (diagnosed 2026-08-19)

`tests/env/test_unified_parity.py` is the gate that would catch a regression in **reset-time state** — exactly what §D4.2 records as the realised per-episode draws and what §D2 records as row `t = 0`. It appeared in this plan's review as a Critical, on the strength of a Known-Bugs entry describing it as red, twice-confirmed and unowned.

It was then actually run, as a full sweep with no early exit. Result:

```
4 failed, 30 passed, 293 skipped   (344 s)

FAILED test_parity[configs__verification__observability_gates_S1]
FAILED test_parity[configs__verification__observability_gates_S2]
FAILED test_parity[configs__verification__observability_gates_S3]
FAILED test_parity[configs__verification__observability_gates_S4]
```

**30 of 34 executed reset-parity scenarios pass.** All four failures are the same scenario family and are byte-identical in form: `agent_pos` at step 0, actual `[4,4]` against fixture `[2,2]`. **No other scenario fails.**

Diagnosed from git, and the answer is benign:

| | |
|---|---|
| Fixtures generated | **2026-05-28**, commit `3d20aab` ("generate CP1 parity fixtures — 31 loadable configs at HEAD") |
| Configs changed | **2026-07-04**, commit `84014e4` ("fix(config): observability gates use fixed start pos (no step-0 contact) — Finding G2") |
| What changed | all four `configs/verification/observability_gates_S{1..4}.yaml` now declare `start_pos: [5, 5]` with `random_start_pos: false` |
| What did not | the four fixtures still record `step000_agent_pos = [2 2]` |

So a **deliberate July config change** — itself a fix for a real bug where the agent could begin an episode already touching an entity — moved the start position in all four gate configs, and the May fixtures were never regenerated. **The environment reset code did not drift.** The test compares today's environment against a snapshot predating an intentional change.

**The four failures are exactly the four configs that change touched, with zero unexplained residue.** That completeness is what makes the diagnosis strong: a partial or scattered failure set would leave room for a second, real regression hiding among stale fixtures. There is none.

**Consequence for this plan, and it is a positive assurance rather than a caveat:** reset-time state — precisely what §D4.2 records as the realised per-episode draws and what §D2 records as row `t = 0` — **is covered by a broadly green gate with no unexplained failures.** This is the strongest independent evidence the plan has that the foundation of its per-episode record is sound, and no earlier draft stated it. Severity **Low**. Fixture regeneration for those four is a `developer` task, not a design change, and is listed as a cheap Phase 0 precondition (§D16) so the gate is fully green before ten million episodes depend on it — not as a blocker on the design.

**Recorded here so nobody re-escalates it.** The commit pair above is the whole story; anyone encountering the red test again should read this section rather than re-triage it.

##### The residual that survives the diagnosis, and is worth more than the fixture fix

The substantive point does not go away just because this particular test turned out benign.

This failure sat in the bug registry as *"reset behaviour drifted, twice-confirmed, unowned"* long enough to surface as a **Critical finding in a plan review** — and the diagnosis that defused it took about four minutes. That is the lesson: **a red test nobody triages is operationally indistinguishable from a red test that matters.** You cannot tell a stale fixture from a live regression without spending those four minutes, and until someone does, every downstream plan has to treat it as the worse case.

The durable version of the concern is an **unstated assumption this pipeline makes**: that environment code is unchanged between a run's training and its collection. Nothing verifies it, and — as §A11 established for the scene format — `train.py` writes **no training-time git SHA** anywhere, so no recorded fact would ever reveal a difference.

**The mitigation is provenance, not more testing:**

1. **Record `collection_git_sha` in `_manifest.json`** (§D14b). The least this pipeline can do is not repeat on the collection side the omission that makes §A11 unresolvable on the training side.
2. Together with `run_dir_name` / `train_config_mtime` (also §D14b), a future analyst has **two bracketing facts** — when the run was trained, and at which commit it was collected — and can bound the code-drift question by inspecting history, instead of guessing.
3. **The schema doc's Known-caveats section states it plainly**: *this store records reset-time state under whatever environment code existed at collection time; the training-time code version is unrecorded; use `run_dir_name` and `collection_git_sha` to bound what may have changed in between.*
4. **Recommended as a separate follow-up outside this plan's scope**: have `train.py` write a training-time git SHA into the saved config, closing the gap permanently for future runs. Flagged for `senior-developer`; deliberately not bundled here, because it modifies the trainer and this plan touches no training code.

#### D16. Phased rollout — pilot before production, and the acceptance gate

Collection is cheap (~1.5 h wall for all ten runs) but a silently-wrong store is expensive, so the sequence is staged and each phase has an exit condition that can fail.

| Phase | Scope | Exit condition |
|---|---|---|
| **0 — preconditions** | No collection. (a) Regenerate the four stale `observability_gates_S{1..4}` parity fixtures so the gate goes from 30/34 to fully green (§D15) — `developer` owns this; (b) confirm the scene-ambiguity guard fires on the 12 known dual-format runs (V10); (c) `df -h /media/nas01`. | Parity suite green; V10 passes; disk confirmed |
| **1 — pilot** | **~25,000 episodes of ONE run, on ONE node.** ~5 blocks, ~10 minutes. | **V1–V5 all pass against the pilot store**, plus the driver's full validation (§D10) and the whole-store draw check below. C6/C7/C8/C8b measured and reported. |
| **2 — first full run** | One complete run at 10⁶ episodes. | Driver validation + whole-store draw check pass; realised size and wall-time within 2× of the §D11/§D6 predictions |
| **3 — remainder** | The other nine runs, packed node-first | Same checks per store |

**The sampled checks run against the PRODUCTION store, not only the pilot.** V1 (realised draws vs. independent replay), V3 (row convention) and V4 (`agent_in_bush` recomputation) are cheap sampled checks — a few hundred episodes out of 10⁶ — and they are **acceptance gates on every production store**, not one-time pre-flight tests. A pilot that passes proves the code works; it does not prove that a specific 10⁶-episode store is sound. Running them per store is the difference.

**Whole-store draw validation (in the driver's final pass, in addition to §D10's structural checks).** The realised draws are the entire point of the store, and V1 samples only ~200 episodes — a degenerate sampler, or a column accidentally wired to a constant, can slip through a sample. Columnar min/max/nunique over the whole `episodes` dataset is cheap (one pass, no step data) and catches it:

1. **Bounds**: every realised-draw column lies within the sampling bounds recorded in the manifest — e.g. every `animal_detect_sampled` value within `[animal_detect_low, animal_detect_high]`. Catches out-of-range wiring and unit errors.
2. **Non-degeneracy**: every realised-draw column has **more than one distinct value wherever the manifest's bounds satisfy `low < high`**. Catches a sampler stuck at a constant, a field silently defaulted, or a column wired to the wrong source. Conversely, where `low == high`, assert exactly one distinct value.
3. **Activation masks**: `animal_active` / `obs_active` / `res_allocated` population counts fall within `[count_low, count_high]`, and — where `count_low < count_high` — more than one distinct count occurs across the store.

A failure here is a hard stop: the store is deleted and the cause found, because a store that passes structure but fails these has *plausible-looking* independent variables, which is the worst possible failure mode for the analyses this exists to serve.

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

def write_shard_atomic(path: Path, table: pa.Table) -> None:
    # Write .tmp -> flush -> os.fsync(fd) -> os.replace -> fsync the DIRECTORY fd.
    # The fsyncs are NOT optional here: results/ is a CIFS mount (//192.168.0.250/
    # cocoanlab01), so POSIX rename atomicity + durability must NOT be assumed. The
    # rename alone survives a process SIGKILL; only the fsync pair also survives a
    # NODE crash, which is the realistic multi-node failure. See V6.
def completed_blocks(store_dir) -> set[int]:

def assert_scene_unambiguous(cfg: dict, run_path, allow_override: bool) -> str:
    # §D14a — raises ValueError when the saved config carries BOTH a modern
    # `entities:` block and a legacy `predators:`/`neutral_animals:` block.
    # Returns the resolved scene_format ("entities"|"legacy"|"none") for the manifest.
    # MUST NOT attempt to pick a scene. Error names the run, the two blocks, commit
    # 828b77e (2026-07-23), and points at §A11.

def assert_restored_tree_matches(restored_tree, model) -> None:
    # F3 — hard-fail if the orbax-restored tree and the built model disagree on ANY
    # key or ANY shape. Catches the silent-unmodulated-agent bug (a missing
    # modulation block builds a structurally different model that restores
    # "successfully") and non-persisted model-size flags. V2 cannot catch either:
    # both its paths rebuild from the same saved config and so agree with the same
    # wrong agent.

def validate_store_draws(store_dir) -> None:
    # §D16 — whole-store columnar bounds / non-degeneracy / activation-count checks.

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

Flow: resolve checkpoint → load `<run>/models/config.yaml` (never from `configs/`) → **`assert_scene_unambiguous` (§D14a)** → build params → compute `env_fp` → create-or-validate the store manifest → load policy → **`assert_restored_tree_matches` (F3)** → for each incomplete block in range: for each chunk of `batch_size`: `vmap(jax_reset)` → PRNG parity guard → `nnx.jit` scan → vectorised flatten → **`assert_obs_representable` range guard (§D12)** → accumulate → write both shards atomically. Policy is deterministic argmax, always.

Note the ordering: the scene guard runs **before** params are built, so an ambiguous run cannot get far enough to create a store directory.

Checkpoint-loading and rollout are kept behind two seams — `load_policy(agent_type, ckpt) -> (model, initial_state_fn)` and `policy_step(model, obs, h) -> (action, h)` — so Dreamer-SRL can be added later. **rPPO is the only supported and tested algorithm in this change**; `agent_type != "rppo"` raises `NotImplementedError` with a pointer to this section. No Dreamer support is claimed.

`load_policy` must call `assert_restored_tree_matches` before returning. A checkpoint that restores "successfully" into a structurally different model is the failure mode behind two recorded bugs (a missing modulation block silently yielding an unmodulated agent; model-size CLI flags possibly not persisted to the saved config), and a strict key+shape comparison catches both with one check.

#### NEW — `scripts/eval/traj_collect/collect_worker.sh`

Per-node worker, modelled on `scripts/eval/dwell_sweep/sweep_worker.sh`. **Invokes the conda interpreter by explicit absolute path — `PY=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` — never `conda run`, `conda activate`, or a bare `python`**, per the project-wide conda rule; the `sweep_worker.sh` precedent already complies (`:27`) and should be copied verbatim. Carries the CPU thread caps (`sweep_worker.sh:29-31`), the per-node persistent XLA compile cache (`:38-40`), `xargs -P npar` over worklist lines, and `_run_markers/{done,fail,prog}_<node>`. Each worklist line is `RUN|CKPT|OUT_ROOT|SEED_BASE|BLOCK_LO:BLOCK_HI|DEVICE|BATCH_SIZE` — a **block range**, so JAX startup is amortised (§D11).

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
3. The complete fixed key list (§D4.1, §D4.2) with dtype, shape, timing, and source. **These tables must be generated from `STEP_COLUMNS` / `EPISODE_COLUMNS`, not hand-transcribed** — a script (`scripts/eval/traj_collect/gen_schema_doc.py`) emits them between marker comments in the doc, and a test asserts the committed doc matches freshly-generated output. Hand-maintained schema tables rot, and this doc is half the deliverable; a doc that disagrees with the code is worse than no doc, because it is trusted.
4. Path scheme and manifest schema (§D3), including how `env_fp` prevents overwrites.
5. Worked reader snippets: load episodes, join steps to episodes on `episode_seed`, reshape a flattened `[A, V]` draw, select "all predators", and compute a per-episode bush-occupancy **fraction in `[0, 1]`** (mean of the boolean `agent_in_bush` over the episode's steps) — the snippet must state that unit explicitly and must not be called "dwell", given the retracted units claim on the existing `bush_dwell` measure (§D8).
6. **When cross-run pairing holds** — the §D13 condition reproduced verbatim, including the "check `env_fp` and the sampling bounds, never assume from run labels" instruction. This is the one thing an analyst is most likely to get wrong and cannot infer from the data.
7. Caveats, in a section titled **Known caveats — read before analysing**:
   - **Resource properties are an episode-level approximation.** `res_property_sampled_init` is the draw *at reset*; the environment re-draws it on every regeneration (`core.py:808-809`). An analysis treating food properties as constant within an episode is making an approximation. **Where it breaks**: any episode in which a resource was consumed and regenerated — detectable per step from `res_cons_count` incrementing and `res_active` toggling `False → True`. The approximation is exact for the window before the first regeneration, and degrades with the number of regenerations, so it is worst in long episodes with a short `res_reg_delay` and best in short ones. Analyses that condition on resource property should either restrict to the pre-first-regeneration window or report the regeneration count as a covariate.
   - The `agent_in_bush` comparability warning (§D8).
   - **Observation precision**: how to read `obs_precision` from the manifest; the `2.44e-04` worst-case absolute error if it says `float16`, stated against the environment's own injected noise (σ = 0.01–0.20, so the quantisation is ~40–800× below it); and the fact that a collection-time range guard hard-fails rather than silently clipping, so an out-of-range channel cannot be sitting in the store unnoticed (§D12).
   - `animal_damage` is per-step, not per-episode (§A2) — only its bounds are in the manifest.
   - **Applicability boundary (§D14c)**: runs trained after 2026-07-23 (commit `828b77e`) are unambiguous; earlier runs may not be, and any run whose saved config carries both scene formats is refused by the collector. Check `scene_format` and `scene_ambiguous` in the manifest before trusting a store built from an older run.
   - **Code drift (§D15)**: this store records reset-time state under whatever environment code existed **at collection time**; the training-time code version is unrecorded anywhere in the project. Use `collection_git_sha` together with `run_dir_name` / `train_config_mtime` to bound what may have changed in between. As of 2026-08-19 the reset-parity gate passed **30 of 34** executed scenarios; the four failures are all `observability_gates_S1`–`S4` and are fully explained by stale fixtures (generated `3d20aab`, 2026-05-28) predating a deliberate start-position change (`84014e4`, 2026-07-04) — not a code regression.
8. A short **"why the schema looks like this"** note carrying the measurements: constant integer / boolean columns are effectively free (211× vs raw binary), floats cost **3.26** compressed bytes per value at `float32` and **1.68** at `float16` regardless of smoothness, hoisting static entity positions was measured at a **0.75 %** net saving and rejected, and **a compression ratio is meaningless without its denominator** — mixing CSV-denominated and raw-denominated figures is what produced two wrong answers during this plan's drafting. This section exists to stop the next reader from "optimising" the schema.
9. **Maintenance Contract**: any change to the fixed key set, any dtype change, and any row-convention change **must** bump `SCHEMA_VERSION` in `src/utils/trajectory_store.py` and update this document in the same commit. Readers hard-fail on an unknown `SCHEMA_VERSION`.

#### NEW — `scripts/eval/traj_collect/README.md`

Operator doc: how to run one run locally, how to run a spec across nodes, how to resume, how to validate, the CPU/GPU flag interaction (§D7), and the `run_command.py` serial-launch constraint.

#### NEW — `tests/test_trajectory_collection.py`

Pytest home for verifications V1, V3, V4, V5, V7, V9 and **V10** (see Verification Plan). Fast variants (small episode counts, tiny configs) so the suite stays runnable. V10 is cheap and high-value — it runs against the 12 real dual-format configs already on disk and needs no rollout at all, since the guard fires before params are built. V2, V6 and V8 are operator-run rather than pytest (they need a real checkpoint, a NAS-hosted `SIGKILL`, and a lab node respectively).

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

- [x] **C0 — Guards fire before anything is written.** DONE — `assert_scene_unambiguous` raises on the real dual-format corpus with no store dir created (V10, 5 pytest cases); `assert_restored_tree_matches` raises on a wrong `hidden_size` and on a structurally different encoder, and passes on the correct model. Original: Confirm, in order: `assert_scene_unambiguous` raises on one of the 12 known dual-format runs with no store directory created (V10); `assert_restored_tree_matches` raises when a deliberately mismatched model is built against a checkpoint. Both must fail *loudly and early* — a guard that runs after side effects is not a guard.
- [x] **C1 — `nnx.jit` entry.** DONE — `_rollout_scan` raises `RuntimeError` when its args are concrete; first 5 actions of 5 seeds match the legacy unbatched loop exactly (see the report's V2 row). Original: Assert the scan is never called eagerly: add a `RuntimeError` if `_rollout_scan` is invoked outside a trace, and confirm the first forward pass of the process goes through `nnx.jit` (§A5). Print the first 5 actions of seed 0 and compare to the legacy path before writing any shard.
- [x] **C2 — Numeric checkpoint selection.** DONE — resolves `59100070` from 591 dirs; lexicographic max is `9900021`, and the test asserts the two disagree so it cannot go vacuous. Original: Print the resolved checkpoint directory for a run with 591 numerically-named dirs and confirm it is `59100070`, not a lexicographic winner (§D9).
- [x] **C3 — Schema round-trip.** DONE for both precisions, and labelled circular in the test itself. Original: Write a 10-episode block, read it back with `open_store`, and assert column names + order + types match `build_step_schema(dims, obs_precision)` exactly. Run once per `obs_precision` value. **Note this check is circular** (reader and writer share `build_step_schema`) — it catches wiring mistakes, not schema-definition mistakes. V9's bare-pyarrow read is the non-circular counterpart; do not treat C3 as sufficient.
- [x] **C4 — Row counts.** DONE (V3 test). Original: For 10 episodes assert `len(steps[seed]) == episodes[seed].length + 1`, `steps[t=0].action == -1`, `steps[t=0].reward == 0.0`.
- [x] **C5 — Realised draws are constant within an episode.** DONE — enforced in `run_chunk` on every chunk (reset block vs final-state block). Original: Assert every episode-level draw read at reset equals the same field on the final state (they must be, per `core.py:818-828`) — a cheap in-loop guard that catches a wrong state being snapshotted.
- [x] **C6 — SUPERSEDED (2026-08-20).** Measured **2,554 MiB** at `batch_size=1024`, `shard_episodes=5000`. The 2.5 GB figure was a self-imposed plan ceiling, not a hardware limit: actual node RAM is 125-128 GB (101/103/104/105), 257 GB (106), 515 GB (113), so `npar: 16` uses ~41 GB — 36 % of the smallest node. **No change made; the buffering is not restructured.** Original: Measure peak RSS of one worker at `batch_size=1024` and confirm ≤ 2.5 GB (§D5 predicts ~1.8 GB). Report the number.
- [x] **C7 — Throughput.** DONE — **47.9-49.6 episodes/s** for one 5,000-episode block, vs the 14.3 eps/s baseline of §A8 (3.4x faster). Original: Time one 5,000-episode block; report episodes/s and compare against the 14.3 eps/s baseline of §A8. Report as a before/after speed number in the Implementation Report.
- [x] **C8 — ESCALATED, AND DECIDED (2026-08-20).** Measured store-wide saving **12.8 %**, below the pre-registered 20 % threshold, so the recommended precision is now **`float32`** (spec template, CLI docstring and schema doc updated; there is still no code-level default anywhere). Original: *(Confirmation, not a gate — the default is already `float16`, §D12.)* Write **the same 5,000 real episodes twice**, once with `obs_precision: float32` and once with `float16`, and report for each: on-disk bytes for the whole shard, on-disk bytes for the two observation columns alone, and compressed bytes per stored float value.
  - Expected from the synthetic benchmark (§D11): **3.26 B/value** at `float32`, **1.68 B/value** at `float16`, **48.6 %** saving on the observation block, **~38 %** store-wide. This checkpoint confirms those hold on real sensor data rather than on a synthetic autocorrelated walk.
  - **Escalate rather than silently proceed** if the store-wide saving comes in **below 20 %** — that would fall under the pre-registered adoption threshold and the default should revert to `float32`. Report the number either way; do not rationalise a near-miss.
  - Also report the realised non-float bytes/row against the ~17 B estimate of §D11, which is the one unmeasured figure in the budget.
  - Cost: ~12 minutes of compute. Do not substitute an estimate.
- [x] **C8b — DONE, and the guard's criterion had to be corrected.** It fires, names the index and the sensor, and leaves no shard. The plan's RELATIVE criterion fired on legitimate near-zero data and was replaced with an absolute one; see the report. Original: Confirm the runtime guard (§D12) raises, names the offending observation index, and resolves it to a sensor name — see V5. Confirm no shard is left on disk after the raise.
- [x] **C9 — Resume is a no-op.** DONE (pytest, plus V6 on the NAS). Original: Run the same block range twice; assert the second run writes nothing and completes in under 30 s.
- [x] **C10 — Zero-slot environment.** DONE, and it forced a schema deviation: Parquet cannot round-trip a zero-width `fixed_size_list`. See the report, Deviation 1. Original: Collect 20 episodes from a config with `A = 0` (no animals) and confirm the animal columns are present as zero-length lists and the reader does not branch.

---

## Verification Plan

Every check below can **fail**, and none of them validates a code path using that same code path.

> **What these checks CANNOT cover — read before trusting them.** V1 and V2 both consume the `params` object the collector built from the saved config. If that object describes the wrong world (§A11), both pass in full agreement while describing the wrong world consistently. **No check in this list can detect a faithfulness failure of the config reload**, which is precisely why §D14's guard refuses ambiguous runs at the door rather than trying to verify its way out afterwards. V10 tests the guard; nothing tests the assumption the guard protects, because nothing can. Same structure applies to F3's restore check: V2 cannot catch a structurally-wrong model, because both of its paths rebuild from the same config, so `assert_restored_tree_matches` is a guard rather than a verification.

### V1 — Realised draws against an independent replay *(the primary correctness check)*

An episode is a pure function of `jax.random.PRNGKey(seed)`, so ground truth is freely available. Sample 200 recorded episodes at random. For each, take the recorded `episode_seed`, call `jax_reset(params, jax.random.PRNGKey(seed))` **unbatched, un-vmapped, outside any scan** — a genuinely different execution path from the batched collector — and assert equality for all 19 episode-level draw columns.

**Tolerance policy, pre-stated so it cannot be quietly weakened.** The assertion is **exact equality, including for float columns.** This is deliberate, not an oversight: both paths execute the identical sampling arithmetic on the identical key, so the outputs are bit-identical or something is genuinely wrong. **If this ever fails on a float column, the correct response is to diagnose the divergence — not to relax it to `allclose`.** A bitwise difference between `vmap(jax_reset)` and `jax_reset` would mean vectorisation is changing the sampler's output, which is exactly the class of bug this check exists to catch and which would shift every recorded draw in the store. Any future loosening of this tolerance requires a documented justification appended to this plan.

**Runs against the production store, not only the pilot** (§D16).

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

### V6 — Resume and atomicity under a hard kill, **on the NAS**

Start a 3-block collection; `SIGKILL` the process partway through block 2; restart. Assert: (a) no `.parquet.tmp` files remain, (b) the final store contains exactly the expected episode count with **no duplicate `episode_seed`**, (c) `steps.groupby(episode_seed).size() == length + 1` holds for every episode, and (d) blocks 0 and 1 are byte-identical to a clean run.

**This must be run on the NAS filesystem, not on local disk.** The entire "a shard on disk is always complete" guarantee (§D3) rides on `os.replace` being atomic and durable, and `results/` lives on a **CIFS mount** (`//192.168.0.250/cocoanlab01`), where POSIX rename semantics must not be assumed. A local-disk pass would prove nothing about the filesystem the store actually uses. Run it in `results/trajectories/_v6_scratch/` and delete afterwards.

Note the failure mode `SIGKILL` alone does **not** exercise: a killed *process* leaves the OS page cache intact, so the rename is durable even without an fsync. A killed *node* — the realistic multi-node failure — can leave a renamed-but-unflushed file that looks complete to `completed_blocks()` and is truncated on read. That is why `write_shard_atomic` fsyncs the file before renaming and the directory after (File Changes). V6 cannot easily simulate a node crash; the fsync is defence for the case the test cannot reach.

**Fails on**: non-atomic writes, a resume that re-does or skips a block, a seed mapping that depends on process state rather than on `(seed_base, shard_episodes)`, or CIFS not honouring rename atomicity — in which case the shard-completeness design needs revisiting before any production collection.

### V7 — Overwrite-hazard guard *(direct regression test for the 2026-07-04 incident)*

Collect 100 episodes into a store. Then attempt to resume **the same store path** with (a) a mutated env config, (b) a different `seed_base`, (c) a bumped `SCHEMA_VERSION`, and (d) a different `obs_precision`. Assert each raises `ValueError` and that **no file in the store is modified** (compare mtimes and hashes before/after).

**Fails on**: a missing or weak manifest guard. Case (d) additionally guarantees no store can end up half `float16` and half `float32` — a silent mixed-precision corpus would be undetectable at read time without it. This is the check that the 2026-07-04 contamination cannot recur.

### V8 — Scale and speed

Time one 5,000-episode block on a lab node. Assert throughput ≥ 10 episodes/s/process (against the 14.3 eps/s baseline of §A8; ≥ 10 allows for the extra recording payload) and peak RSS ≤ 2.5 GB.

**Fails on**: the O(n) Python bottlenecks not actually being removed, or a chunking bug that materialises more than one chunk at a time. Per the project's speed-review rule, a > 15 % throughput regression against the baseline is a blocker unless explicitly accepted here — and it is **not** accepted here.

### V9 — Schema invariance across environments *(the hard requirement's direct test)*

Collect 200 episodes from **three** structurally different saved configs: (i) a probe config with one predator, (ii) a training config with four animal slots, (iii) a config with the location sensor enabled and different `V` / `VV`. Assert the Parquet column **names and order are byte-identical** across all three, that only the `fixed_size_list` widths differ, and that **one** reader function loads all three without branching.

**Fails on**: any conditional column creeping in — the exact failure mode that makes the existing CSV unusable (§A1).

**Plus a non-circular read.** C3 validates a written store by reading it back through `open_store`, which shares `build_step_schema` with the writer — a schema-definition bug would echo itself back as proof. So V9 additionally reads one shard with **bare `pyarrow.parquet`, importing nothing from `trajectory_store`**, and compares the column names, order, and types against the **schema doc's** generated table. That closes the loop between code, store, and documentation without any of the three vouching for itself.

### V10 — The scene-ambiguity guard fires *(direct regression test for the §A11 faithfulness hazard)*

The corpus contains 12 real saved configs carrying both scene formats (§A11), which makes this testable against genuine artifacts rather than synthetic ones.

- Point the collector at `results/JAX_RecurrentPPO/20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage_s42` (or any of the 12) and assert it raises `ValueError`, that the message names both offending blocks and commit `828b77e`, and that **no store directory is created** — the guard runs before params are built (File Changes flow).
- Assert a clean run (any of the 20 rest-premium runs, all `entities:`-only) passes the guard and records `scene_format: "entities"`, `scene_ambiguous: false`.
- Assert a legacy-only config passes and records `scene_format: "legacy"`.
- Assert `--allow-ambiguous-scene` downgrades the failure to a warning and sets `scene_ambiguous: true` in the manifest.

**Fails on**: a missing guard, a guard that warns instead of raising, a guard placed after store creation, or a guard that tries to *resolve* the ambiguity rather than refuse it. This is the check that stands between the project and a million episodes of an agent in the world its trainer discarded — the one failure mode with no internal signal and no post-hoc detection.

---

## Out of Scope (explicit)

- **Any analysis layer.** No aggregation scripts, no plots, no measures, no regression of behaviour on environment factors. The deliverable stops at the documented store. Analyses are written ad hoc afterwards.
- **Dreamer-SRL support.** The checkpoint-loading and rollout seams are kept generic so Dreamer drops in later, but Dreamer is **not** implemented, **not** tested, and **not** claimed. `agent_type != "rppo"` raises.
- **Any change to the existing dwell-sweep pipeline** (`scripts/eval/dwell_sweep/*`, `scripts/eval/eval_rollout.py`, `scripts/behavior_measures/*`). The `bush_dwell` defect of §A7 is documented and avoided in the new store, not fixed in the old pipeline.
- **Reward decomposition and per-step distance columns.** Excluded by design, with per-field reasons in §D4.1.
- **Deleting the dead `build_episode_log_dict`** (`src/behavior/accumulators.py:546`). Noted as pre-existing dead code; removal is a separate change.

---

## Review Response

`plan-reviewer` returned **NOT READY** on 2026-08-19 with two Critical and eight lesser findings ([[plan_trajectory_collection]]). Disposition of each, with where the fix lives:

| # | Sev | Finding | Disposition |
|---|---|---|---|
| F1 | 🔴 | Reloading a saved config can rebuild the scene the trainer discarded; V1/V2 are circular w.r.t. it | **Fixed.** New §A11 (mechanism + corpus scan), §D14 (hard-fail guard, manifest provenance, applicability boundary), V10 (guard-fires test against 12 real dual-format runs). Scope measured, not assumed: 12 of 334 saved configs affected, all 2026-05-29 → 06-11, none after. |
| F2 | 🔴→🟢 | Reset-parity gate red, plan silent on it | **Downgraded to Low on diagnosis, and kept.** Full sweep: 4 failed / 30 passed / 293 skipped. All four failures are `observability_gates_S1`–`S4`, identical in form, fully explained by stale fixtures (`3d20aab`, 2026-05-28) predating a deliberate start-position change (`84014e4`, 2026-07-04) — reset code did not drift, and there is no unexplained residue. §D15 records the diagnosis so nobody re-escalates, adds fixture regeneration as a Phase 0 precondition, and keeps the durable residual: code drift between training and collection is unrecorded, mitigated by `collection_git_sha` + run-date in the manifest. |
| F3 | 🟡 | No strict restore check; V2 circular here too | **Fixed.** `assert_restored_tree_matches` (keys **and** shapes) in the `load_policy` seam; checkpoint C0. |
| F4 | 🟡 | C3 circular; schema doc untied to `STEP_COLUMNS` | **Fixed.** Doc tables generated from code by `gen_schema_doc.py` with a test asserting the committed doc matches; V9 adds a bare-`pyarrow` read importing nothing from `trajectory_store`; C3 relabelled as circular and explicitly not sufficient. |
| F5 | 🟡 | No pilot-to-scale sequence; sampled checks not tied to production store | **Fixed.** §D16 phases 0–3 with per-phase exit conditions; V1/V3/V4 stated as acceptance gates on **every production store**, not just the pilot. |
| F6 | 🟡 | Realised draws get only a 200-episode check | **Fixed.** `validate_store_draws` in the driver: whole-store bounds, `> 1` distinct value wherever `low < high` (and exactly one where `low == high`), activation-count ranges. |
| F7 | 🟢 | GPU buffer stated 1.8 GB | **Fixed** — 2.4 GB; conclusion unchanged. |
| F8 | 🟢 | `damage` cites a stale Open Question | **Fixed** — now points at Decisions Taken #1. |
| F9 | 🟢 | V1 float tolerance could be silently weakened | **Fixed.** Exact-equality policy pre-stated with justification and an explicit prohibition on relaxing to `allclose` without documented cause. |
| F10 | 🟢 | Interpreter path; bush-dwell units | **Fixed.** Worker names the absolute conda interpreter (matching `sweep_worker.sh:27`); reader snippet labelled a `[0,1]` occupancy fraction and renamed away from "dwell". |
| ❓4 | ❓ | `os.replace` atomicity assumed on this mount | **Fixed.** `results/` is a **CIFS** mount, so POSIX semantics are not assumed: `write_shard_atomic` now fsyncs the file before rename and the directory after, and V6 must run on the NAS, not local disk. The plan states which failure mode `SIGKILL` cannot reach (node crash vs. process kill). |

The reviewer also confirmed three known-bug hazards are genuinely closed by the design — fixed-seed episode repetition, stale-data blending, and derived-measure smuggling. Those verdicts are unchanged by this revision.

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

> **Implemented by**: `developer`
> **Date**: 2026-08-20
> **Status**: implemented, tested locally, **uncommitted** (working tree left dirty for review)

### Plain-language summary

The pipeline is built and works end-to-end on a real training run. All ten checkpoints
(C0–C10) and all pytest-able verifications (V1, V3, V4, V5, V7, V9, V10) pass, plus V6
(hard-kill + resume, run on the real NAS) and a 5-seed V2 (parity against the legacy
per-episode loop) run by hand. **46 tests pass in 104 s.**

Four things did not survive contact with reality and are flagged below rather than
quietly absorbed. In decreasing order of importance:

1. **The half-precision size argument does not hold on real data (C8 escalation).** The
   plan adopted `float16` on a predicted ~38 % store-wide saving; measured on the *same*
   5,000 real episodes written twice, the saving is **12.8 %**, below the 20 % threshold
   the plan pre-registered before measuring. The accuracy argument still holds; the size
   argument does not. **The decision is yours, not mine — I changed no default.**
2. **The float16 range guard, as specified, made `float16` collection impossible.** Its
   *relative* error criterion fired on a legitimate reading of `1.34e-06`. I replaced it
   with an *absolute* criterion tied to the store's own error budget, and documented why.
3. **Parquet cannot store a zero-width `fixed_size_list`**, so the array columns are
   variable-size `list<T>` with a writer-enforced width instead. This is stricter than
   the plan asked for (the column *type* is now identical across environments too) and
   measured marginally smaller on disk.
4. **`vmap(jax_reset)` and unbatched `jax_reset` are not bit-identical** on one realised-
   draw column. Diagnosed, bounded at one float32 ULP, and pinned by a test — but it does
   mean V1's pre-stated exact-equality policy cannot hold literally for that one column.

---

### 1. What was implemented, file by file

| File | Status | What it is |
|---|---|---|
| `src/utils/trajectory_store.py` | NEW, 1084 lines | Single source of truth: `SCHEMA_VERSION`, `STEP_COLUMNS` (36) / `EPISODE_COLUMNS` (23), schema builders, `build_table`, `env_fingerprint`, manifest read/write/guard, `write_shard_atomic` (+ the N3 directory-fsync decision), `completed_blocks`, `assert_scene_unambiguous`, `assert_restored_tree_matches`, `validate_store_{structure,shapes,draws}`, `open_store` / `TrajectoryStore`. Every manifest read goes through `_req`; there is no `.get(k, default)` anywhere in the module. |
| `scripts/eval/traj_collect/traj_scan.py` | NEW, 389 lines | `_rollout_scan` (nnx.jit-only, refuses eager entry), `_agent_in_bush`, `_reset_row`, `_draw_block`, `_prng_parity_guard` (`lax.map`, one sync/chunk), `_flatten_to_rows` (fully vectorised), `OBS_ABS_MAX` / `OBS_ABS_ERR_MAX` + `assert_obs_representable`. |
| `scripts/eval/traj_collect/collect_trajectories.py` | NEW, 563 lines | Single-run collector. Flow exactly as §D14/File Changes specify: resolve checkpoint (numeric) → load the run's own `models/config.yaml` → **scene guard before params are built** → params → `env_fp` → create-or-validate manifest → `load_policy` (+ F3) → per block, per chunk: reset → PRNG parity guard → `nnx.jit` scan → vectorised flatten → **range guard** → atomic shard writes. |
| `scripts/eval/traj_collect/collect_worker.sh` | NEW | Per-node worker; explicit absolute conda interpreter, CPU/GPU env split, persistent XLA cache, `xargs -P npar`, `_run_markers/{done,fail,prog,log}_<node>`. Worklist line is a **block range**. |
| `scripts/eval/traj_collect/run_collection.py` | NEW, 240 lines | Multi-node driver: spec → cells → LPT → worklists → **serial** `run_command.py` launches → marker poll → `validate_store_{structure,shapes,draws}` + a per-run size/throughput summary. `--dry-run`, `--validate-only`. |
| `scripts/eval/traj_collect/gen_schema_doc.py` | NEW | Generates the schema doc's tables from `STEP_COLUMNS` / `EPISODE_COLUMNS` between marker comments; `--check` exits 1 when stale. |
| `scripts/eval/traj_collect/README.md` | NEW | Operator guide. |
| `configs/trajectory_collection/example.yaml` | NEW | Spec template; the eight scientific keys are mandatory. |
| `docs/environment/TRAJECTORY_STORE_SCHEMA.md` | NEW | The permanent contract. Tables generated from code; measured numbers corrected (§5, §7). |
| `tests/test_trajectory_collection.py` | NEW, 1087 lines / 46 tests | Every guard tested as **firing**, each with a companion asserting it does *not* fire on good input. |
| `tests/fixtures/trajectory_collection/` | NEW | Byte-identical copy of a real dual-format saved config + a README naming its provenance (N2). |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | MODIFIED | §0 depth note, §1a two new bare-import edges, §3 five new rows, new **Cluster D**, footer date. |
| `tests/env/fixtures/parity/observability_gates_S{1..4}.npz` | MODIFIED | Phase 0(a): the four stale fixtures regenerated. |

Not modified, as the plan requires: `scripts/eval/eval_rollout.py`, `scripts/eval/dwell_sweep/*`,
`src/utils/evaluation_core.py`, `src/utils/eval_recording.py`, `scripts/behavior_measures/*`,
everything under `src/environment/`, and the whole config system.

---

### 2. Test and verification results (actual output)

**New suite — 46 passed in 104 s.**

```
$ JAX_PLATFORMS=cpu python -m pytest tests/test_trajectory_collection.py -q -p no:randomly
.............................................. [100%]
46 passed in 103.64s (0:01:43)
```

**Phase 0(a) — the reset-parity gate is now fully green** (was 4 failed / 30 passed):

```
$ python -m pytest tests/env/test_unified_parity.py -q
34 passed, 293 skipped in 342.88s (0:05:42)
```

The §D15 diagnosis was confirmed at source before regenerating: the four
`observability_gates_S{1..4}` configs were changed at `84014e4` (2026-07-04) from
`random_start_pos: true` to `false`, so the agent now starts at the config's fixed
`start_pos: [5,5]` (stored 0-indexed as `[4,4]`, `config_loader.py:1471` subtracts 1)
while the May fixtures still recorded a random `[2,2]`. Environment reset code did not
drift. Only those four fixtures were regenerated, via a scratch script — the shipped
generator was not modified.

| Check | Result |
|---|---|
| **V1** — realised draws vs an independent unbatched replay | **PASS** on 16 of 17 columns with exact equality; `animal_property_sampled` bounded at one float32 ULP — see §4.4. A companion test confirms a one-row shift *would* be detected, so the assertion is not vacuous. |
| **V2** — trajectory parity vs the legacy per-episode loop (operator-run, 5 seeds, real checkpoint) | **PASS** — action sequence, `(agent_row, agent_col)` sequence and `length` match **exactly** for every seed. Lengths 18/19/219/59/40. |
| **V3** — row-convention self-consistency | **PASS** |
| **V4** — `agent_in_bush` recomputed in pure NumPy | **PASS**, with an added assertion that at least one row *is* in a bush so the comparison cannot be vacuous. |
| **V5a** — precision fidelity | **PASS** (float32 bit-identical; float16 = float32 rounded, nothing else) |
| **V5b** — the range guard fires | **PASS**, 4 cases: injected out-of-range value, a real location-sensor channel, non-finite, and a magnitude below the backstop but above the error budget. Plus two companions: in-range data and a legitimate near-zero value must **not** trip it. |
| **V6** — resume + atomicity under `SIGKILL`, **on the NAS** | **PASS** — see the transcript in §3. |
| **V7** — manifest / overwrite guard | **PASS**, all four cases (mutated env config, different `seed_base`, bumped `SCHEMA_VERSION`, different `obs_precision`); file mtimes+hashes unchanged after each refusal. |
| **V8** — scale and speed | **PASS** — 47.9–49.6 eps/s ≫ the ≥10 eps/s threshold. Peak RSS: see C6 caveat. |
| **V9** — schema invariance + a bare-`pyarrow` read | **PASS** across three structurally different environments (A=1/A=4/location-sensor-on, three distinct dim tuples). The bare read imports nothing from `trajectory_store` and compares against the schema doc's generated table. |
| **V10** — the scene-ambiguity guard fires | **PASS**, 6 cases including the real dual-format corpus, the collector CLI end-to-end (no store dir created), the three unambiguous formats, the `--allow-ambiguous-scene` downgrade, and the N1 empty-`entities:` shape. |

---

### 3. Measurements

All on the same host, CPU, single process, `batch_size=1024`, checkpoint `59100070` of
`20260816-152742_rppo_restpremNH_a10_n112`, `shard_episodes=5000`, one 5,000-episode block.

```
$ python tmp/20260820_c678_measure.py float16
[collect] block 00000: 5000 episodes in 104.5s (47.9 eps/s)
RESULT precision=float16 wall=127.7s eps_per_s=39.16 peak_rss_MB=2554
$ python tmp/20260820_c678_measure.py float32
[collect] block 00000: 5000 episodes in 100.9s (49.6 eps/s)
RESULT precision=float32 wall=117.4s eps_per_s=42.58 peak_rss_MB=2500
```

**C7 — throughput (speed check).** Baseline (§A8, the existing per-episode path):
**14.3 episodes/s**. After: **47.9–49.6 episodes/s** in-block, **39.2–42.6 eps/s**
including the fixed ~17 s JAX-import + model-build + restore. **+235 % in-block.** No
regression anywhere; the batched `vmap` + `lax.scan` kernel is simply much faster than the
per-episode loop it replaces. Mean episode length measured **204.8 steps** (plan assumed
192), so 19.4 core-hours/run in §D6 becomes ≈ 5.6 core-hours/run at this rate.

**C6 — peak RSS: 2,554 MiB.** The plan predicted ~1.8 GB and C6's ceiling is 2.5 GB, so
this is **at or just over the ceiling depending on whether "GB" means GiB (2.49 GiB, pass)
or decimal (2.68 GB, fail)**. Reporting it rather than picking the flattering reading.
The cause is a design choice worth a decision: the collector accumulates **all chunks of a
block** in memory and concatenates once before writing, so peak scales with
`shard_episodes`, not `batch_size`. At `shard_episodes=5000` that is 5 chunks in flight.
Two cheap fixes exist if 2.5 GB matters at 16 workers/node (~40 GB/node): write with a
`ParquetWriter` chunk by chunk, or reduce `shard_episodes`. **Flagged, not fixed** — it
changes the shard-writing design and belongs to `senior-developer`.

**C8 — the paired precision measurement, and the escalation.** Same 5,000 episodes,
1,028,932 step rows, written twice:

| | `float32` | `float16` | saving |
|---|---:|---:|---:|
| whole store | 41.09 MB | 35.82 MB | **12.8 %** |
| steps shard | 40.74 MB | 35.47 MB | 12.9 % |
| the two observation columns alone | 27.36 MB | 22.09 MB | **19.3 %** |
| compressed **bytes per stored float value** | **0.492** | **0.398** | |
| compressed bytes per step row | 39.6 | 34.5 | |
| non-float bytes per step row | 3.81 | 3.81 | |
| episode row | 71.8 B/episode | 71.8 B/episode | |

**This falls under the plan's pre-registered escalation rule and I am escalating rather
than rationalising it.** 12.8 % store-wide is not a near-miss on the 20 % threshold.

Why the synthetic benchmark was wrong by ~7×: it modelled the observation block as
continuous autocorrelated data and measured 3.26 B/value at `float32`. The real block is
**0.49 B/value**, because this observation vector is mostly *not continuous* — of its 27
channels, **5 are identically zero** and **16 more take only {0,1,2}** (collision,
proprioception and visual are one-hot indicators stored as floats). Only 6 olfactory and
interoceptive channels carry genuinely continuous values, so zstd already removes most of
the block and half precision has little left to take.

Two consequences beyond the decision:

- The whole store is **~5× smaller than budgeted**: one 10⁶-episode run extrapolates to
  **≈ 7.1 GB (float16) / 8.2 GB (float32)**, not 28/45 GB. Ten runs ≈ 71–82 GB.
- The plan's one unmeasured figure, non-float bytes/row, was estimated at ~17 B. Measured:
  **3.81 B** — 4.5× smaller, in the same direction as everything else.

**What I did NOT do**: change any default. `obs_precision` is mandatory with no default in
both the CLI and the spec, so nothing silently picks a value. The example spec still says
`float16` and the schema doc now carries a prominent "decision under review" box with the
measured numbers. **Reverting the recommendation to `float32` is a §D12 decision and is
yours.** For what it is worth: the accuracy case for `float16` is unaffected and 12.8 %
of every future full-corpus scan is still 12.8 %; but the plan set the bar at 20 %.

**Disk (Phase 0c)**: `df -h /media/nas01` → `192T size, 134T used, 59T avail (70 %)`.
Unchanged from the plan's figure; the corpus is now ~80 GB, ~0.13 % of free space.

---

### 4. Deviations from the plan — four, all forced, none silent

#### 4.1 Array columns are `list<T>`, not `fixed_size_list<T>[w]` (§D1, §D4)

**Parquet cannot round-trip a zero-width `fixed_size_list`.** pyarrow 24.0.0 writes such a
column and reads it back as `[[None], [None], …]` — silently wrong *data*, not merely a
different type:

```
>>> a = pa.array([[]]*5, type=pa.list_(pa.int16(), 0)); pq.write_table(pa.table({"x":a}), buf)
>>> pq.read_table(buf)
ArrowInvalid: Expected all lists to be of size=0 but index 1 had size=1
```

Zero-width columns are not hypothetical: §A11's own corpus scan found **18 of 334** saved
configs with no animals (`A = 0`), and C10 requires them to work.

So every array column is the variable-size `list<T>`, with the constant width enforced by
the **writer** (offsets built as a ramp of the manifest width, payload length asserted) and
re-checked by `validate_store_shapes` on every shard. This is **stricter than the plan
asked for**: with `fixed_size_list` the column *type* differs between environments
(`fixed_size_list<int16>[4]` vs `[22]`), which V9 explicitly tolerated; with `list<int16>`
the type is byte-identical for every environment, so V9's assertion got stronger. Measured
cost: **823 B vs 835 B** for a 20,000-row × 22-wide `int16` column under zstd — the
variable-size form is marginally *smaller*, because Parquet has no fixed-size-list physical
type either and encodes both as a repeated group. Recorded in the module, in the schema doc
(§3.3) and here.

#### 4.2 The float16 guard's third clause is ABSOLUTE, not relative (§D12)

The plan specified a **relative** round-trip ceiling of `1e-3`. Run against the first real
5,000-episode collection it fired immediately:

```
ValueError: Observation index 1 — sensor Interoceptive Nociception (dims 1..1) — has a
float16 round-trip relative error of 0.0219 (value 1.3415422e-06 -> 1.3709068e-06) …
```

That value's **absolute** error is `2.9e-08`: ~8,000× below the store's own stated error
budget and ~300,000× below the σ = 0.10 noise the environment injects into that channel on
purpose. A relative criterion is the wrong instrument near zero, and as written it made the
plan's *default* precision uncollectable.

Replaced with `OBS_ABS_ERR_MAX = 1e-2` on the **absolute** round-trip error, set from
measurement: worst case anywhere in a real 5,000-episode store is **1.95e-03**, on an
olfaction channel reaching **6.95** — so the plan's "values in `[0,1]`" assumption is also
wrong, and its "2.44e-04 worst case, 400–800× below σ" becomes **1.95e-03, ~100× below
σ = 0.20**. 1e-2 leaves ~5× headroom over measured reality and corresponds to float16's
half-ulp at |x| ≈ 32, so it fires once any channel's magnitude reaches ~32 — about 4.6×
the largest magnitude the reference config produces, which is a realistic drift on a larger
grid or a higher `sensor_radius`. Clause 2 (`OBS_ABS_MAX = 1e4`) is retained as the outer
backstop; see §5 for the honesty note about it.

Two tests pin both directions: one asserts the `1.34e-06` value is *accepted*, one asserts
a magnitude-40 channel is *refused* while still far below the backstop.

#### 4.3 V1's exact-equality policy holds for 16 of 17 columns, not 17

`vmap(jax_reset)` and unbatched `jax_reset` are **not bit-identical** on
`animal_property_sampled`. Measured over 256 episodes: 242 of 5,120 elements differ, by at
most **5.96e-08 absolute** — exactly one float32 ULP at magnitude 1.0. **No other draw
column diverges at all**, including `res_property_sampled` and `obs_property_sampled`,
which are drawn by the *same* `_sample_property` helper.

Diagnosed rather than waved through, as the plan demands. `animal_property_sampled` is the
only one of the three assembled with a `jnp.zeros_like(...).at[idx].set(...)` scatter
(`core.py:1128-1140`, the N2 per-class split). Under `vmap` that scatter lowers differently
and XLA fuses `mean + std * noise` differently (fused multiply-add vs separate multiply and
add), changing the last bit. It is **pre-existing environment/XLA behaviour, independent of
this pipeline**, and invisible to the existing eval tooling only because `eval_rollout.py`
does not record property draws.

`test_v1_animal_property_divergence_is_one_float32_ulp` pins the bound and asserts no other
column joins it, so growth or spread is caught rather than absorbed. Per the plan's rule,
**this loosening is documented here and requires your sign-off.** It is also a candidate
Known-Bugs row — I cannot spawn `bug-curator`; a grep of the registry found no existing row
for it, so **`bug-curator` should record it**.

#### 4.4 N1 resolved by matching the loader exactly

The guard now uses `env.get('entities') is not None` — **identical** to
`config_loader.py:429` — rather than `bool(entities)`. So it refuses the present-but-empty
`entities:` + legacy shape too. A test asserts both the behaviour and that the loader's
predicate string is still present in the source, so a future loader change surfaces here.

#### N3 (directory fsync on CIFS) — resolved empirically, no fallback needed

`_fsync_dir` carries a single documented decision (warn once, continue — with the reasoning
for why failing would trade a real capability for an unattainable guarantee), not a
scattered `try/except`. In practice it never triggers: **directory fsync IS supported on
this CIFS mount** — verified directly (`_DIR_FSYNC_UNSUPPORTED == False` after a real write
into `results/`), and V6 produced no warning.

---

### 5. Checks I suspect cannot actually fail — stated, as asked

1. **`OBS_ABS_MAX = 1e4` (clause 2 of the range guard) is now subsumed by clause 3.** Any
   channel large enough to trip `|x| > 1e4` trips the `1e-2` absolute-error budget first
   (which fires at |x| ≈ 32). Clause 2 is retained because the plan specifies it and it
   names a distinct failure with a distinct message, but it will never be the clause that
   actually catches anything. Clause 3 is where the teeth are.
2. **The PRNG parity guard cannot catch a wrong seed-to-episode *assignment*.** It compares
   `vmap(jax_reset)` against an independent `lax.map(jax_reset(PRNGKey(s)))` over the same
   seed list, so it catches a wrong key *construction* (the `ParallelEnv.reset` failure it
   exists for) but not a wrong seed *list*. Stated in its docstring. V1 and the store's
   no-duplicate-seed check cover the gap; `test_seeds_advance_per_episode_and_are_unique`
   is the one that would catch a regression to the existing harness's fixed-seed behaviour.
3. **C3 is circular** and is labelled so in the test body — reader and writer share
   `build_step_schema`. `test_v9_bare_pyarrow_read_matches_the_schema_doc` is the
   non-circular counterpart and imports nothing from `trajectory_store`.
4. **`assert_restored_tree_matches` against the *restored tree* alone is near-vacuous**,
   because orbax's `partial_restore` filters the checkpoint down to the target's keys. It
   is the **checkpoint metadata tree** comparison that has teeth. If
   `PyTreeCheckpointer().metadata()` ever returns nothing usable, the collector prints a
   warning and degrades to the vacuous comparison — that degradation is loud but it is a
   degradation, and it is worth a reviewer's eye.

---

### 6. The model-size CLI flag question — verified, NOT reproducible

The registry records "model-size CLI flags may not be persisted to the saved config" as
open/probably-fixed/never-verified. **It is fixed, and I verified it two ways.**

*By inspection*: `train.py:759-762` resolves `hidden_size = args.hidden_size or
config.get_mandatory('agent.hidden_size')` and immediately does
`config.set('agent.hidden_size', hidden_size)`; the dump to `<run>/models/config.yaml`
happens later, at `train.py:881`. The `else` branch (non-rPPO/PPO) does the same at
`:764-769`.

*Empirically*: a 2-episode run with `--hidden-size 77` against an agent config declaring
`hidden_size: 128`:

```
$ grep hidden_size results/JAX_RecurrentPPO/20260820-005848_hsizeprobe/models/config.yaml
304:  hidden_size: 77
$ grep hidden_size configs/models/recurrent_ppo/recurrent_ppo.yaml
15:  hidden_size: 128
```

The flag is persisted. The throwaway run directory was deleted. **`bug-curator` should
close that row as verified-fixed.** Note this does not make F3 redundant — F3's real catch
is the *structurally different model* (a missing modulation block), which no amount of
config persistence prevents.

---

### 7. Blockers and follow-ups

| # | Item | Owner |
|---|---|---|
| 1 | ~~C8 escalation~~ **CLOSED 2026-08-20** — recommended precision flipped to `float32` (template + docs; still no code-level default). | done |
| 2 | ~~C6~~ **CLOSED 2026-08-20 as superseded** — measured node RAM is 125-515 GB, so `npar: 16` uses ~41 GB (36 % of the smallest node). The 2.5 GB figure was a plan ceiling, not a hardware limit. Buffering left as-is. | done |
| 3 | **New Known-Bugs row** (still open): `jax_reset` is not bit-reproducible on `animal_property_sampled` across compilations — <=5.96e-08, one float32 ULP, scatter-path FMA fusion at `core.py:1128-1140`. Corroborated independently by `senior-developer` using the shipped fixture generator (unbatched, no `vmap`), so it is compilation-level, not a `vmap` artefact. | `bug-curator` |
| 4 | **Close the model-size-flag row** as verified fixed (§6). | `bug-curator` |
| 5 | The plan's §D11/§D12 size arithmetic is superseded by the measured figures in §3. The schema doc already carries the corrected numbers; the plan text still carries the predictions. | `senior-developer` |
| 6 | §D15's recommendation that `train.py` write a training-time git SHA remains open and out of this plan's scope. | `senior-developer` |
| 7 | Phase 1 (25,000-episode pilot) has **not** been run — the brief said implementation and local verification only, no remote launches. Everything Phase 1 gates on is green locally. | user |

**Working tree left dirty and uncommitted**, as instructed. `docs/project/references/behavior_analysis/`
appears untracked in `git status` and is **not mine** — a parallel session's work; I did not
touch it.

*Implemented by: developer*
---

### 8. Review-response round (2026-08-20) — 19 items, all addressed

All three reviews passed (`senior-developer`: ready for pilot; `code-reviewer`: no
silent-wrongness path in the stored data; `env-config-reviewer`: no Critical). The
consolidated fix list was applied in full. **66 tests pass in 143 s** (was 46).

Two prior open items closed by the reviews rather than by me:

- **The ULP finding got independent corroboration I did not have.** `senior-developer`
  re-derived the four parity fixtures with the project's shipped generator, which uses
  **unbatched `jax_reset` only, no `vmap` anywhere**, and its output still differs on
  `animal_property_sampled` by ≤5.867e-08 with zero integer or boolean differences. Two
  *unbatched* runs disagreeing by exactly the effect I had attributed to `vmap` settles it
  as compilation-level FMA fusion rather than a logic bug. My containment (pinned bound +
  spread test) stands unchanged; the registry row is still owed.
- **C6 is superseded, not fixed.** Real node RAM is 125–515 GB, so `npar: 16` uses ~41 GB
  (36 % of the smallest node). The 2.5 GB figure was a plan ceiling, not a hardware limit.
  I did **not** restructure the block buffering.

#### Tier 1 — correctness and performance

| # | Item | What changed |
|---|---|---|
| 1 | **C5 asserted an invariant the plan contradicts** | The guard compared reset-vs-final over *all* `_draw_block` fields, including the two resource `*_init` draws that `jax_step` re-draws on regeneration (`core.py:526-541`, assigned `808-809`). Split into `traj_scan.CONSTANT_DRAW_FIELDS` / `MUTABLE_DRAW_FIELDS`; the loop now walks the constant set only, **and refuses any field that is in neither** so a new draw column cannot silently escape the check. Line citation corrected (`core.py:800-838`, with the sub-ranges named). New test `test_c5_tolerates_resource_property_redraw` sets `properties_std: 0.3` and `regeneration_delay: 1`, pins one food slot to a single cell, starts the agent on it and drives a **constant eat-action policy**, so the re-draw is guaranteed rather than hoped for — it asserts the re-draw actually fired (which is exactly the condition under which the old loop raised) and that `run_chunk` accepts it. A companion asserts the classification is total and that narrowing C5 did not disarm it. |
| 2 | **`nnx.jit` retraced every chunk** | `make_rollout_fn()` builds the wrapper once in `main`; `run_chunk` now takes it as its first argument and constructs nothing. `test_rollout_fn_is_built_once_and_reused` asserts both structurally (`"nnx.jit" not in inspect.getsource(run_chunk)`) and behaviourally (a shared wrapper traces once over three identical-shape calls; a fresh wrapper traces three times). **Measured, same host/config/block:** in-block **100.9 s → 87.4 s (49.6 → 57.2 eps/s, +15.3 %)** on a 5-chunk block, where the fix avoids 3 of 5 compiles → **~4.5 s per avoided compile**. In production (`blocks_per_cell: 10`, 50 chunks per process) it avoids ~48 compiles ≈ **3.6 min per worker process**; across ~200 worker processes for a ten-run collection that is **~12 core-hours against ~56 core-hours of real work** — roughly a fifth of the total. |
| 3 | **`device` unguarded** | Added to `MANIFEST_GUARDED_FIELDS` with the reasoning inline (bit-level results are lowering-dependent — my own ULP finding proves it). Schema doc §5 gains a "pairing is exact only per-device/per-lowering" paragraph noting that integer and boolean draws are exact everywhere and only float draws are affected. `test_device_is_manifest_guarded`. |
| 4 | **Restore-guard degradation was a printed warning** | Metadata reading moved into `read_checkpoint_tree()`; the bare `except Exception` narrowed to the five concrete failure types. An unreadable checkpoint structure is now a **hard `ValueError`** unless `--allow-weak-restore-check` is passed, in which case the manifest records `restore_check: "weak_allowed"` (guarded, so a store cannot be half strict and half weak). Three tests: refusal by default with **no shard written**, the escape hatch recording itself in the manifest and being guarded, and — the one that matters — `test_c0_metadata_is_readable_so_the_strict_check_is_not_vacuous`, which **fails rather than skips** if orbax stops exposing the tree, so guard and tests cannot go vacuous together in silence. |
| 5 | **Unknown spec keys silently ignored** | `load_spec` now rejects unknown keys **top-level** (against `MANDATORY ∪ OPTIONAL`) and **per-run** (against `RUN_KEYS = path, label, seed_base, allow_ambiguous_scene`), with the error text naming the illusory-pairing hazard. `test_spec_loader_rejects_unknown_keys` covers a top-level typo, the dangerous `seed_bases:` per-run typo, and that each allowed key still passes; `test_spec_loader_accepts_the_shipped_example` is the companion — a validator that rejects its own template is not a validator. |

#### Tier 2 — the precision decision

**6. Recommended precision flipped to `float32`.** There is still no code-level default
anywhere (`required=True` in the CLI, `_req` in the spec loader), so this is the template
plus documentation: `configs/trajectory_collection/example.yaml` now sets `float32` and
carries the measured 12.8 % / 19.3 % figures and the ~100× (not ~400×) noise ratio; the
collector docstring and the operator README say the same; the schema doc's "decision under
review" box became the decision, with the reasoning preserved.

**The dormancy consequence is now written down in three places** (guard code, spec
template, schema doc §5): under `float32`, `assert_obs_representable` returns after the
finiteness check, so **both magnitude clauses go dormant**. That is correct — `float32`
cannot misrepresent these magnitudes — but it means a `float32` store carries **no recorded
evidence about observation range**, so concluding later that a run "could have been
`float16`" is not supported by anything the store contains.

#### Tier 3 — cheap asserts

| # | Item | Note |
|---|---|---|
| 7 | seed-space cliff | Added to `build_manifest`. **One correction to the stated mechanism**: the collision cliff is at **2³², not 2³¹**. Measured — `jnp.asarray(np.int64)` canonicalises `2**31+5` to `-2147483643`, but `PRNGKey` reads that back as uint32 `2147483653`, i.e. the *same* key, so `[2³¹, 2³²)` is a harmless no-op. At 2³² the uint32 itself wraps and `PRNGKey(2**32+5) == PRNGKey(5)` **exactly** — two different recorded `episode_seed` values, one identical episode, invisible to both the parity guard and the duplicate-seed check. I assert the stricter `seed_base + episodes < 2**31` anyway (no canonicalisation at all). `test_seed_space_cliff_is_refused` demonstrates the collision rather than asserting it from memory. |
| 8 | `max_steps < 32768` | Added to `build_manifest`, naming the int16 columns it bounds. `test_max_steps_beyond_int16_is_refused`. |
| 9 | `termination_reason != 0` | Added per chunk in `run_chunk`, with an error naming the documented latent env quirk it corresponds to. |
| 10 | width-enforcement regression test | Two: `test_list_column_width_enforcement_fires` (writer side) and `test_validate_store_shapes_fires_on_a_wrong_width` (reader side, on a deliberately ragged shard). Both have clean-input companions. |

#### Tier 4 — documentation and cosmetics

11. Schema doc gains a **`termination_reason` has two documented latent quirks** section:
    the injury-disabled instant-kill case (reason 0 on a real death — the collector now
    refuses it) and the `overeating_death` case (reason 3 on non-terminal rows), with the
    instruction to read the per-step column jointly with `terminated`, never alone.
12. `example.yaml`: the non-existent `..._rppo_other_n107` replaced with the real
    `20260816-152028_rppo_restpremNH_a03_n107`, clearly marked as carrying the override
    only to show the syntax. "the ONLY permitted per-run key" corrected to the accurate
    four-key list with the reason `seed_base` is the only *scientific* one.
13. `load_spec` pre-flights `<run>/models/config.yaml` for every run, once, before any node
    is launched. `test_spec_loader_preflights_run_paths` also covers a missing `path`
    (previously a bare `KeyError`).
14. `--dry-run` no longer deletes completion markers.
    `test_dry_run_does_not_delete_completion_markers` plants a marker from a notional live
    collection and asserts a dry run leaves it byte-identical.
15. Positivity validators for `episodes`, `seed_base`, `npar`, `batch_size`,
    `shard_episodes`, `blocks_per_cell`, plus type checks on `obs_precision` and `runs`.
    Note the defaults had to move from `spec.get(k) or default` to `is None`, otherwise an
    explicit `npar: 0` was silently replaced by the default before reaching the check — the
    parametrised test caught that.
16. `assert_scene_unambiguous` now returns `(scene_format, scene_ambiguous)`; the collector
    no longer re-derives the predicate a third time.
17. `build_table` derives `n_rows` from the first column and **asserts it is scalar**, so a
    mis-shaped array column can no longer define its own row count.
18. The non-degeneracy failure message now names the clip-boundary caveat (a property whose
    mean sits on a clip bound with small std legitimately collapses to one value).
19. `_stack_rows`'s broadcast branch marked defensive-only, with what it would mean if it
    ever fired.

**Not mine, handled as instructed**: the GPU-index limitation is now a one-line comment in
three places (`load_spec`, `collect_worker.sh`, README) stating plainly that multi-GPU
fan-out does not work today.

#### Re-verification after the changes

```
$ python -m pytest tests/test_trajectory_collection.py -q -p no:randomly
66 passed in 142.82s (0:02:23)

$ python scripts/eval/traj_collect/gen_schema_doc.py --check
docs/environment/TRAJECTORY_STORE_SCHEMA.md: up to date
```

V6 (SIGKILL + resume, re-run on the NAS after the changes): 3 blocks, 3,000 episodes,
642,482 step rows, **zero `.tmp` leftovers, all six shards byte-identical to a clean run**,
structure/shapes/draws all validate. Driver dry-run still expands 60 cells across 4 nodes
and leaves no scratch behind.

#### One thing I disagree with, and one correction

- **Item 7's cliff is at 2³², not 2³¹** (see the table above). The assert I added is the
  stricter 2³¹ bound anyway, so the fix is the same; I am flagging it only because the
  stated mechanism ("a seed past the cliff wraps before key construction while
  `episode_seed` records the unwrapped value") is right about the *consequence* but the
  wrap it describes is benign until 2³².
- **No disagreement on any item.** The one I would have pushed back on — item 2, on the
  grounds that five compiles per block is not obviously fatal — turned out to be
  measurably worth ~20 % of total collection compute once `blocks_per_cell` is taken into
  account, so the reviewer's call was right and my instinct would have been wrong.

*Review round implemented by: developer*


## Verification Report

> **Verified by**: `senior-developer`
> **Date**: 2026-08-20
> **Verdict**: ✅ **APPROVED FOR THE PHASE 1 PILOT.** Three items are routed before Phase 2 (the first full 10⁶-episode run); none of them blocks the pilot.

### Plain-language verdict

The pipeline does what the plan said it would. I re-ran every test myself rather than
taking the developer's word for it, and got the same numbers: **46 of 46 new tests pass**,
and the environment reset-parity gate that was previously failing on four scenarios is now
**fully green (34 passed, 293 skipped)**.

I checked the four things the developer said it had to change, and all four are genuinely
forced rather than convenient. The most important one: Parquet really cannot store an
"empty list" column of the kind the plan specified, and 18 of the project's 334 saved
training configs describe worlds with no animals — so that column type had to change. I
reproduced the failure myself.

The fixture regeneration deserved the most scrutiny, because regenerating a fixture is also
how you would make a real failure disappear. It is legitimate. I regenerated the four
fixtures independently, using the project's own shipped generator rather than the
developer's script, and got the same numbers back to within floating-point rounding noise,
with **every single integer and boolean value identical**.

Three things go back out. (a) One safety check — the one that catches "you loaded the wrong
model into this checkpoint" — quietly weakens itself to a near-useless version if it cannot
read the checkpoint's metadata, and nothing in the resulting data records that it did so;
that should be hardened before a run costing several hours. (b) The example configuration
file still advertises a storage saving (~38 %) that the developer's own measurement
disproved (12.8 %). (c) The memory-usage concern the developer flagged turns out **not** to
be a problem — the lab machines have far more RAM than the job needs.

### Files verified

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/utils/trajectory_store.py` | NEW, 1084 lines | ✅ | Single source of truth as specified. `_req` used for every manifest read; no `.get(k, default)` in the module. `list<T>` deviation documented in-module at the point of use. |
| `scripts/eval/traj_collect/traj_scan.py` | NEW, 389 lines | ✅ | Eager-entry `RuntimeError` present (§A5). Range guard placed after flatten, before write. `OBS_ABS_ERR_MAX` correction documented at the constant. |
| `scripts/eval/traj_collect/collect_trajectories.py` | NEW, 563 lines | ✅ | Guard order matches §D14: scene guard before `load_env_params`. C5 draw-constancy check runs on every chunk. |
| `scripts/eval/traj_collect/collect_worker.sh` | NEW, 71 lines | ✅ | Explicit conda interpreter; `cd ../../..` depth correct for the new two-level nesting. |
| `scripts/eval/traj_collect/run_collection.py` | NEW, 240 lines | ✅ | Serial `run_command.py` launches; `_req` on the eight scientific spec keys. |
| `scripts/eval/traj_collect/gen_schema_doc.py` | NEW, 104 lines | ✅ | Contract verified by regeneration — see below. |
| `scripts/eval/traj_collect/README.md` | NEW | ✅ | Operator doc; no stale size claims. |
| `configs/trajectory_collection/example.yaml` | NEW, 52 lines | ⚠️ | **Stale numbers.** Lines 24–25 still claim `float16` "saves ~38% store-wide" and sits "~400x below" injected noise. The developer's own C8/§4.2 measurements say **12.8 %** and **~100×**. Routed as item R2. |
| `docs/environment/TRAJECTORY_STORE_SCHEMA.md` | NEW, 545 lines | ✅ | Tables generated from code (verified). Carries the corrected measured numbers and a "decision under review" box. |
| `tests/test_trajectory_collection.py` | NEW, 1087 lines / 46 tests | ✅ | Every guard tested as firing, each with a non-firing companion. |
| `tests/fixtures/trajectory_collection/` | NEW, 2 files | ✅ | Real dual-format config + provenance README (closes reviewer note N2). |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | MODIFIED | ✅ | Maintenance Contract satisfied — see below. |
| `tests/env/fixtures/parity/observability_gates_S{1..4}.npz` | MODIFIED | ✅ | Regeneration independently verified legitimate — see below. |
| `docs/develop/INDEX.md` | MODIFIED | ✅ | Regenerated by script, timestamp-only diff. |

**Diff-stat check.** `git diff --stat HEAD` shows 7 files, 362 insertions / 26 deletions, all
in docs plus the four binary fixtures. No tracked source file was modified — consistent with
the plan's "NOT MODIFIED (explicit)" list. No out-of-scope file appears. The untracked
`docs/project/references/behavior_analysis/` is a parallel session's work, correctly
disclaimed by the developer and left alone.

### Gates executed by the verifier (actual output)

| Gate | Command | Result |
|---|---|---|
| New suite | `pytest tests/test_trajectory_collection.py -q -p no:randomly` | **`46 passed in 105.75s`** — matches the reported 103.64 s |
| Parity gate | `pytest tests/env/test_unified_parity.py -q` | **`34 passed, 293 skipped in 68.85s`** — matches the reported result; previously 4 failed |
| Schema-doc contract | `gen_schema_doc.py --check`, then regenerate + `diff` | `up to date`, exit 0; regenerated file **byte-identical** to the committed doc |

**The four guards are tested as FIRING, not merely passing on clean input.** Verified by
reading each test body, not by name:

| Guard | Firing tests | Non-vacuity companion |
|---|---|---|
| Scene ambiguity | 4 × `pytest.raises(ValueError, match="828b77e")`, incl. the whole real 12-config corpus and the collector CLI end-to-end | passes on `entities`-only, `legacy`-only and `none`; `--allow-ambiguous-scene` downgrade asserted via `pytest.warns` |
| Observation range | 4 raising cases: injected out-of-range value, real location-sensor channel, non-finite, magnitude-40 channel below the outer backstop | 2 companions assert in-range data and the legitimate `1.34e-06` value do **not** trip it |
| Manifest mismatch | parametrised over all 4 cases (env config, `seed_base`, `SCHEMA_VERSION`, `obs_precision`), each asserting no store file changed | identical resume asserted to be accepted |
| Strict checkpoint restore | wrong `hidden_size`; `encoding_mode` flat-vs-hierarchical (different key set) | correct model accepted first, so "it raised" means something |

Both restore-guard firing tests pass `checkpoint_tree=_real_checkpoint_tree()`, so the
comparison **with teeth** (against a real on-disk checkpoint's metadata) is the one under
test — not only the near-vacuous restored-tree comparison. That materially reduces the
severity of the developer's honesty item 4; see below for what remains.

**The bare-`pyarrow` read is genuinely non-circular.** `test_v9_bare_pyarrow_read_matches_
the_schema_doc` imports only `pyarrow.parquet`, `json` and `re`; it hardcodes the Arrow type
names (`halffloat`, `int16`, …) rather than importing them, and compares the file against
the *schema doc's* generated table plus the manifest's dims. Confirmed by reading the body.

### Deviation-by-deviation judgement

**D1 — zero-width `fixed_size_list` → `list<T>`: FORCED, correctly handled. ✅**
Reproduced independently on pyarrow 24.0.0. The precise behaviour is slightly different
from the developer's description and worth recording: `pq.write_table` **succeeds** (512
bytes on disk) and `pq.read_table` then fails with `ArrowInvalid: Expected all lists to be
of size=0 but index 1 had size=1`. So it is an unreadable file rather than silently wrong
data — either way the round trip is broken and the deviation is forced. A width-0
`list<int16>` round-trips correctly (`[[], [], [], [], []]`), which is the fix. The width is
**structurally** enforced, not merely asserted: `_list_array` builds offsets as a ramp of
the manifest width, so ragged rows are unrepresentable, and a payload of the wrong total
size raises — verified live: `_list_array(zeros(9), n_rows=5, width=2)` →
`ValueError: list column payload has 9 values, expected 5 rows x width 2 = 10`.
`validate_store_shapes` re-checks every shard. **One gap**: no regression test feeds a
wrong-width payload and asserts the refusal — the width path is exercised only positively
(V9, C10, `validate_store_shapes`). Routed as R3 (low).

**D2 — absolute rather than relative observation-error budget: FORCED, sound. ✅**
A relative criterion is the wrong instrument near zero, and the concrete case is decisive:
`1.34e-06` fails a 1e-3 relative test at 2.2 % while its absolute error is `2.9e-08`. Since
the store's stated error budget is absolute and quoted against absolute injected sensor
noise, an absolute criterion is the internally consistent choice. `OBS_ABS_ERR_MAX = 1e-2`
is falsifiable rather than decorative: float16's half-ulp exceeds 1e-2 from |x| ≈ 32, only
~4.6× above the largest magnitude the reference config produces. The corrected value range
**is** documented — the schema doc records olfaction reaching **6.95** and the worst
measured round-trip error of `1.95e-03`, superseding the plan's "values in [0,1]" and
"2.44e-04". *Minor*: the code comment at `traj_scan.py:65` says the guard fires at |x| ≈ 20
(from the continuous half-ulp formula) while the report says ≈ 32 (the true first failing
binade). 32 is correct; cosmetic only.

**D3 — `vmap(jax_reset)` vs unbatched `jax_reset`: genuinely floating-point, not a logic
bug. ✅ (and I found independent corroboration)**
This was the item asked for my judgement, and the evidence is stronger than the developer
knew. While auditing the fixture regeneration I regenerated the four parity fixtures myself
with the project's shipped generator — a path that uses **unbatched `jax_reset` only, no
`vmap` anywhere**, in a different process. My output differs from the developer's committed
fixtures on `stepNNN_animal_property_sampled` by a maximum of **5.867e-08** — the same
column, the same one-float32-ULP magnitude — with **zero** integer or boolean differences
anywhere. Two unbatched runs disagreeing by exactly the signature attributed to `vmap`
establishes that the effect is compilation-level FMA-fusion non-determinism attached to that
column's scatter assembly, not something `vmap` does to the sampler. A logic bug would not
reproduce between two runs of the same unbatched code, and would not leave every integer
draw bit-identical. The bound is pinned by `test_v1_animal_property_divergence_is_one_
float32_ulp`, which asserts both the magnitude (`≤ 6e-08`) and that **no other column joins
it** — so spread or growth is caught rather than absorbed. The other 16 draw columns are
asserted with `np.array_equal` (exact), including the five other property columns drawn by
the same `_sample_property` helper. The plan's V1 exact-equality policy is therefore
loosened for exactly one column, with a documented mechanism and a pinned bound. **I accept
the loosening.** The Known-Bugs row the developer requested is still correct to file.

**D4 — N1 predicate alignment: correct. ✅** `assert_scene_unambiguous` uses
`env.get("entities") is not None`, byte-identical in meaning to `config_loader.py:429`.
`test_n1_guard_predicate_matches_the_loader_exactly` asserts both the behaviour and that the
loader's predicate string is still present in the source, so a future loader change surfaces
here rather than silently diverging.

### Fixture regeneration — audited independently, legitimate

The mandate asked whether regeneration hid a real failure. Three independent checks say no.

1. **The claimed cause is real.** `git show 84014e4` is a 4-file, 4-line commit changing
   `random_start_pos: true → false` on exactly the four gate configs, with a message
   describing the deliberate fix. Fixtures are from `3d20aab` (2026-05-28), the config
   change from 2026-07-04. The commit archaeology holds.
2. **Test coverage did not shrink.** The regenerated fixtures carry a different key set
   (2618 vs 2921 keys) because they use the post-unified-animal-refactor schema. This is
   **not** a weakening: `_fixture_subset` (pre-existing, documented) prefers the legacy
   `pred_*`/`neutral_*` key and falls back to slicing the unified `animal_*` array by
   `predator_indices`/`neutral_indices`, and **raises loudly if neither exists**. I
   enumerated every key the test actually reads: the 404 "lost" keys are exactly
   `pred_pos`/`neutral_pos`/`pred_property_sampled`/`neutral_property_sampled`, and the 202
   "gained" keys are exactly the `animal_pos`/`animal_property_sampled` arrays they are
   recovered from. `num_pred`/`num_neutral` are unchanged (1/2). 9 of the 34 fixtures were
   already in this schema before the change, so the new form is the modern one, not a
   bespoke one.
3. **The shipped generator reproduces the committed fixtures.** I copied
   `scripts/fixtures/generate_parity_fixtures.py` unmodified (except output dir and repo
   root, which it computes from `__file__`), restricted it to the four gate configs, and
   ran it. Key set: **identical, 2618**. Values: **every integer and boolean array
   identical**; the only differences are float noise — worst relative difference
   **1.413e-07**, all inside the test's own `rtol=atol=1e-5`, and dominated by the same
   `animal_property_sampled` ULP effect discussed above. The developer's scratch script
   produced what the shipped generator produces.

Note for the record: the shipped generator has **no output-directory argument** and would
regenerate all 34 fixtures if run as-is. Using a scratch script scoped to four files was the
right call, not a shortcut.

### Maintenance contracts

- **`SCRIPTS_DEPENDENCY_MAP.md` — satisfied ✅.** New subpackage is exactly this document's
  case, and it was updated in the same change: §0 records `scripts/eval/traj_collect/` as
  the *second* two-level nesting with the `parents[3]` / `cd ../../..` depths spelled out;
  §1a gains two bare-import edges (`collect_trajectories → traj_scan`, and the test's
  `sys.path` insert); §3 gains five rows with stakes and rewrite-triggers; a new **Cluster
  D** records the import/path coupling and states explicitly that it is disjoint from
  Cluster B; the footer date is updated with a summary.
- **Schema doc generated, not hand-written — satisfied ✅.** `gen_schema_doc.py --check`
  exits 0, and regenerating produces a file byte-identical to the committed one.
- **Config-system contracts — correctly not triggered ✅.** No change to
  `config_loader.py`, `state.py` `EnvParams`, or any env YAML schema, so `CONFIG_GUIDE.md`,
  `02_config_schema.md` and the `CONFIG_CRITICAL_SETTINGS.md` change log are not due. The
  plan states this omission is deliberate and the diff confirms it.

### Speed-change review — ✅ no regression

This is a new writer, not a modification of an existing path, so the relevant bar is V8's
`≥ 10 eps/s` floor. Measured **47.9–49.6 eps/s** in-block (39.2–42.6 including the fixed
~17 s startup) against the §A8 per-episode baseline of **14.3 eps/s** — **+235 % in-block**.
The measurement is sound for the purpose: same host, CPU, single process, same checkpoint
(`59100070`) and config, `batch_size=1024`, and a 5,000-episode block is a long enough
budget that the fixed startup is amortised and separately reported rather than dominating.
Mean episode length came in at 204.8 steps against the plan's assumed 192, which the
developer correctly propagated into a revised ≈ 5.6 core-hours/run.

### Assessment of the developer's own honesty list

| # | Self-reported weakness | My assessment | Action |
|---|---|---|---|
| 1 | `OBS_ABS_MAX = 1e4` subsumed by the 1e-2 error budget | **Confirmed and correctly characterised.** Clause 2 is checked first, so it is reachable, but only for values that would also trip clause 3 (which fires from \|x\| ≈ 32). It survives as a distinct message for a gross magnitude. Harmless. **But there is a consequence the developer did not connect** — see R4. | Accept as-is |
| 2 | PRNG parity guard cannot catch a wrong seed *list* | **Correct, and adequately covered elsewhere.** Both sides consume the same seed list by construction, so the gap is real. It is closed by `test_seeds_advance_per_episode_and_are_unique`, V1's seed-to-row check (which has an explicit non-vacuity companion asserting a one-row roll *would* be caught), and `validate_store_structure`'s no-duplicate-seed pass over the whole store. Documented in the guard's own docstring. | Accept |
| 3 | C3 is circular | **Correct, labelled, and genuinely compensated.** V9's bare-`pyarrow` read is non-circular by inspection (imports nothing from `trajectory_store`, hardcodes Arrow type names, compares against the doc and the manifest). | Accept |
| 4 | `assert_restored_tree_matches` is near-vacuous without the checkpoint metadata tree | **Correct, and this is the one that needs strengthening.** The teeth *are* tested — both firing tests pass a real `checkpoint_tree` — so the check works. The problem is the degradation path: `collect_trajectories.py:174-180` catches a bare `Exception`, `print`s a warning, and continues with `checkpoint_tree=None`, which reduces the guard to a comparison of the restored tree against itself. Under `xargs -P 16` across ~160 workers with output going to per-node log files, a printed warning is not a control. Worse, **nothing in `_manifest.json` records which version of the check ran**, so a store built under the degraded check is indistinguishable afterwards from one built under the strict check. That is the same class of defect §D14b was written to prevent for the scene format. | **Route as R1 — strengthen before Phase 2** |

### Flagged issue: peak RSS — ✅ acceptable, no change needed

The developer flagged 2,554 MiB peak RSS against a 2.5 GB ceiling, caused by buffering a
whole 5,000-episode block (5 chunks) before `pa.concat_tables` and writing —
`collect_trajectories.py:527-548`. Confirmed by reading: peak scales with `shard_episodes`,
not `batch_size`, exactly as described. **Checked actual node RAM** rather than reasoning
about it:

| Node | Total RAM | Available | Cores |
|---|---:|---:|---:|
| 101 | 128 GB | 116 GB | 36 |
| 103 | 125 GB | 112 GB | 34 |
| 104 | 125 GB | 112 GB | 36 |
| 105 | 125 GB | 112 GB | 36 |
| 106 | 257 GB | 200 GB | 36 |
| 113 | 515 GB | 489 GB | 64 |

At the default `npar: 16` (cpu), 16 × 2.55 GiB ≈ **41 GB**, which is **36 % of available RAM
on the smallest node measured**. The 2.5 GB figure was a self-imposed plan ceiling, not a
hardware constraint, and the hardware has roughly 3× the headroom needed. **No reduction in
worker count is required and no rewrite to incremental `ParquetWriter` is justified.** The
plan's `≤ 2.5 GB` checkpoint should simply be recorded as superseded by the measured node
capacity. If `shard_episodes` is ever raised above 5,000, re-measure.

### The pending float32 flip — what assumes float16

The pre-registered rule selects `float32`: the measured store-wide saving is **12.8 %**,
below the 20 % adoption threshold, and the rule was registered before the measurement. The
developer correctly escalated rather than rationalising, and correctly changed no default.
Code audit for float16 assumptions:

| Location | Assumes float16? | Verdict |
|---|---|---|
| `collect_trajectories.py` `--obs-precision` | No — `required=True`, `choices=[...]` | ✅ clean |
| `run_collection.py` spec loading | No — `obs_precision` read via `_req` | ✅ clean |
| `trajectory_store.py` `_elem_type` / `_numpy_elem` | No — raises when `obs_precision` is not one of the two | ✅ clean |
| `TRAJECTORY_STORE_SCHEMA.md` | No — carries the corrected 12.8 % figure and a "decision under review" box | ✅ clean |
| `configs/trajectory_collection/example.yaml:23-25` | **Yes** — value `float16` plus comments claiming "~38 % store-wide" and "~400x below … noise" | ⚠️ **R2** |
| `traj_scan.assert_obs_representable` | Structurally — at `float32` it returns after the finiteness check, so **both** magnitude clauses are dormant | ⚠️ **R4** |
| Plan §D11/§D12 text | Yes — superseded predictions retained | Acknowledged as developer follow-up 5 |

R4 is worth stating plainly because it is a consequence nobody has written down: **flipping
the default to `float32` silently retires the observation range guard.** That is defensible
— `float32` has no range problem worth guarding — but it means the four V5b firing cases
cover a code path the production collection will no longer take, and an unbounded or
runaway sensor channel would then be recorded rather than refused. The finiteness check
still runs, which catches `inf`/`nan`.

### Routed items

| # | Item | Severity | Owner | Blocks |
|---|---|---|---|---|
| **R1** | Harden the checkpoint-restore degradation path: make an unreadable checkpoint-metadata tree a hard failure unless an explicit opt-out flag is passed (mirroring `--allow-ambiguous-scene`), and record `restore_check: "strict" \| "degraded"` in `_manifest.json` so a store built under the weak check is self-identifying. Narrow the bare `except Exception` while there. | Medium | `senior-developer` → `developer` | **Phase 2**, not the pilot |
| **R2** | `configs/trajectory_collection/example.yaml:23-25` — replace the "~38 % store-wide / ~400x below noise" comments with the measured 12.8 % / ~100×, and set the example value to whatever precision the user decides (R5). | Low | `developer` | No |
| **R3** | Add one regression test that a wrong-width payload is refused by `build_table` / `_list_array`. The enforcement is structural and works (verified live); it is simply untested. | Low | `developer` | No |
| **R4** | Record in the schema doc that at `obs_precision: float32` the magnitude clauses of the range guard do not run, so only finiteness is checked. | Low | `developer` | No |
| **R5** | **Decision for the user**: precision default. The pre-registered rule selects `float32` (12.8 % < 20 %). Cost of `float32` is ~15 % more bytes on a corpus now measured at ~8 GB/run rather than the budgeted 45 GB — i.e. ~1 GB per run, against a lossless store. My recommendation is to follow the pre-registered rule and use **`float32`**: the threshold existed precisely so this call would not be made after seeing the data, and the measurement removed the only argument for lossy storage. | Decision | user | Phase 1 spec |
| **R6** | Known-Bugs row for the `animal_property_sampled` one-ULP FMA divergence (`core.py:1128-1140`), and closure of the model-size-flag row as verified-fixed (§6). Both confirmed by my own reproduction. | Low | `bug-curator` | No |
| **R7** | C6's "≤ 2.5 GB" checkpoint is superseded by measured node capacity (41 GB of 112 GB available at `npar=16`). Record and close; no code change. | Info | `senior-developer` | No |
| **R8** | §D15's recommendation that `train.py` record a training-time git SHA remains open and out of scope here. | Low | `senior-developer` | No |

### What this verification did NOT cover

- **V6 on the NAS** (`SIGKILL` mid-block, then resume) was operator-run by the developer and
  not repeated by me. I verified its pytest analogues — `test_c9_resume_is_a_noop` and
  `test_resume_after_a_lost_block_is_bit_identical` — both pass. The CIFS mount is confirmed
  (`//192.168.0.250/cocoanlab01`, 70 % used, 62 TB free), so the fsync design is addressing a
  real filesystem property.
- **V2** (5-seed parity against the legacy per-episode loop) was operator-run and not
  repeated; it needs a real checkpoint and a long serial loop.
- **Phase 1 itself.** No collection has been run at any scale beyond the 5,000-episode
  measurement block.
- **The §A11 faithfulness assumption**, which the plan itself states no check can cover.
  §D14's guard refuses ambiguous runs at the door; that guard is verified as firing.

**Conclusion**: ✅ **Ready for the Phase 1 pilot.** All 46 new tests and the previously-red
parity gate pass under my own execution; the four deviations are genuinely forced, correctly
documented, and — for the one that loosens a pre-registered tolerance — independently
corroborated as floating-point rather than logic; the fixture regeneration is legitimate,
verified by re-deriving it with the shipped generator; the maintenance contracts are
satisfied; and the flagged RSS concern is a non-issue against measured lab-node capacity.
Before Phase 2, harden the restore-guard degradation path (R1) and settle the precision
default (R5).

*Verified by: senior-developer*

---

## Feedback from plan-reviewer

> **Date**: 2026-08-19 · **Verdict**: **NOT READY** — two Critical findings, both cheap to fix. Full review: [[plan_trajectory_collection]] (`docs/reviews/plan_trajectory_collection.md`).

1. **🔴 F1 — training-world faithfulness is assumed, never verified.** §A10 declares the saved resolved config the source of truth, but the collector re-loads it through **today's** `load_env_params`, whose legacy-scene precedence (`config_loader.py:429-435`) can rebuild, for pre-`828b77e` runs whose dump carries both scene formats, the scene the trainer *discarded*. V1 and V2 both consume the same `params`, so both are circular with respect to config loading. Required: a hard-fail guard when a saved config contains both a non-empty legacy scene block and an `entities:` block; a documented applicability boundary (runs trained after 2026-07-23); run creation date recorded in the manifest.
2. **🔴 F2 — the red reset-parity gate is unacknowledged.** `tests/env/test_unified_parity.py` fails at step 0 on a clean tree (KNOWN_BUGS.md:73, twice-confirmed, unowned) — standing evidence that env reset behaviour drifted at least once. Triage it before the first production collection, and add the code-drift caveat (training-time git SHA is unrecorded) to the schema doc.
3. **🟡 F3–F6**: strict checkpoint-restore structural check (silent-unmodulated-agent + model-size-flag bugs); a doc↔code schema check plus one bare-pyarrow read (C3 is circular through the shared schema module); a named pilot→validate→scale sequence with V1/V3/V4 run on the *production* store; a whole-store bounds-and-variation check on the realised-draw columns.
4. **🟢 F7–F10**: §D7's GPU scan-buffer figure should be ~2.4 GB, not ~1.8 GB; §D4.1 col 13's "Open Question 1" pointer is stale (damage was decided); pre-state the V1 float-equality tolerance policy; state the explicit conda interpreter in `collect_worker.sh` and label the bush-dwell snippet's output a 0–1 fraction.

Known-bug hazards 2 (fixed-seed repetition), 3 (stale-data blends), and 4 (derived-measure smuggling) are genuinely closed by the design as written.

*Reviewed by: plan-reviewer*

### Re-review addendum (2026-08-19) — verdict revised to **SOUND**

> **Verdict**: **SOUND** — the exit condition of the initial review is met. Full re-review: [[plan_trajectory_collection]] §Re-review.

The revision was verified at source, not taken on trust: the loader-precedence mechanism and the `828b77e` before/after diff match §A11 verbatim; the 334-config corpus scan was **independently reproduced** (153 / 151 / **12 dual-format** / 18, same 12 runs); the F2 commit archaeology checks out (the four failures are exactly the four configs `84014e4` touched, fixtures from `3d20aab` — stale fixtures, not code drift, downgrade to Low justified, durable code-drift residual retained). The §D14 guard is structural (no date heuristic), runs before any store directory can exist, correctly refuses rather than resolves, and V10/C0 test that it *fires*. F3–F10 and the fsync/CIFS handling of open assumption 4 are addressed as claimed, with the untestable node-crash case honestly bounded.

Four new **Low** notes from the revision, none blocking (details and owners in the review doc): (N1) the guard's non-empty-entities predicate differs from the loader's is-not-None predicate on the empty-`entities:`+legacy shape — verified absent from the corpus, harden or comment; (N2) V10-as-pytest depends on gitignored NAS run dirs — copy one dual-format YAML into `tests/fixtures/`; (N3) directory-fsync may be unsupported on CIFS — decide the fallback explicitly, V6-on-NAS will surface it; (N4) the Known Bugs registry row still says the parity failure is untriaged — `bug-curator` to record the §D15 diagnosis.

*Reviewed by: plan-reviewer*
