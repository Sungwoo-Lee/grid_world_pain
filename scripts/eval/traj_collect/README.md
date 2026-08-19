# `traj_collect/` — trajectory collection, operator guide

Put a trained agent back into the **exact world it was trained in**, run it for a very
large number of episodes, and record both the behaviour and the per-episode random draws
the environment made in secret at reset. Output is a sharded Parquet store with a schema
that is identical for every training run, forever.

- **Schema contract (read this before analysing anything)**: [`docs/environment/TRAJECTORY_STORE_SCHEMA.md`](../../../docs/environment/TRAJECTORY_STORE_SCHEMA.md)
- **Plan and rationale**: [`docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md`](../../../docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md)

| File | What it is |
|---|---|
| `collect_trajectories.py` | single-run, single-process collector — the only thing that writes shards |
| `traj_scan.py` | the `nnx.jit` batched rollout kernel + the float16 range guard |
| `collect_worker.sh` | per-node worker: `xargs -P npar` over a worklist of block ranges |
| `run_collection.py` | multi-node driver: spec YAML → LPT partition → serial launch → validate |
| `gen_schema_doc.py` | regenerates the schema doc's column tables from the code |

Schema, writer, reader and every guard live in [`src/utils/trajectory_store.py`](../../../src/utils/trajectory_store.py).
**Nothing else in the codebase may define these column names.**

---

## Run one run locally

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  scripts/eval/traj_collect/collect_trajectories.py \
    --run          results/JAX_RecurrentPPO/20260816-152742_rppo_restpremNH_a10_n112 \
    --checkpoint   final \
    --out-root     results/trajectories \
    --episodes     25000 \
    --seed-base    1000000 \
    --blocks       0:5 \
    --obs-precision float32 \
    --device cpu
```

`--obs-precision` is **required and has no default**: `float16` discards information, so a
collection cannot be launched without someone stating the measurement precision they
accepted. **The recommended value is `float32`**, decided 2026-08-20 by measurement — half
precision saved 12.8 % store-wide on a real paired collection, below the 20 % threshold
pre-registered before the measurement. `float16` stays supported and is a sound *accuracy*
choice (worst measured error 1.95e-03, ~100× below the environment's own injected sensor
noise); note that under `float32` the observation guard's two magnitude clauses go dormant.
See the schema doc §5.

`--checkpoint final` selects by **numeric** maximum, never lexicographic (the reference run
has 591 unpadded checkpoint directories, where a lexicographic maximum picks `9900021`
instead of `59100070`).

`--allow-weak-restore-check` exists but should essentially never be used: without it, a
checkpoint whose own structure cannot be read is **refused**, because the fallback
comparison is structurally blind to a model that differs from the checkpoint (the
silent-unmodulated-agent bug). Using it stamps `restore_check: weak_allowed` into the
manifest so every reader inherits the caveat.

Output lands at `<out-root>/<run_tag>/<ckpt_step>/<env_fp>/`.

## Run a spec across nodes

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  scripts/eval/traj_collect/run_collection.py configs/trajectory_collection/example.yaml
```

`--dry-run` prints the launches without running anything. `--validate-only` skips
launching and validates the stores the spec names.

The driver expands `(run × checkpoint × block-range)` cells, LPT-partitions them across
nodes, writes one worklist per node under `<out_root>/_scratch/<name>/_worklists/`,
launches each node **serially** through `run_command.py`, polls
`<scratch>/_run_markers/done_<node>`, then validates every store.

> **`run_command.py` is not parallel-safe.** Concurrent invocations race through a shared
> SSH control socket and can return one node's answer to every caller. The driver launches
> one node at a time on purpose — do not "speed it up" with a thread pool.

## Resume

Resume is stateless and free: re-run the identical command. A block is complete iff both
of its shards exist, and completed blocks are skipped. Because block `b` covers episodes
`[shard_episodes·b, shard_episodes·(b+1))` and episode `i` uses seed `seed_base + i`, a
redone block is **bit-identical** to what a dead process would have produced — resume can
never produce a mixed population.

If the manifest of an existing store disagrees with what you are about to write
(`env_fp`, `seed_base`, `n_episodes`, `shard_episodes`, `obs_precision`, `dims`,
`max_steps`, `checkpoint_path`, `schema_version`), the collector raises and writes
**nothing**.

## Validate

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python - <<'PY'
from src.utils.trajectory_store import (validate_store_structure, validate_store_shapes,
                                        validate_store_draws)
d = "results/trajectories/<run_tag>/<ckpt_step>/<env_fp>"
print(validate_store_structure(d))   # blocks present, no duplicate seeds, T+1 rows per episode
validate_store_shapes(d)             # schema + list widths equal the manifest
validate_store_draws(d)              # whole-store bounds / non-degeneracy / activation counts
PY
```

`validate_store_draws` is the important one: the realised draws are the entire point of
this store, and a degenerate sampler or a column wired to a constant would produce
*plausible-looking* independent variables — the worst possible failure mode. A failure
here is a hard stop: delete the store and find the cause.

## CPU vs GPU — the two knobs move in opposite directions

|  | `--device cpu` (default) | `--device gpu` |
|---|---|---|
| where the parallelism lives | many single-threaded processes (`npar`) | one big vmap batch |
| `npar` per node | `min(cores − 2, 16)` | **1 per GPU** — a second JAX process on one GPU contends for memory and is *slower* |
| `batch_size` | 1024 (host-RAM bound; ~1.8 GB peak RSS) | 8192 (GPU-memory bound; ~2.4 GB scan buffer) |

Set them **together**; the driver derives both from `device` unless both are given
explicitly, and writes the resolved values into `_manifest.json`.

> **GPU limitation — multi-GPU fan-out does NOT work today.** The worker exports
> `JAX_PLATFORMS=cuda` but sets no `CUDA_VISIBLE_DEVICES`, and the spec has no GPU-index
> field, so every GPU worker on a node lands on **GPU 0**. Only reachable with
> `device: gpu`; the default is `cpu`, where this does not arise.

`device` is a **manifest-guarded** field: a store started on CPU cannot be resumed on GPU.
Bit-level results are lowering-dependent (schema doc §5), so a mixed-device store would
hold bit-inconsistent blocks for what the manifest claims is one homogeneous population.

Before any GPU launch, consult [`docs/environment/LAB_NODE_GPU_SPEC.md`](../../../docs/environment/LAB_NODE_GPU_SPEC.md)
for which GPU indices exist on the target node and check live occupancy with
`scripts/lab/gpu_status.py`. CPU collection can run on nodes that are busy training on
GPU; GPU collection cannot.

## Applicability boundary — which runs this accepts

Runs trained **after 2026-07-23** (commit `828b77e`) are unambiguous. Any run whose saved
config carries **both** a modern `entities:` block and a legacy
`predators:`/`neutral_animals:` block cannot be faithfully reconstructed — the trainer's
scene precedence changed at that commit, and nothing in the run directory records which
branch built its world — so the collector **refuses** it. It deliberately does not pick a
scene: there is no correct scene to reconstruct, only a choice.

`--allow-ambiguous-scene` downgrades the refusal to a loud warning and stamps
`scene_ambiguous: true` into the manifest. Use it only when you have independently
established which scene is correct. Measured against the results tree as of 2026-08-19,
exactly 12 of 334 saved configs are affected, all dated 2026-05-29 to 2026-06-11.

## Data-loss rules (this store is gitignored data on a NAS with no symlink support)

- **Never** `git clean -x` / `-X` / `-fdx` / `-fdX`. `git clean -fd` is safe; always preview
  with `git clean -fdn` first and surface the listed paths.
- **Never** force-checkout or force-switch branches without checking whether the
  destination branch tracks paths currently untracked locally.
- **Avoid** `git stash -u` followed by `git stash drop`.
- Snapshot before any merge / rebase / branch switch. `git reset --hard` alone is safe for
  gitignored data; the danger is the `git clean -x` that often follows it.

One mitigation is built in: because every block is a pure function of
`(seed_base, shard_episodes)`, a partial loss is recoverable by re-running only the missing
shards. Losing `_manifest.json` alone is **not** recoverable that way — it carries the
resolved config, `seed_base` and `obs_precision` — which is why the driver writes it first,
before any shard, and never rewrites it.

## Supported algorithms

**rPPO only.** The checkpoint-loading and rollout seams (`load_policy`, `policy_step`) are
kept generic so Dreamer-SRL can be added later, but Dreamer is not implemented, not
tested, and not claimed: `--algo` anything other than `rppo` raises `NotImplementedError`.
