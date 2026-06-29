# `basic_curriculum/` — frozen 5-stage curriculum source for `--configs-dir`

## What this directory is

This directory holds the **five curriculum stages** (`00`–`04`) that the basic
continual-learning curriculum trains through, in order. It exists to be passed to
`train.py` via `--configs-dir`:

```
--configs-dir configs/environment/experiment/basic_curriculum \
--continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml
```

`train.py --configs-dir <dir>` globs **every** `*.yaml` in the directory and treats each
as one curriculum stage, in filename-sorted order. The number of stage files must equal
the number of `episode_boundaries` in the `--continual-schedule`. The long-L4 schedule
defines **5** boundaries, so this directory deliberately contains **exactly 5** stage
files.

## Why it exists separately from `basic/`

The sibling directory `configs/environment/experiment/basic/` now contains a **6th** file,
`05-random_init_10x10.yaml`, added for a **standalone** (non-curriculum) run that
references that exact path. Because `--configs-dir` globs all YAMLs, pointing it at
`basic/` would pick up the 6th file and break the stage↔boundary count match
(`episode_boundaries length (5) != number of stage configs (6)`).

This directory is the **curriculum-only** view: stages `00`–`04`, no `05`. The standalone
`05-random_init_10x10.yaml` stays isolated in `basic/` and must **not** be copied here —
adding it would re-break the 5-vs-6 count.

## Relationship to `basic/` (keep in sync)

The five files here are **byte-identical copies** of the canonical originals in
`configs/environment/experiment/basic/` (the NAS does not support symlinks, so these are
real `cp` copies, not links):

| Stage file (canonical original under `basic/`) |
|---|
| `00-static_predator_5x5.yaml` |
| `01-slow_predator_5x5.yaml` |
| `02-fast_predator_8x8.yaml` |
| `03-predator_and_rabbit_10x10.yaml` |
| `04-far_sight_predator_10x10.yaml` |

**Canonical source = `configs/environment/experiment/basic/` stages 00–04.** If you edit a
curriculum stage, edit the original under `basic/` and re-copy here (or edit both), or the
two copies will silently drift. Verify parity with:

```
for f in 00-static_predator_5x5 01-slow_predator_5x5 02-fast_predator_8x8 \
         03-predator_and_rabbit_10x10 04-far_sight_predator_10x10; do
  diff configs/environment/experiment/basic/$f.yaml \
       configs/environment/experiment/basic_curriculum/$f.yaml \
    && echo "$f IDENTICAL"
done
```

## Obs / action / sensor parity

All five stages keep `extends: environment/default` (an alias path that resolves the same
from this location as from `basic/`) and inherit the `sensory` / `body` / `noise` /
`behavior_measures` blocks unchanged from `environment/default`. They are byte-identical
copies, so they share **obs_dim = 27**, **action_dim = 6**, and an **identical sensor
fingerprint** with each other and with the `basic/` originals. `--configs-dir` therefore
sees a uniform observation/action interface across the curriculum, as required for a
single recurrent agent to train through all five stages.
