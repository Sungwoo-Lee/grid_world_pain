# Archived: DreamerV3-NNX stack (in-house JAX/Flax-NNX DreamerV3)

**Archived**: 2026-07-10 · **Development stopped**: 2026-05-11 · **Live replacement**: `src/algorithms/dreamer_srl/`

## Why this is archived

This directory holds the project's original in-house DreamerV3 implementation — a
JAX/Flax-NNX world-model agent that was trained through the shared `train.py` entry
point. It was **abandoned as a research vehicle on 2026-05-11**, when a PI call
pivoted the world-model line to a direct port of the reference sheeprl
implementation. The pivot decision is recorded in the memory insight
[sheeprl direct pivot, JAX Dreamer abandoned (2026-05-12)](../../../../docs/memory/memories/dreamer_diagnosis/20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned.md).

The stack was physically archived because it kept attracting real engineering effort
by accident: in the week of 2026-07-06 an entire recipe-alignment fix batch — seven
fixes with regression tests (commits `32c67ca`, `7304e75`, the "WP-NNX" work
package) — was planned, implemented, reviewed, and verified **against this abandoned
implementation**. Archive plan:
`docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/archive_plan_dreamer_v3_nnx.md`.

## Contents

- `dreamer_v3_trainer.py` — trainer + ReplayBuffer + losses (was `src/models/dreamer_v3_trainer.py`)
- `dreamer_v3_nnx.py` — agent/RSSM/WorldModel/ActorCritic NNX modules
- `dreamer_v3_util.py` — symlog/twohot/Ratio/Moments utilities
- `dreamer_v3_network.py` — already-dead flax.linen legacy (0 importers even pre-archive)
- `modulated_layer_norm_gru_cell.py` — NNX RSSM neuromodulation GRU cell (imported only
  by `dreamer_v3_nnx.py`; the live rPPO-NMN cell is the *separate*
  `src/models/modulated_gru_cell.py`, which stays live)
- `tests/` — 9 archived test files + 2 import-support helpers (incl. the WP-NNX F1–F7
  regression tests)
- `scripts/dreamer_offline_wm_test.py` — hand-run offline world-model diagnostic
- Configs: `configs/models/archive/dreamer_v3_nnx/` (8 YAMLs)

The stack stays **importable in place**
(`from src.models.archive.dreamer_v3_nnx.dreamer_v3_trainer import DreamerTrainer`)
for history archaeology. `git log --follow` reaches pre-move history.

## Running the archived tests

Skipped by default (module-level guard). Explicit opt-in:

```bash
GWP_RUN_ARCHIVED_NNX_TESTS=1 /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    -m pytest src/models/archive/dreamer_v3_nnx/tests/ -q
```

## Notes

- `src/algorithms/dreamer_srl/*` provenance comments still cite pre-move
  `src/models/dreamer_v3_trainer.py` line numbers — those refer to **this archived
  copy** (line numbers are as of the move; the file is frozen).
- `train.py` and `evaluation.py` now raise a fail-fast `ValueError` for
  `agent.algorithm: DreamerV3`, pointing here and at `dreamer_srl`.
