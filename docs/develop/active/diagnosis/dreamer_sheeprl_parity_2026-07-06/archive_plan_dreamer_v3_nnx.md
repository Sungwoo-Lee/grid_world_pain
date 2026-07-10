---
title: "Archive the legacy DreamerV3-NNX stack (src/models/dreamer_v3_*) — move to tracked archive, stub train/eval dispatch"
topic: diagnosis
status: active
created: 2026-07-10
last_updated: 2026-07-10
---

# Archive plan: legacy DreamerV3-NNX stack

> **Status**: PLANNED
> **Opened**: 2026-07-10
> **Related**: [[06_nnx_recipe_deviation_register]] · [[fix_plan_nnx_parity]] · [[05_dreamer_v3_nnx_conventions]] · pivot decision memory: [sheeprl direct pivot, JAX Dreamer abandoned (2026-05-12)](../../../../memory/memories/dreamer_diagnosis/20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned.md) · [[KNOWN_BUGS]] (rows listed in §Bug-curator hand-off)

---

## Context

The project has **two** DreamerV3 implementations. The in-house one — a JAX/Flax-NNX
world-model agent living in `src/models/dreamer_v3_*` and trained through the shared
`train.py` entry point — was **abandoned as a research vehicle on 2026-05-11**, when a
PI call pivoted the world-model line to a direct port of the reference sheeprl
implementation (the pivot decision is recorded in the memory insight linked above).
The **live** Dreamer is `src/algorithms/dreamer_srl/`, which has its own trainer,
tests, and configs.

The abandoned stack was never physically separated from the live tree, and it keeps
attracting real engineering effort by accident: this week an entire recipe-alignment
fix batch — seven fixes with regression tests (commits `32c67ca`, `7304e75`, the
"WP-NNX" work package) — was planned, implemented, reviewed, and verified **against
the abandoned implementation**. The user has now decided (recorded this session):
archive the DreamerV3-NNX stack so future sessions stop spending effort on it.

This plan moves the whole stack (4 source modules + 1 support module + 8 configs +
11 test files + 1 hand-run diagnostic script) into a **git-tracked archive location**,
surgically removes/stubs the DreamerV3 dispatch branches in `train.py` and
`evaluation.py` (both heavily shared with the live RecurrentPPO path), keeps the
archived code importable-in-place for history archaeology, and hands the affected
Known-Bugs rows to `bug-curator`. Nothing in the live tree may import the archived
stack afterward.

**Not in scope**: `src/algorithms/dreamer_srl/` (live), `tests/algorithms/dreamer_srl/`
(live), `vendor/sheeprl` (reference), `pytorch_agents/` + `scripts/lab/launch_sheeprl.sh`
(sheeprl launch path), and the `bm_drive_batch` behavior-measure helper (see §Analysis A6).

## Analysis

### A1 — Why the archive location must be chosen carefully

Two traps, both verified against the repo's `.gitignore`:

1. The repo-root `legacy/` directory is **gitignored** (`legacy/` rule). Moving
   tracked files there would silently untrack them and destroy forward history.
2. `.gitignore` also carries the glob **`*legacy*`** — *any* path with "legacy" in a
   component name is ignored. So `src/models/legacy_dreamer/` would be just as fatal.
   The archive path must not contain the substring "legacy".

**Chosen location: `src/models/archive/dreamer_v3_nnx/`** (tests and the diagnostic
script move inside it — fully self-contained). Rationale:

- **Tracked**: matches the repo's existing in-tree archive precedent —
  `configs/environment/experiment/archive/` (tracked, verified via `git ls-files`)
  and `docs/develop/archive/`. No gitignore rule matches `archive/`.
- **Importable-in-place**: `src.models.archive.dreamer_v3_nnx.dreamer_v3_trainer`
  is a valid module path, so history archaeology (re-running an archived test,
  loading an old checkpoint's class definitions) needs no `sys.path` games.
- **`git mv` preserves `git log --follow`** across the move.
- Configs mirror the same precedent: `configs/models/archive/dreamer_v3_nnx/`.

### A2 — Full inventory (verified by repo-wide grep, 2026-07-10)

**Source modules (`src/models/`), all NNX-only:**

| File | Role | Live importers outside the stack? |
|---|---|---|
| `dreamer_v3_trainer.py` | trainer + ReplayBuffer + losses | `train.py:829`, `evaluation.py:415`, `scripts/dreamer/dreamer_offline_wm_test.py:82`, `tests/scripts/test_evaluation_model_rebuild.py:185` — all handled below |
| `dreamer_v3_nnx.py` | agent/RSSM/WM/AC modules | only `dreamer_v3_trainer.py` |
| `dreamer_v3_util.py` | symlog/twohot/Ratio/Moments | `train.py:843` (Ratio), trainer, nnx, offline script, 2 tests |
| `dreamer_v3_network.py` | **already-dead** flax.linen legacy | none (0 importers) |
| `modulated_layer_norm_gru_cell.py` | NNX RSSM neuromodulation GRU cell | **only** `dreamer_v3_nnx.py:8` — goes with the stack. (The rPPO modulated cell is the *separate* `modulated_gru_cell.py`, which stays.) |

`src/models/__init__.py` is empty — no export list to update.

**Configs**: `configs/models/dreamer_v3/` — 8 YAMLs (`dreamer_v3.yaml`,
`dreamer_v3_curriculum.yaml`, `dreamer_v3_curriculum_probe.yaml`, `dreamer_v3_probe.yaml`,
`dreamer_v3_probe_cont10.yaml`, `dreamer_v3_rr06.yaml`, `dreamer_v3_sheeprl_matched.yaml`,
`neuromodulated_dreamer_v3.yaml`).

**Tests (`tests/models/`)** — 9 test files (29 collected tests) + 2 import-support
helpers, including the WP-NNX regression tests added this week:

- `test_dreamer_collect_arrival_alignment.py`, `test_dreamer_continue_truncation.py`,
  `test_dreamer_replay_buffer_wrap.py` (older H6/H7/Finding-B era)
- `test_dreamer_nnx_is_first_reset.py`, `test_dreamer_nnx_stop_gradients.py`,
  `test_dreamer_nnx_obs_loss_sum.py`, `test_dreamer_nnx_online_bootstrap.py`,
  `test_dreamer_nnx_terminal_start_weights.py`, `test_dreamer_nnx_replay_ratio_semantics.py`
  (this week's WP-NNX F1–F7)
- helpers: `dreamer_nnx_fixtures.py`, `dreamer_nnx_rollout_replica.py`

Import mechanics (verified): the tests add their own directory to `sys.path` and import
the helpers by bare name (`from dreamer_nnx_fixtures import ...`) — those survive the
move unchanged. Only the `from src.models.dreamer_* import ...` lines and the fixture's
config path (`dreamer_nnx_fixtures.py:42`, `test_dreamer_collect_arrival_alignment.py:54`
→ `configs/models/dreamer_v3/dreamer_v3.yaml`) need repointing.

**Scripts**: `scripts/dreamer/dreamer_offline_wm_test.py` (imports the NNX trainer at
L82–83; hand-run only per [[SCRIPTS_DEPENDENCY_MAP|docs/environment/SCRIPTS_DEPENDENCY_MAP.md]]
row "referenced in sibling's comments only") — moves into the archive.
The other `scripts/dreamer/*` files (`dreamer_srl_offline_check.py`,
`dreamer_srl_offline_wm_test.py`, `sheeprl_jax_diff.py`, `visualize_dream.py`) import
**only** `src.algorithms.dreamer_srl` — they stay.

**Entry points**: `train.py` DreamerV3 branches (§A3), `evaluation.py` DreamerV3 branch
(§A4), plus one live test that exercises the evaluation branch (§A5).

**References that are mentions-only (no import — enumerated in §File Changes part D as
pointer updates or explicit leave-as-is):** `.claude/agents/training-runner.md`,
`train_command-agent.sh` comments, `train_command-new.sh` (user-owned, gitignored),
`scripts/wandb/{wandb_metrics,compare_wandb_runs,benchmark_wandb_speed}.py` (metric
preset named `dreamer_v3` + old-run-name examples), `scripts/eval/eval_rollout.py`
(detects "dreamer" agent type and already raises `NotImplementedError` at L1011),
`src/algorithms/dreamer_srl/*` provenance comments citing `dreamer_v3_trainer.py`
line numbers, `configs/train/default.yaml:14` comment.

### A3 — `train.py` Dreamer-only regions (per-region decision)

`train.py` is shared with the **live** rPPO path; every region below was read in
place and classified. Strategy: **one early fail-fast stub + wholesale deletion of
the now-unreachable DreamerV3 branches.** The stub goes immediately after
`algorithm = config.get_mandatory('agent.algorithm')` (L452), which makes every
downstream deletion provably safe (the interpreter can never reach a DreamerV3
branch). Leaving the dead branches in place is explicitly rejected — dead
Dreamer-only code in `train.py` is exactly what misdirected this week's effort.

| # | Region (current lines) | What it is | Decision |
|---|---|---|---|
| T1 | L452 (after) | algorithm resolution | **ADD stub**: `if algorithm == "DreamerV3": raise ValueError(...)` with a clear "archived, use dreamer_srl" message (exact text in File Changes) |
| T2 | L463–467 | continual-learning guard tuple `("RecurrentPPO", "DreamerV3")` | **EDIT**: drop `"DreamerV3"` from tuple + message |
| T3 | L501–511 | hyperparam-summary `elif algorithm == "DreamerV3"` (collect_interval / rssm_deter_dim / actor_lr) | **DELETE** |
| T4 | L828–905 | trainer + Ratio + F7 prefill accounting (`learning_starts`, `prefill_env_steps`) + ReplayBuffer + positive buffer + `dreamer_state` init | **DELETE** (whole `elif` incl. the F7/Ratio block and `agent.learning_starts` read — both exist only here) |
| T5 | L1143–1160 | checkpoint-restore DreamerV3 branch | **DELETE** branch; **EDIT** the `else` error message at L1173–1175 ("only RecurrentPPO and DreamerV3 save checkpoints" → "only RecurrentPPO saves checkpoints") |
| T6 | L1263–1285 | stage-swap replay-buffer clear (Dreamer-only) | **DELETE** |
| T7 | L1293–1300 | stage-swap `dreamer_state` re-init `elif` | **DELETE** (keep the rPPO `if` at L1290–1292) |
| T8 | L1542–1904 | the entire DreamerV3 collect/train loop `elif` — includes the H10 BM **Site-2** block (L1660–1676, L1697–1710, L1768–1771) and the F7 train gate (L1827–1856) | **DELETE** wholesale (single contiguous `elif`; verified no rPPO code inside). BM Site-1 (rPPO) is untouched |
| T9 | L2484–2494 | checkpoint-save DreamerV3 `elif` | **DELETE** |
| T10 | L2518, L2533 | `model=model if algorithm == "RecurrentPPO" else trainer.agent` (eval passes 1+2) | **EDIT** to `model=model` (the `else` arm references the deleted `trainer`) |
| T11 | L7, L15 | module docstring + `--algorithm` help mentioning DreamerV3 | **EDIT**: note "DreamerV3-NNX archived → src/models/archive/dreamer_v3_nnx/; use dreamer_srl" |
| T12 | L251, L273 | profiler trace-dir string `"..._dreamer_v3_vs_rppo_profile"` | **LEAVE** (inert tmp-dir naming; renaming would churn profile tooling docs) |
| T13 | L310–313 | comment: `default.yaml` sharing with Dreamer | **LEAVE** (historical comment; `configs/train/default.yaml` has uncommitted user edits — do not touch) |

After T1–T10, `grep -n "dreamer\|Dreamer" train.py` must return only the stub,
T11's pointer, and the T12/T13 leftovers. Any other hit is an implementation error.

### A4 — `evaluation.py` DreamerV3 branch

- L414–445: `elif algorithm == "DreamerV3":` rebuilds a `DreamerTrainer` and restores
  a checkpoint. **Decision: stub, not delete** — replace the body with the same
  archived-stack `ValueError`. Rationale: unlike `train.py` (where the stub at the
  top makes branches unreachable), `evaluation.py` reads `algorithm` from an
  *on-disk results dir* — old DreamerV3-NNX result folders exist under `results/`,
  and a user pointing evaluation at one should get the clear archived message, not
  an `UnboundLocalError`.
- L161 docstring ("both RecurrentPPO and DreamerV3") — pointer note.

### A5 — Live tests touching the stack

- `tests/scripts/test_evaluation_model_rebuild.py::test_dreamer_eval_rebuild_signature`
  (L156–231): spies on `DreamerTrainer.__init__` through the real `evaluation.py`
  branch. With A4's stub it becomes meaningless. **Decision: replace 1:1** with
  `test_dreamer_eval_archived_stub` — same on-disk scaffolding, asserts
  `pytest.raises(ValueError, match="archived")` and that the message names
  `dreamer_srl`. This pins the stub's contract (count-neutral for the suite).
- `tests/environment/test_bm_dreamer_batch_driver.py` — **STAYS.** It tests
  `bm_drive_batch` (`src/behavior/accumulators.py:411`) with synthetic arrays and
  imports nothing from the NNX stack.

### A6 — Known orphan (deliberate, do not delete)

`bm_drive_batch` in `src/behavior/accumulators.py` loses its only production caller
(train.py Site-2, region T8). It is algorithm-agnostic ([T,B] batch driver) and is
the natural building block if `dreamer_srl` ever grows a Site-2 behavior-measure
hook — **keep it and its test**, record it here as a known orphan. Do not "clean it
up" in this change.

### A7 — Archived tests: skip-by-default mechanics

The repo has **no pytest `testpaths`** configured, so a bare `pytest` from the repo
root would collect anything importable. Two layers keep the archive quiet:

1. Tests live at `src/models/archive/dreamer_v3_nnx/tests/` — the normal invocation
   (`pytest tests/`) never sees them.
2. Each archived test module gets a module-level guard so even a bare `pytest` or an
   accidental explicit run is an explicit opt-in:
   ```python
   import os, pytest
   if os.environ.get("GWP_RUN_ARCHIVED_NNX_TESTS") != "1":
       pytest.skip("archived DreamerV3-NNX stack — set GWP_RUN_ARCHIVED_NNX_TESTS=1 to run",
                   allow_module_level=True)
   ```
   (`allow_module_level=True` skips before the heavy JAX imports run.)

They remain runnable for archaeology:
`GWP_RUN_ARCHIVED_NNX_TESTS=1 /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest src/models/archive/dreamer_v3_nnx/tests/ -q`

## Implementation Plan

### Design

1. **Archive location** `src/models/archive/dreamer_v3_nnx/` (rationale §A1), created
   with `git mv` per file so history follows. Self-contained: code + tests + the one
   diagnostic script + a README.
2. **Fail-fast stubs, not silent removal**: `train.py` raises at algorithm resolution;
   `evaluation.py` raises inside its (reachable-from-old-results) branch. Both
   messages name the archive path, the live replacement (`src/algorithms/dreamer_srl/`),
   and the pivot date.
3. **Wholesale deletion of unreachable branches** in `train.py` (T2–T10) so no dead
   Dreamer-only code remains in the shared file.
4. **Archived stack stays importable and its tests runnable via env-var opt-in** (§A7).
5. **Same-change doc maintenance**: `SCRIPTS_DEPENDENCY_MAP.md` (Maintenance
   Contract — a `scripts/` file moves), status notes on
   [[06_nnx_recipe_deviation_register]] and [[fix_plan_nnx_parity]].
   `KNOWN_BUGS.md` is **not** edited here — `bug-curator` owns it (§Hand-off).

### File Changes

#### Part A — moves (`git mv`, history-preserving)

| # | From | To |
|---|---|---|
| 1 | `src/models/dreamer_v3_trainer.py` | `src/models/archive/dreamer_v3_nnx/dreamer_v3_trainer.py` |
| 2 | `src/models/dreamer_v3_nnx.py` | `src/models/archive/dreamer_v3_nnx/dreamer_v3_nnx.py` |
| 3 | `src/models/dreamer_v3_util.py` | `src/models/archive/dreamer_v3_nnx/dreamer_v3_util.py` |
| 4 | `src/models/dreamer_v3_network.py` | `src/models/archive/dreamer_v3_nnx/dreamer_v3_network.py` |
| 5 | `src/models/modulated_layer_norm_gru_cell.py` | `src/models/archive/dreamer_v3_nnx/modulated_layer_norm_gru_cell.py` |
| 6 | `configs/models/dreamer_v3/` (8 YAMLs, whole dir) | `configs/models/archive/dreamer_v3_nnx/` |
| 7 | `tests/models/test_dreamer_collect_arrival_alignment.py` | `src/models/archive/dreamer_v3_nnx/tests/` |
| 8 | `tests/models/test_dreamer_continue_truncation.py` | 〃 |
| 9 | `tests/models/test_dreamer_replay_buffer_wrap.py` | 〃 |
| 10 | `tests/models/test_dreamer_nnx_is_first_reset.py` | 〃 |
| 11 | `tests/models/test_dreamer_nnx_stop_gradients.py` | 〃 |
| 12 | `tests/models/test_dreamer_nnx_obs_loss_sum.py` | 〃 |
| 13 | `tests/models/test_dreamer_nnx_online_bootstrap.py` | 〃 |
| 14 | `tests/models/test_dreamer_nnx_terminal_start_weights.py` | 〃 |
| 15 | `tests/models/test_dreamer_nnx_replay_ratio_semantics.py` | 〃 |
| 16 | `tests/models/dreamer_nnx_fixtures.py` | 〃 |
| 17 | `tests/models/dreamer_nnx_rollout_replica.py` | 〃 |
| 18 | `scripts/dreamer/dreamer_offline_wm_test.py` | `src/models/archive/dreamer_v3_nnx/scripts/dreamer_offline_wm_test.py` |

New files: `src/models/archive/__init__.py` (empty),
`src/models/archive/dreamer_v3_nnx/__init__.py` (short docstring: archived 2026-07-10,
pivot memory link, "do not import from live code"),
`src/models/archive/dreamer_v3_nnx/README.md` (why archived; cite the 2026-05-12 pivot
memory and the misdirected WP-NNX batch `32c67ca`/`7304e75`; how to run the archived
tests; note that `dreamer_srl` provenance comments still cite pre-move
`dreamer_v3_trainer.py` line numbers — those refer to this archived copy).

#### Part B — content edits inside moved files (the only in-file changes)

1. Import repointing, `src.models.dreamer_v3_*` / `src.models.modulated_layer_norm_gru_cell`
   → `src.models.archive.dreamer_v3_nnx.*`, at (pre-move paths):
   `dreamer_v3_nnx.py:7–8`, `dreamer_v3_trainer.py:9–10,124`,
   `dreamer_nnx_fixtures.py:33`, `dreamer_nnx_rollout_replica.py:24,27`,
   `test_dreamer_continue_truncation.py:20`, `test_dreamer_collect_arrival_alignment.py:45`,
   `test_dreamer_nnx_online_bootstrap.py:32`, `test_dreamer_nnx_obs_loss_sum.py:28`,
   `test_dreamer_replay_buffer_wrap.py:38`, `test_dreamer_nnx_replay_ratio_semantics.py:33`,
   `dreamer_offline_wm_test.py:82–83`. (The bare-name `from dreamer_nnx_fixtures import ...`
   lines stay — the tests self-insert their dir into `sys.path`, §A2.)
2. Config-path repointing to `configs/models/archive/dreamer_v3_nnx/dreamer_v3.yaml`:
   `dreamer_nnx_fixtures.py:42`, `test_dreamer_collect_arrival_alignment.py:54`.
3. `_REPO` / repo-root depth fix in the moved files that compute it from `__file__`
   (fixtures, arrival-alignment test, `dreamer_offline_wm_test.py`) — the archive is
   deeper than `tests/models/` and `scripts/dreamer/`; per the depth hazard noted in
   `SCRIPTS_DEPENDENCY_MAP.md`, recount the `os.path.dirname` hops.
4. §A7 module-level skip guard added to the 9 archived test modules.

#### Part C — live-tree edits

##### `train.py`

- **T1 stub** (insert after L452):
  ```python
  if algorithm == "DreamerV3":
      raise ValueError(
          "The in-house DreamerV3 (NNX) stack was archived on 2026-07-10 "
          "(development stopped 2026-05-11; superseded by the sheeprl-parity port). "
          "Use src/algorithms/dreamer_srl/ (entry: src/algorithms/dreamer_srl/"
          "dreamer_srl_main.py) instead. Archived code: src/models/archive/"
          "dreamer_v3_nnx/ — see its README and docs/develop/active/diagnosis/"
          "dreamer_sheeprl_parity_2026-07-06/archive_plan_dreamer_v3_nnx.md.")
  ```
- **T2–T11** exactly per the §A3 table (delete regions T3, T4, T6, T7, T8, T9
  wholesale; edit T2 tuple/message, T5 else-message, T10 both call sites to
  `model=model`, T11 docstring/help pointers). T12/T13 untouched.

##### `evaluation.py`

- L414–445: replace the `elif algorithm == "DreamerV3":` body with the same
  `ValueError` text as T1 (keep the `elif` so old on-disk DreamerV3 results dirs get
  the clear message, §A4). Remove the now-unused `dreamer_mod_config` lines with it.
- L161: docstring pointer update.

##### `tests/scripts/test_evaluation_model_rebuild.py`

- Replace `test_dreamer_eval_rebuild_signature` (L156–231) with
  `test_dreamer_eval_archived_stub` per §A5: reuse the existing checkpoint-scaffold
  code (config with `agent.algorithm: DreamerV3`, minimal orbax save), drop the
  `DreamerTrainer` spy entirely (no NNX import may remain in this file), assert
  `pytest.raises(ValueError, match="archived")` from `ev.main()` and that the
  message contains `dreamer_srl`.

#### Part D — docs + pointer updates (same change)

| File | Change |
|---|---|
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | Maintenance Contract: `scripts/dreamer/dreamer_offline_wm_test.py` row (§ table L147) → moved to `src/models/archive/dreamer_v3_nnx/scripts/`; update the Cluster C note (L166) to drop it; scan for any other mention of the moved path |
| `docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/06_nnx_recipe_deviation_register.md` | Status note under the title: stack archived 2026-07-10 to `src/models/archive/dreamer_v3_nnx/` (this plan); register frozen; open rows **U6 and R1 are moot unless the stack is revived**. Frontmatter `last_updated` bump |
| `docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/fix_plan_nnx_parity.md` | Same status note under the Status banner (WP-NNX landed in `32c67ca`/`7304e75`, then the stack was archived; file paths in the plan now refer to the archive). Frontmatter bump |
| `.claude/agents/training-runner.md` | Remove `dreamer_v3_nnx` as a supported JAX launch path (description + L12 + L26 table) → JAX path is `recurrent_ppo` (+ `dreamer_srl` where applicable); sheeprl path unchanged. **Caution: file currently carries uncommitted user modifications — developer must rebase the edit on the working-tree version, not HEAD** |
| `train_command-agent.sh` | Comment-only: L42/L94 examples cite `configs/models/dreamer_v3/...` — repoint or swap to an rPPO example. **Also has uncommitted user modifications — same caution** |
| `docs/develop/INDEX.md` | regenerated via `scripts/claude/regen_dev_index.py` (never hand-edit) |

**Explicit leave-as-is (checked, mention-only):** `train_command-new.sh` (user-owned
AND gitignored — flag to user only: its commented example L56 cites the old config
path); `scripts/wandb/*` (`dreamer_v3` metric-preset name + old run-name examples —
needed to analyze historical runs); `scripts/eval/eval_rollout.py` (already
`NotImplementedError` for dreamer checkpoints at L1011); `src/algorithms/dreamer_srl/*`
provenance comments citing `dreamer_v3_trainer.py` lines (historical mirrors —
covered by the archive README note); `configs/train/default.yaml:14` comment
(uncommitted user edits); `tests/environment/test_bm_dreamer_batch_driver.py` and
`src/behavior/accumulators.py::bm_drive_batch` (§A6 known orphan);
`configs/models/dreamer_srl/*` ("dreamer_v3" there names the sheeprl algorithm);
`src/graphify-out/` (gitignored, regenerated on demand).

**No new config keys** are introduced by this plan (Configuration Protocol: n/a).

### Test plan + expected suite counts

Baseline measured today: **949 tests collected** under `tests/`; known-red baseline is
**8** (4× dreamer_srl end-to-end A1 parity, 3× stale-config `b093023`, 1× dreamer_srl
offline-WM smoke) — none of the 8 are in the moved set.

After archival:

| Quantity | Before | After |
|---|---|---|
| Collected under `tests/` | 949 | **920** (−29: the 9 archived NNX test files) |
| Known-red | 8 | **8** (unchanged — all live in dreamer_srl/env tests) |
| Everything else | green | green |
| `test_evaluation_model_rebuild.py` | `test_dreamer_eval_rebuild_signature` | `test_dreamer_eval_archived_stub` (1:1 replacement, count-neutral) |
| Archived tests (opt-in) | — | 29 collected & green via `GWP_RUN_ARCHIVED_NNX_TESTS=1 pytest src/models/archive/dreamer_v3_nnx/tests/` |

## Checkpoints

- [x] C1 — All 26 moved paths (18 move rows; row 6 is 8 config files) show `R` in `git status` (renames staged, not delete+add). `git log --follow` is only checkable post-commit (moves are staged, not committed — the archive paths don't exist in HEAD yet); rename staging is the pre-commit equivalent.
- [x] C2 — `git check-ignore -v` on both probe paths returned nothing (exit 1).
- [x] C3 — `from src.models.archive.dreamer_v3_nnx.dreamer_v3_trainer import DreamerTrainer` succeeds.
- [x] C4 — Opt-in run: **26 passed, 3 skipped** (29 collected). The 3 skips are a **deviation from the plan's "29 green"**: `test_dreamer_nnx_replay_ratio_semantics.py`'s 3 train.py-source-pin tests asserted the live train.py Dreamer call sites (Ratio arithmetic / learning_starts gate) that this plan's own T4/T8 deletions removed — they can never pass post-archival, so they carry an explicit `pytest.mark.skip` with the archival reason (see Implementation Report §Deviations). Without the env var → 9 module-level skips in 0.42s.
- [x] C5 — `train.py --agent_config configs/models/archive/dreamer_v3_nnx/dreamer_v3.yaml` raises the archived-stack ValueError at train.py:457 (immediately after algorithm resolution, before env/model construction). Log: `tmp/20260710_c5_stub_check.log`.
- [x] C6 — Zero import hits outside `src/models/archive/`; remaining grep hits are dreamer_srl provenance comments and isolation-rule test strings (which assert the ABSENCE of such imports).
- [x] C7 — `grep -in dreamer train.py` → only T1 stub (L456–463), T11 pointer (L7–9), T12 (L253/275), T13 (L312/315), plus one pre-existing comment referencing the LIVE `dreamer_srl_main.py` (L1399) — not NNX residue.
- [x] C8 — rPPO CPU smoke (20 eps, 4 envs, checkpoint-frequency 10) completed collect→train→checkpoint-save (episode 44) and exited 0; checkpoint `44/` present on disk. Log: `tmp/20260710_rppo_smoke.log`, results: `tmp/20260710_rppo_smoke/`.
- [x] C9 — Suite: 920 collected; failures = exactly the 8 known-red baseline (see Implementation Report §Test results).
- [x] C10 — `regen_dev_index.py` run (169 docs indexed).
- [x] C11 — Speed check waived: deletion of unreachable branches + an early raise cannot affect the rPPO hot path; the only in-loop edit is T10's `model=model` constant-fold in the checkpoint-eval call path (not the collect/train loop). C8 wall-clock sanity confirmed normal.

## Verification checklist (senior-developer, post-implementation)

1. `git diff --stat HEAD` — all files in Parts A–D and nothing else; renames preserved.
2. Re-run C2, C6, C7, C9 personally.
3. Confirm C8 smoke log (rPPO) shows a completed checkpoint save.
4. Confirm the two develop-doc status notes + regenerated INDEX.
5. Fill the Verification Report; diary `verified` row.

## Bug-curator hand-off (registry rows affected — do NOT edit KNOWN_BUGS.md in this change)

Ask `bug-curator` to annotate these rows (current `KNOWN_BUGS.md` line numbers) after
the archival lands — suggested annotation: *"component archived 2026-07-10 →
`src/models/archive/dreamer_v3_nnx/` (fix history preserved in archive)"*; the two
**Open** rows additionally: *"Moot unless stack revived"*:

| Row (KNOWN_BUGS.md line) | Status today | Action |
|---|---|---|
| U6 decoder trailing LayerNorm (L60) | Open | Open → Moot-unless-revived |
| R1 imagined continues soft-vs-hard (L61) | Open | Open → Moot-unless-revived |
| DreamerV3 is_first never told (L98) | Fixed `32c67ca` | archived annotation |
| U1 actor-gradient contamination (L99) | Fixed `32c67ca` | archived annotation |
| U3 obs-loss mean-vs-sum (L100) | Fixed `32c67ca` | archived annotation |
| U2 V2-style value learning (L101) | Fixed `32c67ca` | archived annotation |
| U4 replay-ratio /128 + no prefill (L102) | Fixed `32c67ca`/`7304e75` | archived annotation |
| U5 imagination terminal-start weights (L103) | Fixed `32c67ca` | archived annotation |
| H6 action↔obs misalignment (L120) | Fixed | archived annotation |
| H7 replay wrap splicing (L121) | Fixed | archived annotation |
| H10 batch-loop BM corruption (L124) | Fixed | annotation noting the Site-2 caller was removed with the archived train.py branch; `bm_drive_batch` itself remains live (§A6) |
| Dreamer treats timeout as death, Finding B sibling (L137) | Fixed | archived annotation |
| Old eval script crash on modulated/Dreamer checkpoints, Finding A (L141) | Fixed | annotation: Dreamer arm of the fix now behind the archived-stack stub |

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-10

### What was implemented

**Part A — moves (all via `git mv`, all staged as `R` renames):**
- 5 source modules `src/models/dreamer_v3_{trainer,nnx,util,network}.py` + `modulated_layer_norm_gru_cell.py` → `src/models/archive/dreamer_v3_nnx/`
- 8 config YAMLs `configs/models/dreamer_v3/` → `configs/models/archive/dreamer_v3_nnx/`
- 9 test files + 2 helpers `tests/models/{test_dreamer_*,dreamer_nnx_*}.py` → `src/models/archive/dreamer_v3_nnx/tests/`
- 1 script `scripts/dreamer/dreamer_offline_wm_test.py` → `src/models/archive/dreamer_v3_nnx/scripts/`
- New files: `src/models/archive/__init__.py` (empty), `src/models/archive/dreamer_v3_nnx/__init__.py` (archived docstring), `src/models/archive/dreamer_v3_nnx/README.md` (why/contents/how-to-run per plan).

**Part B — in-archive edits:**
- 16 import lines repointed `src.models.dreamer_v3_*` / `src.models.modulated_layer_norm_gru_cell` → `src.models.archive.dreamer_v3_nnx.*` (all 11 planned sites; sed covered a few extra occurrences of the same patterns). `dreamer_v3_nnx.py:595`'s import of the LIVE `src.models.neuromodulator` left as-is (that module stays live).
- `_REPO` depth 3→6 dirname hops in the 5 files that compute it (fixtures, obs_loss_sum, replay_buffer_wrap, collect_arrival_alignment, offline script).
- Config paths repointed to `configs/models/archive/dreamer_v3_nnx/dreamer_v3.yaml` (fixtures:42, arrival-alignment:54) + the offline script's two usage-example strings.
- §A7 module-level skip guard inserted after the docstring of all 9 test modules (aliased imports `os as _os_guard` / `pytest as _pytest_guard` to avoid clashing with each file's own imports).

**Part C — live-tree edits:**
- `train.py`: T1 stub added after algorithm resolution (fires at L457); T3/T4/T6/T7/T8/T9 deleted wholesale; T2 tuple+message, T5 else-message, T10 both `model=model` call sites, T11 docstring/help edited. T12/T13 untouched. **Net −502 lines (521 deleted, 19 added).**
- `evaluation.py`: DreamerV3 `elif` body → the same archived-stack ValueError (branch kept reachable per §A4); L161 docstring pointer updated. Net −24 lines.
- `tests/scripts/test_evaluation_model_rebuild.py`: `test_dreamer_eval_rebuild_signature` replaced 1:1 with `test_dreamer_eval_archived_stub` (asserts `ValueError` matching "archived" and message naming `dreamer_srl`; no NNX import remains in the file); module docstring item 2 updated.

**Part D — docs:**
- `SCRIPTS_DEPENDENCY_MAP.md`: §3 row for `scripts/dreamer/dreamer_offline_wm_test.py` removed; Cluster C (§4) note updated with the move.
- Status notes + frontmatter bumps on [[06_nnx_recipe_deviation_register]] (register frozen; U6/R1 moot-unless-revived) and [[fix_plan_nnx_parity]] (WP-NNX landed then stack archived).
- `docs/develop/INDEX.md` regenerated via `regen_dev_index.py` (169 docs).

### Test results

| Check | Command | Result |
|---|---|---|
| (b)/C9 full suite | `pytest tests/ -q` (CPU) | **920 collected; 8 failed, 411 passed, 501 skipped** (15:04). Failures = exactly the known-red baseline: 4× `test_unified_parity[observability_gates_S1–S4]`, 3× stale-config `FileNotFoundError` (`test_inactive_animal_offgrid` ×1, `test_truncation_not_death` ×2), 1× `test_dreamer_srl_offline_wm_test::test_offline_wm_smoke`. Composition verified against [[fix_plan_nnx_parity]] §Full-suite gate. Log: `tmp/20260710_full_suite_post_archive.log` |
| (c)/C4 archived opt-in | `GWP_RUN_ARCHIVED_NNX_TESTS=1 pytest src/models/archive/dreamer_v3_nnx/tests/ -q` | 29 collected: **26 passed, 3 skipped** (see Deviation 1). Without env var: 9 module skips in 0.42s. Log: `tmp/20260710_c4_archived_tests.log` |
| replaced eval test | `pytest tests/scripts/test_evaluation_model_rebuild.py -q` | **7 passed** (count-neutral), incl. new `test_dreamer_eval_archived_stub` |
| (d)/C8 rPPO smoke | `train.py` + `recurrent_ppo.yaml`, 20 eps / 4 envs / CPU / ckpt-freq 10 | Completed collect→train→**checkpoint saved (episode 44, on disk)**→exit 0. Logs: `tmp/20260710_rppo_smoke{.log,/}` |
| (e)/C5 train stub | `train.py --agent_config configs/models/archive/dreamer_v3_nnx/dreamer_v3.yaml` | Archived-stack `ValueError` raised at train.py:457, before env/model construction |
| (e) eval stub | `evaluation.py` vs. old `results/JAX_DreamerV3/*` dirs | All on-disk DreamerV3 dirs carry **pre-v2.0 saved configs** and fail in `load_env_params` (`environment.predator_enabled` removed in v2.0) *before* the algorithm branch — pre-existing schema incompatibility, unrelated to this change. Stub contract therefore pinned **unit-level** by `test_dreamer_eval_archived_stub` (the plan's stated fallback), which drives `ev.main()` end-to-end with a modern config + real orbax checkpoint |
| (a)/C6 no-live-imports | plan grep excluding archive | Zero import statements; residual hits are dreamer_srl provenance comments + isolation-rule test strings |

### Speed check

**Waived per C11** (plan-sanctioned): the change deletes branches unreachable after the T1 early raise; the only edit inside live-reachable code is T10's `model=model` constant-fold in the checkpoint-eval call (not the collect/train hot loop) plus stage-swap/restore blocks outside the loop. C8 smoke wall-clock was normal for a CPU run.

### Deviations from the plan

1. **C4 "29 green" → 26 green + 3 explicit skips.** Three tests in `test_dreamer_nnx_replay_ratio_semantics.py` (`test_call_site_counts_env_steps_not_sequences`, `test_learning_starts_mandatory_and_gating`, `test_ratio_first_call_after_prefill_no_backlog_burst`) are **source-level pins on the live `train.py`** — they `open(TRAIN_PY).read()` and assert the Dreamer Ratio/prefill/learning_starts call sites exist. Those call sites are exactly what this plan's T4/T8 deletions removed, so the pins can never hold post-archival. Left red they would poison every future opt-in archaeology run; deleted they would lose the WP-NNX F7 record. Chosen middle: `pytest.mark.skip` with an explicit archival reason + a comment block naming this plan. The 2 remaining tests in that file (pinning the `Ratio` class itself) still run and pass. Flagging for senior-developer sign-off.
2. **`.claude/agents/training-runner.md` and `train_command-agent.sh` NOT touched** (plan Part D rows 4–5): per the parent's instruction they currently carry other sessions' uncommitted edits. Both still reference DreamerV3/`configs/models/dreamer_v3/...` (training-runner.md as a supported JAX launch path; train_command-agent.sh comment examples at L42/L94) — **follow-up for senior-developer** to apply the Part D edits on the current working-tree versions.
3. `git log --follow` (C1 second clause) is only verifiable **post-commit** — the moves are staged, not committed (developer does not commit). All 26 paths staged as `R` renames, which is what preserves `--follow`.

### Blockers / follow-ups

- Part D rows for `training-runner.md` / `train_command-agent.sh` (Deviation 2).
- Bug-curator hand-off (§Bug-curator hand-off) — not done here by design.
- User flag (plan "leave-as-is" list): gitignored `train_command-new.sh` L56 comment cites the old config path.
- Note: `git mv` **stages** the moves (unavoidable — that is how git records renames); all other edits are unstaged. Nothing committed.

> Implemented by: developer

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-10

All greps, both test runs, and the check-ignore probes below were **re-run independently** by the verifier (not read off the Implementation Report). Verifier logs: `tmp/20260710_verif_full_suite.log`, `tmp/20260710_verif_archived_tests.log`, `tmp/20260710_verif_train_diff.txt`.

| File / area | Change | Status | Notes |
|------|--------|:------:|-------|
| Part A moves (25 paths: 5 modules + 8 configs + 11 tests/helpers + 1 script) | `git mv` renames | ⚠️ | All 18 inventory rows executed as pure 0-change renames (`--follow` preserved). **But the staged renames were swept into another session's commit `2a7c6f9`** ("docs(memory): capture 4 insights…", 17:01) — index pollution, not developer error; see Conclusion. Plan's C1 "26 moved paths" is a miscount in the plan text: 17 single-file rows + 8 configs = **25**. |
| Archive location | not gitignored | ✅ | `git check-ignore -v` re-run on `src/models/archive/.../dreamer_v3_trainer.py`, `.../tests/dreamer_nnx_fixtures.py`, `configs/models/archive/.../dreamer_v3.yaml` — all exit 1 (not ignored). No path contains "legacy". |
| New files (`archive/__init__.py`, `dreamer_v3_nnx/__init__.py`, `README.md`) | created (untracked) | ✅ | README accurately states status, the 2026-05-11/12 pivot history (with memory link), the WP-NNX misdirection story, importable-in-place + `GWP_RUN_ARCHIVED_NNX_TESTS=1` revival path, and the dreamer_srl provenance-comment caveat. |
| `train.py` | T1–T11 per §A3; net −502 | ✅ | Diff hunks map 1:1 onto T1–T11; T12/T13 untouched. T4 and T8 deleted regions inspected: both entirely inside `elif algorithm == "DreamerV3":` arms; a scan of the T8 block for `RecurrentPPO\|rppo\|h_state\|optimizer` returned **zero hits**. rPPO Site-1 BM path intact (`make_bm_state`/`bm_step_update`/`bm_reset_env`/finalise at L954–1000). T1 stub names `dreamer_srl` + archive path + this plan. `grep -in dreamer train.py` → only stub (L456–463), T11 pointer (L7–9), T12 (L253/275), T13 (L312/315), live dreamer_srl comment (L1399). |
| `train.py:70` | — | ⚠️ | **Orphaned import**: `bm_drive_batch` is still imported but its only caller (Site-2, deleted with T8) is gone. Harmless (live module); one-line cleanup for `developer` at next touch. The function itself stays per §A6. |
| `evaluation.py` | branch stubbed; −24 | ✅ | `elif algorithm == "DreamerV3":` kept reachable, body replaced by the archived-stack `ValueError` (same text as T1); L158-region docstring updated. Old on-disk results dirs get the clear message. |
| `tests/scripts/test_evaluation_model_rebuild.py` | 1:1 test replacement | ✅ | `test_dreamer_eval_archived_stub` drives real `ev.main()` with an orbax scaffold, asserts `ValueError` matching "archived" and `dreamer_srl` in the message; no NNX import remains. |
| Archived tests (opt-in) | skip guards + repointing | ✅ | Verifier re-run: **26 passed, 3 skipped in 2:14** with the env var; **9 module-skips in 0.08s** without. Depth fixes correct (6 dirname hops verified for the offline script). |
| Deviation 1 (3 replay-ratio skips) | `pytest.mark.skip` | ✅ **signed off** | All three (`test_call_site_counts_env_steps_not_sequences`, `test_learning_starts_mandatory_and_gating`, `test_ratio_first_call_after_prefill_no_backlog_burst`) are source-level pins that `open(TRAIN_PY)` and assert the Dreamer Ratio/learning_starts call sites — exactly the plan's own T4/T8 deletions. They cannot pass post-archival by construction; skip reasons name the archival + this plan. The 2 Ratio-class tests still run and pass. |
| No-live-imports (C6) | grep re-run | ✅ | Zero import statements outside the archive; residual hits are dreamer_srl isolation-rule comments/test strings (which assert the ABSENCE of such imports). `modulated_layer_norm_gru_cell` — no hits outside archive. |
| `src/models/modulated_gru_cell.py` | untouched | ✅ | Zero diff vs HEAD; still imported by `recurrent_ppo_network.py:6`. Live rPPO-NMN path unaffected. |
| rPPO smoke (C8) | implementer artifacts inspected | ✅ | `tmp/20260710_rppo_smoke.log` (16:58 today): collect→train→`[CHECKPOINT] Saving model at episode 44`→"Training complete"; `tmp/20260710_rppo_smoke/models/44/` on disk. |
| Full suite (C9) | verifier re-run | ✅ | **920 total: 8 failed, 418 passed, 494 skipped (28:04)**. Failures = exactly the known-red baseline: 4× `test_unified_parity[observability_gates_S1–S4]`, 3× stale-config `FileNotFoundError`, 1× `test_dreamer_srl_offline_wm_test::test_offline_wm_smoke`. (Pass/skip split differs slightly from the developer's 411/501 — environment-dependent skips; failed set identical.) |
| `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` | row removed + Cluster C note | ✅ | Maintenance Contract satisfied; the moved script's new location recorded in the Cluster C note. |
| `06_nnx_recipe_deviation_register.md` / `fix_plan_nnx_parity.md` | status notes | ✅ | Register frozen, U6/R1 marked moot-unless-revived; fix plan notes WP-NNX landed then archived + opt-in test path. Frontmatter bumped. |
| `docs/develop/INDEX.md` | regenerated | ✅ | 2026-07-10 16:47 regen; 117 active docs (this plan indexed). |
| Bug-curator hand-off | 13 rows listed | ✅ | All 13 rows match the registry (grep-verified). One adjacent row for `bug-curator` to *consider*: KNOWN_BUGS L163 "Silent encode/decode layout drift" cites the now-archived trainer + offline diagnostic; its lesson is general-class, so the omission is defensible. |
| Part D rows 4–5 (`training-runner.md`, `train_command-agent.sh`) | deferred (Deviation 2) | ⚠️ | Still outstanding: `training-runner.md` still advertises `dreamer_v3_nnx` as a JAX launch path (description, L12, L26) and `train_command-agent.sh` L42/L94 comments cite the old config path. Both have since been **committed by other sessions**, so the deferred edits can now be applied cleanly — follow-up for `developer`. Interim risk is low: a `dreamer_v3_nnx` launch now fails fast at the T1 stub. |
| Speed | waived per C11 | ✅ no regression | Waiver justified: deletions are unreachable after the T1 early raise; only live-reachable edit is T10's `model=model` constant-fold outside the collect/train hot loop. C8 smoke wall-clock normal. |

**Deviation sign-offs**: Deviation 1 (3 explicit skips) — **approved** (see row above). Deviation 2 (Part D rows 4–5 deferred) — **accepted as follow-up**, now unblocked. Deviation 3 (`git mv` stages moves) — **overtaken by events**: the staged renames were committed by another session's `git commit` (`2a7c6f9`) with a wrong-scope `docs(memory)` message. Content is correct (pure renames, history preserved); only the commit-message scope is polluted. Rewriting another session's history is out of scope — recorded here as a process incident: **parallel sessions must not commit while another session holds staged work** (or should commit with explicit pathspecs, per the auto-commit rule "stage specific files by name").

**Conclusion**: PASS. The archival is complete and correct: all 25 moves landed as history-preserving renames, every deleted `train.py` region was Dreamer-only, the live rPPO path (Site-1 BM, `modulated_gru_cell`, smoke, full suite) is intact, both stubs raise the clear archived-stack message naming `dreamer_srl`, and the docs contract is satisfied. Open items (non-blocking): (1) apply Part D rows 4–5 now that the blocking uncommitted edits are committed; (2) remove the orphaned `bm_drive_batch` import at `train.py:70`; (3) hand the 13 registry rows (+ optionally L163) to `bug-curator`; (4) commit-message scope pollution in `2a7c6f9` noted for the record.

Verified by: senior-developer
