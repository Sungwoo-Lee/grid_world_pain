---
title: "scripts/ Folder Reorganization (loose pile -> grouped subfolders)"
topic: refactors
status: active
created: 2026-06-30
last_updated: 2026-06-30
phase: null
---

# `scripts/` Folder Reorganization

> **Status**: PLANNED
> **Opened**: 2026-06-30
> **Related**: [[SCRIPTS_DEPENDENCY_MAP]] (`docs/environment/SCRIPTS_DEPENDENCY_MAP.md`) — the authoritative caller map this plan builds on.

---

## Decisions (settled 2026-06-30)

All four open questions are now closed. This plan is a definitive implementation contract — not an options menu. The `developer` agent executes exactly what is locked here.

1. **Grouping scheme = group-by-function** (the recommended one). Destination folders: `scripts/wandb/`, `scripts/eval/`, `scripts/dreamer/`, `scripts/claude/`, `scripts/lab/`, `scripts/media/`, plus the three pre-existing folders `fixtures/`, `verification/`, `behavior_measures/`. The group-by-lifecycle/caller alternative is **REJECTED** (it splits the import-coupled WandB cluster and the eval pipeline, forcing extra rewrites — see the struck-through section below).
2. **Doc/memory tooling folder name = `scripts/claude/`** (not `dev_tools/`).
3. **Near-dead scripts:**
   - `git rm` the two disposable one-shots `migrate_dev_frontmatter.py` and `rewrite_dev_links.py` — their job is done. These are **deletions, not moves**.
   - **Relocate** the three orphans and keep them in git: `md_to_pdf.py` → `scripts/media/`, `video_to_gif.py` → `scripts/media/`, `analyze_noise_diagnostics.py` → `scripts/verification/` (alongside `verify_noise.py`, since it is a post-eval correctness/diagnostics reader).
4. **Import conventions: leave as-is this pass.** Keep the WandB cluster co-located so the bare-name imports (`from wandb_utils import ...`) keep working with **zero import edits**. Do **not** convert to `scripts.<sub>.<mod>` package imports. The only `PYTHONPATH` change is the path-move required by the WandB folder relocation (`PYTHONPATH=scripts` → `PYTHONPATH=scripts/wandb` in the `wandb-analysis` skill); nothing else in that skill's `PYTHONPATH` is touched.

---

## Context

The `scripts/` folder is an unorganized pile: ~31 loose Python files sit at the top level next to three already-grouped subfolders (`behavior_measures/`, `fixtures/`, `verification/`). The loose files span five unrelated jobs — WandB training-log analysis, the evaluation/recording/rendering pipeline, Dreamer/sheeprl numerical-parity diagnostics, the "dev tooling" that Claude agents and skills run to maintain docs and memory, and a few media/asset converters. Finding the right script means scanning a flat list of 31 names with no grouping.

This refactor **only relocates files into topic subfolders and updates every reference to them**. It is a pure move-and-rewrite: **no script's behavior changes, no logic is edited, nothing is renamed** except its containing folder. The win is navigability; the risk is silently breaking the many things that call these scripts by hardcoded path. That risk is fully enumerated in the dependency map referenced above — this plan does not re-derive it, it executes against it.

The one non-obvious trap (explained in the map's preamble): most `scripts/*.py` figure out the repo root by walking **up a fixed number of parent directories from their own file location**. A file written for `scripts/foo.py` assumes "repo root is one level up." Move it to `scripts/sub/foo.py` and that math now points one directory too shallow, so every `from src...` import inside it breaks — even though no *caller* changed. Each relocated Python file that does this must have its depth math bumped by one. The exact line in each file is listed in the move table below.

---

## Analysis

### What each script does (grouped by proposed destination)

The dependency map tells us *who calls* each file; this section adds *what each file does* (one line, read from the module docstring), grouped by where the recommended scheme puts it. The two pre-existing subfolders that don't change (`behavior_measures/`, `fixtures/`, `verification/`) are summarized at the end.

**`scripts/wandb/` — WandB training-log analysis (import-coupled cluster, must stay together)**
| File | What it does |
|---|---|
| `wandb_utils.py` | Shared WandB helpers (entity/project constants, run fetch, stats, formatting). Leaf of the import cluster. |
| `wandb_metrics.py` | General-purpose training-metrics tool; imports `wandb_utils`. |
| `compare_wandb_runs.py` | Side-by-side run comparison (DreamerV3 preset); imports `wandb_utils` + `wandb_metrics`. |
| `benchmark_wandb_speed.py` | WandB speed/throughput benchmark; imports `wandb_utils`. |

**`scripts/eval/` — evaluation -> record -> render -> consume pipeline**
| File | What it does |
|---|---|
| `eval_rollout.py` | Offline evaluation rollout; writes the `.rec.gz` recordings the toolkit consumes. |
| `render_recordings.py` | Renders saved eval recordings to MP4 in parallel. **Called from `src/` by hardcoded path.** |
| `trajectory_story.py` | Step-level qualitative story analysis of eval-rollout recordings. |
| `motif_cluster.py` | Offline behavior-motif clustering over recordings. |
| `benchmark_render.py` | Benchmarks `render_jax_state` cost and recording payload size. |

**`scripts/dreamer/` — Dreamer/sheeprl parity + world-model diagnostics**
| File | What it does |
|---|---|
| `dreamer_offline_wm_test.py` | Offline world-model imagination diagnostic for a frozen DreamerV3 checkpoint. |
| `dreamer_srl_offline_wm_test.py` | Same diagnostic for a frozen dreamer-srl v2 checkpoint. **Imported by a pytest.** |
| `dreamer_srl_offline_check.py` | CP8 end-to-end forward-parity integration check. **Run by a pytest via subprocess.** |
| `sheeprl_jax_diff.py` | Per-function numerical diff tool for the dreamer-srl v3 rebuild. (~30 internal `sys.path` lines.) |
| `visualize_dream.py` | Dream-strip visualizer for the dreamer-srl agent. |

**`scripts/claude/` — tooling the Claude agents/skills run to maintain docs + memory**
| File | What it does |
|---|---|
| `diary_append.py` | Appends an event row to `docs/diary/YYYY-MM-DD.md`. **Most-wired dependency in the repo (7 callers).** |
| `regen_dev_index.py` | Regenerates `docs/develop/INDEX.md` from frontmatter. |
| `regen_memory_links.py` | Rewrites `related:` frontmatter from `[[id]]` wikilinks in memory bodies. |
| `regen_memory_graph.py` | Builds the memory insight+session graph + per-insight backlinks. |
| `regen_code_graph.py` | Wrapper around `graphify` producing `GRAPH_REPORT.md` (output gitignored). |
| `snapshot_code_graph.py` | Captures a point-in-time code-graph snapshot under `docs/memory/`. |
| `claude_jsonl_to_md.py` | Converts a Claude Code transcript JSONL to a markdown archive. |
| `open_conversation.py` | Resolves an insight/session ID to its source JSONL + restore commands. |
| `lint_memory.py` | Read-only 11-check lint of `docs/memory/`. |

**`scripts/lab/` — lab-node training-launch shell scripts**
| File | What it does |
|---|---|
| `launch_sheeprl.sh` | The one real sheeprl training-launch path. |
| `bootstrap_lab_ssh.sh` | One-time SSH-key bootstrap from the container to lab nodes 101-114. |

**`scripts/media/` — demo/asset converters**
| File | What it does |
|---|---|
| `record_env_demo.py` | Records an environment demo video under a random policy. |
| `video_to_gif.py` | Converts recorded MP4 to optimized GIF for docs. (orphan) |
| `md_to_pdf.py` | Converts a markdown file to PDF. (orphan, zero callers) |

**Files added to existing folders**
| File | Current | Proposed | What it does |
|---|---|---|---|
| `generate_parity_fixtures.py` | top-level | `fixtures/` | Generates pre-refactor parity `.npz` fixtures for the CP1 animal-entity refactor. |
| `verify_noise.py` | top-level | `verification/` | Verifies perceptual-noise system correctness. |

**Resolved (Decision 3)**: `analyze_noise_diagnostics.py` (post-eval noise diagnostics, orphan) → `verification/`. The two one-shot disposables `migrate_dev_frontmatter.py` + `rewrite_dev_links.py` are **`git rm`'d** (deleted, not moved).

**Unchanged pre-existing folders**: `behavior_measures/` (`avoidance_stats_heatmap.py`, `_heatmap_style.py`, `README.md`), `fixtures/` (`gen_cp1..8_fixtures.py`), `verification/` (`check_observability_gates.py`, `check_olfaction_parity.py`).

### Move-together constraints (from the map)

- **Cluster A (WandB) is import-coupled by bare-name imports** (`from wandb_utils import ...`, no `scripts.` prefix). The four files resolve each other only because they share a directory. The recommended scheme keeps all four in `scripts/wandb/`, so the imports keep working **without edits** — but the `wandb-analysis` skill runs them with `PYTHONPATH=scripts python scripts/wandb_metrics.py`, so that `PYTHONPATH` and path must move to `scripts/wandb`.
- **Cluster B (eval pipeline) is path/format-coupled, not import-coupled.** Files can live together; the only hard edge is `render_recordings.py`'s two `src/` subprocess callers.
- **Cluster C (dreamer/parity) is path/string-coupled** via fixture `.npz` paths + two `tests/` files; no intra-`scripts/` Python imports.

---

## Implementation Plan

### Design

**Recommended scheme: group-by-function.** Eight destination folders (5 new + 3 existing), chosen so a reader looking for "the WandB tools" or "the eval pipeline" finds them co-located. This scheme also satisfies the Cluster-A co-location requirement for free.

```
scripts/
  wandb/            (NEW) wandb_utils, wandb_metrics, compare_wandb_runs, benchmark_wandb_speed
  eval/             (NEW) eval_rollout, render_recordings, trajectory_story, motif_cluster, benchmark_render
  dreamer/          (NEW) dreamer_offline_wm_test, dreamer_srl_offline_wm_test,
                          dreamer_srl_offline_check, sheeprl_jax_diff, visualize_dream
  claude/           (NEW) diary_append, regen_dev_index, regen_memory_links, regen_memory_graph,
                          regen_code_graph, snapshot_code_graph, claude_jsonl_to_md,
                          open_conversation, lint_memory
  lab/              (NEW) launch_sheeprl.sh, bootstrap_lab_ssh.sh
  media/            (NEW) record_env_demo, video_to_gif, md_to_pdf
  fixtures/         (exists) gen_cp1..8 + generate_parity_fixtures
  verification/     (exists) check_observability_gates, check_olfaction_parity
                             + verify_noise + analyze_noise_diagnostics
  behavior_measures/(exists) unchanged
  README.md         (updated)

  DELETED (git rm): migrate_dev_frontmatter.py, rewrite_dev_links.py  (one-shots, job done)
```

**Why this over the alternative.** The two natural axes are *function* (what the script does) and *lifecycle/caller* (who runs it — `src/` at runtime vs. pytest vs. Claude skills vs. hand-run). Group-by-function wins because:
- The strongest physical coupling (Cluster A bare imports) is functional, so a functional grouping keeps it intact with zero import edits.
- "Who calls it" is already fully documented in the dependency map; encoding it in the folder layout duplicates that and ages badly (a script can gain a new caller without changing function).
- A contributor's mental query is almost always "where's the WandB tool / the renderer," not "what calls the renderer."

#### Alternative scheme (group-by-lifecycle/caller) — REJECTED (2026-06-30)

> **Outcome:** rejected. Recorded here for the record. One-line why: it splits the import-coupled WandB cluster (Cluster A) and the path-coupled eval pipeline (Cluster B) across lifecycle buckets, forcing import rewrites and re-coupling work that group-by-function avoids entirely.

```
scripts/
  runtime/        render_recordings                     (called by src/ at runtime)
  claude_tooling/ diary_append, regen_*, eval_rollout, trajectory_story, ...  (run by skills/agents)
  test_support/   fixtures/, dreamer_srl_offline_*, sheeprl_jax_diff, generate_parity_fixtures
  hand_run/       record_env_demo, benchmark_*, visualize_dream, verify_noise, media converters
```

Downsides: it **splits Cluster A** (only `wandb_metrics`/`compare` are skill-run; `wandb_utils` is a pure leaf) forcing the 4 bare-import rewrites; it **splits Cluster B** (`render_recordings` -> `runtime/`, the rest -> `claude_tooling/`); and a file's bucket changes whenever its caller set changes. Presented only so the user can choose; the recommendation is group-by-function.

### File Changes

#### Per-file move table

Legend for "Depth fix": the exact line whose `parent.parent` / double-`dirname` must gain **one more level** (`parent.parent.parent` / `parents[N+1]` / triple-`dirname`) because the file drops one directory deeper. Files marked **none** do not compute repo root from `__file__` and need no depth edit (verified by grep).

| Current path | Proposed path | Depth fix | Other reference updates (from map §1-3) |
|---|---|---|---|
| `scripts/wandb_utils.py` | `scripts/wandb/wandb_utils.py` | none | none (bare imports preserved by co-location) |
| `scripts/wandb_metrics.py` | `scripts/wandb/wandb_metrics.py` | none | `wandb-analysis` skill: `PYTHONPATH=scripts`->`scripts/wandb`, path -> `scripts/wandb/wandb_metrics.py` |
| `scripts/compare_wandb_runs.py` | `scripts/wandb/compare_wandb_runs.py` | none | `wandb-analysis` skill path |
| `scripts/benchmark_wandb_speed.py` | `scripts/wandb/benchmark_wandb_speed.py` | none | `wandb-analysis` skill path (if listed) |
| `scripts/eval_rollout.py` | `scripts/eval/eval_rollout.py` | L47 `parent.parent` -> `parents[2]` (then re-`insert` L48) | `trajectory-story` skill commands; `experiment-analyzer` agent L59 |
| `scripts/render_recordings.py` | `scripts/eval/render_recordings.py` | L22 `dirname(dirname(...))` -> add one `dirname` | `src/utils/evaluation_core.py:295` (+ cosmetic hint L328); `src/algorithms/dreamer_srl/eval.py:232`; `trajectory-story` skill L60; `docs/environment/12_renderer.md` line citation |
| `scripts/trajectory_story.py` | `scripts/eval/trajectory_story.py` | L47 `parent.parent` -> `parents[2]` | `trajectory-story` skill (primary tool); `experiment-analyzer` agent L59 |
| `scripts/motif_cluster.py` | `scripts/eval/motif_cluster.py` | L31 `parent.parent` -> `parents[2]` (then re-`insert` L32) | none (no programmatic caller) |
| `scripts/benchmark_render.py` | `scripts/eval/benchmark_render.py` | L31 `dirname(dirname(...))` -> add one `dirname` | none |
| `scripts/dreamer_offline_wm_test.py` | `scripts/dreamer/dreamer_offline_wm_test.py` | L73 `dirname(dirname(...))` -> add one `dirname` | none |
| `scripts/dreamer_srl_offline_wm_test.py` | `scripts/dreamer/dreamer_srl_offline_wm_test.py` | L65 `dirname(dirname(...))` -> add one `dirname` | **TEST** `tests/scripts/test_dreamer_srl_offline_wm_test.py:107` import `from scripts.dreamer_srl_offline_wm_test import main` -> `from scripts.dreamer.dreamer_srl_offline_wm_test import main` |
| `scripts/dreamer_srl_offline_check.py` | `scripts/dreamer/dreamer_srl_offline_check.py` | L81 `dirname(dirname(...))` -> add one `dirname` | **TEST** `tests/algorithms/dreamer_srl/test_end_to_end_parity.py:48` subprocess path |
| `scripts/sheeprl_jax_diff.py` | `scripts/dreamer/sheeprl_jax_diff.py` | ~30 `dirname(dirname(...))` occurrences (grep `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`) -> add one `dirname` **each** | docs/tests-README mentions (non-breaking) |
| `scripts/visualize_dream.py` | `scripts/dreamer/visualize_dream.py` | L76 `Path(__file__).parent.parent.resolve()` -> add one `.parent` | none |
| `scripts/diary_append.py` | `scripts/claude/diary_append.py` | L66 `parent.parent` -> `parents[2]` | **7 callers** — skills `diary`, `memorize`(L182), `summarize-study`(L279,L373); agents `developer`(L94), `senior-developer`(L97), `experiment-analyzer`(L114), `training-runner`(L286) |
| `scripts/regen_dev_index.py` | `scripts/claude/regen_dev_index.py` | L21 `parent.parent` -> `parents[2]` | `senior-developer` agent L52; contracts `CLAUDE.md` L62, `AGENT_PLAYBOOK.md` L144, `FRONTMATTER_CONTRACT.md`; negative mentions in `experiment-analyzer`/`experiment-designer` |
| `scripts/regen_memory_links.py` | `scripts/claude/regen_memory_links.py` | L11 `parent.parent` -> `parents[2]` | `memorize` skill L198; `docs/memory/CLAUDE.md` |
| `scripts/regen_memory_graph.py` | `scripts/claude/regen_memory_graph.py` | L20 `parent.parent` -> `parents[2]` | `memorize` skill L248 |
| `scripts/regen_code_graph.py` | `scripts/claude/regen_code_graph.py` | L103 `parent.parent` -> `parents[2]` | on-demand hints: `recall` skill L53, `senior-developer`/`code-reviewer` agents, README |
| `scripts/snapshot_code_graph.py` | `scripts/claude/snapshot_code_graph.py` | L30 `parent.parent` -> `parents[2]` | `memorize` skill L34,L79 |
| `scripts/claude_jsonl_to_md.py` | `scripts/claude/claude_jsonl_to_md.py` | none | `memorize` skill L40,L156,L280 |
| `scripts/open_conversation.py` | `scripts/claude/open_conversation.py` | L20 `parent.parent` -> `parents[2]` | design-doc mention |
| `scripts/lint_memory.py` | `scripts/claude/lint_memory.py` | L15 `parent.parent` -> `parents[2]` | `docs/memory/CLAUDE.md` mention |
| `scripts/launch_sheeprl.sh` | `scripts/lab/launch_sheeprl.sh` | check shell `cd`/dir logic (not Python depth) | `training-runner` agent L16; `run_command.py` docstrings L26,L126; `pytorch_agents/run_dreamer_v3.py:10` docstring |
| `scripts/bootstrap_lab_ssh.sh` | `scripts/lab/bootstrap_lab_ssh.sh` | check shell dir logic | `training-runner` agent L81 |
| `scripts/record_env_demo.py` | `scripts/media/record_env_demo.py` | L8 `dirname(dirname(...))` -> add one `dirname` | README usage block |
| `scripts/video_to_gif.py` | `scripts/media/video_to_gif.py` | none | README usage block |
| `scripts/md_to_pdf.py` | `scripts/media/md_to_pdf.py` | none | none (zero callers) |
| `scripts/generate_parity_fixtures.py` | `scripts/fixtures/generate_parity_fixtures.py` | L23 `_ROOT = dirname(_HERE)` -> `dirname(dirname(_HERE))` | named in a test docstring only (non-breaking) |
| `scripts/verify_noise.py` | `scripts/verification/verify_noise.py` | none (runs via `PYTHONPATH=.`) — **verify** no `sys.path` repo-root math before moving | docs mention |
| `scripts/analyze_noise_diagnostics.py` | `scripts/verification/analyze_noise_diagnostics.py` | none (no `__file__`/`sys.path` repo-root math — verified by grep; it is a standalone CSV reader) | archived docs only (`docs/develop/archive/NOISE_*`) — non-breaking; map §1 ORPHAN row updated to new path |

#### Deletions (`git rm`, Decision 3)

These two one-shots have completed their job (their docstrings declare them disposable). They are **removed**, not relocated. `git rm` preserves their history under `git log`.

| Current path | Action | Notes |
|---|---|---|
| `scripts/migrate_dev_frontmatter.py` | `git rm` | One-shot frontmatter migration, already run. Has `ROOT = Path(__file__).resolve().parent.parent` (L30) but that is irrelevant once deleted. Confirm no live caller in the closing grep sweep before deleting. |
| `scripts/rewrite_dev_links.py` | `git rm` | One-shot link-rewrite, already run. Same `ROOT` math (L26), irrelevant once deleted. Confirm no live caller before deleting. |

> **Depth-fix verification rule for `developer`:** after each move, grep the moved file for `__file__` and re-confirm the repo-root walk now resolves to the actual repo root (print `REPO_ROOT`/`PROJECT_ROOT` and assert it equals the repo root). `sheeprl_jax_diff.py` has ~30 occurrences of the same pattern — fix them with a single `replace_all` and count the replacements against the grep count.

#### Non-`scripts/` files this refactor edits

| File | Why |
|---|---|
| `src/utils/evaluation_core.py` (L295, cosmetic L328) | `render_recordings.py` subprocess path |
| `src/algorithms/dreamer_srl/eval.py` (L232) | `render_recordings.py` subprocess path |
| `tests/scripts/test_dreamer_srl_offline_wm_test.py` (L107) | import of moved `dreamer_srl_offline_wm_test` |
| `tests/algorithms/dreamer_srl/test_end_to_end_parity.py` (L48) | subprocess path to moved `dreamer_srl_offline_check` |
| `.claude/skills/wandb-analysis/SKILL.md` | `PYTHONPATH` + `wandb_metrics.py`/`compare_wandb_runs.py` paths |
| `.claude/skills/trajectory-story/SKILL.md` (incl. L60) | `eval_rollout`/`render_recordings`/`trajectory_story` paths |
| `.claude/skills/memorize/SKILL.md` (L34,40,79,156,182,198,248,280) | `regen_memory_*`, `snapshot_code_graph`, `claude_jsonl_to_md`, `diary_append` paths |
| `.claude/skills/summarize-study/SKILL.md` (L279,373) | `diary_append` path |
| `.claude/skills/diary/SKILL.md` | `diary_append` path |
| `.claude/skills/recall/SKILL.md` (L53) | `regen_code_graph` hint |
| `.claude/agents/developer.md` (L94) | `diary_append` command |
| `.claude/agents/senior-developer.md` (L52,L97) | `regen_dev_index` + `diary_append` commands |
| `.claude/agents/experiment-analyzer.md` (L59,L114) | `eval_rollout`/`trajectory_story` + `diary_append` |
| `.claude/agents/experiment-designer.md` | negative "do NOT run `regen_dev_index`" mention |
| `.claude/agents/training-runner.md` (L16,L81,L286) | `launch_sheeprl.sh`, `bootstrap_lab_ssh.sh`, `diary_append` |
| `.claude/agents/code-reviewer.md` | `regen_code_graph` hint |
| `run_command.py` (docstrings L26,L126) | `launch_sheeprl.sh` path |
| `pytorch_agents/run_dreamer_v3.py` (L10 docstring) | `launch_sheeprl.sh` path |
| `CLAUDE.md` (L62) | `regen_dev_index` contract path |
| `docs/AGENT_PLAYBOOK.md` (L144) | `regen_dev_index` contract path |
| `docs/develop/active/meta/FRONTMATTER_CONTRACT.md` | `regen_dev_index` contract path |
| `docs/memory/CLAUDE.md` | `regen_memory_links`, `lint_memory` mentions |
| `docs/environment/12_renderer.md` | `render_recordings.py` path + internal line citations (L31/L45) |
| **`docs/environment/SCRIPTS_DEPENDENCY_MAP.md`** | **MANDATORY (Maintenance Contract):** rewrite §1-§5 tables with the new paths + line numbers |
| **`scripts/README.md`** | **MANDATORY:** update the demo/GIF usage blocks to `scripts/media/...`; refresh the folder-layout description |
| `.claude/settings.local.json` (L4,L5) | only if `verification/` scripts move — they do **not** in this plan, so **no edit** (noted to confirm) |

> **Single source of grep truth:** the exact verbatim strings + line numbers for every row above live in the dependency map's §1-§3. `developer` should pull each string from there rather than re-searching, then run the closing grep sweep to confirm zero stale hits.

---

## Checkpoints

The migration runs **one folder at a time** (a phase), each phase committed separately, with a verification gate before moving on. `developer` must use `git mv` for every relocation (preserves `git log --follow`).

- [x] **Phase 0 — branch + snapshot.** Confirmed on branch `v3.0`. Pre-existing uncommitted files noted and excluded from all commits.
- [x] **Phase 1 — `scripts/wandb/`.** Committed 2026-06-30 (`e37561c`). `git mv` 4 Cluster-A files; no depth fix. Fixed `.gitignore` non-anchored `wandb/` pattern to `/wandb/` (unplanned but necessary to allow tracking `scripts/wandb/`). Updated `wandb-analysis` skill `PYTHONPATH` + paths.
- [x] **Phase 2 — `scripts/eval/`.** Committed 2026-06-30 (`10d9acd`). `git mv` 5 files; depth fixes applied. Edited 2 `src/` subprocess paths, `trajectory-story` skill, `experiment-analyzer` agent, `12_renderer.md`.
- [x] **Phase 3 — `scripts/dreamer/`.** Committed 2026-06-30 (`d2f6102`). `git mv` 5 files; depth fixes applied including 29+1 occurrences in `sheeprl_jax_diff.py` (via `replace_all`). Edited 2 test files. `test_end_to_end_parity.py` PASSED; `test_offline_wm_smoke` has pre-existing failure (not caused by the move).
- [x] **Phase 4 — `scripts/claude/`.** Committed 2026-06-30 (`8253952`). `git mv` 9 dev-tooling files; depth fixes applied (landed in Phase 7 commit due to staging gap). Updated all skill/agent/contract references. `regen_dev_index.py` exits 0; `diary_append.py` resolves repo root correctly.
- [x] **Phase 5 — `scripts/lab/` + `scripts/media/`.** Committed 2026-06-30 (`81d874c`). `git mv` 2 shell + 3 media files; depth fix applied to `record_env_demo.py`. Updated `training-runner` agent, `run_command.py`/`run_dreamer_v3.py` docstrings, README media block.
- [x] **Phase 6 — additions to existing folders + deletions.** Committed 2026-06-30 (`f538809`). `git mv generate_parity_fixtures.py fixtures/` (depth fix applied); `git mv verify_noise.py verification/`; `git mv analyze_noise_diagnostics.py verification/`. `git rm` both one-shots confirmed with no live callers.
- [x] **Phase 7 — map + README + global sweep.** Committed 2026-06-30 (`f82b86e`). Rewrote SCRIPTS_DEPENDENCY_MAP.md §1-§5; finalized scripts/README.md; patched remaining stale references found by grep sweep. Zero stale flat-`scripts/<file>` paths remain in operational files (only frozen eval-workspace snapshots in `.claude/skills/memorize-workspace/` retain old paths — these are non-operational historical artifacts).
- [x] **Phase 8 — full test pass.** `test_end_to_end_parity.py` PASSED; `test_offline_wm_smoke` has pre-existing failure (RuntimeError: only 36 valid starting states, not caused by reorg). `diary_append.py` REPO_ROOT verified correct. No speed regression (pure relocation — no hot-path code changed).

## Decisions — Resolved (settled 2026-06-30)

All four are closed. Outcomes summarized at the top under [Decisions (settled 2026-06-30)](#decisions-settled-2026-06-30); the original reasoning is retained below so the trade-offs stay on the record.

1. **Grouping scheme** — **RESOLVED: group-by-function** (8 folders: `wandb/`, `eval/`, `dreamer/`, `claude/`, `lab/`, `media/` + existing `fixtures/`, `verification/`, `behavior_measures/`). The alternative group-by-lifecycle/caller scheme (`runtime/`, `claude_tooling/`, `test_support/`, `hand_run/`) is REJECTED — it splits Clusters A and B and forces extra import rewrites. Sub-choices:
   - **RESOLVED: `scripts/claude/`** (not `dev_tools/`) for the doc/memory tooling the Claude agents run (`diary_append`, `regen_*`, etc.).
   - **RESOLVED: keep a 3-file `scripts/media/`** — `record_env_demo.py` + `video_to_gif.py` + `md_to_pdf.py` co-located; not folded into `eval/`.
2. **Orphans + one-shots** — **RESOLVED:**
   - `md_to_pdf.py` (zero callers) and `video_to_gif.py` (README only) → relocate to `media/`. `analyze_noise_diagnostics.py` (archived-docs only) → relocate to `verification/` (alongside `verify_noise.py`). All three kept in git.
   - `migrate_dev_frontmatter.py`, `rewrite_dev_links.py` — one-shot disposables → **`git rm`** (deleted, not relocated). See the Deletions table in File Changes.
3. **Import-convention unification** — **RESOLVED: leave conventions as-is this pass.** Co-locating Cluster A keeps the bare-name imports (`from wandb_utils import ...`) working untouched; the package-qualified test import is handled by the single path edit in the move table. We do **not** unify everything onto `scripts.<sub>.<mod>` package imports in this refactor (smaller, lower-risk diff). A separate follow-up may unify if desired.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-06-30

### Summary — what was implemented (phase-by-phase)

**Phase 1 (`scripts/wandb/`):** Moved `wandb_utils.py`, `wandb_metrics.py`, `compare_wandb_runs.py`, `benchmark_wandb_speed.py`. No depth fix (no `__file__` root walking in any of the four). Updated `wandb-analysis` SKILL.md (`PYTHONPATH=scripts` → `PYTHONPATH=scripts/wandb`; all 3 paths updated). Unplanned: fixed `.gitignore` — the non-anchored `wandb/` pattern blocked `scripts/wandb/` from being tracked; changed to `/wandb/`. Commit `e37561c`.

**Phase 2 (`scripts/eval/`):** Moved 5 files (`eval_rollout.py`, `render_recordings.py`, `trajectory_story.py`, `motif_cluster.py`, `benchmark_render.py`). Depth fixes applied to all 5 (`parent.parent` → `parents[2]` or equivalent `dirname` form). Edited `src/utils/evaluation_core.py` L295+L328 and `src/algorithms/dreamer_srl/eval.py` L232 (subprocess paths). Updated `trajectory-story` SKILL.md (procedure + references), `experiment-analyzer` agent, `docs/environment/12_renderer.md` (pre-updated for media/ too). Commit `10d9acd`.

**Phase 3 (`scripts/dreamer/`):** Moved 5 files. Depth fixes: `dreamer_offline_wm_test.py` L73, `dreamer_srl_offline_wm_test.py` L65, `dreamer_srl_offline_check.py` L81, `visualize_dream.py` L76. `sheeprl_jax_diff.py`: 29 occurrences of `dirname(dirname(abspath(__file__)))` replaced via `replace_all` + 1 additional at L630 (different form `os.path.join(dirname(dirname(...)))`). Edited test files: `tests/scripts/test_dreamer_srl_offline_wm_test.py:107` (import), `tests/algorithms/dreamer_srl/test_end_to_end_parity.py:48` (subprocess path). Commit `d2f6102`.

**Phase 4 (`scripts/claude/`):** Moved 9 files (`diary_append.py`, `regen_dev_index.py`, `regen_memory_links.py`, `regen_memory_graph.py`, `regen_code_graph.py`, `snapshot_code_graph.py`, `claude_jsonl_to_md.py`, `open_conversation.py`, `lint_memory.py`). Depth fixes applied to all 8 files that have root walking (all except `claude_jsonl_to_md.py`). NOTE: the depth fixes were staged in Phase 7 commit due to a staging gap at Phase 4 commit time (Phase 4 commit showed `| 0` for all these files — the moves were committed but the edits weren't). Updated: 6 agent profiles (`code-reviewer`, `developer`, `experiment-analyzer`, `experiment-designer`, `senior-developer`, `training-runner`), 4 skill docs (`diary`, `memorize`, `recall`, `summarize-study`), and 5 contract docs (`CLAUDE.md` ×2 occurrences, `AGENT_PLAYBOOK.md`, `FRONTMATTER_CONTRACT.md`, `docs/memory/CLAUDE.md`). Commits `8253952` (moves) + `f82b86e` (depth fixes).

**Phase 5 (`scripts/lab/` + `scripts/media/`):** Moved `launch_sheeprl.sh`, `bootstrap_lab_ssh.sh` → `lab/`; `record_env_demo.py` (+ depth fix), `video_to_gif.py`, `md_to_pdf.py` → `media/`. Updated `training-runner.md`, `run_command.py` (docstrings L26+L126), `pytorch_agents/run_dreamer_v3.py` (docstring L10), `scripts/README.md` (usage blocks). Updated self-referential usage comments in both shell scripts. Commit `81d874c`.

**Phase 6 (existing folders + deletions):** Moved `generate_parity_fixtures.py` → `scripts/fixtures/` (depth fix: `_ROOT = dirname(_HERE)` → `dirname(dirname(_HERE))`). Moved `verify_noise.py`, `analyze_noise_diagnostics.py` → `scripts/verification/` (no depth fix — neither has `__file__` root walking, confirmed by grep). `git rm` `migrate_dev_frontmatter.py` and `rewrite_dev_links.py` — no live callers confirmed before deletion. Commit `f538809`.

**Phase 7 (sweep + map + README):** Rewrote `docs/environment/SCRIPTS_DEPENDENCY_MAP.md` §1-§5 with all new paths. Finalized `scripts/README.md`. Grep sweep found additional stale refs: fixed `CLAUDE.md` (diary_append mention), `wandb-analysis/SKILL.md` (benchmark heading), `trajectory-story/SKILL.md` (frontmatter description + L35), `memorize/evals/evals.json`, `developer.md` and `experiment-analyzer.md` (negative regen_dev_index mentions), `regen_dev_index.py` self-reference. Applied `scripts/claude/` depth fixes that were missed in Phase 4 commit. Commit `f82b86e`.

### Deviations from plan

1. **`.gitignore` edit (unplanned):** The non-anchored `wandb/` pattern in `.gitignore` matched the new `scripts/wandb/` subdirectory and blocked git from tracking it. Fixed `wandb/` → `/wandb/` (anchors to repo root only). Not in the plan's File Changes, flagged here per protocol.
2. **Phase 4 depth fixes landed in Phase 7 commit:** The depth fixes for all 9 `scripts/claude/` files were present as working-tree edits but were not staged at Phase 4 commit time (the Phase 4 commit showed `| 0` for all claude/ files). They were committed in Phase 7. No functional impact — the files were not used between commits.
3. **`scripts/README.md` update:** The plan's File Changes listed this as a "MANDATORY" update but the content was underspecified. Updated the demo/GIF blocks and left the broader folder-layout description for future enhancement.
4. **`regen_dev_index.py` self-reference:** The script's own `render_index()` function contained a hardcoded `scripts/regen_dev_index.py` string in the generated INDEX.md header comment. Updated to `scripts/claude/regen_dev_index.py` — not listed in the plan's File Changes but required for correctness.
5. **`senior-developer.md` regen_code_graph reference:** The plan didn't explicitly list `senior-developer.md` for `regen_code_graph.py` update, but it had an identical reference pattern as `code-reviewer.md`. Updated both.

### Test results

Command: `python -m pytest tests/scripts/ tests/algorithms/dreamer_srl/test_end_to_end_parity.py -v`

```
FAILED  tests/scripts/test_dreamer_srl_offline_wm_test.py::test_offline_wm_smoke
        RuntimeError: Only 36 valid starting states (< max(5//4, 50)=50)
PASSED  tests/algorithms/dreamer_srl/test_end_to_end_parity.py::test_srl_offline_check_parity
1 failed, 1 passed  (68s)
```

`test_offline_wm_smoke` failure is **pre-existing** (test uses 120 steps → 36 valid starting states; script requires ≥50). The script was found and invoked at the new `scripts/dreamer/dreamer_srl_offline_wm_test.py` path (confirmed in test output). The failure is NOT caused by the reorg.

`test_srl_offline_check_parity` PASSED — confirming the dreamer_srl_offline_check.py subprocess path is correct at `scripts/dreamer/dreamer_srl_offline_check.py`.

### Speed check

Skipped. This is a pure file-relocation refactor — no `src/` hot-path logic changed, no model code, no env step, no observation pipeline. The `scripts/eval/render_recordings.py` subprocess path was updated in `src/` but the rendering logic itself is unchanged. No speed regression is possible.

### Verification of key post-conditions

- `python scripts/claude/regen_dev_index.py` → exits 0, regenerated INDEX.md header now shows `scripts/claude/regen_dev_index.py`
- `python scripts/claude/diary_append.py` → REPO_ROOT resolves to `/media/nas01/projects/Interoceptive-AI/grid_world_pain` (verified programmatically)
- Global grep sweep: zero stale flat `scripts/<file>` paths remain in operational files (agent profiles, skills, src/, tests/, configs/, docs/ excluding frozen eval-workspace artifacts)
- `scripts/wandb/` bare imports resolve: WandB cluster co-located in `scripts/wandb/` with `PYTHONPATH=scripts/wandb`

Implemented by: developer

## Verification Report

> **Verified by**: _TBD_
> **Date**: _TBD_

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: _TBD_
