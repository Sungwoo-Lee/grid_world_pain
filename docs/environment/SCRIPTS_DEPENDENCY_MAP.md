# `scripts/` Dependency Map — Safe-Reorganization Reference

## Purpose (read this first)

The `scripts/` folder holds ~45 loose helper scripts (training launchers, evaluation/rendering tools, WandB analysis, fixture generators, and "dev-tooling" that the Claude agents/skills run). The goal is to tidy it into subfolders **without breaking anything that calls these scripts**. This document is the safety net for that move: it lists, for every script, **who calls it and exactly how**, so that when a file is relocated you know precisely which references must be updated.

A script can be referenced in four different ways, and a reorg must respect all four:

1. **Python imports between scripts** — e.g. `wandb_metrics.py` does `from wandb_utils import ...`. Move one without the other and the import fails.
2. **Subprocess/shell paths** — e.g. the training code shells out to `scripts/render_recordings.py` by a hardcoded path string.
3. **Claude tooling** — skills and agent profiles hardcode commands like `python scripts/diary_append.py ...`. These break **silently** (no error until an agent next runs the command).
4. **Permission allowlist** — `.claude/settings.local.json` pins two scripts by exact path; moving them re-triggers permission prompts.

> **The one hazard that has nothing to do with callers — read before moving ANY file.**
> Most `scripts/*.py` compute the repo root by walking **up a fixed number of parent directories** from `__file__` (e.g. `Path(__file__).resolve().parent.parent`, or `os.path.dirname(os.path.dirname(__file__))`). That math assumes the file sits at `scripts/<file>.py` (one level under the repo root). **Moving a file into a new subfolder — `scripts/<subdir>/<file>.py` — adds a directory level and silently makes the script import the wrong repo root**, breaking its `from src...` imports, *regardless of who calls it*. The two folders that already live one level deeper (`behavior_measures/`, `verification/`) compensate: `behavior_measures/avoidance_stats_heatmap.py` uses `parents[2]`, and `verification/check_*.py` use a triple-`dirname`. **Any file moved one level down must have its repo-root computation re-depthed the same way.** This affects essentially every relocation and is the single most likely cause of a "worked before, broken after" reorg.

Bottom line up front: training and data-sync are almost fully decoupled from `scripts/` (only `launch_sheeprl.sh` matters). The real coupling lives in (a) the WandB import cluster, (b) `render_recordings.py` (called from `src/`), (c) a handful of test files, (d) the Claude skills/agents, and (e) the `sys.path` depth issue above.

---

## How this map was built

Three read-only agent sweeps over the three caller surfaces, cross-checked against a full `find scripts -type f` inventory (47 files incl. READMEs):

- **Training / data-sync surface** — `train_command-*.sh`, `generate_demo.sh`, `sync-*.sh`, and the `train.py` / `evaluation.py` chain.
- **Eval / render / analysis surface** — the Python tools and their script-to-script imports.
- **Claude-tooling surface** — `.claude/skills`, `.claude/agents`, `.claude/settings*.json`, `.git/hooks`, and canonical `docs/`.

Verbatim reference strings and line numbers are preserved so a relocation knows exactly what to rewrite.

---

## 1. Hard edges that break code on move (highest priority)

### 1a. Script-to-script Python imports (WandB cluster)

These use **bare-name imports** (no `scripts.` prefix), which resolve only because the running script's own directory is on `sys.path[0]`. Moving the target **or** the importer into a different folder breaks them unless you convert to package-qualified imports or fix `sys.path`.

| Importer | Exact import line | Target |
|---|---|---|
| `scripts/wandb_metrics.py:53` | `from wandb_utils import (WANDB_ENTITY, WANDB_PROJECT, compute_stats, fetch_wandb_runs, fmt, match_wandb_run)` | `scripts/wandb_utils.py` |
| `scripts/compare_wandb_runs.py:23` | `from wandb_utils import WANDB_ENTITY, WANDB_PROJECT, fetch_wandb_runs, match_wandb_run` | `scripts/wandb_utils.py` |
| `scripts/compare_wandb_runs.py:24` | `from wandb_metrics import PRESETS, pull_and_analyze, print_compare_preset` | `scripts/wandb_metrics.py` |
| `scripts/benchmark_wandb_speed.py:25` | `from wandb_utils import (WANDB_ENTITY, WANDB_PROJECT, fetch_wandb_runs, format_duration, match_wandb_run)` | `scripts/wandb_utils.py` |

**Rule:** keep `wandb_utils.py`, `wandb_metrics.py`, `compare_wandb_runs.py`, `benchmark_wandb_speed.py` **in the same folder** (move them together), or rewrite the 4 imports above. The `wandb-analysis` skill also runs these with `PYTHONPATH=scripts python scripts/wandb_metrics.py ...`, so if they move into `scripts/<sub>/`, update both the skill's `PYTHONPATH` and the path.

> **Name-collision caution:** `scripts/wandb_utils.py` is a *different file* from `src/utils/wandb_utils.py`. The `src/` code (`train.py`, `evaluation_core.py`, `dreamer_srl/eval.py`) imports `from src.utils.wandb_utils` and does **not** touch the `scripts/` one. Moving `scripts/wandb_utils.py` does not affect `src/`.

### 1b. `scripts/` paths hardcoded inside `src/` (subprocess)

| Caller | Exact reference | Target |
|---|---|---|
| `src/utils/evaluation_core.py:295` | `render_script = os.path.join(project_root, "scripts", "render_recordings.py")` → `subprocess.run([...])` | `scripts/render_recordings.py` |
| `src/algorithms/dreamer_srl/eval.py:232` | `render_script = os.path.join(_project_root, 'scripts', 'render_recordings.py')` → `subprocess.run([...])` | `scripts/render_recordings.py` |

**Rule:** moving `render_recordings.py` requires editing both lines above. (`evaluation_core.py:328` also prints a cosmetic "Render with: python scripts/render_recordings.py ..." hint — non-breaking, but update for tidiness. `eval.py:229` separately hardcodes the project root as a literal absolute path — unrelated to the move.)

### 1c. `scripts/` referenced from the test suite

| Caller | Exact reference | Target |
|---|---|---|
| `tests/scripts/test_dreamer_srl_offline_wm_test.py:107` | `from scripts.dreamer_srl_offline_wm_test import main` (uses `sys.path.insert(0, _REPO)`, `scripts.` package prefix) | `scripts/dreamer_srl_offline_wm_test.py` |
| `tests/algorithms/dreamer_srl/test_end_to_end_parity.py:48` | `OFFLINE_CHECK_SCRIPT = os.path.join(REPO_ROOT, "scripts", "dreamer_srl_offline_check.py")` → `subprocess.run([...])` | `scripts/dreamer_srl_offline_check.py` |

**Rule:** moving either target requires editing the matching test. Note the test uses the `scripts.`-package import style, while the WandB trio (§1a) uses bare names — **two inconsistent intra-`scripts/` import conventions coexist**; a reorg is a good moment to unify them.

### 1d. Permission allowlist (exact-path pins)

| Caller | Exact entry | Target |
|---|---|---|
| `.claude/settings.local.json:4` | `Bash(/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/verification/check_observability_gates.py)` | `scripts/verification/check_observability_gates.py` |
| `.claude/settings.local.json:5` | `Bash(.../bin/python scripts/verification/check_olfaction_parity.py)` | `scripts/verification/check_olfaction_parity.py` |

**Rule:** these two already live in `verification/`; if moved, update the allowlist or the next run re-prompts for permission (non-fatal, but annoying).

---

## 2. Claude tooling edges (break silently — no error until an agent runs them)

These are commands hardcoded in skills and agent profiles with the full conda interpreter path. Nothing surfaces an error on move; the failure appears the next time that skill/agent fires. **No git hook and no `settings.json` invokes any of these** — `.git/hooks/` contains only `*.sample` files, so nothing runs on commit.

| Target | Executable callers (must update on move) |
|---|---|
| `diary_append.py` | **7 callers** — skills: `diary`, `memorize` (L182), `summarize-study` (L279,373); agents: `developer` (L94), `senior-developer` (L97), `experiment-analyzer` (L114), `training-runner` (L286). Most-wired dependency in the repo. |
| `regen_dev_index.py` | agent `senior-developer` (L52, executable); contracts in `CLAUDE.md` (L62), `AGENT_PLAYBOOK.md` (L144), `FRONTMATTER_CONTRACT.md`; `experiment-analyzer`/`experiment-designer` name it in *negative* "do NOT run" instructions. |
| `regen_memory_links.py` | skill `memorize` Step 9 (L198, executable); contract `docs/memory/CLAUDE.md`. |
| `regen_memory_graph.py` | skill `memorize` (L248, executable). |
| `snapshot_code_graph.py` | skill `memorize` (L34,79, optional step). |
| `claude_jsonl_to_md.py` | skill `memorize` (L40,156,280). |
| `trajectory_story.py` | skill `trajectory-story` (primary tool); agent `experiment-analyzer` (L59). |
| `eval_rollout.py` | skill `trajectory-story` (writes the `.rec.gz` the toolchain consumes); agent `experiment-analyzer` (L59). |
| `render_recordings.py` | skill `trajectory-story` (L60); plus the two `src/` subprocess paths in §1b; `docs/environment/12_renderer.md` cites its **internal lines L31/L45**. |
| `launch_sheeprl.sh` | agent `training-runner` (L16); `run_command.py` docstring examples (L26,126); `pytorch_agents/run_dreamer_v3.py:10` docstring. The one real training-launch path. |
| `bootstrap_lab_ssh.sh` | agent `training-runner` (L81). |
| `regen_code_graph.py` | on-demand hint only — skill `recall` (L53), agents `senior-developer`/`code-reviewer`, README. Output is gitignored; low stakes. |

**Lower-stakes (doc-mention only, no executable caller):** `lint_memory.py`, `open_conversation.py`, `migrate_dev_frontmatter.py` (one-shot, disposable), `rewrite_dev_links.py` (one-shot, disposable).

---

## 3. Per-file roll-up (the move-safety table)

Stakes legend: **CODE** = breaks Python/subprocess; **TOOL** = breaks a skill/agent silently; **TEST** = breaks pytest; **SETTINGS** = re-prompts permission; **HAND** = hand-run only (move is free except the §0 depth fix); **ORPHAN** = no inbound reference at all.

| File | Inbound callers | Stakes | Must update on move |
|---|---|---|---|
| `wandb_utils.py` | wandb_metrics, compare_wandb_runs, benchmark_wandb_speed (bare imports) | CODE | move with WandB cluster or fix 4 imports |
| `wandb_metrics.py` | compare_wandb_runs; `wandb-analysis` skill | CODE+TOOL | WandB cluster; skill `PYTHONPATH`/path |
| `compare_wandb_runs.py` | `wandb-analysis` skill | TOOL | WandB cluster; skill |
| `benchmark_wandb_speed.py` | `wandb-analysis` skill | CODE+TOOL | WandB cluster; skill |
| `render_recordings.py` | `src/` ×2 subprocess; `trajectory-story` skill; renderer doc | CODE+TOOL | `evaluation_core.py:295`, `eval.py:232`, skill L60, `12_renderer.md` |
| `eval_rollout.py` | `trajectory-story` skill; `experiment-analyzer`; testbed configs (comment) | TOOL | skill + agent commands |
| `trajectory_story.py` | `trajectory-story` skill (primary); `experiment-analyzer` | TOOL | skill + agent |
| `diary_append.py` | 4 agents + 3 skills | TOOL | all 7 commands (see §2) |
| `regen_dev_index.py` | `senior-developer` agent; 3 contract docs | TOOL | agent cmd + contract docs |
| `regen_memory_links.py` | `memorize` skill; memory contract | TOOL | skill + doc |
| `regen_memory_graph.py` | `memorize` skill | TOOL | skill |
| `snapshot_code_graph.py` | `memorize` skill | TOOL | skill |
| `claude_jsonl_to_md.py` | `memorize` skill | TOOL | skill |
| `regen_code_graph.py` | on-demand hints (recall/sr-dev/reviewer) | TOOL (low) | doc hints |
| `launch_sheeprl.sh` | `training-runner`; `run_command.py` docstrings | TOOL | agent + docstrings |
| `bootstrap_lab_ssh.sh` | `training-runner` (L81) | TOOL | agent |
| `dreamer_srl_offline_wm_test.py` | `tests/scripts/...:107` (import) | TEST | test import |
| `dreamer_srl_offline_check.py` | `tests/.../test_end_to_end_parity.py:48` (subprocess) | TEST | test path |
| `verification/check_observability_gates.py` | `settings.local.json:4` | SETTINGS | allowlist |
| `verification/check_olfaction_parity.py` | `settings.local.json:5` | SETTINGS | allowlist |
| `record_env_demo.py` | README only | HAND | depth fix only |
| `benchmark_render.py` | docs only | HAND | depth fix only |
| `visualize_dream.py` | design doc + diary | HAND | depth fix only |
| `motif_cluster.py` | none (test reimplements KMeans, no import) | HAND | depth fix only |
| `verify_noise.py` | docs only | HAND | depth fix only |
| `behavior_measures/avoidance_stats_heatmap.py` | README + study doc | HAND | already `parents[2]` |
| `dreamer_offline_wm_test.py` | referenced in sibling's comments only | HAND | depth fix only |
| `generate_parity_fixtures.py` | named in a test *docstring* only (not invoked) | HAND | depth fix only |
| `sheeprl_jax_diff.py` | tests README + docs (mentions) | HAND | depth fix only |
| `fixtures/gen_cp1..8_fixtures.py` | hand-run generators; gen_cp8 in hint strings | HAND | already `scripts/fixtures/` depth |
| `lint_memory.py` | `docs/memory/CLAUDE.md` (mention) | HAND | doc mention |
| `open_conversation.py` | design doc (mention) | HAND | doc mention |
| `migrate_dev_frontmatter.py` | reorg plan doc (one-shot) | HAND | disposable — consider `git rm` |
| `rewrite_dev_links.py` | reorg plan doc (one-shot) | HAND | disposable — consider `git rm` |
| `analyze_noise_diagnostics.py` | archived docs only | ORPHAN | none |
| `video_to_gif.py` | README only | ORPHAN | none |
| `md_to_pdf.py` | **none anywhere** | ORPHAN | none |

---

## 4. Move-together clusters

- **Cluster A — WandB analysis (import-coupled, MUST stay together):** `wandb_utils.py` (leaf) ← `wandb_metrics.py` ← `compare_wandb_runs.py`; `wandb_utils.py` ← `benchmark_wandb_speed.py`. Bare-name imports; splitting across folders breaks resolution unless rewritten.
- **Cluster B — eval → record → render pipeline (path/format-coupled, not imports):** `eval_rollout.py` (writes `.rec.gz`) → `render_recordings.py` (renders) → `trajectory_story.py`, `behavior_measures/avoidance_stats_heatmap.py`, `motif_cluster.py` (consume recordings). Coupling is the recording format (`src/utils/eval_recording.py`) plus the hardcoded `render_recordings.py` subprocess path in `src/`. They can live in separate folders, but `render_recordings.py`'s move forces the §1b edits.
- **Cluster C — dreamer/sheeprl parity (path/string-coupled):** `fixtures/gen_cp*.py` → `sheeprl_jax_diff.py`, `dreamer_srl_offline_check.py`, `dreamer_srl_offline_wm_test.py`, `dreamer_offline_wm_test.py`. Coupling is fixture `.npz` paths under `tests/fixtures/dreamer_srl/` and hint strings, plus the two `tests/` files (§1c). No intra-`scripts/` Python imports.
- **Dev-tooling group (TOOL stakes, no imports between them):** `diary_append.py`, `regen_dev_index.py`, `regen_memory_links.py`, `regen_memory_graph.py`, `regen_code_graph.py`, `snapshot_code_graph.py`, `claude_jsonl_to_md.py`, `open_conversation.py`, `lint_memory.py`, `migrate_dev_frontmatter.py`, `rewrite_dev_links.py`. Independent files, but every executable one is hardcoded in a skill/agent (§2).

---

## 5. Orphans (zero inbound reference — safe to move/retire, ask owner first)

- `md_to_pdf.py` — no caller anywhere in code, configs, skills, agents, or docs.
- `video_to_gif.py` — only `scripts/README.md`.
- `analyze_noise_diagnostics.py` — only archived docs (`docs/develop/archive/NOISE_*`).

All three remain hand-runnable; "orphan" means no programmatic inbound edge. They still need the §0 depth fix if moved one level deeper.

---

## 6. Reorg checklist (when the move actually happens)

1. **Apply the §0 depth fix to every relocated `.py`** that derives repo root from `__file__` — re-depth `parent.parent` / double-`dirname` to match the new level (`parents[2]` / triple-`dirname` for one-level-deeper).
2. **Keep Cluster A together** or convert its 4 bare imports to package-qualified form (and pick one convention — bare vs `scripts.` — to unify with the test in §1c).
3. **Edit the two `src/` subprocess paths** if `render_recordings.py` moves (§1b); also `docs/environment/12_renderer.md`'s line-number citation.
4. **Update the 2 test files** in §1c if their targets move.
5. **Update the `.claude/settings.local.json` allowlist** if the `verification/` scripts move.
6. **Grep skills + agent profiles** for each moved dev-tooling filename and rewrite the hardcoded `scripts/<file>` command (§2). These fail silently — verify by re-running the affected skill/agent.
7. **Retire decisions:** confirm with the owner before deleting the 3 orphans (§5) and the 2 one-shot scripts (`migrate_dev_frontmatter.py`, `rewrite_dev_links.py`).
8. After moving, `grep -rn "scripts/" --include=*.md --include=*.py --include=*.json . | grep -v worktrees` and reconcile every remaining hit against the new layout.

---

*Generated from a three-surface read-only dependency sweep. The verbatim reference strings and line numbers above are the authoritative "what to rewrite" list; re-run the sweep if `scripts/` callers change substantially.*
