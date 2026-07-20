# `scripts/` Dependency Map — Safe-Reorganization Reference

## Purpose (read this first)

The `scripts/` folder holds helper scripts organized into topic subfolders. Each subfolder groups scripts by function: `wandb/` (WandB analysis), `eval/` (evaluation/rendering), `dreamer/` (Dreamer parity tools), `claude/` (Claude agent/skill tooling), `lab/` (cluster launch), `media/` (video/GIF/PDF), `fixtures/` (test fixture generators), `verification/` (environment soundness checks), `behavior_measures/` (behavioral stats). This document lists **who calls each script and exactly how**, so that when a file is relocated you know precisely which references must be updated.

A script can be referenced in four different ways, and a reorg must respect all four:

1. **Python imports between scripts** — e.g. `wandb_metrics.py` does `from wandb_utils import ...`. Move one without the other and the import fails.
2. **Subprocess/shell paths** — e.g. the training code shells out to `scripts/eval/render_recordings.py` by a hardcoded path string.
3. **Claude tooling** — skills and agent profiles hardcode commands like `python scripts/claude/diary_append.py ...`. These break **silently** (no error until an agent next runs the command).
4. **Permission allowlist** — `.claude/settings.local.json` pins two scripts by exact path; moving them re-triggers permission prompts.

> **The one hazard that has nothing to do with callers — read before moving ANY file.**
> Most `scripts/**/*.py` compute the repo root by walking **up a fixed number of parent directories** from `__file__`. Scripts at `scripts/<subdir>/<file>.py` (one level under the repo root) must use `parents[2]` or triple-`dirname`; scripts at `scripts/<subdir>/<subdir2>/<file>.py` must use `parents[3]` or four-deep `dirname`. The two existing folders `behavior_measures/` and `verification/` already use the correct depth. **Any file moved one level down must have its repo-root computation re-depthed.** This is the single most likely cause of a "worked before, broken after" reorg.

Bottom line up front: training and data-sync are almost fully decoupled from `scripts/` (only `scripts/lab/launch_sheeprl.sh` matters). The real coupling lives in (a) the WandB import cluster, (b) `render_recordings.py` (called from `src/`), (c) a handful of test files, (d) the Claude skills/agents, and (e) the `sys.path` depth issue above.

## Maintenance Contract

**This map must stay in step with the code.** Any change that **adds, moves, renames, or deletes a file under `scripts/`** — or that adds/removes a caller of a `scripts/` file (a new subprocess path in `src/`, a new skill/agent command, a new test import, a new settings allowlist entry) — must update the affected rows of this document **in the same change**, not later.

- **Who owns the update.** The `senior-developer` agent must list this file in the **File Changes** section of any plan that touches `scripts/` layout or callers; the `developer` agent then updates it as part of implementing that plan. A direct `scripts/` change made outside a plan (by top-level Claude or any agent) carries the same duty.
- **What to update.** The hard-edge tables (§1), the Claude-tooling table (§2), the per-file roll-up (§3), the clusters (§4), and the orphan list (§5) — whichever the change touches. Keep the verbatim line numbers accurate; they are the authoritative "what to rewrite" list.
- **Cheap re-verification.** When in doubt, re-run the sweep: `grep -rn "scripts/" --include=*.py --include=*.md --include=*.json . | grep -v worktrees` and reconcile every hit against this map.

---

## How this map was built

Three read-only agent sweeps over the three caller surfaces, cross-checked against a full `find scripts -type f` inventory, updated after the 2026-06-30 `scripts/` reorganization (Phases 1–6):

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
| `scripts/wandb/wandb_metrics.py:53` | `from wandb_utils import (WANDB_ENTITY, WANDB_PROJECT, compute_stats, fetch_wandb_runs, fmt, match_wandb_run)` | `scripts/wandb/wandb_utils.py` |
| `scripts/wandb/compare_wandb_runs.py:23` | `from wandb_utils import WANDB_ENTITY, WANDB_PROJECT, fetch_wandb_runs, match_wandb_run` | `scripts/wandb/wandb_utils.py` |
| `scripts/wandb/compare_wandb_runs.py:24` | `from wandb_metrics import PRESETS, pull_and_analyze, print_compare_preset` | `scripts/wandb/wandb_metrics.py` |
| `scripts/wandb/benchmark_wandb_speed.py:25` | `from wandb_utils import (WANDB_ENTITY, WANDB_PROJECT, fetch_wandb_runs, format_duration, match_wandb_run)` | `scripts/wandb/wandb_utils.py` |
| `scripts/eval/parity_check_eval_rollout.py` | `sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "behavior_measures"))` then `from avoidance_stats_heatmap import episode_measures, KEYS` (bare-name import, added 2026-07-06 — Tier 2 batched-rollout parity gate) | `scripts/behavior_measures/avoidance_stats_heatmap.py` |

**Rule:** keep `wandb_utils.py`, `wandb_metrics.py`, `compare_wandb_runs.py`, `benchmark_wandb_speed.py` **in the same folder** (all now in `scripts/wandb/`). The `wandb-analysis` skill runs these with `PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py ...`. Moving any of the four out of `scripts/wandb/` requires updating both the 4 imports above and the skill's `PYTHONPATH`/path.

**Rule (parity harness):** `parity_check_eval_rollout.py` imports `episode_measures`/`KEYS` from `avoidance_stats_heatmap.py` **deliberately** — it is the acceptance test for Tier 2 batched-rollout parity and must compute the 11 avoidance measures through the SAME function the real stats pipeline uses, never a re-derived copy. Moving `avoidance_stats_heatmap.py` out of `scripts/behavior_measures/` requires updating the harness's `sys.path.insert` line above.

> **Name-collision caution:** `scripts/wandb/wandb_utils.py` is a *different file* from `src/utils/wandb_utils.py`. The `src/` code (`train.py`, `evaluation_core.py`, `dreamer_srl/eval.py`) imports `from src.utils.wandb_utils` and does **not** touch the `scripts/wandb/` one.

### 1b. `scripts/` paths hardcoded inside `src/` (subprocess)

| Caller | Exact reference | Target |
|---|---|---|
| `src/utils/evaluation_core.py:295` | `render_script = os.path.join(project_root, "scripts", "eval", "render_recordings.py")` → `subprocess.run([...])` | `scripts/eval/render_recordings.py` |
| `src/algorithms/dreamer_srl/eval.py:232` | `render_script = os.path.join(_project_root, 'scripts', 'eval', 'render_recordings.py')` → `subprocess.run([...])` | `scripts/eval/render_recordings.py` |
| `scripts/eval/parity_check_eval_rollout.py` (`EVAL_ROLLOUT` constant, `_run_eval_rollout()`) | `EVAL_ROLLOUT = PROJECT_ROOT / "scripts" / "eval" / "eval_rollout.py"` → `subprocess.run([python, str(EVAL_ROLLOUT), --config, --checkpoint, --output-root, --batched?, ...])`, invoked once for the legacy path and once for the `--batched` path (added 2026-07-06, Tier 2) | `scripts/eval/eval_rollout.py` |

**Rule:** moving `render_recordings.py` requires editing both lines above. (`evaluation_core.py:328` also prints a cosmetic "Render with: python scripts/eval/render_recordings.py ..." hint — non-breaking, but update for tidiness.) Moving `eval_rollout.py` requires updating the harness's `EVAL_ROLLOUT` constant (it lives in the same `scripts/eval/` folder today, so this is currently a same-folder reference, not a cross-folder one).

### 1c. `scripts/` referenced from the test suite

| Caller | Exact reference | Target |
|---|---|---|
| `tests/scripts/test_dreamer_srl_offline_wm_test.py:107` | `from scripts.dreamer.dreamer_srl_offline_wm_test import main` (uses `sys.path.insert(0, _REPO)`, `scripts.` package prefix) | `scripts/dreamer/dreamer_srl_offline_wm_test.py` |
| `tests/algorithms/dreamer_srl/test_end_to_end_parity.py:48` | `OFFLINE_CHECK_SCRIPT = os.path.join(REPO_ROOT, "scripts", "dreamer", "dreamer_srl_offline_check.py")` → `subprocess.run([...])` | `scripts/dreamer/dreamer_srl_offline_check.py` |
| `tests/scripts/test_eval_rollout_online_replay.py` and `tests/scripts/test_eval_rollout_stage_config.py` | `sys.path.insert(0, os.path.join(_REPO, "scripts", "eval"))` + `import eval_rollout as er` (bare-name import) | `scripts/eval/eval_rollout.py` |

**Rule:** moving either target requires editing the matching test.

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
| `scripts/claude/diary_append.py` | **7 callers** — skills: `diary`, `memorize` (L182), `summarize-study` (L279,373); agents: `developer` (L94), `senior-developer` (L97), `experiment-analyzer` (L114), `training-runner` (L286). Most-wired dependency in the repo. |
| `scripts/claude/regen_dev_index.py` | agent `senior-developer` (L52, executable); contracts in `CLAUDE.md`, `AGENT_PLAYBOOK.md`, `FRONTMATTER_CONTRACT.md`; `experiment-analyzer`/`experiment-designer` name it in *negative* "do NOT run" instructions. |
| `scripts/claude/regen_memory_links.py` | skill `memorize` Step 9 (L198, executable); contract `docs/memory/CLAUDE.md`. |
| `scripts/claude/regen_memory_graph.py` | skill `memorize` (L248, executable). |
| `scripts/claude/snapshot_code_graph.py` | skill `memorize` (L34,79, optional step). |
| `scripts/claude/claude_jsonl_to_md.py` | skill `memorize` (L40,156,280). |
| `scripts/eval/trajectory_story.py` | skill `trajectory-story` (primary tool); agent `experiment-analyzer` (L59). |
| `scripts/eval/eval_rollout.py` | skill `trajectory-story` (writes the `.rec.gz` the toolchain consumes); agent `experiment-analyzer` (L59). |
| `scripts/eval/render_recordings.py` | skill `trajectory-story` (L60); plus the two `src/` subprocess paths in §1b; `docs/environment/12_renderer.md` cites its **internal lines L31/L45 — now stale** (added 2026-07-06, Tier 3: an FD-limit fix + `--max-episodes`/`--stride` flags shifted the two renderer imports to **L78/L93**; `12_renderer.md`'s citation was out of scope for this change and needs a follow-up edit). Neither `src/` caller (`evaluation_core.py`, `dreamer_srl/eval.py`) passes the new flags, so their behavior (render all episodes, default `--workers`) is unchanged. |
| `scripts/lab/launch_sheeprl.sh` | agent `training-runner` (L16); `run_command.py` docstring examples (L26,126); `pytorch_agents/run_dreamer_v3.py:10` docstring. The one real training-launch path. |
| `scripts/lab/bootstrap_lab_ssh.sh` | agent `training-runner` (L81). |
| `scripts/lab/gpu_status.py` | skill `gpu-status` (.claude/skills/gpu-status/SKILL.md); top-level Claude's GPU-assignment-before-launch flow; maintenance command in `docs/environment/LAB_NODE_GPU_SPEC.md`. Read-only direct-SSH nvidia-smi query across nodes 101-114. |
| `scripts/claude/regen_code_graph.py` | on-demand hint only — skill `recall` (L53), agents `senior-developer`/`code-reviewer`, README. Output is gitignored; low stakes. |

**Lower-stakes (doc-mention only, no executable caller):** `scripts/claude/lint_memory.py`, `scripts/claude/open_conversation.py`.

---

## 3. Per-file roll-up (the move-safety table)

Stakes legend: **CODE** = breaks Python/subprocess; **TOOL** = breaks a skill/agent silently; **TEST** = breaks pytest; **SETTINGS** = re-prompts permission; **HAND** = hand-run only (move is free except the §0 depth fix); **ORPHAN** = no inbound reference at all.

| File | Inbound callers | Stakes | Must update on move |
|---|---|---|---|
| `scripts/wandb/wandb_utils.py` | wandb_metrics, compare_wandb_runs, benchmark_wandb_speed (bare imports) | CODE | move with WandB cluster or fix 4 imports |
| `scripts/wandb/wandb_metrics.py` | compare_wandb_runs; `wandb-analysis` skill | CODE+TOOL | WandB cluster; skill `PYTHONPATH`/path |
| `scripts/wandb/compare_wandb_runs.py` | `wandb-analysis` skill | TOOL | WandB cluster; skill |
| `scripts/wandb/benchmark_wandb_speed.py` | `wandb-analysis` skill | CODE+TOOL | WandB cluster; skill |
| `scripts/eval/render_recordings.py` | `src/` ×2 subprocess; `trajectory-story` skill; renderer doc | CODE+TOOL | `evaluation_core.py:295`, `eval.py:232`, skill L60, `12_renderer.md` |
| `scripts/eval/eval_rollout.py` | `trajectory-story` skill; `experiment-analyzer`; tests `test_eval_rollout_online_replay.py` + `test_eval_rollout_stage_config.py` (§1c bare import); `scripts/eval/parity_check_eval_rollout.py` (subprocess ×2, added 2026-07-06 — Tier 2 `--batched` parity gate). Since 2026-07-20, ALSO the Dreamer eval entry point (agent-type dispatch on `--agent_config`/checkpoint layout): its Dreamer branch imports `src/algorithms/dreamer_srl/{agent,checkpoint,eval}.py` (plus `utils.moments_init` in the restore-target helper) branch-locally (mirrors the rPPO branch's own branch-local `src/models/recurrent_ppo_network.py` import) — a Dreamer checkpoint is passed as `--checkpoint <run_dir>/checkpoints/<episode>` (episode parsed from the basename; `--episode` disambiguates otherwise) and `--batched` is rejected with a warning + single-env fallback for Dreamer. | TOOL+TEST | skill + agent commands + 2 test `sys.path` inserts; harness `EVAL_ROLLOUT` constant; Dreamer branch's 4 branch-local `src/algorithms/dreamer_srl/*` imports (agent, checkpoint, eval, utils.moments_init) |
| `scripts/eval/dreamer_srl_probe_eval.py` | none (hand-run driver, added 2026-07-14, for running dreamer_srl checkpoints through behavior-probe env configs). **Left untouched by the 2026-07-20 unification** (a long-running distributed sweep was using it at the time — converting it to a thin shim over `eval_rollout.py` is a separate future follow-up); `eval_rollout.py`'s own Dreamer branch now ports the same restore/rollout logic independently rather than importing this file. | HAND | depth fix only (`scripts/eval/`, already `parents[2]`) |
| `scripts/eval/parity_check_eval_rollout.py` | none (hand-run acceptance test for the Tier 2 batched-rollout change, added 2026-07-06) | HAND | depth fix only (`scripts/eval/`, already `parents[2]`) |
| `scripts/eval/trajectory_story.py` | `trajectory-story` skill (primary); `experiment-analyzer` | TOOL | skill + agent |
| `scripts/eval/motif_cluster.py` | none (test reimplements KMeans, no import) | HAND | depth fix only |
| `scripts/eval/benchmark_render.py` | docs only | HAND | depth fix only |
| `scripts/claude/diary_append.py` | 4 agents + 3 skills | TOOL | all 7 commands (see §2) |
| `scripts/claude/regen_dev_index.py` | `senior-developer` agent; 3 contract docs | TOOL | agent cmd + contract docs |
| `scripts/claude/regen_memory_links.py` | `memorize` skill; memory contract | TOOL | skill + doc |
| `scripts/claude/regen_memory_graph.py` | `memorize` skill | TOOL | skill |
| `scripts/claude/snapshot_code_graph.py` | `memorize` skill | TOOL | skill |
| `scripts/claude/claude_jsonl_to_md.py` | `memorize` skill | TOOL | skill |
| `scripts/claude/regen_code_graph.py` | on-demand hints (recall/sr-dev/reviewer) | TOOL (low) | doc hints |
| `scripts/claude/lint_memory.py` | `docs/memory/CLAUDE.md` (mention) | HAND | doc mention |
| `scripts/claude/open_conversation.py` | design doc (mention) | HAND | doc mention |
| `scripts/lab/launch_sheeprl.sh` | `training-runner`; `run_command.py` docstrings | TOOL | agent + docstrings |
| `scripts/lab/bootstrap_lab_ssh.sh` | `training-runner` (L81) | TOOL | agent |
| `scripts/lab/gpu_status.py` | `gpu-status` skill; launch GPU-assignment; spec-doc maint | TOOL | skill + doc |
| `scripts/dreamer/dreamer_srl_offline_wm_test.py` | `tests/scripts/...:107` (import) | TEST | test import |
| `scripts/dreamer/dreamer_srl_offline_check.py` | `tests/.../test_end_to_end_parity.py:48` (subprocess) | TEST | test path |
| `scripts/dreamer/sheeprl_jax_diff.py` | tests README + docs (mentions) | HAND | depth fix only |
| `scripts/dreamer/visualize_dream.py` | design doc + diary | HAND | depth fix only |
| `scripts/media/record_env_demo.py` | README only | HAND | depth fix only |
| `scripts/media/video_to_gif.py` | README only | HAND | depth fix only |
| `scripts/media/md_to_pdf.py` | none anywhere | ORPHAN | none |
| `scripts/fixtures/generate_parity_fixtures.py` | named in a test *docstring* only (not invoked) | HAND | depth fix only |
| `scripts/fixtures/gen_cp1..8_fixtures.py` | hand-run generators; gen_cp8 in hint strings | HAND | already `scripts/fixtures/` depth |
| `scripts/verification/verify_noise.py` | docs only | HAND | depth fix only |
| `scripts/verification/analyze_noise_diagnostics.py` | archived docs only (2026-07-06: gained the `obs_intero_nociception`/`true_intero_nociception` modality pair as the name-based consumer of the H8 stats-CSV fix; still no runtime caller) | ORPHAN | none |
| `scripts/verification/check_observability_gates.py` | `settings.local.json:4` | SETTINGS | allowlist |
| `scripts/verification/check_olfaction_parity.py` | `settings.local.json:5` | SETTINGS | allowlist |
| `scripts/behavior_measures/avoidance_stats_heatmap.py` | README + study doc; `scripts/eval/parity_check_eval_rollout.py` (bare import of `episode_measures`/`KEYS`, added 2026-07-06) | HAND+CODE | already `parents[2]`; harness's `sys.path.insert` line (§1a) |

---

## 4. Move-together clusters

- **Cluster A — WandB analysis (import-coupled, MUST stay together):** `scripts/wandb/wandb_utils.py` (leaf) ← `scripts/wandb/wandb_metrics.py` ← `scripts/wandb/compare_wandb_runs.py`; `scripts/wandb/wandb_utils.py` ← `scripts/wandb/benchmark_wandb_speed.py`. Bare-name imports; splitting across folders breaks resolution unless rewritten. The `wandb-analysis` skill uses `PYTHONPATH=scripts/wandb`.
- **Cluster B — eval → record → render pipeline (path/format-coupled, not imports):** `scripts/eval/eval_rollout.py` (writes `.rec.gz`) → `scripts/eval/render_recordings.py` (renders) → `scripts/eval/trajectory_story.py`, `scripts/behavior_measures/avoidance_stats_heatmap.py`, `scripts/eval/motif_cluster.py` (consume recordings). Coupling is the recording format (`src/utils/eval_recording.py`) plus the hardcoded `scripts/eval/render_recordings.py` subprocess path in `src/`. **`scripts/eval/parity_check_eval_rollout.py` (added 2026-07-06)** joins this cluster from both ends: it subprocess-invokes `eval_rollout.py` twice (once legacy, once `--batched`) and imports `avoidance_stats_heatmap.episode_measures` directly (§1a) to score both runs identically — it is the Tier 2 acceptance gate for `eval_rollout.py`'s batched rollout path (docs/develop/active/refactors/EVAL_ROLLOUT_BATCHING_PERF.md). **`scripts/eval/dreamer_srl_probe_eval.py` (added 2026-07-14)** is the Dreamer-side counterpart to `eval_rollout.py`'s rPPO path: it calls `src/algorithms/dreamer_srl/eval.py::dreamer_srl_eval_rollout` directly (not a subprocess) to write the same `.rec.gz` format, so it joins Cluster B as an alternate producer feeding the same consumer scripts. **As of 2026-07-20, `eval_rollout.py` also calls `dreamer_srl_eval_rollout` directly** (in its own Dreamer branch, agent-type-dispatched alongside the existing rPPO branch), so `eval_rollout.py` alone now produces `.rec.gz` recordings for BOTH algorithms; `dreamer_srl_probe_eval.py` remains a separate, untouched, hand-run driver (see its row in §3) rather than being folded into a shim, pending the in-flight sweep that depends on its current form.
- **Cluster C — dreamer/sheeprl parity (path/string-coupled):** `scripts/fixtures/gen_cp*.py` → `scripts/dreamer/sheeprl_jax_diff.py`, `scripts/dreamer/dreamer_srl_offline_check.py`, `scripts/dreamer/dreamer_srl_offline_wm_test.py`. Coupling is fixture `.npz` paths under `tests/fixtures/dreamer_srl/` and hint strings, plus the two `tests/` files (§1c). (`dreamer_offline_wm_test.py` — the NNX offline diagnostic — moved out of `scripts/` on 2026-07-10 to `src/models/archive/dreamer_v3_nnx/scripts/` with the archived DreamerV3-NNX stack.)
- **Dev-tooling group (TOOL stakes, no imports between them):** all scripts under `scripts/claude/`. Independent files, but every executable one is hardcoded in a skill/agent (§2).

---

## 5. Orphans (zero inbound reference — safe to move/retire, ask owner first)

- `scripts/media/md_to_pdf.py` — no caller anywhere in code, configs, skills, agents, or docs.
- `scripts/verification/analyze_noise_diagnostics.py` — only archived docs (`docs/develop/archive/NOISE_*`).

Both remain hand-runnable; "orphan" means no programmatic inbound edge. They still need the §0 depth fix if moved one level deeper.

---

## 6. Reorg checklist (for future reorganizations within `scripts/`)

1. **Apply the §0 depth fix to every relocated `.py`** that derives repo root from `__file__` — re-depth `parent.parent` / double-`dirname` to match the new level (`parents[2]` / triple-`dirname` for one level under a subfolder; `parents[3]` / quadruple-`dirname` for two levels deep).
2. **Keep Cluster A together** (all four WandB scripts in the same folder) or convert its 4 bare imports to package-qualified form.
3. **Edit the two `src/` subprocess paths** if `render_recordings.py` moves (§1b); also `docs/environment/12_renderer.md`'s line-number citation.
4. **Update the 2 test files** in §1c if their targets move.
5. **Update the `.claude/settings.local.json` allowlist** if the `verification/` scripts move.
6. **Grep skills + agent profiles** for each moved dev-tooling filename and rewrite the hardcoded `scripts/<subdir>/<file>` command (§2). These fail silently — verify by re-running the affected skill/agent.
7. After moving, `grep -rn "scripts/" --include=*.md --include=*.py --include=*.json . | grep -v worktrees` and reconcile every remaining hit against the new layout.

---

*Verbatim reference strings and line numbers above are the authoritative "what to rewrite" list. Last updated: 2026-06-30 after the scripts/ folder reorganization (Phases 1–6). Re-run the sweep above if `scripts/` callers change substantially.*
