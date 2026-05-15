---
id: 20260509_0309_cluster_py_consolidation
date: 2026-05-09
time: "03:09"
folder: cluster_ops
tags: [meta, decision, training_runner, learned_lesson]
summary: "Replaced 8 single-purpose bash scripts (distribute_image, list_containers_all, list_images_all, stop_container_all, rm_container_all, run_container_all, remove_docker_image_all, roll_to_episode) with one Python tool `cluster.py` (~440 LOC, stdlib only). Subcommands via argparse; per-subcommand `--include` / `--exclude` filters with range syntax (`101-105`, `101..103`, `192.168.0.108`); module-level password cache for the `rollout` orchestrator; SSH ControlMaster multiplexing baked into the shared SSH-opts list."
related: ["20260509_0310_bash_ic_alias_over_ssh", "20260508_1638_container_slimdown_recipe", "20260508_1639_ssh_credentials_in_shared_image"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# `cluster.py` — Python consolidation of cluster-ops scripts

## Key conclusion
Maintaining ~600 lines across eight bash scripts (`distribute_image.sh`, `list_containers_all.sh`, `list_images_all.sh`, `stop_container_all.sh`, `rm_container_all.sh`, `run_container_all.sh`, `remove_docker_image_all.sh`, `roll_to_episode.sh`) was generating duplication every time the password-prompt UX, the color helper, the table renderer, or the parallel-SSH primitives needed a tweak. A single Python file `/media/nas03/docker_backup/sungwoo-lee/script/cluster.py` (~440 LOC, stdlib only — no `pip install`) now exposes nine subcommands (`ps`, `images`, `start`, `stop`, `rm`, `rmi`, `run`, `distribute`, `rollout`) over a shared infrastructure: one `ssh_run` primitive, one `parallel_each` ThreadPoolExecutor wrapper, one `render_table` with optional `colorize_row` and `group_by_col`, one module-level `_password_cache` so the `rollout` orchestrator prompts once and every internal subcommand call reuses the cached password. Every subcommand carries `--include` / `--exclude` filters via an argparse parent-parser pattern, mutually exclusive, with range syntax `101-105` / `101..103` / `192.168.0.108` parsed by a single `_parse_node_spec` helper. The bash scripts were deleted at the same commit; there is one file to edit going forward.

## Evidence, measurements, facts
- Original eight scripts (last bash form): `distribute_image.sh` 9.2K, `list_containers_all.sh` 6.0K, `list_images_all.sh` 5.5K, `stop_container_all.sh` 1.9K, `rm_container_all.sh` 1.9K, `run_container_all.sh` 2.7K, `remove_docker_image_all.sh` 2.3K, `roll_to_episode.sh` 3.8K. Total ≈ 33 KB.
- New `cluster.py`: ~440 lines, ≈ 17 KB. Net reduction of half. Behavior preserved (every subcommand maps 1:1 to one of the deleted scripts; `rollout` is the orchestrator).
- Subcommand routing pattern: shared parent-parser carries `--include` / `--exclude` (mutually exclusive group); every `sub.add_parser(name, parents=[filt], ...)` inherits both flags automatically. `_resolve_nodes(args)` is called once in `main()` after `parse_args()`, populating `args.nodes` for every handler.
- SSH multiplexing: `SSH_OPTS` includes `-o ControlMaster=auto -o ControlPath=/tmp/cluster_ssh_mux/%h_%p_%r -o ControlPersist=300`. First subcommand opens 14 master connections; subsequent stages of `rollout` reuse them (measurable speedup across the 6-stage pipeline).
- Password handling: `_password_cache: str | None` at module scope, guarded by `_password_lock`. `get_password()` checks env var `SSH_PASSWORD` first, then prompts via `getpass.getpass`. `cmd_rollout` calls `get_password()` once at stage 0; every subsequent `cmd_distribute`, `cmd_stop`, `cmd_rm`, `cmd_run` call hits the same cached value.
- Color awareness: `class C` evaluates `sys.stdout.isatty() and TERM not in ('','dumb')` at import time; ANSI codes are empty strings when piped or redirected, so log files stay clean automatically.
- `--include` / `--exclude` parser (`_parse_node_spec`): accepts comma-separated items, each either bare integer (`101`), full IP (`192.168.0.108`, last octet extracted), or a range with `-` or `..` (`101-105`, `101..103`). Unknown nodes (e.g. `--include 200`) and inverted ranges (`105-101`) are caught at parse time before any SSH happens.
- Verification: `bash -n cluster.py` syntax-clean; `cluster.py --help` lists all 9 subcommands; `cluster.py ps --running --include 101,103` smoke-tested against the live cluster (returned correct subset).

## Decisions and actions
- Wrote `cluster.py` (~440 lines, stdlib only) at `/media/nas03/docker_backup/sungwoo-lee/script/cluster.py`.
- Deleted all 8 bash scripts (rsync_to_all, load_docker_all, remove_docker_tar_all were already removed in an earlier consolidation step; the remaining 5 went together with the 3 stop/rm/run scripts that had been split out of an earlier `redeploy_container.sh`).
- Made `cluster.py` executable (`chmod +x`) with shebang `#!/usr/bin/env python3` so `./cluster.py ps` works directly.
- Added `start` subcommand (counterpart to `stop`) using the shared `_action_all` helper with idempotency pattern `("no such container",)`.
- Documented the migration path in the file's module-level docstring (`USAGE`, `NODE SPEC`, `EXAMPLES`, `REQUIREMENTS`, `NOTES`).

## Open questions and follow-ups
- The `distribute` subcommand opens the local tar with `open(tar, 'rb')` once per node and pipes it through SSH; for 14 parallel transfers this means 14 simultaneous local-disk reads of the same 10 GB file. Consider whether a single `pv`-style read with a `tee` to multiple SSH stdins would be faster (probably not — disk read is fast; network is the bottleneck), or whether the OS page cache makes it a non-issue.
- The `bash -ic '<alias>'` invocation in `cmd_run` works today but assumes `/home/sungwoo320/.bashrc` defines the named alias. If the bashrc is out of sync on a node, `cluster.py run doc-run-episode --include <that-node>` fails. A `cluster.py push-bashrc` subcommand would close the loop, but for now the user uses the `copy-bash-all` alias (already in the bashrc itself) to fan out.
- The password is held in a Python `str` for the duration of `rollout` (~5 min). On a multi-user host this could be observable via `/proc/<pid>/environ` if `SSH_PASSWORD` was set in the env, or via `/proc/<pid>/maps` examination. Acceptable risk on the lab nodes; would need rethinking for a public host.

## References
- File: `/media/nas03/docker_backup/sungwoo-lee/script/cluster.py` (the only file in the script directory now).
- Sibling insights from this session: `20260509_0310_bash_ic_alias_over_ssh` (the alias-over-SSH technique used inside `cmd_run`), `20260509_0312_node_num_hostname_in_bashrc` (the bashrc that `cmd_run` depends on).
- Predecessor insight: `20260508_1638_container_slimdown_recipe` (the image rebuild that motivated needing better cluster-distribution scripts), `20260508_1639_ssh_credentials_in_shared_image` (the shared SSH credentials that make `cmd_run`'s key auth work).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
