---
id: 20260509_0310_bash_ic_alias_over_ssh
date: 2026-05-09
time: "03:10"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner]
summary: "To invoke a bash alias defined in a remote `~/.bashrc` over SSH, you must use `bash -ic '<alias>'`, not plain `ssh host '<alias>'` or even `ssh host 'source ~/.bashrc; <alias>'`. Reason: non-interactive shells skip alias expansion entirely (and the standard Ubuntu `~/.bashrc` returns early in non-interactive mode via `case $- in *i*) ;; *) return ;; esac`). The harmless 'no job control in this shell' notice can be filtered with `grep -v`."
related: ["20260509_0309_cluster_py_consolidation", "20260509_0312_node_num_hostname_in_bashrc"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# `bash -ic '<alias>'` to expand a remote `.bashrc` alias over SSH

## Key conclusion
A bash alias defined in `~/.bashrc` is only registered when the shell is interactive. SSH commands of the form `ssh host '<alias>'` start a non-interactive remote shell that (a) doesn't source `~/.bashrc` by default, and (b) even if you `source ~/.bashrc` explicitly, the standard Ubuntu bashrc returns early via `case $- in *i*) ;; *) return ;; esac` so aliases are never registered. The fix is `ssh host "bash -ic '<alias>'"` — `-i` forces interactive mode (sources bashrc, registers aliases), `-c '<alias>'` runs the alias as the command. Some bash builds emit a one-line "no job control in this shell" notice on stderr because there's no controlling TTY; harmless, filter with `grep -v 'no job control'`.

## Evidence, measurements, facts
- Tested `ssh sungwoo320@192.168.0.102 'doc-run-episode'` from inside the docker-102 container with bashrc properly distributed: command failed with `bash: doc-run-episode: command not found`.
- Tested `ssh sungwoo320@192.168.0.102 'source ~/.bashrc && doc-run-episode'`: same failure. The `source ~/.bashrc` ran but the `case $- in *i*` early return at line 6-9 of the standard bashrc skipped everything below it (including the alias definitions).
- Tested `ssh sungwoo320@192.168.0.102 "bash -ic 'doc-run-episode'"`: alias resolved correctly, container started, returned the docker container ID. The `${NODE_NUM}` variable inside the alias body resolved at use-time against the host's environment (set near the top of the bashrc, before the early-return check).
- 'no job control' notice: emitted by some bash builds when `bash -i` runs without a controlling terminal. Harmless. The script filters it with `2>&1 | grep -v 'no job control'` so the docker container ID is the last meaningful line of output.
- Used in `cluster.py run <alias>` (subcommand `cmd_run`) at `/media/nas03/docker_backup/sungwoo-lee/script/cluster.py`: builds the remote command as `bash -ic {shlex.quote(alias)} 2>&1 | grep -v 'no job control'` and parses the trailing 12-char hex token from stdout as the container ID.
- The same technique works for any user-defined function in `~/.bashrc`, `~/.bash_aliases`, or anywhere sourced inside the bashrc's interactive-only block.

## Decisions and actions
- Adopted `bash -ic '<alias>'` as the canonical pattern in `cluster.py run` and any future automation that needs to invoke shared shell aliases on cluster nodes.
- Documented in `cluster.py` source comments: "non-interactive SSH does not source .bashrc and aliases aren't expanded; `-ic` forces interactive mode so the alias is recognized".
- Rejected the alternative of inlining the full `docker run …` command in the script: the alias body is the source of truth (kept in `~/.bashrc` so the user can edit it once and have it propagate via `copy-bash-all`); duplicating it in the script would create a sync hazard.

## Open questions and follow-ups
- For aliases that produce noisy output (multi-line warnings, progress bars, etc.) the simple `grep -v 'no job control'` filter may strip too much or not enough. So far only the one notice has been observed; revisit if a future alias starts shadowing useful output.
- This pattern has a dependency on the user's shell being bash. If a node's sungwoo320 default shell were ever changed to zsh, `bash -ic` would still work (we explicitly invoke bash) but the bashrc itself would need to coexist with zshrc, which it does today.

## References
- File: `/media/nas03/docker_backup/sungwoo-lee/script/cluster.py` (`cmd_run` function — uses the pattern).
- Standard Ubuntu bashrc: see `/etc/skel/.bashrc` lines 5-9 for the `case $- in *i*) ;; *) return ;; esac` early return.
- Sibling insights: `20260509_0309_cluster_py_consolidation` (the cluster.py tool that uses this technique), `20260509_0312_node_num_hostname_in_bashrc` (the bashrc whose aliases this technique invokes).
- Bash manpage: `man bash` → `INVOCATION` section discusses interactive vs. non-interactive shells and which startup files are sourced.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
