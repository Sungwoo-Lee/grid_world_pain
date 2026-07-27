---
id: 20260509_0312_node_num_hostname_in_bashrc
date: 2026-05-09
time: "03:12"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner]
summary: "The host's `/home/sungwoo320/.bashrc` derives `NODE_NUM` from the primary lab IP's last octet (`hostname -I | grep '^192\\.168\\.0\\.' | awk -F. '{print $4}'`) and every `doc-run-*` alias passes `--hostname=\"docker-${NODE_NUM}\"`. After rollout, the in-container prompt reads `vncuser@docker-102` on node 102 and `vncuser@docker-114` on node 114 — instant disambiguation when shelling between cluster containers. Variable expands at alias-use time (not definition time) because aliases are macro-substituted before re-parsing."
related: ["20260508_1639_ssh_credentials_in_shared_image", "20260509_0309_cluster_py_consolidation", "20260509_0310_bash_ic_alias_over_ssh"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# `NODE_NUM`-suffixed container hostname in shared bashrc

## Key conclusion
Before this change, every container on every lab node had hostname `docker` (set by `--hostname='docker'` in the bashrc aliases). When SSH'd into multiple containers in parallel, the prompt read `vncuser@docker` on every node — visually indistinguishable. The fix: add a single line near the top of the shared `/home/sungwoo320/.bashrc` that derives the node number from the host's primary lab IP, then reference it inside every `doc-run-*` alias body. The alias expands `${NODE_NUM}` at use time (not definition time), so the variable resolves against each node's environment when the alias is invoked. After this change rolls out via `copy-bash-all`, container prompts read `vncuser@docker-101` through `vncuser@docker-114` — instant disambiguation. Falls back to `XX` if the host has no `192.168.0.0/24` IP (e.g. a laptop on a different network).

## Evidence, measurements, facts
- Derivation line, near the top of `/media/nas03/docker_backup/sungwoo-lee/bashrc_260508`:
  ```bash
  NODE_NUM=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^192\.168\.0\.' | awk -F. '{print $4}')
  NODE_NUM=${NODE_NUM:-XX}
  ```
- The `tr ' ' '\n' | grep -m1 '^192\.168\.0\.'` filter avoids picking up docker bridge IPs (`172.17.0.x`) or other interfaces; on a multi-homed host it still resolves to the lab IP.
- `${NODE_NUM:-XX}` fallback ensures `--hostname=docker-XX` rather than `--hostname=docker-` (Docker would actually reject the latter as malformed).
- Alias expansion semantics that make this work: bash aliases are macro-substituted into the command line BEFORE the shell re-parses. So `alias foo='echo "host-${NODE_NUM}"'` substitutes to the literal string `echo "host-${NODE_NUM}"` at use time, then the shell parses THAT line and expands `${NODE_NUM}` against the current environment. The alternative — using a function — works too but doesn't compose with the existing single-line alias style of the bashrc.
- Single-quoted vs double-quoted alias body: the entire alias is in single quotes (`alias foo='...'`), and `"${NODE_NUM}"` (double-quoted) inside is treated as part of the literal alias body. That's correct — single quotes preserve the `${NODE_NUM}` syntax for runtime expansion; double quotes inside ensure the variable expansion respects whitespace if NODE_NUM ever contained any (it never will, but defensive).
- Verified before distribution: `ssh -G vncuser@192.168.0.103` returned the right port + key for the `Match user` SSH-config block; the actual hostname change is observable only after the new `episode:v1` containers are launched on each node (the running `evaaa:v3` containers still have the old `docker` hostname).
- Three places where `--hostname="docker-${NODE_NUM}"` now appears in the bashrc: `doc-run-evaaa`, `doc-run-sungwoo`, `doc-run-episode`. All three project containers get node-suffixed hostnames going forward.

## Decisions and actions
- Edited `/media/nas03/docker_backup/sungwoo-lee/bashrc_260508`:
  - Removed all `iai`-related aliases (the project is dead).
  - Cleaned the broken multi-line `doc-run-evaaa` (was split across two lines with an embedded newline; collapsed back to one line).
  - Added `NODE_NUM` derivation block.
  - Replaced `--hostname='docker'` (with fragile single-quote nesting) with `--hostname="docker-${NODE_NUM}"` in all three `doc-run-*` aliases.
  - Added `--restart=unless-stopped` to `doc-run-sungwoo` for parity with the other two.
  - Added `doc-run-episode` and `doc-exec-episode` aliases.
  - Fixed the duplicate `copy-vim-node12` typo (was repeated 3× instead of having `node13` and `node14`).
  - Added `copy-bash-all` and `copy-vim-all` fan-out aliases using `for ip in 192.168.0.{101..114}; do scp ...; done`.
- Distribution path: this bashrc is at `/media/nas03/...` (NAS, not yet on any node's host filesystem). After the user runs `copy-bash-all` from a node that has the new bashrc as `~/.bashrc`, all 14 lab nodes get the update.

## Open questions and follow-ups
- The `NODE_NUM` resolution runs every time bash starts (login + interactive), which involves a `hostname -I` call + pipe. On a slow host the cost is ~1 ms — negligible. If it ever becomes annoying, cache the value in `/etc/profile.d/node-num.sh` once at boot.
- For containers run from a host that's NOT on the lab subnet (e.g. a developer's laptop), `NODE_NUM=XX` and the container hostname becomes `docker-XX`. That's intentionally ugly so the user notices they're not on the lab.
- For multi-container hosts (e.g. a node running both evaaa and sungwoo), both containers now share the same hostname suffix (`docker-102` for evaaa AND `docker-102` for sungwoo). Differentiation falls back to `--name` (which is unique). Probably fine; revisit if it confuses anyone.

## References
- File: `/media/nas03/docker_backup/sungwoo-lee/bashrc_260508` (the canonical bashrc, distributed via `copy-bash-all`).
- Sibling insight: `20260509_0310_bash_ic_alias_over_ssh` (how `cluster.py run` invokes these aliases over SSH; depends on this bashrc being present on each node).
- Sibling insight: `20260509_0309_cluster_py_consolidation` (the script that calls these aliases).
- Predecessor insight: `20260508_1639_ssh_credentials_in_shared_image` (the shared docker-image story this bashrc rolls out alongside).
- Bash manpage on aliases: `man bash` → `ALIASES` section: "The first word of each simple command, if unquoted, is checked to see if it has an alias … the rules concerning the definition and use of aliases are somewhat confusing".
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
