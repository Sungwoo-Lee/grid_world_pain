---
id: 20260508_1717_ssh_config_match_user_scoping
date: 2026-05-08
time: "17:17"
folder: cluster_ops
tags: [meta, decision, learned_lesson, training_runner]
summary: "OpenSSH `Host` directives match by destination only — not by user. The lab SSH config written by `bootstrap_lab_ssh.sh` (commit d7b4f1f) silently rewrote `ssh sungwoo320@192.168.0.10X` from port 22 to port 1800, hitting the docker container's sshd (which has no sungwoo320 account) and producing a 'Permission denied' that masquerades as a wrong-password error. Fix: replace the global `Host` block with `Match user vncuser host 192.168.0.10?,192.168.0.11?` so only vncuser routes to 1800. All four propagation surfaces (`~/.ssh/config`, `scripts/bootstrap_lab_ssh.sh`, `.claude/agents/training-runner.md` lines 38 + 49) updated together. Audit confirmed every SSH-using script (`run_command.py`, `terminate_command.py`, `bootstrap_lab_ssh.sh`) passes `-p 1800` explicitly, so the config block is convenience-only — agents are unaffected by the scoping."
related: ["20260508_1638_container_slimdown_recipe", "20260508_1639_ssh_credentials_in_shared_image", "20260508_1428_node_env_recovery_recipe", "20260508_1434_terminate_command_key_auth_refactor"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# Scope the lab SSH config block to `vncuser` via `Match user`

## Key conclusion
The convenience block that `bootstrap_lab_ssh.sh` writes to `~/.ssh/config` (`Host 192.168.0.10? 192.168.0.11?` → `Port 1800`, `User vncuser`, `IdentityFile ~/.ssh/id_ed25519_gridworld`) is **destination-scoped, not user-scoped**. Every SSH to a lab IP — including `ssh sungwoo320@192.168.0.10X` for host-level admin — gets the port silently rewritten to 1800, lands on the docker container's sshd where `sungwoo320` does not exist, and fails authentication. The fix is `Match user vncuser host 192.168.0.10?,192.168.0.11?` — same effect for the vncuser→container path that `run_command.py` and ad-hoc workflows depend on, but `sungwoo320` and other usernames fall through to default port 22 against the host's sshd. The fix is safe because every SSH-using automation path in the project passes `-p 1800` explicitly; the `~/.ssh/config` block is convenience for human typing only.

## Evidence, measurements, facts
- Reproduction symptom: from inside the running evaaa container on node 102, `ssh sungwoo320@192.168.0.102` (with the correct password) returned "Permission denied, please try again." The same command from the user's MacBook on the lab network worked. `ssh -vvv` revealed the cause in two lines: `debug1: Reading configuration data /home/vncuser/.ssh/config` and `debug1: Connecting to 192.168.0.102 [192.168.0.102] port 1800.` — the port was being rewritten before the connection was made.
- The Match block in OpenSSH semantics: `Match user X host A,B` (no spaces in the host list, lowercase `user`/`host` keywords) applies only when the resolved login user matches and the destination matches. The original `Host A B` block applies for any user. With the `Match user vncuser` form, `ssh -G vncuser@192.168.0.103` returned `port 1800` + the gridworld key; `ssh -G sungwoo320@192.168.0.102` returned `port 22` + the default `~/.ssh/id_rsa` chain. Both verified via dry-run before the user re-tested.
- Source-of-truth files identified in audit:
  - `scripts/bootstrap_lab_ssh.sh` lines 76-92 — the `cat >> "$SSH_CONFIG"` heredoc that writes the Host block on first run. Idempotency check at line 79 (`grep -qF "$CONFIG_SENTINEL"`) gates re-writes by looking for the sentinel comment `# lab-nodes 101-114 (managed by bootstrap_lab_ssh.sh)`, NOT the actual block content. This is what allowed a duplicate block (a hand-written one without the sentinel + the script-written one with the sentinel) to coexist in the user's config.
  - `run_command.py` line 40 (`SSH_PORT = 1800`) and line 76 (`-p str(SSH_PORT)`) — explicit port. Not affected by config rewrite.
  - `terminate_command.py` line 39 (`SSH_PORT = 1800`) and line 98 (`-p str(SSH_PORT)`) — explicit port. Not affected.
  - `sync-agent-data.sh` — uses rsync against the local CIFS-mounted NAS only, no inter-node SSH, not affected.
  - `.claude/agents/training-runner.md` line 38 — doc claim about how `~/.ssh/config` makes plain `ssh vncuser@…` work; updated to mention the new `Match user` form.
  - `.claude/agents/training-runner.md` line 49 — precondition check `grep -q '^Host 192.168.0.10? 192.168.0.11?' ~/.ssh/config`; updated to grep for the new `Match` line.
- Git provenance:
  - Commit `d7b4f1f` (Sungwoo-Lee, 2026-05-07): "feat(agent): ✨ add training-runner agent and lab SSH bootstrap" introduced both `bootstrap_lab_ssh.sh` and the global `Host` block pattern.
  - Commit `bc09dd8` (Sungwoo-Lee, 2026-03-06): "feat: Add SSH multiplexing options to run_command.py" — the run_command.py path was always explicit-port; the `~/.ssh/config` block is the convenience layer added later.
- Agent-impact audit: every `Agent` profile in `.claude/agents/` was scanned. The only profile that touches SSH at all is `training-runner.md`. It uses `run_command.py` (explicit port) and the precondition grep (we updated). Other profiles (`developer`, `senior-developer`, `experiment-analyzer`, `experiment-designer`, `code-reviewer`, `math-reviewer`, `env-config-auditor`, `agent-manager`, all Researchers) do not invoke SSH. **Zero functional impact on any agent.**
- Final precondition check verified end-to-end: `test -f ~/.ssh/id_ed25519_gridworld && grep -q '^Match user vncuser host 192.168.0.10?,192.168.0.11?' ~/.ssh/config && ssh -o BatchMode=yes -o ConnectTimeout=5 vncuser@192.168.0.101 'true' && echo OK` returned `OK`.

## Decisions and actions
- Edited `/home/vncuser/.ssh/config` to replace both duplicate `Host` blocks with one `Match user vncuser host 192.168.0.10?,192.168.0.11?` block. Sentinel comment retained so `bootstrap_lab_ssh.sh`'s idempotency guard still recognizes the block on future runs.
- Edited `scripts/bootstrap_lab_ssh.sh` lines 82-90 (the `cat >> "$SSH_CONFIG"` heredoc) to emit the new `Match user` form on future runs. Sentinel preserved.
- Edited `.claude/agents/training-runner.md` line 38 (doc claim) to describe the `Match user` scoping and explicitly note that `sungwoo320` falls through to port 22.
- Edited `.claude/agents/training-runner.md` line 49 (precondition check `grep` pattern) to match the new `Match user vncuser host …` line so the agent's precondition still says `OK` after the change.
- Did NOT edit `~/.ssh/config` to remove the rewrite entirely — chose `Match user` over removal so the daily ergonomic of bare `ssh vncuser@…` and `ssh 192.168.0.103` (defaulting to local user vncuser) is preserved.
- Did NOT edit `.claude-memory/memories/cluster_ops/20260508_1428_node_env_recovery_recipe.md` — the bare-ssh examples there implicitly use vncuser (the local user inside the container), which still routes correctly via the new `Match` block. No edit needed.
- All four edits live in either the docker writable layer (`/home/vncuser/.ssh/config`) or the NAS-mounted repo (`scripts/`, `.claude/agents/`), so they propagate via the next `docker export | docker import` to `episode:v1`.

## Open questions and follow-ups
- After `episode:v1` rolls to all 14 nodes, verify the precondition check returns `OK` on a node other than 102 (where the fix was authored). Reason to expect it works: the `~/.ssh/config` is in the writable layer captured by `docker export`, and the script-side change is repo-side. But a sanity check on first rollout target is worth doing.
- Decide whether `bootstrap_lab_ssh.sh` should clean up old non-sentinel blocks during a re-run (current behavior leaves them in place — only blocks new sentinel-bearing additions). Low priority because the new block now subsumes any old block's effect via the more-specific `Match user` rule, and SSH applies all matching directives in order.
- If the lab ever needs sungwoo320 to also have ergonomic SSH between hosts (without typing `-p 22` explicitly), add a separate `Match user sungwoo320 host …` block. Not needed today because `sungwoo320`'s default fall-through (port 22 + default key chain) already does the right thing.

## References
- Commit history: `d7b4f1f` (training-runner agent + lab SSH bootstrap), `bc09dd8` (run_command.py SSH multiplexing).
- Source files updated: `/home/vncuser/.ssh/config`, `scripts/bootstrap_lab_ssh.sh` (lines 76-92 region), `.claude/agents/training-runner.md` (lines 34-52 region).
- Sibling insights from this session: `20260508_1638_container_slimdown_recipe` (image flatten that propagates `~/.ssh/config`), `20260508_1639_ssh_credentials_in_shared_image` (the broader SSH-credential propagation pattern that this scoping decision lives within).
- Predecessor insight: `20260508_1428_node_env_recovery_recipe` (uses bare `ssh 192.168.0.<node>` for vncuser; remains correct under the new `Match user` block — examples not affected).
- Related insight: `20260508_1434_terminate_command_key_auth_refactor` (terminate_command.py mirrors run_command.py's explicit-port pattern; same audit conclusion applies).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
