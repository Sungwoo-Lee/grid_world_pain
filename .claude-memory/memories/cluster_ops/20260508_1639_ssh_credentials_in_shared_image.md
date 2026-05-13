---
id: 20260508_1639_ssh_credentials_in_shared_image
date: 2026-05-08
time: "16:39"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner]
summary: "`docker export` snapshots the writable filesystem including `/etc/ssh/ssh_host_*` and `~vncuser/.ssh/{id_rsa,authorized_keys,known_hosts}`. When the same flattened image is rolled to all 14 lab nodes, this is what enables `run_command.py`'s passwordless SSH between containers (shared id_rsa + authorized_keys + pre-trusted known_hosts), but it also collapses host keys to a single shared identity and creates a host-key-reconciliation gotcha if the prior image regenerated host keys per-container. `/tmp/ssh_mux_*` ControlMaster sockets stay on tmpfs and do not propagate."
related: ["20260508_1637_nas_automount_fstab_actimeo", "20260508_1638_container_slimdown_recipe", "20260508_1434_terminate_command_key_auth_refactor"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# SSH credential propagation when rolling a shared Docker image to 14 nodes

## Key conclusion
`docker export <container> | docker import - <new>:<tag>` snapshots the entire writable filesystem, which includes the SSH server's host keys at `/etc/ssh/ssh_host_*_key` and vncuser's client-side identity at `~/.ssh/`. When the same flattened image (e.g. `episode:v1`) is rolled to nodes 101–114, all 14 containers will present **identical** host keys and carry **identical** vncuser private keys, public keys, and `authorized_keys`. This is the mechanism that lets `run_command.py` passwordless-SSH between any pair of containers without per-node key dances. The risk is host-key reconciliation: if the prior image regenerated host keys on first boot per-container, then the v5/v1 export collapses them all to one identity and existing entries in everyone's `~/.ssh/known_hosts` will fail-closed with `REMOTE HOST IDENTIFICATION HAS CHANGED!`. The mitigation, when needed, is one `ssh-keygen -R <ip>:<port>` line per node in the rollout script. tmpfs paths (`/tmp/ssh_mux_*` ControlMaster sockets, `/tmp/cleanup_*.sh`) do not propagate — `docker export` excludes tmpfs and bind mounts.

## Evidence, measurements, facts
- `run_command.py` (line 67-79) uses ControlMaster + ControlPath multiplexing at `/tmp/ssh_mux_%h_%p_%r` with `ControlPersist=600`. SSH connects to `192.168.0.<node>:1800` (host port → docker container port 22 via `-p 1800:22` in `doc-run-evaaa`). Authentication is as `vncuser` via SSH key (no password — confirmed by the runner profile and the `terminate_command.py` refactor on 2026-05-08 that mirrored the same key-auth path: `20260508_1434_terminate_command_key_auth_refactor`).
- Files in writable layer that `docker export` includes:
  - `/etc/ssh/ssh_host_rsa_key` and `/etc/ssh/ssh_host_*_key.pub` — server identity. **Shared after rollout.**
  - `~vncuser/.ssh/id_rsa`, `id_rsa.pub` — client identity. **Shared after rollout** (this is what enables run_command.py).
  - `~vncuser/.ssh/authorized_keys` — who can SSH in. **Shared after rollout** (correct: all containers should trust the same vncuser key).
  - `~vncuser/.ssh/known_hosts` — pre-trusted hosts. **Shared after rollout**, pre-loaded with all 14 nodes' fingerprints from prior connections. Pro: no first-connection prompts. Con: see host-key-reconciliation below.
- Files NOT in writable layer (tmpfs / bind mount):
  - `/tmp/ssh_mux_*` (tmpfs `--tmpfs /tmp`) — fresh on every container start. ✓
  - `/tmp/.X11-unix` (bind `-v /tmp/.X11-unix:/tmp/.X11-unix:rw`) — host-side socket dir, not in image. ✓
  - `/sys/fs/cgroup` (bind `-v /sys/fs/cgroup:/sys/fs/cgroup:rw`) — host kernel state. ✓
- Host-key-reconciliation case analysis:
  - **If `evaaa:v3` already shared host keys across 14 containers** (most likely — that's why the existing setup works without manual fingerprint dances): `episode:v1` will continue presenting the same single host key, and existing `known_hosts` entries on each node still match. **No action needed.**
  - **If `evaaa:v3` regenerated host keys on first boot** (less likely given the working setup): each node's current container has a unique key. `episode:v1` will collapse them all to whatever this container has now. After rollout, every node's `known_hosts` will fail with "host key changed". Fix: `ssh-keygen -R 192.168.0.<n>:1800` for n in 101..114, or the equivalent loop in the rollout script.
- Diagnostic probe (one-minute test before flatten): from this container, compare `cat /etc/ssh/ssh_host_rsa_key.pub` with the entry in `~/.ssh/known_hosts` for any peer node (e.g., `ssh-keygen -F '[192.168.0.103]:1800' -f ~/.ssh/known_hosts`). If they match → all containers were already sharing keys → safe. If they differ → need the post-rollout `ssh-keygen -R` cleanup.

## Decisions and actions
- Documented in the session that the rollout sequence's smoke-test step should include both `docker exec episode mount | grep nas0[123]` (NAS auto-mount) and an SSH peer-reachability probe (`ssh -o BatchMode=yes vncuser@192.168.0.<peer>:1800 'echo ok'`).
- Pending: run the host-key-comparison probe inside the live container before flatten, to determine which case applies and whether the rollout script needs a `ssh-keygen -R` step.
- No security action taken in-session — the shared-key model is acceptable for the closed lab network. Documented for future awareness only.

## Open questions and follow-ups
- Run the diagnostic probe before flattening `episode:v1`. Decide based on result whether to add `ssh-keygen -R` to the rollout script.
- Consider regenerating host keys on first boot via a systemd unit (`/etc/systemd/system/sshd-keygen-once.service`) if the lab ever moves to a less-trusted network. Trade-off: per-container unique host keys break run_command.py's pre-trusted `known_hosts`, requiring a host-key collection step at provisioning.
- For the next major image rebuild, consider injecting SSH keys via Docker secrets or runtime env vars rather than baking them in. Same trade-off discussed for SMB credentials in `20260508_1638_container_slimdown_recipe` — convenient bake-in vs. registry-leak risk.

## References
- `run_command.py` lines 67-79 (ControlMaster path + ssh_opts).
- Sibling insights: `20260508_1637_nas_automount_fstab_actimeo` (CIFS auto-mount baked alongside SSH creds), `20260508_1638_container_slimdown_recipe` (the flatten step that propagates these credentials), `20260508_1434_terminate_command_key_auth_refactor` (terminate_command.py mirrors run_command.py's key-auth path).
- Built-in MEMORY.md: `feedback_runner_post_launch_pgrep.md`, `feedback_runner_cifs_bypass.md` (one-line operational rules for the runner; this insight is the rationale chain behind the multi-node SSH model that makes those rules work).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
