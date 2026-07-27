---
id: 20260508_1637_nas_automount_fstab_actimeo
date: 2026-05-08
time: "16:37"
folder: cluster_ops
tags: [meta, decision, learned_lesson, training_runner]
summary: "Bake CIFS auto-mount with `actimeo=1,_netdev,nofail` into the container image via `/etc/fstab` + `/etc/smb-credentials` (mode 0600). systemd-in-docker generates `.mount` units from fstab automatically; the runner's `/tmp/<unique>.sh` bypass stays as belt-and-suspenders since `actimeo=1` shrinks the staleness window from ~60 s to ~1 s but does not fully close it."
related: ["20260508_1433_cifs_bypass_for_run_command", "20260508_1446_cifs_race_node112_recurrence", "20260508_1638_container_slimdown_recipe"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# CIFS auto-mount via fstab in systemd-in-docker container

## Key conclusion
The CIFS attribute-cache race that recurred across multiple Claude sessions on multiple lab nodes is fixed at the mount layer with `actimeo=1` (default cache TTL is ~60 s). For an `--entrypoint=/sbin/init` container, `/etc/fstab` is the right mechanism because systemd auto-generates `.mount` units from it — no custom service or script needed. Credentials go in `/etc/smb-credentials` (mode 0600 root:root) inside the image, so the auto-mount is non-interactive at boot. `_netdev` orders the mount after `network-online.target`, and `nofail` prevents a NAS hiccup at boot from dropping the container into emergency mode.

## Evidence, measurements, facts
- Trigger: `docs/diary/2026-05-08.md` showed the CIFS staleness pattern (`20260508_1433_cifs_bypass_for_run_command`, `20260508_1446_cifs_race_node112_recurrence`) appearing across 3+ Claude sessions on different nodes (102, 112, 114). The /tmp bypass alone did not prevent recurrence — it required runner-side discipline that was not always applied.
- Mount options used: `credentials=/etc/smb-credentials,uid=1000,gid=1000,rw,rsize=130048,wsize=130048,vers=3.0,actimeo=1,_netdev,nofail 0 0`. Same options, three lines (cocoanlab01/02/03 → /media/nas01/02/03).
- vncuser uid=1000, gid=1000 (confirmed via `id vncuser`), so fstab can hardcode rather than evaluating `id -u` at runtime.
- After running `mount -a` inside the live container, `mount | grep nas0[23]` confirmed both NAS02 and NAS03 came up with `actimeo=1,_netdev` in the kernel options. NAS01 was deliberately not touched (the running shell's CWD lives there) — it stays on the manual-alias mount and will pick up the fstab entry on next container boot.
- Safety pattern when testing in-container changes from a session whose CWD lives on a NAS-mounted path: test the mechanism on a non-CWD share first (`mount /media/nas02 && mount /media/nas03`, NOT `mount -a`) to avoid yanking the CWD if the fstab line is malformed. Recover-on-failure: run `mount -a` only after the per-share probes succeed.
- Trade-off retained: the runner's `/tmp/train_cmd_<unique>.sh` bypass (`20260508_1433_cifs_bypass_for_run_command`) and post-launch `pgrep` check stay in place. Reason: `actimeo` controls **attribute** freshness (mtime, size); page-cache (content) coherency under `cache=strict` is governed by SMB oplocks and can still race under concurrent writes. `actimeo=1` makes the bug rare, the bypass eliminates it.

## Decisions and actions
- Created `/etc/smb-credentials` (mode 0600, root:root) in the running container with the SMB password.
- Appended three CIFS lines to `/etc/fstab`. NAS01 entry added but not activated by the test script (CWD safety); takes effect on next container boot.
- Updated `~/.zshrc` aliases `mnt_nas_0{1,2,3}` to include `actimeo=1` for parity with the fstab auto-mount.
- Container `doc-run-evaaa` flags unchanged: `--privileged` already covers all caps CIFS needs (SYS_ADMIN, DAC_READ_SEARCH, AppArmor), so no docker-run changes required.
- Per-host prereq for nodes 101–114: `cifs` kernel module loaded and made sticky via `/etc/modules-load.d/cifs.conf`. Without this, the in-container fstab fires but `mount.cifs` silently fails.
- Same fstab + credentials baked into the new `episode:v1` image (renamed from `evaaa`) via `docker export | docker import` — see sibling insight on container slim-down.

## Open questions and follow-ups
- After episode:v1 rolls to all 14 nodes, monitor whether the runner's CIFS-bypass discipline can be relaxed (post-launch `pgrep` likely still needed for >1-PID detection). Decision should be data-driven: count CIFS-staleness incidents in the diary over a 30-day window post-rollout.
- Consider adding a `findmnt --verify-fstab` smoke test to a future container-build pipeline so a malformed fstab line is caught before image commit.

## References
- Sibling insight: `20260508_1638_container_slimdown_recipe` (this fstab change shipped together with the slim-down in the episode:v1 image).
- Predecessor insights: `20260508_1433_cifs_bypass_for_run_command` (the /tmp bypass — still load-bearing), `20260508_1446_cifs_race_node112_recurrence` (cross-node confirmation that motivated this fix).
- Built-in MEMORY.md entry: `feedback_runner_cifs_bypass.md` (one-liner, harness-managed; rationale chain lives in the cifs_bypass insight, not here).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
