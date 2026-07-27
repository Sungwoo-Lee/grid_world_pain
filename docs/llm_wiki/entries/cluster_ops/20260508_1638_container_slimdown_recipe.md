---
id: 20260508_1638_container_slimdown_recipe
date: 2026-05-08
time: "16:38"
folder: cluster_ops
tags: [meta, decision, learned_lesson]
summary: "Reusable 5-tier recipe that took the running evaaa container from 53.2 GB image footprint to ~11 GB on disk in /home/vncuser. Critical traps: `docker commit` cannot shrink (must flatten via `docker export | docker import`); ncdu sums hardlinks so deleting one path frees no disk; `conda clean --all` is conservative — `conda clean --force-pkgs-dirs` is needed to evict orphan extracted dirs in `pkgs/`; pip-installed packages don't have pkgs/ hardlinks; `pyproject.toml` may declare deps the code never imports (this project listed `torch>=2.9.1` with zero imports anywhere in src/scripts/configs/tests/notebooks)."
related: ["20260508_1637_nas_automount_fstab_actimeo", "20260508_1639_ssh_credentials_in_shared_image"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# Container slim-down recipe — 5-tier cleanup with hardlink and conda gotchas

## Key conclusion
A long-lived development container can shed 30–40 GB by working through five tiers in order: (1) regenerable caches (`pip` cache, `conda` pkgs cache, playwright, /var/log, apt lists); (2) orphan envs and IDE installs (older conda envs, .cursor-server, .antigravity-server, .gemini); (3a) stale ML packages in the active env when imports prove they are unused; (3b) ML packages in the **base** conda env when only one non-base env is in use, plus `conda clean --force-pkgs-dirs` to evict the orphan extracted package directories that `conda clean --all` leaves behind; (3c) IDE remote-server caches (.vscode-server, .cursor-server) since they auto-reinstall on first remote connection. The actual image-size reduction only materializes after `docker export <container> | docker import - <new>:<tag> --change ...` because `docker commit` is append-only and deletions become whiteouts that grow the new layer. Re-supply ENTRYPOINT, CMD, WORKDIR, USER, EXPOSE, ENV via `--change` flags since `docker import` strips image metadata.

## Evidence, measurements, facts
- Starting image: `evaaa:v4` at **53.2 GB** (baked over ~2 months of dev). Container's `/home/vncuser` was at 26.9 GB after the first round of cleanups — final at 8.1 GB after `.vscode-server` removal. miniconda3 went from 23 GB → 7.3 GB.
- Per-tier reclaim measured inside the live container:
  - **Tier 1 (regenerable caches)**: ~21 GB — `~/.cache/pip` (9.9 G), `miniconda3/pkgs` via `conda clean --all` (9.8 G), `~/.cache/ms-playwright*` (757 M), misc `~/.cache/{cloud-code,google-chrome,jedi,tracker3,...}` (~400 M), `/var/log` truncate (873 M), `/var/lib/apt/lists` (73 M).
  - **Tier 2 (orphan envs/IDEs)**: ~7.6 GB — `miniconda3/envs/evaaa` (5.6 G), `.cursor-server` (1.4 G), `.antigravity-server` + `.cache/antigravity` + `.gemini` (~620 M).
  - **Tier 3a (stale env packages)**: ~4 GB — pip-uninstalled `torch torchgen triton captum tensorflow tensorflow-cpu` from grid_world_pain env. Verified zero imports first via `grep -rn --include='*.py' -E '^\s*(import torch|from torch|import tensorflow|...)' src/ scripts/ configs/ tests/`. Also edited `pyproject.toml` to drop the `torch>=2.9.1` line so a future `pip install -e .` does not reinstall it.
  - **Tier 3b (base env + claude versions)**: ~6.7 GB — `conda remove -n base pytorch torchtriton pytorch-cuda pytorch-mutex` (3.5 G expected, but `conda clean --all` left orphan extracted dirs in pkgs/ — required `conda clean --force-pkgs-dirs -y` to actually free; that command's output was the giveaway: "Will remove 1 package cache(s)"); plus deleted `~/.local/share/claude/versions/{2.1.129,2.1.131,2.1.132}` keeping only the symlinked-active `2.1.133` (~700 M).
  - **Tier 3c**: 2.4 GB — `rm -rf ~/.vscode-server` (auto-reinstalls on next VS Code remote connect, so safe).
- **Hardlink trap (confirmed)**: ncdu's JSON export summed hardlinks per visited path, so `pkgs/pytorch-2.5.1-...` (2.8 G) and base env's `lib/python3.12/site-packages/torch` (2.8 G) both counted, suggesting 5.6 G when actually 2.8 G of disk. Deleting only `pkgs/pytorch-...` would have freed nothing because the env-side link kept the inode alive. Real reclaim required removing from BOTH locations.
- **`conda clean --all` is conservative**: after `conda remove -n base pytorch ...`, `conda clean --all` reported "There are no unused tarball(s) to remove" yet pytorch's extracted dir was still in `pkgs/`. `conda clean --force-pkgs-dirs -y` (separate flag, not in `--all`) finally evicted it. The miniconda3 footprint dropped from 14 G → 7.3 G after this single step.
- **Stale `pyproject.toml` dependency**: `dependencies = [..., "torch>=2.9.1", ...]` was present despite `grep -rn '^\s*(import torch|from torch)' src/ scripts/ configs/ tests/` returning zero hits and 23 files importing JAX/Flax. `pip show torch` confirmed `Required-by: captum, grid_world_pain` — both unused. Removed the line, uninstalled both packages, environment intact (`jax 0.9.0.1, devices=2` post-cleanup).
- **`docker export | docker import` flatten** is the correct mechanism (not `docker commit`). `--change` flags must re-supply: `ENTRYPOINT ["/sbin/init"]`, `CMD ["/bin/zsh"]`, `WORKDIR /home/vncuser`, `USER root`, `EXPOSE 22 5901-5910 6080`, `ENV LANG=en_US.UTF-8`. `docker export` excludes tmpfs and bind mounts (`--tmpfs /tmp /run /run/lock`, `-v /sys/fs/cgroup`, `-v /tmp/.X11-unix`), so transient `/tmp` artifacts like cleanup scripts and ssh ControlMaster sockets do not propagate.
- **Audit-before-delete protocol used**: ncdu JSON export (18 MB, 240k lines) parsed by a small Python walker that ranked paths ≥50 MB. Top entries surfaced the hardlink-doubled pytorch and the orphan extracted dirs that `conda clean --all` failed to remove.

## Decisions and actions
- Cleaned the running `evaaa` container in five tiers (above).
- Edited `pyproject.toml`: dropped the stale `torch>=2.9.1` line.
- Renamed image lineage at the slim-down moment: `evaaa:v3/v4` → `episode:v1` (project name match; clean version reset). Container `--name`, `--hostname`, alias name (`doc-run-episode`) similarly renamed for consistency. Added `--restart=unless-stopped` to alias for host-reboot survival.
- Documented the host-side flatten command sequence with `--change` flags re-supplying lost metadata. Sequence: `docker export evaaa | docker import - episode:v1 --change ...`, smoke-test as `episode-v1-test` in a separate container before promoting, then push to private registry and roll per-node.
- Per-node prereqs noted: `sudo systemctl enable docker`, `echo cifs | sudo tee /etc/modules-load.d/cifs.conf`, `sudo modprobe cifs`. Without the cifs module loaded on the host, in-container fstab silently fails to mount.

## Open questions and follow-ups
- Verify episode:v1 is actually ~10–14 GB after flatten (vs. 53.2 GB for v4). If significantly larger, investigate which tmpfs/bind-mount paths leaked into the export tar.
- Decide whether to write a `Dockerfile.episode` that recreates v1 from a clean base + the captured artifacts (`/etc/fstab`, `/etc/smb-credentials`, conda env pyproject, `~/.zshrc` deltas). The flattened image is faster to ship today; a Dockerfile is more reproducible for the next major rebuild and can use multi-stage to keep the SMB password out of intermediate layers.
- Re-evaluate whether base env still has cleanup headroom after the v1 baseline stabilizes (e.g., `mkl` 532 MB stays because some base packages depend on it — could probably remove some of those base packages too if truly unused).

## References
- Sibling insights: `20260508_1637_nas_automount_fstab_actimeo` (NAS auto-mount baked in alongside the slim-down in the same v1 image), `20260508_1639_ssh_credentials_in_shared_image` (SSH-cred propagation considered when rolling the shared image to 14 nodes).
- Predecessor insights: `20260508_1428_node_env_recovery_recipe` (peer-version-matched env recovery; sibling-cluster-ops insight that informed why grid_world_pain env is the only one to keep), `20260508_1433_cifs_bypass_for_run_command`.
- File touched: `pyproject.toml` (line: `"torch>=2.9.1"` removed).
- Conda safe-uninstall sequence used: `pip uninstall -y torch torchgen triton captum tensorflow tensorflow-cpu` (env-side); `conda remove -n base pytorch torchtriton pytorch-cuda pytorch-mutex -y` (base-env side); `conda clean --all -y` then `conda clean --force-pkgs-dirs -y`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
