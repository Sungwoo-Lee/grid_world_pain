# _topic_index.md — `cluster_ops` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `cluster_ops` topic.

**Folder definition**: Lab cluster ops and env mgmt
**Insights**: 4
**Last updated**: 2026-05-08

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-08 | 14:46 | `20260508_1446_cifs_race_node112_recurrence` | CIFS attribute-cache race recurred on node 112 (cross-node confirmation), and the runner did NOT auto-apply the canonical /tmp/<unique>.sh bypass on first launch of either round — duplicates `gh1cz01q` (Round 1) and `f5d933ro` (Round 2 Cell C) caught only by post-launch pgrep. Bypass must be applied upfront. |
| 2026-05-08 | 14:34 | `20260508_1434_terminate_command_key_auth_refactor` | Refactored terminate_command.py from pexpect+getpass to subprocess+SSH-key-auth (mirrors run_command.py). Added `--yes` flag for non-interactive use; kept two-stage scan/confirm default. |
| 2026-05-08 | 14:33 | `20260508_1433_cifs_bypass_for_run_command` | Node 114 CIFS client caches `train_command-agent.sh` inode content; fresh NAS edits invisible to remote bash → silent duplicate launches. Workaround: write same content to `/tmp/train_cmd_<unique>.sh` on the node and bash that path. |
| 2026-05-08 | 14:28 | `20260508_1428_node_env_recovery_recipe` | Lab node 101 had a 3-layer broken conda env (missing JAX, then flax/optax/orbax/chex, then ptxas/cuda-nvcc) plus a JAX 0.10 vs 0.9.0.1 version mismatch with peers; the recovery recipe is a peer-version-matched surgical pip-install, not a full freeze-file mirror. |

---

## Change history

- 2026-05-08: Added 1 insight from the hypervigilance/sameProp session: `20260508_1446_cifs_race_node112_recurrence` (cross-node confirmation of the CIFS race + compliance-gap finding).
- 2026-05-08: Added 2 insights from the dreamer-hypervigilance session: `20260508_1433_cifs_bypass_for_run_command`, `20260508_1434_terminate_command_key_auth_refactor`.
- 2026-05-08: Folder created. Added insight `20260508_1428_node_env_recovery_recipe` (node 101 env recovery).
