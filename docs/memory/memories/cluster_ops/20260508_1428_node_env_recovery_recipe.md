---
id: 20260508_1428_node_env_recovery_recipe
date: 2026-05-08
time: "14:28"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner]
summary: "Lab node 101 had a 3-layer broken conda env (missing JAX, then flax/optax/orbax/chex, then ptxas/cuda-nvcc) plus a JAX 0.10 vs 0.9.0.1 version mismatch with peers; the recovery recipe is a peer-version-matched surgical pip-install, not a full freeze-file mirror."
related: []
session_origin: claude_code
session_label: "nmn_noise_heterogeneity_sweep_launch_2026-05-07/08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/c7ee226b-2162-4e7a-95e9-257a5b19d713.jsonl
raw_completeness: full
---

# Node 101 env recovery — peer-version-match recipe + pre-flight check upgrade

## Key conclusion
On 2026-05-07/08, launching cells 1–2 of the NMN heterogeneity sweep on node 101 surfaced **three successive layers of env breakage** plus a JAX version mismatch with the rest of the cluster. The full freeze-file mirror approach (`pip install -r <peer>_freeze.txt`) failed twice with pip resolver conflicts (grid-world-pain numpy>=2.4.1 vs flax<2.4 ; tensorflow-cpu vs protobuf 6.x). The recipe that worked was a **surgical, version-pinned pip install of just the JAX/Flax stack**, matching node 102 exactly. Pre-flight check has been upgraded: import-only checks are insufficient because they miss `ptxas` (XLA compile fails only at first GPU op).

## Evidence, measurements, facts
- Cluster SSH form: `ssh 192.168.0.<node>` (101–114 → 192.168.0.101–192.168.0.114), per `~/.ssh/config`. Plain `ssh <node>` resolves to garbage (`0.0.0.<node>`).
- Node 102 (working peer) versions: jax 0.9.0.1, jaxlib 0.9.0.1, jax-cuda12-pjrt 0.9.0.1, jax-cuda12-plugin 0.9.0.1, nvidia-cuda-nvcc-cu12 12.9.86, flax 0.12.4, optax 0.2.6, orbax-checkpoint 0.11.33, chex 0.1.91, jaxtyping 0.3.9, treescope 0.1.10, absl-py 2.4.0, numpy 2.3.5.
- Node 101 starting state: jax 0.10.0, numpy 2.4.1, wandb 0.24.0 — only 77 packages vs 102's 160. Missing: flax, optax, orbax, chex, ptxas (cuda-nvcc).
- Failure mode A — full freeze install: pip resolver rejects because `grid-world-pain==0.1.0` pins `numpy>=2.4.1` but `flax 0.12.4` requires `numpy<2.4.0`. Filtering grid-world-pain (it's a local editable install, not a pypi dep) fixes A.
- Failure mode B — full freeze install (after A fix): pip resolver rejects because `tensorflow-cpu 2.18.0` pins `protobuf<6.0` but freeze pins `protobuf==6.33.4`. Tensorflow-cpu isn't needed for training; surgical install bypasses.
- Failure mode C — Python imports fine but JAX GPU compile crashes with `UNAVAILABLE: No PTX compilation provider is available. Neither ptxas/nvlink nor nvjtlink is available.` Caused by missing `nvidia-cuda-nvcc-cu12` (provides ptxas at `<env>/bin/ptxas`).
- Working recipe (after authorisation):
  ```
  ssh 192.168.0.<node> '/home/vncuser/miniconda3/envs/grid_world_pain/bin/pip install --upgrade \
    numpy==2.3.5 jax==0.9.0.1 jaxlib==0.9.0.1 \
    jax-cuda12-pjrt==0.9.0.1 jax-cuda12-plugin==0.9.0.1 \
    nvidia-cuda-nvcc-cu12==12.9.86 \
    flax==0.12.4 optax==0.2.6 orbax-checkpoint==0.11.33 chex==0.1.91 \
    jaxtyping==0.3.9 treescope==0.1.10 absl-py==2.4.0'
  ```
- Stronger pre-flight check (replaces import-only): `cd <project>/ && python -c "import jax, flax, optax, orbax.checkpoint, chex; from src.environment import config_loader; x = jax.numpy.ones((4,4)); y = (x @ x).block_until_ready(); print(jax.__version__, y.sum())"` — forces a real GPU JIT, catches all three failure layers in one shot.
- Cost of the original import-only check: 2 wasted launch attempts on cells 1–2 + ~20 minutes of diagnosis time before root cause found.

## Decisions and actions
- Updated `~/.claude/.../memory/MEMORY.md` (auto-memory layer) with the upgraded pre-flight recipe and the JAX/Flax version-match recovery command. Distinct from this insight: the auto-memory layer holds the *rule* future runners must obey; this insight holds the *story and rationale* — see CLAUDE.md §2 coexistence rule.
- Cells 1–2 launched successfully after the surgical install + cuda-nvcc add. PIDs 673589 and 674259 verified alive on 192.168.0.101 with t+60s liveness check.
- The lab node's hostname-from-inside is "docker" because nodes are Docker containers — this is normal and not a routing mistake. Don't be alarmed when `hostname` reports "docker" instead of "node101".

## Open questions and follow-ups
- **How did node 101's env get into this state?** Other nodes presumably went through a setup script that put the full stack in place. Worth checking if there's a `bootstrap_lab_ssh.sh` or similar that should have included the JAX/Flax install — and updating it so future-101-style breakage doesn't happen.
- Should the project's `pyproject.toml` loosen `numpy>=2.4.1` to `>=2.3.5` so it doesn't conflict with flax 0.12.4? The cluster runs numpy 2.3.5 successfully — the >=2.4.1 pin appears to be stale.
- Tensorflow-cpu is in the freeze on 102 but not actually used by training. If anyone audits dependencies, it could be removed to simplify future env mirroring.

## References
- Auto-memory rule file: `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/feedback_runner_node_env_preflight.md` (the canonical pre-flight rule for the runner agent).
- Project SSH config: `~/.ssh/config` (defines `192.168.0.10? / 192.168.0.11?` host pattern, port 1800, user vncuser, identity `id_ed25519_gridworld`).
- Why a new folder: closest existing folder is `subagent_engineering` (which is about subagent + worktree mechanics), but this insight is about lab cluster infrastructure (conda envs, CUDA toolchain, node-level recovery). The two are different operational domains. No `cluster_ops` folder existed yet — this is the first cluster-infra insight.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume c7ee226b-2162-4e7a-95e9-257a5b19d713` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view). Shared JSONL with companion insights `20260508_1426_v8_noise_bug_refuted` and `20260508_1427_nmn_heterogeneity_sweep_design`.
