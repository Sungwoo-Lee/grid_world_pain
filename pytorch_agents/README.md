# pytorch_agents

PyTorch-side agent code for the grid_world_pain project. The JAX-side code lives under `src/` at the repository root; this folder hosts everything that uses PyTorch / Lightning Fabric / sheeprl.

## Current contents

- `pytorch_agents/envs/grid_world_pain.py` — gymnasium wrapper that lets sheeprl train on our JAX-implemented 5×5 (and 10×10) grid-world env. The wrapper forces JAX onto CPU (sheeprl's torch owns the GPU), loads our YAML env config, and exposes the `GridWorldPainWrapper` class.
- `pytorch_agents/configs/{env,exp,logger}/*.yaml` — Hydra config additions discovered by sheeprl via the `SHEEPRL_SEARCH_PATH=pkg://pytorch_agents.configs` env var (set automatically by `scripts/launch_sheeprl.sh`).

## Install

```bash
# Into the sheeprl_bridge conda env on the target training node:
pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain/pytorch_agents

# Also re-install the main project (so `from src.xxx import ...` works inside the bridge):
pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain
```

This single install pulls in sheeprl (pinned to commit `33b6366`, the validated smoke-run tree), JAX CPU, wandb, and pyyaml. See §5 of [`docs/develop/active/diagnosis/sheeprl_training_howto.md`](../docs/develop/active/diagnosis/sheeprl_training_howto.md) for the full per-node setup recipe.

## Future direction

This folder is designed to accommodate the user's longer-term PyTorch migration — rPPO and other algorithms will live alongside the sheeprl bridge here. See [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md) §Layout for the restructure rationale. The NMN-port (porting the neuromodulator + FiLM hooks into sheeprl's PyTorch DreamerV3 agent) is the next planned addition, gated on the experiment that authorizes it.
