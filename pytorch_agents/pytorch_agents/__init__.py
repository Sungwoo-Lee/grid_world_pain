"""PyTorch-side agents for grid_world_pain.

Currently houses the sheeprl-DreamerV3 bridge wrapper (under :mod:`pytorch_agents.envs`)
and the Hydra config tree consumed by sheeprl (under :mod:`pytorch_agents.configs`).
Eventually rPPO and other PyTorch ports will live alongside.

The JAX side of the project lives under :mod:`src` at the repository root.
"""

__version__ = "0.1.0"
