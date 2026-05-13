"""Shared episode-level behavior-measure accumulators.

Used by both the JAX trainer (``train.py``) and the sheeprl bridge
(``pytorch_agents/envs/grid_world_pain.py``).  Pure-numpy — no JAX,
no PyTorch, no Lightning.
"""
