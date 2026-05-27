"""Smoke test for scripts/dreamer_srl_offline_wm_test.py.

Exercises the full pipeline (build agent → save checkpoint → run main()) with
synthetic data on CPU.  No real checkpoint or GPU required.  Must complete in
< 60 seconds.

Tests:
  1. main() exits with code 0.
  2. JSON output is written and parseable.
  3. JSON contains metrics_by_horizon.1 and metrics_by_horizon.5 with finite floats.
  4. neg_pos_ratio_by_horizon keys match the requested horizons {"1", "5"}.
  5. Markdown file exists and contains the per-horizon table header.
"""
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

# Add repo root to path
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from src.utils.config import Config, get_default_config
from src.environment.config_loader import load_env_params
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.utils import moments_init
from src.algorithms.dreamer_srl.checkpoint import (
    make_checkpoint_manager, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_FOOD_ONLY_ENV_CFG = os.path.join(_REPO, "configs/experiment/dreamer_curriculum/01_food_only.yaml")
_FOOD_ONLY_AGENT_CFG = os.path.join(_REPO, "configs/models/dreamer_srl/01_food_only.yaml")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Build a tiny food-only dreamer-srl agent and save a checkpoint.

    Returns (checkpoint_parent_dir, env_cfg_path, agent_cfg_path).
    The checkpoint_parent_dir has a checkpoints/ subdirectory.
    """
    tmp_path = tmp_path_factory.mktemp("srl_smoke_ckpt")

    env_cfg = get_default_config()
    env_cfg.merge(Config.load_yaml(_FOOD_ONLY_ENV_CFG))
    agent_cfg = Config.load_yaml(_FOOD_ONLY_AGENT_CFG)

    env_params = load_env_params(env_cfg)
    obs_dim = 19  # food-only 5x5 obs dim (known fixture from test_checkpoint.py)
    action_dim = 4 + int(env_params.rest_action_enabled) + int(env_params.eat_action_enabled)

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=agent_cfg.to_dict(),
        rngs=rngs,
    )
    moments = moments_init()

    manager = make_checkpoint_manager(str(tmp_path), max_to_keep=5)
    save_checkpoint(
        manager=manager,
        episode=1,
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        moments=moments,
        key=key,
        iter_num=1,
        policy_step=100,
        total_episodes_completed=1,
        cumulative_grad_steps=1,
    )

    return str(tmp_path), _FOOD_ONLY_ENV_CFG, _FOOD_ONLY_AGENT_CFG


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_offline_wm_smoke(tiny_checkpoint, tmp_path):
    """Full pipeline smoke: CPU, tiny config, minimal steps.

    Asserts the script runs without error and produces valid output.
    Must finish in < 60 seconds.
    """
    t0 = time.time()

    checkpoint_dir, env_cfg, agent_cfg = tiny_checkpoint
    json_out = str(tmp_path / "smoke_out.json")

    # Import here so the test can run as a subprocess alternative
    from scripts.dreamer_srl_offline_wm_test import main

    argv = [
        "--checkpoint", checkpoint_dir,
        "--env-config", env_cfg,
        "--agent-config", agent_cfg,
        "--source", "real_env",
        "--num-real-steps", "120",
        "--num-starts", "5",
        "--horizon-max", "5",
        "--horizons", "1,5",
        "--seed", "7",
        "--device", "cpu",
        "--output", json_out,
    ]

    ret = main(argv)
    elapsed = time.time() - t0
    print(f"[smoke] main() returned {ret} in {elapsed:.1f}s")

    # 1. Exit code 0
    assert ret == 0, f"main() returned {ret} (expected 0)"

    # 2. JSON written and parseable
    assert os.path.exists(json_out), f"JSON not written: {json_out}"
    with open(json_out) as f:
        data = json.load(f)

    # 3. Finite floats for h=1 and h=5
    mbh = data.get("metrics_by_horizon", {})
    assert "1" in mbh, f"metrics_by_horizon missing key '1'. Keys: {list(mbh.keys())}"
    assert "5" in mbh, f"metrics_by_horizon missing key '5'. Keys: {list(mbh.keys())}"
    mae1 = mbh["1"]["reward_mae_total"]
    mae5 = mbh["5"]["reward_mae_total"]
    assert math.isfinite(mae1), f"h=1 reward_mae_total is not finite: {mae1}"
    assert math.isfinite(mae5), f"h=5 reward_mae_total is not finite: {mae5}"

    # 4. neg_pos_ratio_by_horizon keys match requested horizons
    ratio_keys = set(data.get("neg_pos_ratio_by_horizon", {}).keys())
    assert ratio_keys == {"1", "5"}, f"Expected ratio keys {{1,5}}, got {ratio_keys}"

    # 5. Markdown file exists with per-horizon table header
    md_out = json_out.replace(".json", ".md")
    assert os.path.exists(md_out), f"Markdown not written: {md_out}"
    md_text = open(md_out).read()
    assert "MAE total" in md_text, "Markdown missing per-horizon table header 'MAE total'"

    # 6. Timing guard
    assert elapsed < 60, f"Smoke test took {elapsed:.1f}s (limit: 60s)"
