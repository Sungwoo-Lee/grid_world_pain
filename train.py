"""
JAX Training Script for GridWorld RL Agents.

This script mirrors train.py but uses JAX-native components:
1. Loads configuration from YAML files.
2. Initializes the JAX ParallelEnv for massive parallelization.
3. Trains RecurrentPPO (or future DreamerV3) using Flax NNX.
4. Supports WandB logging, checkpointing, and results directory management.

Arguments:
- `--config <path>`: Path to environment/ablation config YAML (Required).
- `--episodes <int>`: Number of training episodes (converted to timesteps).
- `--num-envs <int>`: Number of parallel environments (default: 256).
- `--total-timesteps <int>`: Total timesteps to train.
- `--algorithm <str>`: Algorithm to use (RecurrentPPO, DreamerV3).
- `--seed <int>`: Random seed.
- `--wandb-name <str>`: WandB run name.
- `--no-wandb`: Disable WandB logging.
- `--checkpoint-frequency <int>`: Save frequency (as % of total).
- `--results-dir <path>`: Custom results directory.

GPU Memory Management:
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to prevent JAX from taking 90% VRAM per process.
- Use `CUDA_VISIBLE_DEVICES` to isolate processes on specific GPUs.
- Example: `XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_jax.py ...`

Usage:
    python train_jax.py --config configs/ablation/homeostatic/04_nociception.yaml --total-timesteps 100000
"""
import argparse
import os
import time
import signal
import sys

# --- Pre-parse arguments for Device Selection ---
# To properly set JAX_PLATFORMS, we must do this BEFORE importing jax.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--device", type=str, default="gpu")
_args, _ = _pre_parser.parse_known_args()

# Disable JAX memory pre-allocation (crucial for shared GPU environments)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

if _args.device.lower().startswith("cpu"):
    os.environ["JAX_PLATFORMS"] = "cpu"
elif ":" in _args.device:
    backend, index = _args.device.lower().split(":")
    if backend in ["gpu", "cuda"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        os.environ["JAX_PLATFORMS"] = "cuda"

import yaml
import numpy as np
from datetime import datetime
from collections import deque
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import NamedTuple, List, Optional
from dataclasses import dataclass
import glob

from src.environment.config_loader import load_env_params, load_behavior_measure_cfg
from src.behavior.accumulators import (
    make_bm_state, bm_step_update, bm_reset_env,
    bm_finalise_episode as _bm_finalise_episode_shared,
)
from src.environment.wrapper import ParallelEnv
from src.environment.sensor import get_observation, get_observation_breakdown
from src.environment.core import jax_step
from src.models.recurrent_ppo_network import ActorCriticRNN
from src.models.recurrent_ppo_trainer import train_iteration
from src.models.dqn_network import DQNNetwork, get_action_dqn_nnx
from src.models.dqn_trainer import ReplayBuffer as DQNReplayBuffer, update_step_dqn
from src.models.drqn_network import DRQNNetwork, get_action_drqn_nnx
from src.models.drqn_trainer import RecurrentReplayBuffer as DRQNReplayBuffer, update_step_drqn
from src.models.ppo_network import ActorCriticMLP, get_action_and_value_ppo_nnx
from src.models.ppo_trainer import train_iteration_ppo
from src.utils.config import get_default_config, Config

# Orbax
import orbax.checkpoint as ocp

# Optional WandB
try:
    import wandb
    from src.utils.wandb_utils import wandb_login
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class PPOConfig(NamedTuple):
    num_steps: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    lr: float
    rnn_type: str = "LSTM"
    activation: str = "tanh"
    return_mode: str = "MC"
    max_grad_norm: float = 0.5

# --- ANSI Color Codes ---
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
NC = "\033[0m" # No Color

# --- Graceful Shutdown ---
stop_requested = False

def signal_handler(sig, frame):
    global stop_requested
    if not stop_requested:
        print(f"\n{YELLOW}⚠️  Shutdown signal received (signal {sig}). Finishing current iteration...{NC}")
        stop_requested = True
    else:
        print(f"\n{RED}🚨 Force quitting!{NC}")
        sys.exit(1)

# Defaults
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "environment", "default.yaml")


# ---------------------------------------------------------------------------
# Continual Learning: schedule dataclass and loader
# ---------------------------------------------------------------------------

@dataclass
class ContinualSchedule:
    stage_config_paths: List[str]   # absolute paths, alphabetic order
    stage_names: List[str]           # file stem, e.g. "01_predator_intro"
    stage_configs: List[Config]      # pre-loaded Config per stage (base + stage overlay)
    episode_boundaries: List[int]    # cumulative, strictly increasing
    checkpoint_frequencies: List[int]  # parallel to stages

    @property
    def num_stages(self) -> int:
        return len(self.stage_config_paths)

    def stage_for_episode(self, episode: int) -> int:
        """Return stage index for the given (0-based) episode count."""
        for i, b in enumerate(self.episode_boundaries):
            if episode < b:
                return i
        return self.num_stages - 1   # past the end -> stay in final stage


def _build_continual_schedule(base_config: Config,
                              configs_dir: str,
                              schedule_path: str) -> ContinualSchedule:
    # 1. Discover stage files
    if not os.path.isdir(configs_dir):
        raise ValueError(f"--configs-dir '{configs_dir}' is not a directory.")
    paths = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    if not paths:
        raise ValueError(f"No *.yaml files found in {configs_dir}.")
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    # 2. Load schedule YAML
    schedule = Config.load_yaml(schedule_path)
    boundaries = schedule.get_mandatory("continual.episode_boundaries")
    ckpt_freqs  = schedule.get_mandatory("continual.checkpoint_frequencies")

    if len(boundaries) != len(paths):
        raise ValueError(
            f"episode_boundaries length ({len(boundaries)}) != number of stage configs "
            f"({len(paths)}) in {configs_dir}.")
    if len(ckpt_freqs) != len(paths):
        raise ValueError(
            f"checkpoint_frequencies length ({len(ckpt_freqs)}) != number of stage configs "
            f"({len(paths)}).")
    if sorted(boundaries) != list(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError(f"episode_boundaries must be strictly increasing: {boundaries}")
    if boundaries[0] <= 0:
        raise ValueError(
            f"episode_boundaries[0] must be > 0 (got {boundaries[0]}); "
            "the first boundary is the upper bound of stage 0, so stage 0 must run "
            "for at least one episode.")
    if any(f <= 0 for f in ckpt_freqs):
        raise ValueError(f"checkpoint_frequencies must be > 0: {ckpt_freqs}")

    # 3. Pre-build per-stage Config objects by cloning base and merging each stage YAML
    stage_configs = []
    for p in paths:
        stage_cfg = Config(yaml.safe_load(yaml.dump(base_config.to_dict())))  # deep copy
        stage_cfg.merge(Config.load_yaml(p))
        stage_configs.append(stage_cfg)

    return ContinualSchedule(
        stage_config_paths=paths,
        stage_names=names,
        stage_configs=stage_configs,
        episode_boundaries=list(boundaries),
        checkpoint_frequencies=list(ckpt_freqs),
    )


def main():
    parser = argparse.ArgumentParser(description="Train JAX RL Agents")
    
    # Matching train.py flags
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to base config YAML (Environment/Ablation)")
    parser.add_argument("--configs-dir", type=str, default=None,
                        help="Directory of stage config YAMLs for continual learning. "
                             "Files are ordered alphabetically; prefix names with 01_, 02_, ... to control order. "
                             "Mutually exclusive with --config.")
    parser.add_argument("--continual-schedule", type=str, default=None,
                        help="Path to schedule YAML (episode_boundaries, checkpoint_frequencies). "
                             "Required when --configs-dir is used.")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent config YAML (Required)")
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    parser.add_argument("--device", type=str, help="Device to use (JAX handles this; kept for parity)")
    parser.add_argument("--no-satiation", action="store_true", help="Disable satiation (conventional mode)")
    parser.add_argument("--no-overeating-death", action="store_true", help="Disable death by overeating")
    parser.add_argument("--wandb-project", type=str, help="WandB Project Name")
    parser.add_argument("--wandb-group", type=str, help="WandB Group Name")
    parser.add_argument("--wandb-job-type", type=str, help="WandB Job Type")
    parser.add_argument("--wandb-name", type=str, help="WandB Run Name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress output and progress bar")
    parser.add_argument("--debug", action="store_true", help="Show verbose step-by-step progress logging")
    parser.add_argument("--checkpoint-frequency", type=int, help="Save checkpoint every N episodes/evals")
    parser.add_argument("--load-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-resume-id", type=str, help="WandB Run ID to resume logging")

    # JAX-specific but useful flags
    parser.add_argument("--total-timesteps", type=int, help="Total training timesteps (overrides episodes if provided)")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, help="Steps per iteration (rollout length)")
    parser.add_argument("--hidden-size", type=int, help="Hidden layer size")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides agent config if provided)")
    parser.add_argument("--results-dir", type=str, help="Custom results directory")
    parser.add_argument("--wandb-entity", type=str, help="WandB Entity Name")
    parser.add_argument("--log-interval", type=int, help="WandB logging interval in iterations (default: 1)")
    parser.add_argument("--log-accumulate", action=argparse.BooleanOptionalAction, default=None,
                        help="Accumulate episode metrics across log interval (default: true). Use --no-log-accumulate for hard interval.")
    parser.add_argument("--profile", action="store_true",
                        help="Enable jax.profiler trace of the training loop. "
                             "Trace written to tmp/<timestamp>_dreamer_v3_vs_rppo_profile/<algo>_trace/. "
                             "Forces --no-wandb and --quiet, runs warm-up + trace window then exits.")

    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.debug:
        print(f"[DEBUG] Script started. CLI arguments: {args}", flush=True)

    # --- Profiler constants ---
    PROFILE_WARMUP_ITERS = 20
    PROFILE_TRACE_ITERS = 200
    PROFILE_TOTAL_ITERS = PROFILE_WARMUP_ITERS + PROFILE_TRACE_ITERS  # = 220

    if args.profile:
        args.no_wandb = True
        args.quiet = True
        from datetime import datetime as _dt
        _profile_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        profile_parent = os.path.join("tmp", f"{_profile_ts}_dreamer_v3_vs_rppo_profile")
        os.makedirs(profile_parent, exist_ok=True)
        # The per-algo subdirectory is created after `algorithm` is known (below).

    # --- Finalize Device Selection ---
    try:
        device_str = (args.device or _args.device).lower()
        if ":" in device_str or device_str in ["gpu", "cuda"]:
            # If we set CUDA_VISIBLE_DEVICES, JAX sees the target GPU as index 0
            jax.config.update("jax_default_device", jax.devices("cuda")[0])
            if not args.quiet:
                print(f"Device: gpu ({jax.devices('cuda')[0]})")
        elif device_str == "cpu":
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
            if not args.quiet:
                print(f"Device: cpu")
    except Exception as e:
        if not args.quiet:
            print(f"Warning: Device configuration for '{args.device}' failed ({e}). Using JAX default: {jax.devices()[0]}")

    # 1. Configuration Loading
    if args.debug: print(f"[DEBUG] Phase 1: Configuration Loading...", flush=True)
    
    # Matching train.py logic (Strictly no safe defaults)
    base_config_path = args.config or DEFAULT_CONFIG_PATH
    if args.debug: print(f"[DEBUG] Loading base config from {base_config_path}", flush=True)
    config = get_default_config()

    # Merge Training Defaults
    train_config_path = os.path.join(os.path.dirname(__file__), "configs", "train", "default.yaml")
    if os.path.exists(train_config_path):
        if not args.quiet:
            print(f"Loading train config from {train_config_path}")
        train_defaults = Config.load_yaml(train_config_path)
        config.merge(train_defaults)

    # Merge Evaluation Defaults
    eval_config_path = os.path.join(os.path.dirname(__file__), "configs", "evaluation", "default.yaml")
    if os.path.exists(eval_config_path):
        if not args.quiet:
            print(f"Loading eval config from {eval_config_path}")
        eval_defaults = Config.load_yaml(eval_config_path)
        config.merge(eval_defaults)

    # Merge Logger Config (WandB settings)
    logger_config_path = os.path.join(os.path.dirname(__file__), "configs", "logger", "wandb.yaml")
    if os.path.exists(logger_config_path):
        if not args.quiet:
            print(f"Loading logger config from {logger_config_path}")
        logger_config = Config.load_yaml(logger_config_path)
        config.merge(logger_config)

    # Merge Visualization Defaults
    vis_config_path = os.path.join(os.path.dirname(__file__), "configs", "visualization", "default.yaml")
    if os.path.exists(vis_config_path):
        if not args.quiet:
            print(f"Loading visualization config from {vis_config_path}")
        vis_defaults = Config.load_yaml(vis_config_path)
        config.merge(vis_defaults)

    # Merge Base/User/Ablation Config (--config) OR load continual schedule
    schedule: Optional[ContinualSchedule] = None
    if args.configs_dir is not None:
        if args.config:
            raise ValueError("--configs-dir and --config are mutually exclusive.")
        if args.continual_schedule is None:
            raise ValueError("--continual-schedule is required when --configs-dir is set.")
        schedule = _build_continual_schedule(config, args.configs_dir, args.continual_schedule)
        # Fix 1: propagate CLI overrides to every stage config so load_env_params(stage_configs[i])
        # honours --no-satiation / --no-overeating-death regardless of what the stage YAML says.
        # These are properties of the run, not of any individual stage.
        if args.no_satiation:
            for _sc in schedule.stage_configs:
                _sc.set('body.with_satiation', False)
        if args.no_overeating_death:
            for _sc in schedule.stage_configs:
                _sc.set('body.overeating_death', False)
        # Merge stage-0 into the live config so downstream code sees a fully-populated Config
        # for the starting stage.
        config = schedule.stage_configs[0]
        if not args.quiet:
            print(f"Continual mode: {schedule.num_stages} stages from {args.configs_dir}")
            for i, (n, b, f) in enumerate(zip(schedule.stage_names,
                                              schedule.episode_boundaries,
                                              schedule.checkpoint_frequencies)):
                print(f"  [{i:02d}] {n:30s}  until_ep={b:>6d}  ckpt_freq={f}")
    elif args.config:
        if not args.quiet:
            print(f"Loading override config from: {args.config}")
        user_config = Config.load_yaml(args.config)
        config.merge(user_config)

    # Merge Agent Config (--agent_config) - REQUIRED
    agent_config_path = args.agent_config
    if args.debug: print(f"[DEBUG] Loading agent config from {agent_config_path}", flush=True)
    if not args.quiet:
        print(f"Loading agent config from: {agent_config_path}")
    agent_config = Config.load_yaml(agent_config_path)
    config.merge(agent_config)

    # CLI Overrides (Synchronized with train.py)
    if args.wandb_project: config.set('wandb.project', args.wandb_project)
    if args.wandb_group: config.set('wandb.group', args.wandb_group)
    if args.wandb_job_type: config.set('wandb.job_type', args.wandb_job_type)
    if args.wandb_name: config.set('wandb.name', args.wandb_name)
    if args.no_wandb: config.set('wandb.disabled', True)
    if args.tag: config.set('tag', args.tag)
    if args.seed is not None: config.set('seed', args.seed)
    if args.no_satiation: config.set('environment.with_satiation', False)
    if args.no_overeating_death: config.set('environment.overeating_death', False)

    # 1.5 Print Combined Configuration (Always)
    if not args.quiet:
        def print_pretty_config(config_dict):
            """Stylized configuration audit log."""
            width = 64
            print("\n" + "=" * width)
            print(" SYSTEM CONFIGURATION AUDIT ".center(width, "="))
            print("=" * width)
            
            def print_recursive(d, indent=0):
                keys = sorted(d.keys())
                for k in keys:
                    v = d[k]
                    prefix = "  " * indent
                    if isinstance(v, dict):
                        print(f"\n{prefix}[ {k.upper()} ]")
                        print_recursive(v, indent + 1)
                    elif isinstance(v, list):
                        if len(v) > 0 and isinstance(v[0], dict):
                            # List of dicts (like predators/resources)
                            names = [item.get('name', 'unnamed') for item in v]
                            print(f"{prefix}● {k: <20} : {len(v)} items ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")
                        elif len(v) > 5:
                            print(f"{prefix}● {k: <20} : [List of {len(v)} items]")
                        else:
                            print(f"{prefix}● {k: <20} : {v}")
                    else:
                        print(f"{prefix}● {k: <20} : {v}")
            
            # Separate dicts from non-dicts at top level for grouping
            top_dicts = {k: v for k, v in config_dict.items() if isinstance(v, dict)}
            top_leafs = {k: v for k, v in config_dict.items() if not isinstance(v, dict)}
            
            if top_leafs:
                print("\n[ GLOBAL SETTINGS ]")
                print_recursive(top_leafs, indent=1)
            
            print_recursive(top_dicts, indent=0)
            print("\n" + "=" * width + "\n")

        print_pretty_config(config.to_dict())

    # Determine Algorithm
    algorithm = config.get_mandatory('agent.algorithm')

    if args.profile:
        profile_trace_dir = os.path.join(profile_parent, f"{algorithm}_trace")
        os.makedirs(profile_trace_dir, exist_ok=True)
        print(f"[PROFILE] trace dir: {profile_trace_dir}")
        print(f"[PROFILE] warm-up: {PROFILE_WARMUP_ITERS} iters, "
              f"trace window: {PROFILE_TRACE_ITERS} iters")

    # Strictly Resolve Parameters (No Safe Defaults)
    if schedule is not None:
        if algorithm not in ("RecurrentPPO", "DreamerV3"):
            raise ValueError(
                f"Continual learning (--configs-dir) is only supported for "
                f"RecurrentPPO and DreamerV3, got algorithm='{algorithm}'. "
                "Use Option A: restrict to supported algorithms at startup.")
        if args.episodes is not None:
            raise ValueError("--episodes is incompatible with --configs-dir; "
                             "episode budget is set by the schedule's last boundary.")
        episodes = schedule.episode_boundaries[-1]
    else:
        episodes = args.episodes if args.episodes is not None else config.get_mandatory('episodes')
    env_max_steps = config.get_mandatory('environment.max_steps')
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')
    log_interval = args.log_interval or config.get('training.log_interval', 1)
    log_accumulate = args.log_accumulate if args.log_accumulate is not None else config.get('training.log_accumulate', True)
    
    # Budget scales with parallelization: episodes * steps per episode * num environments
    total_timesteps = args.total_timesteps or (episodes * env_max_steps * num_envs)
    
    if algorithm in ["RecurrentPPO", "PPO"]:
        if algorithm == "RecurrentPPO":
            num_steps = args.num_steps or config.get_mandatory('agent.sequence_length')
        else:
            num_steps = args.num_steps or config.get_mandatory('agent.num_steps')
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr_actor')
    elif algorithm == "DreamerV3":
        # collect_interval: how many env steps to collect per iteration per env.
        # 1 = sheeprl-style (canonical, fine-grained), 128 = full sequence (JAX-optimized).
        # The replay buffer samples sequence_length-step sequences for BPTT training regardless.
        num_steps = args.num_steps or config.get_mandatory('agent.collect_interval')
        # Dreamer has many hidden sizes; using rssm_deter_dim as a proxy for summary/logging
        hidden_size = args.hidden_size or config.get_mandatory('agent.rssm_deter_dim')
        lr = args.lr or config.get_mandatory('agent.actor_lr')
    else:
        num_steps = args.num_steps or config.get_mandatory('agent.num_steps')
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr')

    seed = args.seed if args.seed is not None else config.get_mandatory('seed')

    
    # Re-load EnvParams with full merged config for JAX core
    params = load_env_params(config)  # stage 0 (or single-config)

    # Apply CLI overrides to params if they exist in params (redundant now but safe)
    if args.no_satiation: params = params.replace(with_satiation=False)
    if args.no_overeating_death: params = params.replace(overeating_death=False)

    # Validate obs/action dim AND modality fingerprint consistency across all stages
    # BEFORE training starts.
    def _modality_fingerprint(p):
        """Tuple of all sensor-enable flags and key shape params that affect obs layout.
        If any two stages produce different fingerprints the input semantics differ even
        when obs_dim happens to be the same (same-total-dim modality swap)."""
        return (
            p.visual_sensor_enabled,
            p.visual_sensor_range,
            p.local_view_size,
            p.olfactory_enabled,
            p.olfactory_vector_size,
            p.nociception_enabled,
            p.nociception_size,
            p.interoceptive_nociception_enabled,
            p.location_sensor_enabled,
            p.proprioception_enabled,
            p.injury_observable,
            p.nutrition_observable,
            p.sensor_range,
        )

    if schedule is not None:
        env_probe = ParallelEnv(params)
        probe_key = jax.random.PRNGKey(0)
        _, probe_obs = env_probe.reset(probe_key, 1)
        stage0_obs_dim = int(probe_obs.shape[-1])
        stage0_action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)
        stage0_fingerprint = _modality_fingerprint(params)
        for i in range(1, schedule.num_stages):
            p_i = load_env_params(schedule.stage_configs[i])
            env_i = ParallelEnv(p_i)
            _, obs_i = env_i.reset(probe_key, 1)
            a_i = 4 + int(p_i.rest_action_enabled) + int(p_i.eat_action_enabled)
            if int(obs_i.shape[-1]) != stage0_obs_dim or a_i != stage0_action_dim:
                raise ValueError(
                    f"Stage {i} ({schedule.stage_names[i]}) changes "
                    f"obs_dim ({stage0_obs_dim} -> {int(obs_i.shape[-1])}) or "
                    f"action_dim ({stage0_action_dim} -> {a_i}). "
                    "Continual learning forbids architecture-visible dimension changes.")
            fp_i = _modality_fingerprint(p_i)
            if fp_i != stage0_fingerprint:
                raise ValueError(
                    f"Stage {i} ({schedule.stage_names[i]}) has a different sensor modality "
                    f"fingerprint than stage 0, which would scramble the observation semantics "
                    f"even if obs_dim is unchanged.\n"
                    f"  Stage 0 fingerprint: {stage0_fingerprint}\n"
                    f"  Stage {i} fingerprint: {fp_i}")
        del env_probe
        if not args.quiet:
            print(f"Continual mode: obs_dim={stage0_obs_dim}, action_dim={stage0_action_dim} "
                  f"validated consistent across {schedule.num_stages} stages.")

    # 2. Setup Results Directory
    if args.debug: print(f"[DEBUG] Phase 2: Results Directory Setup...", flush=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or config.get_mandatory('tag')
    run_name = f"{timestamp}_{tag}"
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join("results", f"JAX_{algorithm}", run_name)
    
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Orbax Setup (New API)
    _missing = object()
    max_checkpoints = config.get('training.max_checkpoints_to_keep', _missing)
    if max_checkpoints is _missing:
        raise ValueError("Strict Config: Configuration key 'training.max_checkpoints_to_keep' is required but missing.")
    checkpointer = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
    )

    # Save config
    config_save_path = os.path.join(models_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    if not args.quiet:
        print(f"Config saved to: {config_save_path}")

    # In continual mode: also dump each stage config and the schedule for auditability
    if schedule is not None:
        for i, (name, cfg) in enumerate(zip(schedule.stage_names, schedule.stage_configs)):
            out = os.path.join(models_dir, f"stage_{i:02d}_{name}.yaml")
            with open(out, "w") as f:
                yaml.dump(cfg.to_dict(), f, default_flow_style=False)
        sched_dump = {
            "continual": {
                "episode_boundaries": schedule.episode_boundaries,
                "checkpoint_frequencies": schedule.checkpoint_frequencies,
                "stage_names": schedule.stage_names,
            }
        }
        with open(os.path.join(models_dir, "schedule.yaml"), "w") as f:
            yaml.dump(sched_dump, f, default_flow_style=False)
        if not args.quiet:
            print(f"Stage configs and schedule saved under {models_dir}")

    # 3. Initialize WandB
    if args.debug: print(f"[DEBUG] Phase 3: WandB Initialization...", flush=True)
    wandb_enabled = WANDB_AVAILABLE and not args.no_wandb and not config.get_mandatory('wandb.disabled')
    if wandb_enabled:
        wandb_login(quiet=True)
        
        wandb_kwargs = {
            "project": args.wandb_project or config.get_mandatory('wandb.project'),
            "entity": args.wandb_entity or config.get_mandatory('wandb.entity'),
            "group": args.wandb_group or config.get_mandatory('wandb.group'),
            "name": args.wandb_name or tag,
            "reinit": True
        }
        wandb_config_payload = {
            "algorithm": algorithm,
            "framework": "JAX/Flax NNX",
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "lr": lr,
            "hidden_size": hidden_size,
            "seed": seed,
            **config.to_dict(),
        }
        if schedule is not None:
            wandb_config_payload["continual"] = {
                "num_stages": schedule.num_stages,
                "stage_names": schedule.stage_names,
                "episode_boundaries": schedule.episode_boundaries,
                "checkpoint_frequencies": schedule.checkpoint_frequencies,
            }
        wandb_kwargs["config"] = wandb_config_payload

        if args.wandb_resume_id:
            wandb_kwargs['id'] = args.wandb_resume_id
            wandb_kwargs['resume'] = "allow"

        wandb.init(**wandb_kwargs)

        wandb.define_metric("iteration")
        wandb.define_metric("timesteps")
        wandb.define_metric("Episode/Number")
        wandb.define_metric("*", step_metric="timesteps")
        wandb.define_metric("Episode/*", step_metric="Episode/Number")
        wandb.define_metric("loss/*", step_metric="iteration")
        wandb.define_metric("modulator/*", step_metric="iteration")
        wandb.define_metric("behavior/*", step_metric="timesteps")
        wandb.define_metric("stage/index",      step_metric="Episode/Number")
        wandb.define_metric("stage/transition", step_metric="Episode/Number")

        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # 4. Print Summary
    if not args.quiet:
        width = 60
        header = " JAX/FLAX RL CONFIGURATION "
        print("\n" + "=" * width)
        print(header.center(width, "="))
        print("=" * width)
        
        def print_section(title, data):
            print(f"\n[{title}]")
            for k, value in data.items():
                print(f"  \u25cf {k:.<25} {value}")
        
        with_satiation = config.get_mandatory('body.with_satiation')
        with_injury = config.get_mandatory('body.with_injury')
        use_homeostatic_reward = config.get_mandatory('body.use_homeostatic_reward')
        
        env_data = {
            "Grid Size": f"{params.height}x{params.width}",
            "Max Steps": env_max_steps,
            "Mode": "Interoceptive (Homeostasis)" if with_satiation else "Conventional (Goal-driven)",
        }
        if with_satiation:
            env_data["Reward Logic"] = "Homeostatic (Drive Reduction)" if use_homeostatic_reward else "Survival Step"
        if with_injury:
            env_data["Injury System"] = "ENABLED"
        print_section("Environment", env_data)
        
        train_data = {
            "Framework": "JAX/Flax NNX",
            "Total Timesteps": f"{total_timesteps:,}",
            "Parallel Envs": num_envs,
            "Steps/Iteration": num_steps,
            "Seed": seed,
            "Results": results_dir,
            "WandB": "Enabled" if wandb_enabled else "Disabled"
        }
        print_section("Training", train_data)

    # 5. Training Setup
    if args.debug: print(f"[DEBUG] Phase 5: Training Setup...", flush=True)
    env = ParallelEnv(params)
    key = jax.random.PRNGKey(seed)
    key, model_key, env_key = jax.random.split(key, 3)

    if args.debug: print(f"[DEBUG] Performing initial environment reset for {num_envs} envs...", flush=True)
    env_state, obs = env.reset(env_key, num_envs)
    input_dim = obs.shape[-1]
    
    rest_enabled = params.rest_action_enabled
    eat_enabled = params.eat_action_enabled
    action_dim = 4 + int(rest_enabled) + int(eat_enabled)

    obs_breakdown = get_observation_breakdown(params)
    total_dim = sum(obs_breakdown.values())

    if not args.quiet:
        print("\n--- RL API Specifications ---")
        print(f"Action Dim: {action_dim}")
        breakdown_str = ", ".join([f"{k}={v}" for k, v in obs_breakdown.items()])
        print(f"Observation Dim: {total_dim} ({breakdown_str})")
        print(f"Dimension Breakdown:")
        for sensor_name, dim in obs_breakdown.items():
            print(f"  {sensor_name:.<20} {dim}")
        print(f"Hidden Size: {hidden_size}")
        print(f"Learning Rate: {lr}")
        print("--------------------------------------------\n")
    
    # 6. Algorithm Initialization
    if args.debug: print(f"[DEBUG] Phase 6: Algorithm Initialization ({algorithm})...", flush=True)
    if algorithm == "RecurrentPPO":
        key, init_key = jax.random.split(key)
        
        # Read parity options from config
        rnn_type = config.get_mandatory('agent.rnn_type')
        activation = config.get_mandatory('agent.activation')
        return_mode = config.get_mandatory('agent.return_mode')
        
        # Read neuromodulation config (MUST be defined in config, even if empty/null)
        # Using get() because get_mandatory() raises ValueError for null/None values
        modulation_config = config.get('agent.modulation')
        if modulation_config is not None and modulation_config.get('type') is None:
            modulation_config = None

        if not args.quiet:
            print(f"RNN Type: {rnn_type}, Activation: {activation}, Return Mode: {return_mode}")
            if modulation_config is not None:
                print(f"Neuromodulation: ENABLED (type={modulation_config['type']}, "
                      f"mod_hidden={modulation_config['mod_hidden_size']}, "
                      f"grouping={modulation_config['grouping_size']})")
            else:
                print(f"Neuromodulation: DISABLED (baseline)")
        
        model = ActorCriticRNN(
            input_dim=input_dim, 
            action_dim=action_dim, 
            hidden_size=hidden_size, 
            rngs=nnx.Rngs(init_key),
            rnn_type=rnn_type,
            activation=activation,
            modulation_config=modulation_config,
            observation_breakdown=obs_breakdown,
            encoding_config=config.to_dict().get('agent', {})
        )
        
        # Use optax.chain for gradient clipping (Option A)
        max_grad_norm = config.get_mandatory('agent.max_grad_norm')
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr),
            ),
            wrt=nnx.Param,
        )
        
        ppo_config = PPOConfig(
            num_steps=num_steps,
            num_epochs=config.get_mandatory('agent.K_epochs'),
            gamma=config.get_mandatory('agent.gamma'),
            gae_lambda=config.get_mandatory('agent.gae_lambda'),
            clip_eps=config.get_mandatory('agent.eps_clip'),
            ent_coef=config.get_mandatory('agent.entropy_coef'),
            vf_coef=config.get_mandatory('agent.vf_coef'),
            lr=lr,
            rnn_type=rnn_type,
            activation=activation,
            return_mode=return_mode,
            max_grad_norm=config.get_mandatory('agent.max_grad_norm')
        )
        
        # Initialize hidden state via model (handles modulator state automatically)
        h_state = model.initial_state(num_envs)

        if not args.quiet:
            print("JIT compiling train_iteration...")
        jit_train = nnx.jit(train_iteration, static_argnums=(6,))

        
    elif algorithm == "DreamerV3":
        from src.models.dreamer_v3_trainer import DreamerTrainer, ReplayBuffer
        
        # Read neuromodulation config (MUST be defined in config, even if empty/null)
        # Using get() because get_mandatory() raises ValueError for null/None values
        dreamer_mod_config = config.get('agent.modulation')
        if dreamer_mod_config is not None and dreamer_mod_config.get('type') is None:
            dreamer_mod_config = None
        
        key, init_key = jax.random.split(key)
        # Use agent_config (Config object) directly to support get_mandatory inside trainer
        trainer = DreamerTrainer(input_dim, action_dim, agent_config, rngs=nnx.Rngs(init_key),
                                 obs_breakdown=obs_breakdown,
                                 modulation_config=dreamer_mod_config)

        from src.models.dreamer_v3_util import Ratio
        ratio_scaled_updates = Ratio(config.get_mandatory('agent.replay_ratio'))
        cumulative_gradient_steps = 0


        buffer_device = config.get_mandatory('agent.buffer_device')
        buffer_capacity = config.get_mandatory('agent.buffer_capacity')
        buffer = ReplayBuffer(
            capacity=buffer_capacity, 
            sequence_length=config.get_mandatory('agent.sequence_length'), 
            obs_dim=input_dim, 
            action_dim=action_dim,
            device=buffer_device
        )

        # Positive-reward buffer (only created if mixture mode)
        sampling_mode = config.get_mandatory('agent.sampling_mode')
        positive_buffer = None
        if sampling_mode == 'mixture':
            pos_cap = config.get_mandatory('agent.positive_buffer_capacity')
            # Round capacity to multiple of sequence_length
            seq_len = config.get_mandatory('agent.sequence_length')
            pos_cap = (pos_cap // seq_len) * seq_len
            positive_buffer = ReplayBuffer(
                capacity=pos_cap,
                sequence_length=seq_len,
                obs_dim=input_dim,
                action_dim=action_dim,
                device=buffer_device
            )
        if algorithm == "DreamerV3":
            # Initial Dreamer state (reset on every collect if we want, but better to persist)
            # Initialize with zeros instead of None to avoid JIT re-trace on first call
            dreamer_state = trainer.agent.wm.rssm.initial(num_envs)
            # Add prev_action for consistency
            dreamer_state['prev_action'] = jnp.zeros(
                (num_envs, trainer.agent.ac.actor.net.layers[-1].out_features))
            # Initial step is always 'first'
            dreamer_state['is_first'] = jnp.ones((num_envs, 1))
            # Initial modulator state if enabled
            if trainer.agent.wm.modulation_enabled:
                dreamer_state['mod_h'] = trainer.agent.wm.modulator.initial_state(num_envs)
        else:
            dreamer_state = None

    elif algorithm == "DQN":
        key, init_key = jax.random.split(key)
        fc_layers = config.get_mandatory('agent.fc_layers')
        model = DQNNetwork(input_dim, action_dim, fc_layers, rngs=nnx.Rngs(init_key))
        target_model = DQNNetwork(input_dim, action_dim, fc_layers, rngs=nnx.Rngs(init_key))
        # NNX state sync
        nnx.update(target_model, nnx.state(model))
        
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        
        buffer_capacity = config.get_mandatory('agent.buffer_size')
        dqn_buffer = DQNReplayBuffer(capacity=buffer_capacity, obs_dim=input_dim)
        
        epsilon = config.get_mandatory('agent.epsilon_start')
        epsilon_end = config.get_mandatory('agent.epsilon_end')
        epsilon_decay = config.get_mandatory('agent.epsilon_decay')
        target_update_freq = config.get_mandatory('agent.target_update_freq')
        batch_size = config.get_mandatory('agent.batch_size')
        
        # Use num_steps as "collection steps per iteration"
        num_steps = args.num_steps or config.get_mandatory('agent.num_steps')

    elif algorithm == "DRQN":
        key, init_key = jax.random.split(key)
        fc_layers = config.get_mandatory('agent.fc_layers')
        recurrent_layers = config.get_mandatory('agent.recurrent_layers')
        hidden_size = recurrent_layers[0] # NNX LSTM/GRU use single hidden size
        rnn_type = config.get_mandatory('agent.rnn_type')
        
        model = DRQNNetwork(input_dim, action_dim, hidden_size, rnn_type, fc_layers, rngs=nnx.Rngs(init_key))
        target_model = DRQNNetwork(input_dim, action_dim, hidden_size, rnn_type, fc_layers, rngs=nnx.Rngs(init_key))
        nnx.update(target_model, nnx.state(model))
        
        optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        
        buffer_capacity = config.get_mandatory('agent.buffer_size')
        drqn_buffer = DRQNReplayBuffer(capacity=buffer_capacity, obs_dim=input_dim)
        
        epsilon = config.get_mandatory('agent.epsilon_start')
        epsilon_end = config.get_mandatory('agent.epsilon_end')
        epsilon_decay = config.get_mandatory('agent.epsilon_decay')
        target_update_freq = config.get_mandatory('agent.target_update_freq')
        batch_size = config.get_mandatory('agent.batch_size')
        trace_length = config.get_mandatory('agent.trace_length')
        burn_in_length = config.get_mandatory('agent.burn_in_length')
        
        # Hidden state for collection
        h_state = model.initial_state(num_envs)
        
        # Reuse num_steps as "collection steps per iteration"
        num_steps = args.num_steps or config.get_mandatory('agent.num_steps')

    elif algorithm == "PPO":
        key, init_key = jax.random.split(key)
        
        activation = config.get_mandatory('agent.activation')
        return_mode = config.get_mandatory('agent.return_mode')
        actor_fc_layers = config.get_mandatory('agent.actor_fc_layers')
        critic_fc_layers = config.get_mandatory('agent.critic_fc_layers')
        
        # We can support dual LR by choosing one or using a complex optimizer
        # For now, let's use lr_actor as primary
        lr_actor = config.get_mandatory('agent.lr_actor')
        
        if not args.quiet:
            print(f"Activation: {activation}, Return Mode: {return_mode}")
            print(f"Actor Layers: {actor_fc_layers}, Critic Layers: {critic_fc_layers}")
        
        model = ActorCriticMLP(
            input_dim=input_dim, 
            action_dim=action_dim, 
            actor_fc_layers=actor_fc_layers,
            critic_fc_layers=critic_fc_layers,
            rngs=nnx.Rngs(init_key),
            activation=activation
        )
        optimizer = nnx.Optimizer(model, optax.adam(lr_actor), wrt=nnx.Param)
        
        ppo_config = PPOConfig(
            num_steps=num_steps,
            num_epochs=config.get_mandatory('agent.K_epochs'),
            gamma=config.get_mandatory('agent.gamma'),
            gae_lambda=config.get_mandatory('agent.gae_lambda'),
            clip_eps=config.get_mandatory('agent.eps_clip'),
            ent_coef=config.get_mandatory('agent.entropy_coef'),
            vf_coef=config.get_mandatory('agent.vf_coef'),
            lr=lr_actor,
            activation=activation,
            return_mode=return_mode
        )
        
        if not args.quiet:
            print("JIT compiling train_iteration_ppo...")
        jit_train = nnx.jit(train_iteration_ppo, static_argnums=(5,))
        
    # 7. Training Loop Initialization
    if args.debug: print(f"[DEBUG] Phase 7: Training Loop Initialization...", flush=True)
    global_step = 0
    iteration = 0
    total_episodes_completed = 0
    current_stage = 0
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_info_buffer = deque(maxlen=100)
    
    # Behavioral event accumulators (per-env, reset on episode done)
    BEHAVIOR_KEYS = ['ate_food', 'hit_predator', 'hit_hiding_predator', 'hit_neutral',
                     'event_collided', 'rested',
                     'damage', 'damage_predator', 'damage_hiding_predator', 'damage_obstacle']
    BEHAVIOR_DIST_KEYS = ['dist_to_food', 'dist_to_pred',
                          'dist_to_neutral', 'dist_to_hiding_predator']  # Need mean, not sum

    episode_behavior = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_KEYS}
    episode_dist_sums = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_DIST_KEYS}

    # Per-instance (tag-based) accumulators for Round-2 metrics.
    neutral_tags  = tuple(params.neutral_tags)   # static; possibly empty
    predator_tags = tuple(params.predator_tags)
    num_neutral_for_log  = len(neutral_tags)
    num_predator_for_log = len(predator_tags)
    episode_dist_per_neutral_sums  = np.zeros((num_envs, num_neutral_for_log),  dtype=np.float32)
    episode_dist_per_predator_sums = np.zeros((num_envs, num_predator_for_log), dtype=np.float32)

    # === Behavior-measure toolkit v1: online M1/M2/M5 state ===
    # Loaded once from config; None when behavior_measures: block is absent (backwards-compat).
    bm_cfg = load_behavior_measure_cfg(config)
    bm_enabled = bm_cfg is not None and bm_cfg.enabled
    if bm_enabled:
        bm_R = float(bm_cfg.cue_radius)
        bm_K = int(bm_cfg.obs_window)
    else:
        bm_R = 0.0
        bm_K = 0

    # Behavior-measure toolkit v1: M1/M2/M5 state via shared module (src.behavior.accumulators).
    # All 16 per-class/per-tag numpy arrays and K-buffer state are encapsulated in BMState.
    if bm_enabled:
        _bm_state = make_bm_state(
            num_envs=num_envs,
            num_predator_tags=num_predator_for_log,
            num_neutral_tags=num_neutral_for_log,
            bm_R=bm_R,
            bm_K=bm_K,
        )
    else:
        _bm_state = None

    def _bm_reset_env(i):
        """Reset all behavior-measure accumulators for env i at episode end."""
        if _bm_state is not None:
            bm_reset_env(_bm_state, i)

    def _bm_step_update(info_np_t, done_mask):
        """One-step update of M1/M2/M5 counters. Delegates to shared module."""
        if _bm_state is not None:
            bm_step_update(_bm_state, info_np_t, done_mask)

    def _bm_finalise_episode(i, ep_data):
        """Compute per-episode BM scalars for env i and add to ep_data dict.
        Delegates to shared module."""
        if _bm_state is not None:
            ep_data.update(_bm_finalise_episode_shared(_bm_state, i, predator_tags, neutral_tags))

    def _append_per_measure_mean(ep_log, iteration_episodes, ep_key_raw, wandb_key):
        """Mean the same per-episode raw scalar (skipping NaN) across the iteration's episodes."""
        vals = [ep[ep_key_raw] for ep in iteration_episodes
                if ep_key_raw in ep and not (isinstance(ep[ep_key_raw], float) and
                                              ep[ep_key_raw] != ep[ep_key_raw])]  # isnan check
        if vals:
            ep_log[wandb_key] = float(np.mean(vals))

    def _bm_log_wandb(ep_log, iteration_episodes):
        """Append all BM WandB keys to ep_log from the iteration's episodes."""
        for cname in ("predator", "rabbit"):
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"interrupted_feeding_rate_{cname}_raw", f"Episode/InterruptedFeedingRate_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"interrupted_feeding_denom_{cname}_raw", f"Episode/InterruptedFeedingDenominator_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"bush_dive_rate_{cname}_raw", f"Episode/BushDiveRate_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"bush_dive_denom_{cname}_raw", f"Episode/BushDiveDenominator_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_ratio_{cname}_raw", f"Episode/EatUnderThreatRatio_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_rate_{cname}_raw", f"Episode/EatUnderThreatRate_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_safe_rate_{cname}_raw", f"Episode/EatSafeRate_{cname}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_safe_steps_{cname}_raw", f"Episode/EatUnderThreatSafeSteps_{cname}")
        for tag in predator_tags:
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"interrupted_feeding_rate_predator_{tag}_raw", f"Episode/InterruptedFeedingRate_predator_{tag}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"bush_dive_rate_predator_{tag}_raw", f"Episode/BushDiveRate_predator_{tag}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_ratio_predator_{tag}_raw", f"Episode/EatUnderThreatRatio_predator_{tag}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_rate_predator_{tag}_raw", f"Episode/EatUnderThreatRate_predator_{tag}")
        for tag in neutral_tags:
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"interrupted_feeding_rate_rabbit_{tag}_raw", f"Episode/InterruptedFeedingRate_rabbit_{tag}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"bush_dive_rate_rabbit_{tag}_raw", f"Episode/BushDiveRate_rabbit_{tag}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_ratio_rabbit_{tag}_raw", f"Episode/EatUnderThreatRatio_rabbit_{tag}")
            _append_per_measure_mean(ep_log, iteration_episodes,
                f"eat_under_threat_rate_rabbit_{tag}_raw", f"Episode/EatUnderThreatRate_rabbit_{tag}")

    def _append_per_tag_means(ep_log, iteration_episodes, tags, ep_key_prefix, wandb_key_prefix):
        """Group ep_data['<ep_key_prefix>_<tag>_raw'] by tag, mean across instances
        then mean across episodes, write into ep_log[f'{wandb_key_prefix}_{tag}']."""
        for tag in sorted(set(tags)):
            matching = [j for j, t in enumerate(tags) if t == tag]
            per_ep = []
            for ep in iteration_episodes:
                vals = [ep[f'{ep_key_prefix}_{tags[j]}_raw']
                        for j in matching
                        if f'{ep_key_prefix}_{tags[j]}_raw' in ep]
                if vals:
                    per_ep.append(np.mean(vals))
            if per_ep:
                ep_log[f'{wandb_key_prefix}_{tag}'] = float(np.mean(per_ep))

    # Buffer for episodes that finish across iterations (Stage 3)
    iteration_episodes = []

    # --- Checkpoint Restoration (Continual Learning / Transfer) ---
    if args.load_checkpoint:
        if args.debug: print(f"[DEBUG] Phase 6.5: Restoring Checkpoint from {args.load_checkpoint}...", flush=True)
        if not args.quiet:
            print(f"Restoring checkpoint from {args.load_checkpoint}...")
            
        try:
            restore_mngr = ocp.CheckpointManager(os.path.abspath(args.load_checkpoint))
            # orbax CheckpointManager.latest_step() returns the largest step number
            step = restore_mngr.latest_step()
            
            if step is not None:
                # Load the raw state tree using PyTreeRestore, avoiding StandardRestore shape panic
                restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())
                
                # Check if this is a standard NNX model/optimizer setup, or Dreamer
                if algorithm == "DreamerV3":
                    nnx.update(trainer.agent.wm, restored['wm'])
                    nnx.update(trainer.agent.ac.actor, restored['actor'])
                    nnx.update(trainer.agent.ac.critic, restored['critic'])
                    key = restored['key']
                    global_step = restored['step']
                    iteration = restored['iteration']
                    total_episodes_completed = restored['episode']
                    if schedule is not None:
                        current_stage = restored.get('stage', 0)
                    if not args.quiet: print(f"  -> DreamerV3 Model fully restored (Step: {step}).")
                elif 'model' in locals() and 'optimizer' in locals():
                    # Enforce Strict Architecture Matching
                    restored_model_state = restored['model']
                    current_model_state = nnx.state(model)
                    
                    # We flatten both states and map matching keys/shapes
                    flat_restored, tree_def = jax.tree_util.tree_flatten_with_path(restored_model_state)
                    flat_current, current_def = jax.tree_util.tree_flatten_with_path(current_model_state)
                    
                    # Convert paths to string keys for easy lookup
                    restored_dict = {str(k): v for k, v in flat_restored}
                    current_dict = {str(k): v for k, v in flat_current}
                    
                    mismatches = []
                    
                    # Check for mismatches or missing layers
                    for k, cur_v in current_dict.items():
                        if k not in restored_dict:
                            mismatches.append(f"Layer '{k}': Missing in Checkpoint (Current expects shape {getattr(cur_v, 'shape', 'No Shape')})")
                        else:
                            res_v = restored_dict[k]
                            cur_shape = getattr(cur_v, 'shape', None)
                            res_shape = getattr(res_v, 'shape', None)
                            
                            if cur_shape != res_shape:
                                mismatches.append(f"Layer '{k}': Checkpoint Shape {res_shape} != Current Shape {cur_shape}")
                                
                    for k in restored_dict.keys():
                        if k not in current_dict:
                            res_shape = getattr(restored_dict[k], 'shape', 'No Shape')
                            mismatches.append(f"Layer '{k}': Missing in Current (Checkpoint has shape {res_shape})")

                    if mismatches:
                        error_msg = "Architecture mismatch detected between checkpoint and current environment!\n"
                        error_msg += "The following structure differences were found:\n"
                        error_msg += "\n".join([f"  - {m}" for m in mismatches])
                        raise ValueError(error_msg)
                    
                    # If we survived, the structures are identical. Reconstruct and apply.
                    valid_flat = [restored_dict[str(k)] for k, _ in flat_current]
                    valid_tree = jax.tree_util.tree_unflatten(current_def, valid_flat)
                    
                    nnx.update(model, valid_tree)
                    nnx.update(optimizer, restored['optimizer'])
                    if not args.quiet: print(f"  -> Model and Optimizer strictly matched and fully restored (Step: {step}).")
                    
                    # Also restore standard training counters
                    if 'h_state' in restored: h_state = restored['h_state']
                    if 'key' in restored: key = restored['key']
                    global_step = restored.get('step', global_step)
                    iteration = restored.get('iteration', iteration)
                    total_episodes_completed = restored.get('episode', total_episodes_completed)
                    if schedule is not None:
                        current_stage = restored.get('stage', 0)
                    
            else:
                if not args.quiet: print(f"Warning: No valid checkpoint steps found at {args.load_checkpoint}.")
        except Exception as e:
            if not args.quiet:
                print(f"Error restoring checkpoint: {e}")

    start_time = datetime.now()
    if args.debug: print(f"[DEBUG] Loop start time: {start_time.strftime('%H:%M:%S')}", flush=True)

    def _stage_tag() -> dict:
        """Return stage WandB tag dict; empty in single-config mode."""
        if schedule is None:
            return {}
        return {"stage/index": current_stage,
                "stage/name": schedule.stage_names[current_stage]}

    with tqdm(total=episodes, disable=args.quiet, desc="Training") as pbar:

        try:
            while (total_episodes_completed < episodes) if episodes > 0 else (global_step < total_timesteps):
                if stop_requested:
                    break

                iteration += 1
                if args.debug: print(f"\n[DEBUG] --- Iteration {iteration} Start (Step: {global_step}) ---", flush=True)

                # === CONTINUAL LEARNING: stage-transition check ===
                # Same per-iteration granularity as the checkpoint scheduler; drift is accepted.
                if schedule is not None:
                    new_stage = schedule.stage_for_episode(total_episodes_completed)
                    if new_stage != current_stage:
                        old_name = schedule.stage_names[current_stage]
                        new_name = schedule.stage_names[new_stage]
                        if not args.quiet:
                            pbar.write(f"[STAGE] {current_stage}:{old_name} -> {new_stage}:{new_name} "
                                       f"at ep={total_episodes_completed} "
                                       f"(boundary was {schedule.episode_boundaries[current_stage]})")

                        # Rebuild env with new params.
                        # Model, optimizer, key persist; recurrent state is reset below (Fix 4).
                        # NO forced save here — the periodic scheduler below is
                        # the single source of truth for saves, same drift as today.
                        params = load_env_params(schedule.stage_configs[new_stage])
                        env = ParallelEnv(params)
                        key, reset_key = jax.random.split(key)
                        env_state, obs = env.reset(reset_key, num_envs)
                        # Wipe in-flight episode accumulators.
                        # Mid-episode envs' partial episodes are silently dropped
                        # per user decision.
                        episode_returns[:] = 0.0
                        episode_lengths[:] = 0
                        for _bk in BEHAVIOR_KEYS:
                            episode_behavior[_bk][:] = 0.0
                        for _bk in BEHAVIOR_DIST_KEYS:
                            episode_dist_sums[_bk][:] = 0.0
                        if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[:, :]  = 0.0
                        if num_predator_for_log > 0: episode_dist_per_predator_sums[:, :] = 0.0
                        # Behavior-measure toolkit v1: stage-transition wipe
                        if bm_enabled:
                            m1_candidates[:, :]  = 0; m1_interrupted[:, :]  = 0
                            m2_onsets[:, :]      = 0; m2_dives[:, :]        = 0
                            m5_threat_steps[:, :] = 0; m5_safe_steps[:, :]  = 0
                            m5_eat_threat[:, :]  = 0; m5_eat_safe[:, :]     = 0
                            m1_candidates_tag[:, :]  = 0; m1_interrupted_tag[:, :]  = 0
                            m2_onsets_tag[:, :]      = 0; m2_dives_tag[:, :]        = 0
                            m5_threat_steps_tag[:, :] = 0; m5_safe_steps_tag[:, :]  = 0
                            m5_eat_threat_tag[:, :]  = 0; m5_eat_safe_tag[:, :]     = 0
                            m1_candidate_age[:, :] = -1; m1_candidate_tag_idx[:, :] = -1
                            m2_onset_age[:, :]     = -1; m2_onset_tag_idx[:, :]     = -1
                            m2_in_bush_seen[:, :]  = False
                            m_prev_threat_in_R[:, :]     = False
                            m_prev_threat_in_R_tag[:, :] = False
                            m1_steps_since_eat[:] = 0

                        # --- DreamerV3 only: clear replay buffers to prevent
                        # cross-stage dynamics contamination of the world model.
                        if algorithm == "DreamerV3":
                            # Cheap reset: mark as empty. sample() gates on self.size so
                            # the (now stale) array contents become unreachable.
                            pre_size = buffer.size
                            buffer.idx = 0
                            buffer.size = 0
                            pos_pre_size = 0
                            if positive_buffer is not None:
                                pos_pre_size = positive_buffer.size
                                positive_buffer.idx = 0
                                positive_buffer.size = 0
                            if not args.quiet:
                                pbar.write(f"[STAGE] Cleared Dreamer replay buffer "
                                           f"({pre_size} transitions) and positive buffer "
                                           f"({pos_pre_size} transitions).")
                            if wandb_enabled:
                                wandb.log({
                                    "stage/buffer_cleared_main":     pre_size,
                                    "stage/buffer_cleared_positive": pos_pre_size,
                                    "Episode/Number": total_episodes_completed,
                                })

                        # --- Fix 4: reset agent recurrent state at stage transition.
                        # Symmetric with replay-buffer clearing — the env is fresh, so the
                        # agent's memory of the old env should not contaminate new-stage rollouts.
                        if algorithm == "RecurrentPPO":
                            # Same init as training startup (train.py:717)
                            h_state = model.initial_state(num_envs)
                        elif algorithm == "DreamerV3":
                            # Same init as training startup (train.py:772-780)
                            dreamer_state = trainer.agent.wm.rssm.initial(num_envs)
                            dreamer_state['prev_action'] = jnp.zeros(
                                (num_envs, trainer.agent.ac.actor.net.layers[-1].out_features))
                            dreamer_state['is_first'] = jnp.ones((num_envs, 1))
                            if trainer.agent.wm.modulation_enabled:
                                dreamer_state['mod_h'] = trainer.agent.wm.modulator.initial_state(num_envs)

                        current_stage = new_stage

                        if wandb_enabled:
                            wandb.log({
                                "stage/index":      current_stage,
                                "stage/transition": 1,
                                "Episode/Number":   total_episodes_completed,
                            })
                # ==================================================

                # === PROFILER GATING ===
                if args.profile:
                    if iteration == PROFILE_WARMUP_ITERS + 1:  # i.e. iter 21
                        jax.block_until_ready(env_state.agent_pos)
                        jax.profiler.start_trace(profile_trace_dir)
                        print(f"[PROFILE] start_trace at iter {iteration}", flush=True)
                    if iteration > PROFILE_TOTAL_ITERS:
                        jax.block_until_ready(env_state.agent_pos)
                        jax.profiler.stop_trace()
                        print(f"[PROFILE] trace complete: {profile_trace_dir}", flush=True)
                        break
                # ======================

                # Reset behavior depends on accumulation mode
                if not log_accumulate or (iteration - 1) % log_interval == 0:
                    iteration_episodes = []

                if algorithm == "RecurrentPPO":
                    if args.debug: print(f"  [DEBUG] Collecting {num_steps * num_envs} steps of experience...", end="", flush=True)
                    with jax.named_scope("rppo_train_iteration"):
                        env_state, h_state, key, losses, num_completed, trajectories = jit_train(
                            model, optimizer, params, env_state, h_state, key, ppo_config
                        )
                    
                    step_info = getattr(trajectories, 'step_info', None)
                    # Convert step_info fields to numpy (all shape [T, B])
                    info_np = {}
                    if step_info is not None:
                        for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason']:
                            info_np[k] = np.array(getattr(step_info, k))
                        # Per-instance arrays: shape [num_steps, num_envs, num_entity]
                        if num_neutral_for_log  > 0: info_np['dist_per_neutral']  = np.array(step_info.dist_per_neutral)
                        if num_predator_for_log > 0: info_np['dist_per_predator'] = np.array(step_info.dist_per_predator)
                        # Behavior-measure toolkit v1: agent_in_bush (explicit extraction — Site 1 anti-pattern guard)
                        info_np['agent_in_bush'] = np.array(step_info.agent_in_bush)

                    rollout_rew = trajectories.reward
                    rollout_done = trajectories.done
                    mod_info = trajectories.mod_info
                    if args.debug: print(f" Done.", flush=True)
                    
                    if args.debug and mod_info is not None:
                        z_uni = float(jnp.mean(mod_info.z_unimodal))
                        z_multi = float(jnp.mean(mod_info.z_multimodal))
                        z_mem = float(jnp.mean(mod_info.z_memory))
                        temp = float(jnp.mean(mod_info.temperature)) if hasattr(mod_info, 'temperature') else 1.0
                        print(f"  [Modulator] Mean Uni: {z_uni:.3f}, Multi: {z_multi:.3f}, Mem: {z_mem:.3f}, Temp: {temp:.2f}")
                    
                    steps_this_iter = num_steps * num_envs
                    global_step += steps_this_iter
                    
                    rew_np = np.array(rollout_rew)
                    done_np = np.array(rollout_done)
                    
                    for t in range(num_steps):
                        episode_returns += rew_np[t]
                        episode_lengths += 1

                        if info_np:
                            for k in BEHAVIOR_KEYS:
                                episode_behavior[k] += info_np[k][t]
                            for k in BEHAVIOR_DIST_KEYS:
                                episode_dist_sums[k] += info_np[k][t]
                            # === Per-tag accumulation (Site 1: RecurrentPPO main) ===
                            if 'dist_per_neutral' in info_np and num_neutral_for_log > 0:
                                episode_dist_per_neutral_sums  += info_np['dist_per_neutral'][t]
                            if 'dist_per_predator' in info_np and num_predator_for_log > 0:
                                episode_dist_per_predator_sums += info_np['dist_per_predator'][t]
                            # === Behavior-measure toolkit v1: per-step accumulation (Site 1) ===
                            if bm_enabled and 'agent_in_bush' in info_np:
                                _bm_info_t = {k: info_np[k][t] for k in ('ate_food', 'agent_in_bush')}
                                if 'dist_per_predator' in info_np: _bm_info_t['dist_per_predator'] = info_np['dist_per_predator'][t]
                                if 'dist_per_neutral'  in info_np: _bm_info_t['dist_per_neutral']  = info_np['dist_per_neutral'][t]
                                _bm_step_update(_bm_info_t, done_np[t].astype(bool))

                        dones_t = done_np[t].astype(bool)

                        if np.any(dones_t):
                            completed_indices = np.where(dones_t)[0]
                            for i in completed_indices:
                                total_episodes_completed += 1
                                ep_reward = float(episode_returns[i])
                                ep_length = int(episode_lengths[i])

                                ep_data = {'r': ep_reward, 'l': ep_length}
                                if info_np:
                                    for k in BEHAVIOR_KEYS:
                                        ep_data[k] = float(episode_behavior[k][i])
                                    for k in BEHAVIOR_DIST_KEYS:
                                        ep_data[k] = float(episode_dist_sums[k][i] / max(ep_length, 1))
                                    ep_data['termination_reason'] = int(info_np['termination_reason'][t][i])
                                    # Per-tag finalisation
                                    ep_l_safe = max(ep_length, 1)
                                    if num_neutral_for_log > 0:
                                        means = episode_dist_per_neutral_sums[i] / ep_l_safe
                                        for j, tag in enumerate(neutral_tags):
                                            ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
                                    if num_predator_for_log > 0:
                                        means = episode_dist_per_predator_sums[i] / ep_l_safe
                                        for j, tag in enumerate(predator_tags):
                                            ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])
                                    # Behavior-measure toolkit v1: per-episode finalisation (Site 1)
                                    if bm_enabled:
                                        _bm_finalise_episode(i, ep_data)

                                # Store for moving average (tqdm)
                                ep_info_buffer.append(ep_data)
                                # Store for iteration-level logging (Stage 3)
                                iteration_episodes.append(ep_data)

                                # Reset for next episode in this slot
                                episode_returns[i] = 0.0
                                episode_lengths[i] = 0
                                if info_np:
                                    for k in BEHAVIOR_KEYS:
                                        episode_behavior[k][i] = 0.0
                                    for k in BEHAVIOR_DIST_KEYS:
                                        episode_dist_sums[k][i] = 0.0
                                if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
                                if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
                                # Behavior-measure toolkit v1: per-env reset (Site 1)
                                if bm_enabled:
                                    _bm_reset_env(i)

                    # Log AGGREGATED stats for the iteration (Stage 3)
                    if wandb_enabled and iteration_episodes and iteration % log_interval == 0:
                        rewards = [ep['r'] for ep in iteration_episodes]
                        lengths = [ep['l'] for ep in iteration_episodes]
                        ep_log = {
                            "Episode/Reward": np.mean(rewards),
                            "Episode/Reward_Min": np.min(rewards),
                            "Episode/Reward_Max": np.max(rewards),
                            "Episode/Steps": np.mean(lengths),
                            "Episode/Number": total_episodes_completed,
                            **_stage_tag(),
                        }
                        # Behavioral metrics
                        if 'ate_food' in iteration_episodes[0]:
                            ep_log.update({
                                "Episode/FoodEaten": np.mean([ep['ate_food'] for ep in iteration_episodes]),
                                "Episode/PredatorHits": np.mean([ep['hit_predator'] for ep in iteration_episodes]),
                                # Note: WandB labels like 'Episode/DangerHits' are kept for dashboard-history continuity
                                "Episode/DangerHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                "Episode/RestCount": np.mean([ep['rested'] for ep in iteration_episodes]),
                                "Episode/Collisions": np.mean([ep['event_collided'] for ep in iteration_episodes]),
                                "Episode/TotalDamage": np.mean([ep['damage'] for ep in iteration_episodes]),
                                "Episode/DamagePredator": np.mean([ep['damage_predator'] for ep in iteration_episodes]),
                                "Episode/DamageDanger": np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes]),
                                "Episode/DamageObstacle": np.mean([ep['damage_obstacle'] for ep in iteration_episodes]),
                                "Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in iteration_episodes]),
                                "Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in iteration_episodes]),
                                "Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in iteration_episodes]),
                                "Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in iteration_episodes]),
                                "Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in iteration_episodes]),
                                "Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                            })
                            # Termination reason distribution (fraction of episodes ending each way)
                            term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
                            for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                                ep_log[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
                            # Per-tag fan-out
                            _append_per_tag_means(ep_log, iteration_episodes, neutral_tags,
                                                  'mean_dist_rabbit',   'Episode/MeanDistRabbit')
                            _append_per_tag_means(ep_log, iteration_episodes, predator_tags,
                                                  'mean_dist_predator', 'Episode/MeanDistPredator')
                            # Behavior-measure toolkit v1: WandB fan-out (Site 1)
                            if bm_enabled:
                                _bm_log_wandb(ep_log, iteration_episodes)
                        wandb.log(ep_log)

                    # Update progress bar based on total episodes completed
                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.refresh()

                    avg_policy_loss = jnp.mean(jnp.array([l[1][0] for l in losses]))
                    avg_value_loss = jnp.mean(jnp.array([l[1][1] for l in losses]))
                    avg_ent_loss = jnp.mean(jnp.array([l[1][2] for l in losses]))
                    avg_grad_norm = jnp.mean(jnp.array([l[1][3] for l in losses]))
                    avg_mod_grad_norm = jnp.mean(jnp.array([l[1][4] for l in losses]))
                    total_loss = jnp.mean(jnp.array([l[0] for l in losses]))
                    
                    if wandb_enabled and iteration % log_interval == 0:
                        wandb_logs = {
                            "loss/total": total_loss,
                            "loss/policy": avg_policy_loss,
                            "loss/value": avg_value_loss,
                            "loss/entropy": avg_ent_loss,
                            "loss/grad_norm": avg_grad_norm,
                        }

                        # Add Modulator metrics if enabled
                        if mod_info is not None:
                            wandb_logs.update({
                                "modulator/grad_norm": float(avg_mod_grad_norm),
                                "modulator/gamma_uni_mean": float(jnp.mean(mod_info.z_unimodal)),
                                "modulator/gamma_uni_std": float(jnp.std(mod_info.z_unimodal)),
                                "modulator/gamma_multi_mean": float(jnp.mean(mod_info.z_multimodal)),
                                "modulator/gamma_multi_std": float(jnp.std(mod_info.z_multimodal)),
                                "modulator/z_memory_mean": float(jnp.mean(mod_info.z_memory)),
                                "modulator/z_memory_std": float(jnp.std(mod_info.z_memory)),
                                "modulator/temperature_mean": float(jnp.mean(mod_info.temperature)),
                                "modulator/temperature_min": float(jnp.min(mod_info.temperature)),
                                "modulator/temperature_max": float(jnp.max(mod_info.temperature)),
                            })
                            if modulation_config is not None and modulation_config.get('type') in ("PreActivation", "FiLM"):
                                wandb_logs.update({
                                    "modulator/beta_uni_mean": float(jnp.mean(mod_info.z_unimodal_add)),
                                    "modulator/beta_uni_std": float(jnp.std(mod_info.z_unimodal_add)),
                                    "modulator/beta_multi_mean": float(jnp.mean(mod_info.z_multimodal_add)),
                                    "modulator/beta_multi_std": float(jnp.std(mod_info.z_multimodal_add)),
                                })
                        
                        wandb_logs.update({
                            "timesteps": global_step,
                            "iteration": iteration,
                            **_stage_tag(),
                        })
                        wandb.log(wandb_logs)

                    postfix = {
                        "Iter": iteration,
                        "Loss": f"{total_loss:.4f}",
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    }
                    if mod_info is not None:
                        postfix["T"] = f"{float(jnp.mean(mod_info.temperature)):.2f}"
                    pbar.set_postfix(postfix)
                        
                elif algorithm == "DreamerV3":
                    # Use JITTED collect_sequence (collect_interval steps per env per iteration)
                    key, collect_key = jax.random.split(key)
                    with jax.named_scope("dreamer_collect_sequence"):
                        env_state, dreamer_state, key, transitions = trainer.collect_sequence(
                            env_state, params, num_steps, collect_key, dreamer_state)

                    # Convert transitions to NumPy and add to buffer.
                    if buffer.device == "gpu":
                        # STAY ON GPU: perform transpose/reshape in JAX (Zero Copy)
                        T, B = transitions['obs'].shape[0], transitions['obs'].shape[1]
                        with jax.named_scope("dreamer_buffer_add"):
                            obs_flat = transitions['obs'].transpose(1, 0, 2).reshape(B * T, -1)
                            act_flat = transitions['action'].transpose(1, 0, 2).reshape(B * T, -1)
                            rew_flat = transitions['reward'].transpose(1, 0).reshape(B * T)
                            done_flat = transitions['terminal'].transpose(1, 0).reshape(B * T)
                            is_first_arr = transitions['is_first']
                            if is_first_arr.ndim == 3:
                                is_first_flat = is_first_arr.transpose(1, 0, 2).reshape(B * T)
                            else:
                                is_first_flat = is_first_arr.transpose(1, 0).reshape(B * T)
                            buffer.add_batch(obs_flat, act_flat, rew_flat, done_flat, is_first_flat)

                        with jax.named_scope("dreamer_positive_buffer_copy"):
                            # Copy positive-reward blocks to the dedicated positive buffer
                            if positive_buffer is not None:
                                seq_len = buffer.sequence_length
                                num_items = obs_flat.shape[0]
                                num_written_blocks = num_items // seq_len

                                for b in range(num_written_blocks):
                                    blk_start = b * seq_len
                                    blk_end = blk_start + seq_len
                                    blk_rewards = rew_flat[blk_start:blk_end]

                                    # Check if this block contains any positive reward
                                    if buffer._on_gpu:
                                        has_positive = bool(jnp.any(blk_rewards > 0.0))
                                    else:
                                        has_positive = bool(np.any(blk_rewards > 0.0))

                                    if has_positive:
                                        positive_buffer.add_batch(
                                            obs_flat[blk_start:blk_end],
                                            act_flat[blk_start:blk_end],
                                            rew_flat[blk_start:blk_end],
                                            done_flat[blk_start:blk_end],
                                            is_first_flat[blk_start:blk_end]
                                        )
                        # Still need numpy for cpu-side stats calculation
                        transitions_np = jax.device_get(transitions)
                    else:
                        # CPU path: existing logic
                        transitions_np = jax.device_get(transitions)
                        T, B = transitions_np['obs'].shape[0], transitions_np['obs'].shape[1]
                        with jax.named_scope("dreamer_buffer_add"):
                            obs_flat = transitions_np['obs'].transpose(1, 0, 2).reshape(B * T, -1)
                            act_flat = transitions_np['action'].transpose(1, 0, 2).reshape(B * T, -1)
                            rew_flat = transitions_np['reward'].transpose(1, 0).reshape(B * T)
                            done_flat = transitions_np['terminal'].transpose(1, 0).reshape(B * T)
                            is_first_arr = transitions_np['is_first'].astype(bool)
                            if is_first_arr.ndim == 3:
                                is_first_flat = is_first_arr.transpose(1, 0, 2).reshape(B * T)
                            else:
                                is_first_flat = is_first_arr.transpose(1, 0).reshape(B * T)
                            buffer.add_batch(obs_flat, act_flat, rew_flat, done_flat, is_first_flat)

                        with jax.named_scope("dreamer_positive_buffer_copy"):
                            # Copy positive-reward blocks to the dedicated positive buffer
                            if positive_buffer is not None:
                                seq_len = buffer.sequence_length
                                num_items = obs_flat.shape[0]
                                num_written_blocks = num_items // seq_len

                                for b in range(num_written_blocks):
                                    blk_start = b * seq_len
                                    blk_end = blk_start + seq_len
                                    blk_rewards = rew_flat[blk_start:blk_end]

                                    # Check if this block contains any positive reward
                                    has_positive = bool(np.any(blk_rewards > 0.0))

                                    if has_positive:
                                        positive_buffer.add_batch(
                                            obs_flat[blk_start:blk_end],
                                            act_flat[blk_start:blk_end],
                                            rew_flat[blk_start:blk_end],
                                            done_flat[blk_start:blk_end],
                                            is_first_flat[blk_start:blk_end]
                                        )
                    
                    # Update statistics (Vectorized where possible)
                    rew_steps = transitions_np['reward'] # (T, B)
                    done_steps = transitions_np['terminal'] # (T, B)

                    # Extract behavioral info arrays [T, B] (or [T, B, num_entity] for per-instance)
                    info_steps = {}
                    for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason']:
                        if k in transitions_np:
                            info_steps[k] = transitions_np[k]
                    # Per-instance keys: shape [T, B, num_entity]
                    dist_per_neutral_steps  = transitions_np.get('dist_per_neutral')   # [T, B, num_neutral] or None
                    dist_per_predator_steps = transitions_np.get('dist_per_predator')  # [T, B, num_predator] or None
                    # Behavior-measure toolkit v1: agent_in_bush (Site 2 direct extraction)
                    agent_in_bush_steps = transitions_np.get('agent_in_bush')          # [T, B] or None

                    # Behavior-measure toolkit v1: per-step sequential update for Site 2 (DreamerV3 batch)
                    # The K-buffer must be driven step-by-step so done-discarding is episode-accurate.
                    if bm_enabled and agent_in_bush_steps is not None:
                        T2, B2 = done_steps.shape
                        for t2 in range(T2):
                            _bm_info_t2 = {
                                'ate_food': transitions_np['ate_food'][t2].astype(bool),
                                'agent_in_bush': agent_in_bush_steps[t2].astype(bool),
                            }
                            if dist_per_predator_steps is not None: _bm_info_t2['dist_per_predator'] = dist_per_predator_steps[t2]
                            if dist_per_neutral_steps  is not None: _bm_info_t2['dist_per_neutral']  = dist_per_neutral_steps[t2]
                            _bm_step_update(_bm_info_t2, done_steps[t2].astype(bool))

                    # More vectorized stats handling
                    done_indices = np.where(done_steps) # (t_idxs, env_idxs)

                    if done_indices[0].size > 0:
                        # Track episode returns/lengths
                        for i in np.unique(done_indices[1]):
                            d_idxs = done_indices[0][done_indices[1] == i]
                            curr_start = 0
                            for d_idx in d_idxs:
                                ep_reward = float(episode_returns[i] + np.sum(rew_steps[curr_start:d_idx+1, i]))
                                ep_length = int(episode_lengths[i] + (d_idx + 1 - curr_start))

                                ep_data = {'r': ep_reward, 'l': ep_length}
                                if info_steps:
                                    for k in BEHAVIOR_KEYS:
                                        ep_data[k] = float(episode_behavior[k][i] + np.sum(info_steps[k][curr_start:d_idx+1, i]))
                                    for k in BEHAVIOR_DIST_KEYS:
                                        ep_data[k] = float((episode_dist_sums[k][i] + np.sum(info_steps[k][curr_start:d_idx+1, i])) / max(ep_length, 1))
                                    ep_data['termination_reason'] = int(info_steps['termination_reason'][d_idx, i])
                                    # Per-tag finalisation (Site 2: DreamerV3 batch)
                                    ep_l_safe = max(ep_length, 1)
                                    if num_neutral_for_log > 0 and dist_per_neutral_steps is not None:
                                        means = (episode_dist_per_neutral_sums[i] + np.sum(dist_per_neutral_steps[curr_start:d_idx+1, i], axis=0)) / ep_l_safe
                                        for j, tag in enumerate(neutral_tags):
                                            ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
                                    if num_predator_for_log > 0 and dist_per_predator_steps is not None:
                                        means = (episode_dist_per_predator_sums[i] + np.sum(dist_per_predator_steps[curr_start:d_idx+1, i], axis=0)) / ep_l_safe
                                        for j, tag in enumerate(predator_tags):
                                            ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])
                                    # Behavior-measure toolkit v1: per-episode finalisation (Site 2)
                                    if bm_enabled:
                                        _bm_finalise_episode(i, ep_data)

                                ep_info_buffer.append(ep_data)
                                iteration_episodes.append(ep_data)
                                total_episodes_completed += 1
                                episode_returns[i] = 0
                                episode_lengths[i] = 0
                                if info_steps:
                                    for k in BEHAVIOR_KEYS:
                                        episode_behavior[k][i] = 0.0
                                    for k in BEHAVIOR_DIST_KEYS:
                                        episode_dist_sums[k][i] = 0.0
                                if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
                                if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
                                # Behavior-measure toolkit v1: per-env reset (Site 2)
                                if bm_enabled:
                                    _bm_reset_env(i)
                                curr_start = d_idx + 1

                            # Add leftover
                            if curr_start < num_steps:
                                episode_returns[i] += np.sum(rew_steps[curr_start:, i])
                                episode_lengths[i] += (num_steps - curr_start)
                                if info_steps:
                                    for k in BEHAVIOR_KEYS:
                                        episode_behavior[k][i] += np.sum(info_steps[k][curr_start:, i])
                                    for k in BEHAVIOR_DIST_KEYS:
                                        episode_dist_sums[k][i] += np.sum(info_steps[k][curr_start:, i])
                                if num_neutral_for_log  > 0 and dist_per_neutral_steps is not None:
                                    episode_dist_per_neutral_sums[i]  += np.sum(dist_per_neutral_steps[curr_start:, i], axis=0)
                                if num_predator_for_log > 0 and dist_per_predator_steps is not None:
                                    episode_dist_per_predator_sums[i] += np.sum(dist_per_predator_steps[curr_start:, i], axis=0)

                        # Environments with NO dones in this batch
                        no_done_mask = np.ones(num_envs, dtype=bool)
                        no_done_mask[done_indices[1]] = False
                        episode_returns[no_done_mask] += np.sum(rew_steps[:, no_done_mask], axis=0)
                        episode_lengths[no_done_mask] += num_steps
                        if info_steps:
                            for k in BEHAVIOR_KEYS:
                                episode_behavior[k][no_done_mask] += np.sum(info_steps[k][:, no_done_mask], axis=0)
                            for k in BEHAVIOR_DIST_KEYS:
                                episode_dist_sums[k][no_done_mask] += np.sum(info_steps[k][:, no_done_mask], axis=0)
                        if num_neutral_for_log  > 0 and dist_per_neutral_steps is not None:
                            episode_dist_per_neutral_sums[no_done_mask]  += np.sum(dist_per_neutral_steps[:, no_done_mask], axis=0)
                        if num_predator_for_log > 0 and dist_per_predator_steps is not None:
                            episode_dist_per_predator_sums[no_done_mask] += np.sum(dist_per_predator_steps[:, no_done_mask], axis=0)
                    else:
                        # No episodes finished at all
                        episode_returns += np.sum(rew_steps, axis=0)
                        episode_lengths += num_steps
                        if info_steps:
                            for k in BEHAVIOR_KEYS:
                                episode_behavior[k] += np.sum(info_steps[k], axis=0)
                            for k in BEHAVIOR_DIST_KEYS:
                                episode_dist_sums[k] += np.sum(info_steps[k], axis=0)
                        if num_neutral_for_log  > 0 and dist_per_neutral_steps is not None:
                            episode_dist_per_neutral_sums  += np.sum(dist_per_neutral_steps, axis=0)
                        if num_predator_for_log > 0 and dist_per_predator_steps is not None:
                            episode_dist_per_predator_sums += np.sum(dist_per_predator_steps, axis=0)

                    global_step += num_envs * num_steps

                    if wandb_enabled and iteration_episodes and iteration % log_interval == 0:
                        rewards = [ep['r'] for ep in iteration_episodes]
                        lengths = [ep['l'] for ep in iteration_episodes]
                        ep_log = {
                            "Episode/Reward": np.mean(rewards),
                            "Episode/Reward_Min": np.min(rewards),
                            "Episode/Reward_Max": np.max(rewards),
                            "Episode/Steps": np.mean(lengths),
                            "Episode/Number": total_episodes_completed,
                            **_stage_tag(),
                        }
                        # Behavioral metrics
                        if 'ate_food' in iteration_episodes[0]:
                            ep_log.update({
                                "Episode/FoodEaten": np.mean([ep['ate_food'] for ep in iteration_episodes]),
                                "Episode/PredatorHits": np.mean([ep['hit_predator'] for ep in iteration_episodes]),
                                # Note: WandB labels like 'Episode/DangerHits' are kept for dashboard-history continuity
                                "Episode/DangerHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                "Episode/RestCount": np.mean([ep['rested'] for ep in iteration_episodes]),
                                "Episode/Collisions": np.mean([ep['event_collided'] for ep in iteration_episodes]),
                                "Episode/TotalDamage": np.mean([ep['damage'] for ep in iteration_episodes]),
                                "Episode/DamagePredator": np.mean([ep['damage_predator'] for ep in iteration_episodes]),
                                "Episode/DamageDanger": np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes]),
                                "Episode/DamageObstacle": np.mean([ep['damage_obstacle'] for ep in iteration_episodes]),
                                "Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in iteration_episodes]),
                                "Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in iteration_episodes]),
                                "Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in iteration_episodes]),
                                "Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in iteration_episodes]),
                                "Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in iteration_episodes]),
                                "Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                            })
                            # Termination reason distribution (fraction of episodes ending each way)
                            term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
                            for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                                ep_log[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
                            # Per-tag fan-out
                            _append_per_tag_means(ep_log, iteration_episodes, neutral_tags,
                                                  'mean_dist_rabbit',   'Episode/MeanDistRabbit')
                            _append_per_tag_means(ep_log, iteration_episodes, predator_tags,
                                                  'mean_dist_predator', 'Episode/MeanDistPredator')
                            # Behavior-measure toolkit v1: WandB fan-out (Site 2)
                            if bm_enabled:
                                _bm_log_wandb(ep_log, iteration_episodes)
                        wandb.log(ep_log)

                    # Update progress bar
                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.refresh()

                    metrics = {}
                    loss_msg = ""
                    with jax.named_scope("dreamer_train_multiple"):
                        if buffer.size > max(config.get_mandatory('agent.batch_size') * 2, config.get_mandatory('agent.sequence_length')):
                            # Dynamic gradient steps based on replay_ratio.
                            # With collect_interval=1 (sheeprl-style): global_step increments by num_envs per iter,
                            # ratio returns num_envs gradient steps. With collect_interval=128: increments by
                            # num_envs*128, so we normalize to count sequences, not individual timesteps.
                            train_steps = ratio_scaled_updates(global_step // num_steps)

                            if buffer.device == "gpu":
                                # GPU path: sample + train all inside one JIT call
                                metrics, key = trainer.train_multiple_gpu(buffer, train_steps, key,
                                                                           positive_buffer=positive_buffer)
                            else:
                                # CPU path: pre-sample on CPU, bulk transfer, then JIT train
                                if config.get_mandatory('agent.sampling_mode') == 'mixture':
                                    stacked = trainer._sample_mixture_cpu(buffer, positive_buffer, train_steps, config.get_mandatory('agent.batch_size'))
                                else:
                                    stacked = buffer.sample_multiple(train_steps, config.get_mandatory('agent.batch_size'))
                                metrics, key = trainer.train_multiple_cpu(stacked, key)

                            cumulative_gradient_steps += train_steps
                            loss_msg = f"L: {metrics.get('loss_model', 0):.2f}"
                    
                    if wandb_enabled and iteration % log_interval == 0:
                        wandb_logs = {
                            "timesteps": global_step,                             "iteration": iteration,
                             "Params/effective_replay_ratio": cumulative_gradient_steps / max(1, global_step)
                        }
                        if positive_buffer is not None:
                            pos_blocks = positive_buffer.size // positive_buffer.sequence_length
                            pos_cap_blocks = positive_buffer.capacity // positive_buffer.sequence_length
                            wandb_logs.update({
                                "Params/positive_buffer_blocks": pos_blocks,
                                "Params/positive_buffer_utilization": pos_blocks / max(pos_cap_blocks, 1),
                                "Params/main_buffer_blocks": buffer.size // buffer.sequence_length,
                            })
                        for mk, mv in metrics.items():
                            if mk.startswith('loss_actor') or mk.startswith('loss_critic') or \
                               mk.startswith('mean_') or mk.startswith('entropy'):
                                wandb_logs[f"Behavior/{mk}"] = float(mv)
                            elif mk.startswith('loss_model') or mk.startswith('loss_recon') or \
                                 mk.startswith('loss_kl') or mk.startswith('loss_rew') or \
                                 mk.startswith('loss_cont') or mk.startswith('loss_dyn') or \
                                 mk.startswith('loss_rep') or mk.startswith('model_') or \
                                 mk.startswith('imagined_'):
                                wandb_logs[f"WorldModel/{mk}"] = float(mv)
                            elif mk.startswith('mod_'):
                                wandb_logs[f"Modulator/{mk}"] = float(mv)
                            else:
                                wandb_logs[mk] = float(mv)
                        wandb_logs.update(_stage_tag())
                        wandb.log(wandb_logs)
                    
                    postfix = {
                        "Iter": iteration,
                        "Loss": loss_msg,
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}",
                    }
                    if metrics:
                        if 'model_reward_mae' in metrics:
                            postfix["R_MAE"] = f"{float(metrics['model_reward_mae']):.3f}"
                        if 'mean_entropy' in metrics:
                            postfix["Ent"] = f"{float(metrics['mean_entropy']):.2f}"
                        if dreamer_mod_config is not None and 'mod_z_reward_mean' in metrics:
                            postfix["R_mod"] = f"{float(metrics['mod_z_reward_mean']):.2f}"
                    pbar.set_postfix(postfix)
                    if args.debug: print(f" Done.", flush=True)

                elif algorithm == "DQN":
                    if args.debug: print(f"  [DEBUG] DQN Step Collection...", end="", flush=True)
                    
                    # 1. Collect Step
                    key, act_key = jax.random.split(key)
                    # vmapped action selection
                    action = jax.vmap(get_action_dqn_nnx, in_axes=(None, 0, 0, None))(
                        model, obs, jax.random.split(act_key, num_envs), epsilon
                    )
                    
                    from src.environment.core import jax_step
                    step_fn = jax.vmap(lambda s, a: jax_step(s, a, params))
                    next_env_state, reward, done, info = step_fn(env_state, action)
                    
                    next_obs = jax.vmap(get_observation, in_axes=(0, None))(next_env_state, params)
                    
                    # 2. Add to Buffer
                    dqn_buffer.add(
                        obs=np.array(obs),
                        action=np.array(action),
                        reward=np.array(reward),
                        next_obs=np.array(next_obs),
                        done=np.array(done)
                    )
                    
                    # 3. Auto-Reset and state transition
                    from src.environment.core import jax_reset
                    reset_key, key = jax.random.split(key)
                    reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(params, jax.random.split(reset_key, num_envs))
                    
                    def select_done(d, r, n):
                        d_expanded = d.reshape((d.shape[0],) + (1,) * (r.ndim - 1))
                        return jnp.where(d_expanded, r, n)
                        
                    env_state = jax.tree_util.tree_map(
                        lambda r, n: select_done(done, r, n),
                        reset_state, next_env_state
                    )
                    obs = jax.vmap(get_observation, in_axes=(0, None))(env_state, params)
                    
                    global_step += num_envs
                    
                    # Stats tracking
                    episode_returns += np.array(reward)
                    episode_lengths += 1
                    
                    if info:
                        info_np_step = {k: np.array(info[k]) for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason']}
                        for k in BEHAVIOR_KEYS:
                            episode_behavior[k] += info_np_step[k]
                        for k in BEHAVIOR_DIST_KEYS:
                            episode_dist_sums[k] += info_np_step[k]
                        # === Per-tag accumulation (Site 3: DQN step) ===
                        if 'dist_per_neutral' in info and num_neutral_for_log > 0:
                            episode_dist_per_neutral_sums  += np.array(info['dist_per_neutral'])
                        if 'dist_per_predator' in info and num_predator_for_log > 0:
                            episode_dist_per_predator_sums += np.array(info['dist_per_predator'])
                        # === Behavior-measure toolkit v1: per-step accumulation (Site 3: DQN) ===
                        if bm_enabled and 'agent_in_bush' in info:
                            _bm_info_dqn = {'ate_food': np.array(info['ate_food']).astype(bool),
                                            'agent_in_bush': np.array(info['agent_in_bush']).astype(bool)}
                            if 'dist_per_predator' in info: _bm_info_dqn['dist_per_predator'] = np.array(info['dist_per_predator'])
                            if 'dist_per_neutral'  in info: _bm_info_dqn['dist_per_neutral']  = np.array(info['dist_per_neutral'])
                            _bm_step_update(_bm_info_dqn, np.array(done).astype(bool))

                    dones_np = np.array(done).astype(bool)
                    if np.any(dones_np):
                        completed_indices = np.where(dones_np)[0]
                        for i in completed_indices:
                            total_episodes_completed += 1
                            ep_reward = float(episode_returns[i])
                            ep_length = int(episode_lengths[i])

                            ep_data = {'r': ep_reward, 'l': ep_length}
                            if info:
                                for k in BEHAVIOR_KEYS:
                                    ep_data[k] = float(episode_behavior[k][i])
                                for k in BEHAVIOR_DIST_KEYS:
                                    ep_data[k] = float(episode_dist_sums[k][i] / max(ep_length, 1))
                                ep_data['termination_reason'] = int(info_np_step['termination_reason'][i])
                                # Per-tag finalisation
                                ep_l_safe = max(ep_length, 1)
                                if num_neutral_for_log > 0:
                                    means = episode_dist_per_neutral_sums[i] / ep_l_safe
                                    for j, tag in enumerate(neutral_tags):
                                        ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
                                if num_predator_for_log > 0:
                                    means = episode_dist_per_predator_sums[i] / ep_l_safe
                                    for j, tag in enumerate(predator_tags):
                                        ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])
                                # Behavior-measure toolkit v1: per-episode finalisation (Site 3: DQN)
                                if bm_enabled:
                                    _bm_finalise_episode(i, ep_data)

                            ep_info_buffer.append(ep_data)
                            iteration_episodes.append(ep_data)
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0
                            if info:
                                for k in BEHAVIOR_KEYS:
                                    episode_behavior[k][i] = 0.0
                                for k in BEHAVIOR_DIST_KEYS:
                                    episode_dist_sums[k][i] = 0.0
                            if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
                            if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
                            # Behavior-measure toolkit v1: per-env reset (Site 3: DQN)
                            if bm_enabled:
                                _bm_reset_env(i)

                    # 4. Update Step
                    loss_val = 0.0
                    if dqn_buffer.size > batch_size:
                        key, sample_key = jax.random.split(key)
                        batch = dqn_buffer.sample(batch_size, sample_key)
                        loss_val = update_step_dqn(model, target_model, optimizer, batch, config.get_mandatory('agent.gamma'))
                        
                        # Epsilon Decay
                        epsilon = max(epsilon_end, epsilon * epsilon_decay)
                        
                        # Target Sync
                        if iteration % target_update_freq == 0:
                            nnx.update(target_model, nnx.state(model))
                    
                    if wandb_enabled and iteration % log_interval == 0:
                        logs = {
                            "iteration": iteration,
                            "timesteps": global_step,
                            "train/epsilon": float(epsilon),
                            "loss/dqn": float(loss_val)
                        }
                        if iteration_episodes:
                            rewards_list = [ep['r'] for ep in iteration_episodes]
                            lengths_list = [ep['l'] for ep in iteration_episodes]
                            ep_logs = {
                                "Episode/Reward": np.mean(rewards_list),
                                "Episode/Reward_Min": np.min(rewards_list),
                                "Episode/Reward_Max": np.max(rewards_list),
                                "Episode/Steps": np.mean(lengths_list),
                                "Episode/Number": total_episodes_completed
                            }
                            # Behavioral metrics
                            if 'ate_food' in iteration_episodes[0]:
                                ep_logs.update({
                                    "Episode/FoodEaten": np.mean([ep['ate_food'] for ep in iteration_episodes]),
                                    "Episode/PredatorHits": np.mean([ep['hit_predator'] for ep in iteration_episodes]),
                                    # Note: WandB labels like 'Episode/DangerHits' are kept for dashboard-history continuity
                                    "Episode/DangerHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/RestCount": np.mean([ep['rested'] for ep in iteration_episodes]),
                                    "Episode/Collisions": np.mean([ep['event_collided'] for ep in iteration_episodes]),
                                    "Episode/TotalDamage": np.mean([ep['damage'] for ep in iteration_episodes]),
                                    "Episode/DamagePredator": np.mean([ep['damage_predator'] for ep in iteration_episodes]),
                                    "Episode/DamageDanger": np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/DamageObstacle": np.mean([ep['damage_obstacle'] for ep in iteration_episodes]),
                                    "Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in iteration_episodes]),
                                    "Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in iteration_episodes]),
                                    "Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in iteration_episodes]),
                                    "Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in iteration_episodes]),
                                    "Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                })
                                # Termination reason distribution
                                term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
                                for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                                    ep_logs[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
                                # Per-tag fan-out (Site 3: DQN)
                                _append_per_tag_means(ep_logs, iteration_episodes, neutral_tags,
                                                      'mean_dist_rabbit',   'Episode/MeanDistRabbit')
                                _append_per_tag_means(ep_logs, iteration_episodes, predator_tags,
                                                      'mean_dist_predator', 'Episode/MeanDistPredator')
                                # Behavior-measure toolkit v1: WandB fan-out (Site 3: DQN)
                                if bm_enabled:
                                    _bm_log_wandb(ep_logs, iteration_episodes)

                            wandb.log(ep_logs)

                        wandb.log(logs)

                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.set_postfix({
                        "Iter": iteration,
                        "Loss": f"{float(loss_val):.4f}",
                        "Eps": f"{float(epsilon):.2f}",
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    })
                    pbar.refresh()

                elif algorithm == "DRQN":
                    if args.debug: print(f"  [DEBUG] DRQN Step Collection...", end="", flush=True)
                    
                    # 1. Collect Step
                    key, act_key = jax.random.split(key)
                    # vmapped action selection
                    # h_state is a PyTree (B, H) or ((B, H), (B, H))
                    # we vmap over first dimension (which is B)
                    if rnn_type.upper() == "LSTM":
                        action, h_state_new = jax.vmap(get_action_drqn_nnx, in_axes=(None, 0, (0, 0), 0, None))(
                            model, obs, h_state, jax.random.split(act_key, num_envs), epsilon
                        )
                    else:
                        action, h_state_new = jax.vmap(get_action_drqn_nnx, in_axes=(None, 0, 0, 0, None))(
                            model, obs, h_state, jax.random.split(act_key, num_envs), epsilon
                        )
                    
                    from src.environment.core import jax_step
                    step_fn = jax.vmap(lambda s, a: jax_step(s, a, params))
                    next_env_state, reward, done, info = step_fn(env_state, action)
                    
                    # 2. Add to Buffer
                    drqn_buffer.add(
                        obs=np.array(obs),
                        action=np.array(action),
                        reward=np.array(reward),
                        done=np.array(done)
                    )
                    
                    # 3. Auto-Reset and state transition
                    from src.environment.core import jax_reset
                    reset_key, key = jax.random.split(key)
                    reset_state = jax.vmap(jax_reset, in_axes=(None, 0))(params, jax.random.split(reset_key, num_envs))
                    
                    def select_done(d, r, n):
                        d_expanded = d.reshape((d.shape[0],) + (1,) * (r.ndim - 1))
                        return jnp.where(d_expanded, r, n)
                        
                    env_state = jax.tree_util.tree_map(
                        lambda r, n: select_done(done, r, n),
                        reset_state, next_env_state
                    )
                    obs = jax.vmap(get_observation, in_axes=(0, None))(env_state, params)
                    
                    # Reset hidden state for completed envs
                    if rnn_type.upper() == "LSTM":
                        h_state = (
                            jnp.where(done[:, None], 0.0, h_state_new[0]),
                            jnp.where(done[:, None], 0.0, h_state_new[1])
                        )
                    else:
                        h_state = jnp.where(done[:, None], 0.0, h_state_new)
                    
                    global_step += num_envs
                    
                    # Stats tracking
                    episode_returns += np.array(reward)
                    episode_lengths += 1
                    
                    if info:
                        info_np_step = {k: np.array(info[k]) for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason']}
                        for k in BEHAVIOR_KEYS:
                            episode_behavior[k] += info_np_step[k]
                        for k in BEHAVIOR_DIST_KEYS:
                            episode_dist_sums[k] += info_np_step[k]
                        # === Per-tag accumulation (Site 4: DRQN step) ===
                        if 'dist_per_neutral' in info and num_neutral_for_log > 0:
                            episode_dist_per_neutral_sums  += np.array(info['dist_per_neutral'])
                        if 'dist_per_predator' in info and num_predator_for_log > 0:
                            episode_dist_per_predator_sums += np.array(info['dist_per_predator'])
                        # === Behavior-measure toolkit v1: per-step accumulation (Site 4: DRQN) ===
                        if bm_enabled and 'agent_in_bush' in info:
                            _bm_info_drqn = {'ate_food': np.array(info['ate_food']).astype(bool),
                                             'agent_in_bush': np.array(info['agent_in_bush']).astype(bool)}
                            if 'dist_per_predator' in info: _bm_info_drqn['dist_per_predator'] = np.array(info['dist_per_predator'])
                            if 'dist_per_neutral'  in info: _bm_info_drqn['dist_per_neutral']  = np.array(info['dist_per_neutral'])
                            _bm_step_update(_bm_info_drqn, np.array(done).astype(bool))

                    dones_np = np.array(done).astype(bool)
                    if np.any(dones_np):
                        completed_indices = np.where(dones_np)[0]
                        for i in completed_indices:
                            total_episodes_completed += 1
                            ep_reward = float(episode_returns[i])
                            ep_length = int(episode_lengths[i])

                            ep_data = {'r': ep_reward, 'l': ep_length}
                            if info:
                                for k in BEHAVIOR_KEYS:
                                    ep_data[k] = float(episode_behavior[k][i])
                                for k in BEHAVIOR_DIST_KEYS:
                                    ep_data[k] = float(episode_dist_sums[k][i] / max(ep_length, 1))
                                ep_data['termination_reason'] = int(info_np_step['termination_reason'][i])
                                # Per-tag finalisation
                                ep_l_safe = max(ep_length, 1)
                                if num_neutral_for_log > 0:
                                    means = episode_dist_per_neutral_sums[i] / ep_l_safe
                                    for j, tag in enumerate(neutral_tags):
                                        ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
                                if num_predator_for_log > 0:
                                    means = episode_dist_per_predator_sums[i] / ep_l_safe
                                    for j, tag in enumerate(predator_tags):
                                        ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])
                                # Behavior-measure toolkit v1: per-episode finalisation (Site 4: DRQN)
                                if bm_enabled:
                                    _bm_finalise_episode(i, ep_data)

                            ep_info_buffer.append(ep_data)
                            iteration_episodes.append(ep_data)
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0
                            if info:
                                for k in BEHAVIOR_KEYS:
                                    episode_behavior[k][i] = 0.0
                                for k in BEHAVIOR_DIST_KEYS:
                                    episode_dist_sums[k][i] = 0.0
                            if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
                            if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
                            # Behavior-measure toolkit v1: per-env reset (Site 4: DRQN)
                            if bm_enabled:
                                _bm_reset_env(i)

                    # 4. Update Step
                    loss_val = 0.0
                    if drqn_buffer.size > (trace_length + burn_in_length + batch_size):
                        key, sample_key = jax.random.split(key)
                        obs_seq, actions, rewards, next_obs_seq, dones = drqn_buffer.sample_sequences(
                            batch_size, trace_length + burn_in_length, sample_key
                        )
                        loss_val = update_step_drqn(
                            model, target_model, optimizer, 
                            obs_seq, actions, rewards, next_obs_seq, dones, 
                            config.get_mandatory('agent.gamma'), 
                            burn_in_length
                        )
                        
                        # Epsilon Decay
                        epsilon = max(epsilon_end, epsilon * epsilon_decay)
                        
                        # Target Sync
                        if iteration % target_update_freq == 0:
                            nnx.update(target_model, nnx.state(model))
                    
                    if wandb_enabled and iteration % log_interval == 0:
                        logs = {
                            "iteration": iteration,
                            "timesteps": global_step,
                            "train/epsilon": float(epsilon),
                            "loss/drqn": float(loss_val)
                        }
                        if iteration_episodes:
                            rewards_list = [ep['r'] for ep in iteration_episodes]
                            lengths_list = [ep['l'] for ep in iteration_episodes]
                            ep_logs = {
                                "Episode/Reward": np.mean(rewards_list),
                                "Episode/Reward_Min": np.min(rewards_list),
                                "Episode/Reward_Max": np.max(rewards_list),
                                "Episode/Steps": np.mean(lengths_list),
                                "Episode/Number": total_episodes_completed
                            }
                            # Behavioral metrics
                            if 'ate_food' in iteration_episodes[0]:
                                ep_logs.update({
                                    "Episode/FoodEaten": np.mean([ep['ate_food'] for ep in iteration_episodes]),
                                    "Episode/PredatorHits": np.mean([ep['hit_predator'] for ep in iteration_episodes]),
                                    # Note: WandB labels like 'Episode/DangerHits' are kept for dashboard-history continuity
                                    "Episode/DangerHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/RestCount": np.mean([ep['rested'] for ep in iteration_episodes]),
                                    "Episode/Collisions": np.mean([ep['event_collided'] for ep in iteration_episodes]),
                                    "Episode/TotalDamage": np.mean([ep['damage'] for ep in iteration_episodes]),
                                    "Episode/DamagePredator": np.mean([ep['damage_predator'] for ep in iteration_episodes]),
                                    "Episode/DamageDanger": np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/DamageObstacle": np.mean([ep['damage_obstacle'] for ep in iteration_episodes]),
                                    "Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in iteration_episodes]),
                                    "Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in iteration_episodes]),
                                    "Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in iteration_episodes]),
                                    "Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in iteration_episodes]),
                                    "Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                })
                                # Termination reason distribution
                                term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
                                for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                                    ep_logs[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
                                # Per-tag fan-out (Site 4: DRQN)
                                _append_per_tag_means(ep_logs, iteration_episodes, neutral_tags,
                                                      'mean_dist_rabbit',   'Episode/MeanDistRabbit')
                                _append_per_tag_means(ep_logs, iteration_episodes, predator_tags,
                                                      'mean_dist_predator', 'Episode/MeanDistPredator')
                                # Behavior-measure toolkit v1: WandB fan-out (Site 4: DRQN)
                                if bm_enabled:
                                    _bm_log_wandb(ep_logs, iteration_episodes)

                            wandb.log(ep_logs)

                        wandb.log(logs)

                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.set_postfix({
                        "Iter": iteration,
                        "Loss": f"{float(loss_val):.4f}",
                        "Eps": f"{float(epsilon):.2f}",
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    })
                    pbar.refresh()

                elif algorithm == "PPO":
                    if args.debug: print(f"  [DEBUG] Collecting {num_steps * num_envs} steps of experience...", end="", flush=True)
                    env_state, key, losses, num_completed, trajectories = jit_train(
                        model, optimizer, params, env_state, key, ppo_config
                    )
                    if args.debug: print(f" Done.", flush=True)
                    
                    steps_this_iter = num_steps * num_envs
                    global_step += steps_this_iter
                    
                    step_info = trajectories.step_info
                    info_np = {}
                    if step_info is not None:
                        for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason']:
                            info_np[k] = np.array(getattr(step_info, k))
                        # Per-instance arrays: shape [num_steps, num_envs, num_entity]
                        if num_neutral_for_log  > 0: info_np['dist_per_neutral']  = np.array(step_info.dist_per_neutral)
                        if num_predator_for_log > 0: info_np['dist_per_predator'] = np.array(step_info.dist_per_predator)
                        # Behavior-measure toolkit v1: agent_in_bush (explicit extraction — Site 5 anti-pattern guard)
                        if hasattr(step_info, 'agent_in_bush'):
                            info_np['agent_in_bush'] = np.array(step_info.agent_in_bush)

                    rew_np = np.array(trajectories.reward)
                    done_np = np.array(trajectories.done)

                    for t in range(num_steps):
                        episode_returns += rew_np[t]
                        episode_lengths += 1

                        if info_np:
                            for k in BEHAVIOR_KEYS:
                                episode_behavior[k] += info_np[k][t]
                            for k in BEHAVIOR_DIST_KEYS:
                                episode_dist_sums[k] += info_np[k][t]
                            # === Per-tag accumulation (Site 5: PPO non-recurrent) ===
                            if 'dist_per_neutral' in info_np and num_neutral_for_log > 0:
                                episode_dist_per_neutral_sums  += info_np['dist_per_neutral'][t]
                            if 'dist_per_predator' in info_np and num_predator_for_log > 0:
                                episode_dist_per_predator_sums += info_np['dist_per_predator'][t]
                            # === Behavior-measure toolkit v1: per-step accumulation (Site 5: PPO) ===
                            if bm_enabled and 'agent_in_bush' in info_np:
                                _bm_info_t5 = {k: info_np[k][t] for k in ('ate_food', 'agent_in_bush')}
                                if 'dist_per_predator' in info_np: _bm_info_t5['dist_per_predator'] = info_np['dist_per_predator'][t]
                                if 'dist_per_neutral'  in info_np: _bm_info_t5['dist_per_neutral']  = info_np['dist_per_neutral'][t]
                                _bm_step_update(_bm_info_t5, done_np[t].astype(bool))

                        dones_t = done_np[t].astype(bool)
                        if np.any(dones_t):
                            completed_indices = np.where(dones_t)[0]
                            for i in completed_indices:
                                total_episodes_completed += 1
                                ep_reward = float(episode_returns[i])
                                ep_length = int(episode_lengths[i])

                                ep_data = {'r': ep_reward, 'l': ep_length}
                                if info_np:
                                    for k in BEHAVIOR_KEYS:
                                        ep_data[k] = float(episode_behavior[k][i])
                                    for k in BEHAVIOR_DIST_KEYS:
                                        ep_data[k] = float(episode_dist_sums[k][i] / max(ep_length, 1))
                                    ep_data['termination_reason'] = int(info_np['termination_reason'][t][i])
                                    # Per-tag finalisation
                                    ep_l_safe = max(ep_length, 1)
                                    if num_neutral_for_log > 0 and 'dist_per_neutral' in info_np:
                                        means = episode_dist_per_neutral_sums[i] / ep_l_safe
                                        for j, tag in enumerate(neutral_tags):
                                            ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
                                    if num_predator_for_log > 0 and 'dist_per_predator' in info_np:
                                        means = episode_dist_per_predator_sums[i] / ep_l_safe
                                        for j, tag in enumerate(predator_tags):
                                            ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])
                                    # Behavior-measure toolkit v1: per-episode finalisation (Site 5: PPO)
                                    if bm_enabled:
                                        _bm_finalise_episode(i, ep_data)

                                ep_info_buffer.append(ep_data)
                                iteration_episodes.append(ep_data)
                                episode_returns[i] = 0.0
                                episode_lengths[i] = 0
                                if info_np:
                                    for k in BEHAVIOR_KEYS:
                                        episode_behavior[k][i] = 0.0
                                    for k in BEHAVIOR_DIST_KEYS:
                                        episode_dist_sums[k][i] = 0.0
                                if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
                                if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
                                # Behavior-measure toolkit v1: per-env reset (Site 5: PPO)
                                if bm_enabled:
                                    _bm_reset_env(i)

                    if wandb_enabled and iteration % log_interval == 0:
                        logs = {
                            "iteration": iteration,
                            "timesteps": global_step,
                            **_stage_tag(),
                        }
                        if iteration_episodes:
                            rewards_list = [ep['r'] for ep in iteration_episodes]
                            lengths_list = [ep['l'] for ep in iteration_episodes]
                            ep_logs = {
                                "Episode/Reward": np.mean(rewards_list),
                                "Episode/Reward_Min": np.min(rewards_list),
                                "Episode/Reward_Max": np.max(rewards_list),
                                "Episode/Steps": np.mean(lengths_list),
                                "Episode/Number": total_episodes_completed,
                                **_stage_tag(),
                            }
                            # Behavioral metrics
                            if 'ate_food' in iteration_episodes[0]:
                                ep_logs.update({
                                    "Episode/FoodEaten": np.mean([ep['ate_food'] for ep in iteration_episodes]),
                                    "Episode/PredatorHits": np.mean([ep['hit_predator'] for ep in iteration_episodes]),
                                    # Note: WandB labels like 'Episode/DangerHits' are kept for dashboard-history continuity
                                    "Episode/DangerHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/RestCount": np.mean([ep['rested'] for ep in iteration_episodes]),
                                    "Episode/Collisions": np.mean([ep['event_collided'] for ep in iteration_episodes]),
                                    "Episode/TotalDamage": np.mean([ep['damage'] for ep in iteration_episodes]),
                                    "Episode/DamagePredator": np.mean([ep['damage_predator'] for ep in iteration_episodes]),
                                    "Episode/DamageDanger": np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/DamageObstacle": np.mean([ep['damage_obstacle'] for ep in iteration_episodes]),
                                    "Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in iteration_episodes]),
                                    "Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in iteration_episodes]),
                                    "Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in iteration_episodes]),
                                    "Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in iteration_episodes]),
                                    "Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in iteration_episodes]),
                                    "Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in iteration_episodes]),
                                })
                                # Termination reason distribution
                                term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
                                for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                                    ep_logs[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
                                # Per-tag fan-out (Site 5: PPO non-recurrent)
                                _append_per_tag_means(ep_logs, iteration_episodes, neutral_tags,
                                                      'mean_dist_rabbit',   'Episode/MeanDistRabbit')
                                _append_per_tag_means(ep_logs, iteration_episodes, predator_tags,
                                                      'mean_dist_predator', 'Episode/MeanDistPredator')
                                # Behavior-measure toolkit v1: WandB fan-out (Site 5: PPO)
                                if bm_enabled:
                                    _bm_log_wandb(ep_logs, iteration_episodes)

                            wandb.log(ep_logs)

                        if losses:
                            # losses is a list of (total_loss, (p_loss, v_loss, e_loss))
                            avg_total = np.mean([l[0] for l in losses])
                            logs.update({"loss/ppo_total": float(avg_total)})
                        
                        wandb.log(logs)

                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    loss_msg = f"L: {float(losses[-1][0]):.2f}" if losses else ""
                    pbar.set_postfix({
                        "Iter": iteration,
                        "Loss": loss_msg,
                        "Rew": f"{np.mean([ep['r'] for ep in ep_info_buffer]) if ep_info_buffer else 0.0:.2f}"
                    })
                    pbar.refresh()

                # Checkpoint Logic
                if schedule is not None:
                    checkpoint_freq = schedule.checkpoint_frequencies[current_stage]
                else:
                    checkpoint_freq = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')
                
                # Check if we've crossed an episode boundary for checkpointing
                # We save if the current episode count has reached the next checkpoint milestone
                if not hasattr(main, 'last_checkpoint_save'):
                    main.last_checkpoint_save = 0
                
                should_checkpoint = (total_episodes_completed >= main.last_checkpoint_save + checkpoint_freq)
                
                if should_checkpoint:
                    main.last_checkpoint_save = (total_episodes_completed // checkpoint_freq) * checkpoint_freq
                    
                    ckpt_data = {}
                    if algorithm == "RecurrentPPO":
                        ckpt_data = {
                            'model': nnx.state(model, nnx.Param),
                            'optimizer': nnx.state(optimizer),
                            'h_state': h_state,
                            'key': key,
                            'iteration': iteration,
                            'step': global_step,
                            'episode': total_episodes_completed,
                            'stage': current_stage,
                        }
                    elif algorithm == "DreamerV3":
                        ckpt_data = {
                            'wm': nnx.state(trainer.agent.wm, nnx.Param),
                            'actor': nnx.state(trainer.agent.ac.actor, nnx.Param),
                            'critic': nnx.state(trainer.agent.ac.critic, nnx.Param),
                            'key': key,
                            'iteration': iteration,
                            'step': global_step,
                            'episode': total_episodes_completed,
                            'stage': current_stage,
                        }

                    if ckpt_data:
                        pbar.write(f"[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iteration})...")
                        checkpointer.save(total_episodes_completed, args=ocp.args.StandardSave(ckpt_data))
                        checkpointer.wait_until_finished()  # Ensure sync for stability
                        
                        # Trigger evaluation after checkpoint
                        vis_flag = config.get_mandatory('visualization.enabled')
                        eval_v_flag = config.get_mandatory('training.video_during_training')
                        eval_s_flag = config.get_mandatory('training.stats_during_training')

                        if eval_v_flag or eval_s_flag:
                            if args.debug:
                                print(f"  [DEBUG] Starting evaluation... Video={eval_v_flag}, Stats={eval_s_flag}", flush=True)
                            try:
                                from src.utils.evaluation_core import evaluate_jax_checkpoint
                                
                                # Pass 1: Video
                                if eval_v_flag:
                                    video_eps = config.get_mandatory('training.eval_video_episodes')
                                    if args.debug:
                                        print(f"  [EVAL] Pass 1: Video (eps={video_eps})")
                                    evaluate_jax_checkpoint(
                                        model=model if algorithm == "RecurrentPPO" else trainer.agent,
                                        params=params, config=config, num_episodes=video_eps, seed=seed,
                                        results_dir=results_dir, checkpoint_pct=total_episodes_completed,
                                        render_video=True, record_stats=False, wandb_enabled=wandb_enabled, debug=args.debug,
                                        quiet=not args.debug, num_envs=1, device=jax.config.values['jax_default_device']
                                    )
                                
                                # Pass 2: Stats
                                eval_results = None
                                if eval_s_flag:
                                    stats_eps = config.get_mandatory('training.eval_stats_episodes')
                                    stats_envs = config.get_mandatory('training.eval_stats_num_envs')
                                    if args.debug:
                                        print(f"  [EVAL] Pass 2: Stats (eps={stats_eps}, envs={stats_envs})")
                                    eval_results = evaluate_jax_checkpoint(
                                        model=model if algorithm == "RecurrentPPO" else trainer.agent,
                                        params=params, config=config, num_episodes=stats_eps, seed=seed,
                                        results_dir=results_dir, checkpoint_pct=total_episodes_completed,
                                        render_video=False, record_stats=True, wandb_enabled=wandb_enabled, debug=args.debug,
                                        quiet=not args.debug, num_envs=stats_envs, device=jax.config.values['jax_default_device']
                                    )
                                    if wandb_enabled and eval_results:
                                        wandb.log({"Eval/MeanReward": eval_results["mean_reward"], "Eval/MeanLength": eval_results["mean_length"], "iteration": iteration, "timesteps": global_step, **_stage_tag()})
                                
                                # Auto Analysis Trigger
                                if config.get_mandatory('training.auto_analysis') and eval_s_flag:
                                    if args.debug:
                                        print(f"  [ANALYSIS] Triggering automated behavior analysis...")
                                    import subprocess, sys
                                    analysis_cmd = [
                                        sys.executable, "analysis/agentActionAnalysis.py",
                                        "--results_dir", results_dir,
                                        "--checkpoint", str(total_episodes_completed),
                                        "--output", os.path.join(results_dir, "stats", str(total_episodes_completed), "action_scatter.png"),
                                        "--title", f"Episode {total_episodes_completed}"
                                    ]
                                    if not args.debug:
                                        analysis_cmd.append("--quiet")
                                    result = subprocess.run(analysis_cmd, check=False, capture_output=True, text=True)
                                    if result.returncode != 0:
                                        print(f"Warning: Analysis script failed with return code {result.returncode}")
                                        print(f"Error output: {result.stderr}")
                                    
                                    if wandb_enabled:
                                        plot_path = os.path.join(results_dir, "stats", str(total_episodes_completed), "action_scatter.png")
                                        from src.utils.wandb_utils import upload_image
                                        upload_image(
                                            plot_path, 
                                            step=total_episodes_completed, 
                                            episode=total_episodes_completed, 
                                            caption=f"Episode {total_episodes_completed}",
                                            quiet=True,
                                            extra_data={"iteration": iteration, "timesteps": global_step}
                                        )

                            except Exception as e:
                                print(f"Warning: Evaluation failed: {e}")
                                if args.debug:
                                    import traceback

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            
    # Final cleanup
    print(f"Training complete. Results saved to {results_dir}")
    if wandb_enabled:
        wandb.finish()
    checkpointer.close()

if __name__ == "__main__":
    main()
