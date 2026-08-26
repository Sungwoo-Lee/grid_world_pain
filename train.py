"""
JAX Training Script for GridWorld RL Agents.

This script mirrors train.py but uses JAX-native components:
1. Loads configuration from YAML files.
2. Initializes the JAX ParallelEnv for massive parallelization.
3. Trains RecurrentPPO using Flax NNX. (DreamerV3-NNX was archived 2026-07-10 →
   src/models/archive/dreamer_v3_nnx/; the live world-model agent is
   src/algorithms/dreamer_srl/.)
4. Supports WandB logging, checkpointing, and results directory management.

Arguments:
- `--config <path>`: Path to environment/ablation config YAML (Required).
- `--episodes <int>`: Number of training episodes (converted to timesteps).
- `--num-envs <int>`: Number of parallel environments (default: 256).
- `--total-timesteps <int>`: Total timesteps to train.
- `--algorithm <str>`: Algorithm to use (RecurrentPPO).
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
import json
import os
import subprocess
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

from src.environment.config_loader import load_env_params, load_behavior_measure_cfg, load_env_config
from src.behavior.accumulators import (
    make_bm_state, bm_step_update, bm_reset_env, bm_drive_batch,
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
from src.utils.config import get_default_config, Config, dump_config_yaml
from src.utils.checkpoint_restore import restore_rppo_training_state
from src.utils.async_render import new_render_state, poll_render, drain_render
from src.utils.provenance import write_provenance

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

    # 3. Pre-build per-stage Config objects by cloning base and merging each stage YAML.
    # Resolve `extends:` per stage (load_env_config) for the same reason as the single
    # --config path — see docs/develop/archive/EXTENDS_NOT_RESOLVED_IN_TRAINING.md.
    # A stage file with no `extends:` key loads byte-identically to Config.load_yaml.
    stage_configs = []
    for p in paths:
        stage_cfg = Config(yaml.safe_load(yaml.dump(base_config.to_dict())))  # deep copy
        stage_cfg.merge(load_env_config(p))
        stage_configs.append(stage_cfg)

    return ContinualSchedule(
        stage_config_paths=paths,
        stage_names=names,
        stage_configs=stage_configs,
        episode_boundaries=list(boundaries),
        checkpoint_frequencies=list(ckpt_freqs),
    )


# ---------------------------------------------------------------------------
# Experiment eval during training (async, on-node CPU; RecurrentPPO only).
# See docs/develop/active/behavior/EXPERIMENT_EVAL_DURING_TRAINING.md for the full design.
# The eval subprocess (scripts/eval/experiment_eval_checkpoint.py) NEVER touches WandB --
# this trainer is the sole WandB writer, polling for its result.json files.
# ---------------------------------------------------------------------------

# Hardcoded, NOT a 6th config key (nit fix -- avoids a fallback default on a mandatory-key
# path): bounded wait for an in-flight experiment-eval subprocess at a NORMAL exit, before
# wandb.finish(). On a Ctrl-C exit this is NOT used -- see EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED.
EXPERIMENT_EVAL_DRAIN_TIMEOUT_S = 300
# Fix #4 (Ctrl-C policy): a user hitting Ctrl-C should not wait up to 300s for a
# background CPU eval before the process actually exits.
EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED = 5

# Which measures/conditions to surface LIVE on WandB is config-driven (experiment.
# log_measures / log_conditions in configs/evaluation/default.yaml), resolved once at startup
# into experiment_eval_cfg["log_measures"] (tuple of measure names) and
# experiment_eval_cfg["log_conditions"] (None = no filter, log every condition, or a tuple of
# stems to restrict) and threaded through to _poll_and_log_experiment_results /
# _drain_experiment_results below. The full 11 measures x 12 conditions are always written to
# CSV regardless (see experiment_eval_checkpoint.py) -- this filter is a live-panel choice
# only, not data loss.

_EXPERIMENT_EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "scripts", "eval", "experiment_eval_checkpoint.py")


def _experiment_eval_env():
    """Belt-and-braces GPU isolation (nit fix) for the experiment-eval subprocess TREE: even
    though `experiment_eval_checkpoint.py` also caps its own child (`eval_rollout.py`), setting
    JAX_PLATFORMS=cpu + CUDA_VISIBLE_DEVICES="" here too means nothing in the eval tree can
    ever reach the training GPU, regardless of which layer would otherwise miss a path."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["TF_NUM_INTEROP_THREADS"] = "1"
    return env


def _maybe_dispatch_experiment_eval(experiment_cfg, experiment_state, results_dir, models_dir,
                                ep, global_step, iteration, quiet):
    """Called once per checkpoint save (RecurrentPPO only; gated by the caller on
    experiment_cfg is not None). Increments a monotonic checkpoint index; returns early
    unless this is the Nth checkpoint. At most one concurrent experiment-eval subprocess:
    if the previous one is still running, this checkpoint's eval is SKIPPED (logged,
    non-fatal) rather than queued -- rPPO keeps every checkpoint on disk, and the
    offline `run_sweep.py` pipeline can incrementally backfill any skipped checkpoint
    later, so a skipped live-eval is never a data-loss event."""
    proc = experiment_state.get("proc")
    if proc is not None and proc.poll() is None:
        experiment_state["ckpt_index"] += 1
        if not quiet:
            print(f"[experiment-eval] checkpoint {ep}: previous experiment eval (pid={proc.pid}) "
                  f"still running -- skipping this checkpoint (offline-recoverable via "
                  f"run_sweep.py's incremental backfill).")
        return
    experiment_state["proc"] = None

    experiment_state["ckpt_index"] += 1
    if experiment_state["ckpt_index"] % experiment_cfg["every_n_checkpoints"] != 0:
        return

    ep_dir = os.path.join(results_dir, "experiment_eval", str(ep))
    os.makedirs(ep_dir, exist_ok=True)
    cmd = [
        sys.executable, _EXPERIMENT_EVAL_SCRIPT,
        "--checkpoint", os.path.abspath(os.path.join(models_dir, str(ep))),
        "--result-json", os.path.abspath(os.path.join(ep_dir, "result.json")),
        "--out-root", os.path.abspath(ep_dir),
        "--conditions", str(experiment_cfg["conditions"]),
        "--episodes", str(experiment_cfg["episodes"]),
        "--checkpoint-key", str(ep),
        "--global-step", str(global_step),
        "--iteration", str(iteration),
    ]
    log_f = open(os.path.join(ep_dir, "log.txt"), "w")
    proc = subprocess.Popen(cmd, env=_experiment_eval_env(), stdout=log_f, stderr=subprocess.STDOUT)
    experiment_state["proc"] = proc
    experiment_state["proc_log_f"] = log_f
    if not quiet:
        print(f"[experiment-eval] dispatched checkpoint {ep} (pid={proc.pid}); "
              f"log: {os.path.join(ep_dir, 'log.txt')}")


def _poll_and_log_experiment_results(experiment_state, results_dir, iteration, global_step,
                                 wandb_enabled, quiet, focus_measures, focus_conds):
    """Cheap, once-per-iteration poll for finished result.json files. Fix #1
    (failure isolation): every internal step below has its OWN try/except, so a
    malformed JSON, a NAS glob hiccup, or a wandb.log raise on ONE checkpoint's result
    can never block or crash processing of the others -- and the call site in the main
    loop ALSO wraps this whole function, as defense in depth.

    focus_measures: tuple of measure names to surface on WandB (from
    experiment.log_measures). focus_conds: None (log every condition) or a
    tuple of condition stems to restrict to (from experiment.log_conditions).
    Both are config-driven filters on the LIVE panel only; CSVs always carry all 11x12."""
    experiment_dir = os.path.join(results_dir, "experiment_eval")
    try:
        result_paths = glob.glob(os.path.join(experiment_dir, "*", "result.json"))
    except Exception as e:
        print(f"[experiment-eval] WARNING: glob failed (non-fatal): {e}")
        return

    for rp in result_paths:
        ep_str = os.path.basename(os.path.dirname(rp))
        if ep_str in experiment_state["logged"]:
            continue
        try:
            with open(rp) as f:
                data = json.load(f)
        except Exception as e:
            # Truncated / mid-write read racing the atomic os.replace write in
            # experiment_eval_checkpoint.py -- skip WITHOUT marking handled, so the next
            # iteration retries (a fully-written file is never truncated; this only
            # guards a read racing an in-progress write).
            print(f"[experiment-eval] WARNING: could not read {rp} (non-fatal, will retry): {e}")
            continue

        try:
            if data.get("status") == "ok" and wandb_enabled:
                flat = {}
                for cond, meas in data.get("measures", {}).items():
                    short = cond.replace("avoid_", "")
                    if focus_conds is not None and short not in focus_conds:
                        continue
                    for k, v in meas.items():
                        if v is None or k not in focus_measures:
                            continue
                        flat[f"Experiment/{k}/{short}"] = v
                if flat:
                    # Experiment metrics are EPISODE-LEVEL: plot against Episode/Number (the
                    # project's canonical episode x-axis), so they sit with the other
                    # Episode/* metrics. checkpoint_key == total_episodes_completed.
                    flat["Episode/Number"] = data.get("checkpoint_key", int(ep_str))
                    wandb.log(flat)
            elif data.get("status") != "ok" and not quiet:
                print(f"[experiment-eval] checkpoint {ep_str}: eval failed, nothing logged "
                      f"({str(data.get('error', ''))[-300:]})")
        except Exception as e:
            print(f"[experiment-eval] WARNING: failed to log results for checkpoint {ep_str} "
                  f"(non-fatal): {e}")
        experiment_state["logged"].add(ep_str)  # mark handled either way (ok, failed, or log error)


def _drain_experiment_results(experiment_state, results_dir, iteration, global_step, wandb_enabled,
                          quiet, interrupted, focus_measures, focus_conds):
    """Final drain before wandb.finish(). Fix #4 (Ctrl-C + drain policy):
    - Normal exit: bounded wait (EXPERIMENT_EVAL_DRAIN_TIMEOUT_S) for an in-flight eval so its
      series reach WandB before the run closes. If it's STILL running past the timeout,
      do NOT kill it -- leave it running (orphaned, on-node CPU only, harmless) so its
      CSV still completes; only its final WandB point is missed.
    - Ctrl-C (KeyboardInterrupt) exit: sharply shortened wait
      (EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED) -- the user asked training to stop NOW,
      not in up to 300s."""
    proc = experiment_state.get("proc")
    timeout_s = EXPERIMENT_EVAL_DRAIN_TIMEOUT_S_INTERRUPTED if interrupted else EXPERIMENT_EVAL_DRAIN_TIMEOUT_S
    if proc is not None and proc.poll() is None:
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if not quiet:
                print(f"[experiment-eval] in-flight eval (pid={proc.pid}) did not finish within "
                      f"{timeout_s}s at shutdown -- leaving it running in the background "
                      f"(its CSV will still complete; its final WandB point will be missed).")
    try:
        _poll_and_log_experiment_results(experiment_state, results_dir, iteration, global_step,
                                     wandb_enabled, quiet, focus_measures, focus_conds)
    except Exception as e:
        print(f"[experiment-eval] WARNING: final drain poll failed (non-fatal): {e}")


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
    parser.add_argument("--eval-config", type=str, default=None,
                        help="Path to evaluation config YAML (selector layer: conditions/episodes/"
                             "log filters/opt-in for the during-training behavior-probe experiment, "
                             "rPPO only). Defaults to configs/evaluation/default.yaml. Supports "
                             "`extends:` chains (e.g. configs/evaluation/experiment_on.yaml extends "
                             "evaluation/default to turn the experiment on).")
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
    parser.add_argument("--log-interval", type=int,
                        help="DEPRECATED (use the config `logging:` block; ignored when that "
                             "block is present). Legacy WandB logging interval in ITERATIONS "
                             "— note 1 rPPO iteration = num_steps*num_envs env-steps, 128x a "
                             "Dreamer iteration. See "
                             "docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md")
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

    # Merge rPPO-specific training defaults (configs/train/recurrent_ppo.yaml).
    # default.yaml above is SHARED with dreamer; rPPO needs sparser logging + all-checkpoint
    # retention, so peek the agent algorithm and merge the rPPO overrides for RecurrentPPO
    # only. Precedence: above train/default.yaml, below the experiment --config and the
    # --agent_config (and below CLI flags), so they stay overridable defaults. Dreamer never
    # loads this file. NOTE: continual (--configs-dir) rPPO runs set per-stage frequencies
    # from the schedule and replace `config` below, so this default applies to single --config runs.
    if Config.load_yaml(args.agent_config).get("agent.algorithm") == "RecurrentPPO":
        rppo_train_path = os.path.join(os.path.dirname(__file__), "configs", "train", "recurrent_ppo.yaml")
        if os.path.exists(rppo_train_path):
            if not args.quiet:
                print(f"Loading rPPO train defaults from {rppo_train_path}")
            config.merge(Config.load_yaml(rppo_train_path))

    # Merge Evaluation config (selector layer -- default.yaml, or a preset selected via
    # --eval-config, e.g. configs/evaluation/experiment_on.yaml to opt into the during-training
    # behavior-probe experiment). Resolve `extends:` chains (load_env_config) exactly like
    # --config above, so a preset's `extends: evaluation/default` actually pulls in the base's
    # keys rather than silently dropping them.
    eval_config_path = args.eval_config or os.path.join(
        os.path.dirname(__file__), "configs", "evaluation", "default.yaml")
    if not os.path.exists(eval_config_path):
        raise ValueError(f"--eval-config path not found: {eval_config_path!r}")
    if not args.quiet:
        print(f"Loading eval config from {eval_config_path}")
    eval_defaults = load_env_config(eval_config_path)
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
        # Resolve `extends:` chains (load_env_config) — a plain Config.load_yaml here would
        # silently drop every layer inherited from a non-default `extends` parent (noise,
        # random-init, all-combined predators, etc.). See
        # docs/develop/archive/EXTENDS_NOT_RESOLVED_IN_TRAINING.md.
        # A config with NO `extends:` key loads byte-identically to Config.load_yaml
        # (load_env_config's own documented standalone behaviour), so this is backward-compatible.
        user_config = load_env_config(args.config)
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
    # Fix (Finding G3/L4): these must land on the SAME keys load_env_params() reads
    # ('body.with_satiation' / 'body.overeating_death'), not 'environment.*' -- the previous
    # keys were never read back by anything, so the saved config.yaml kept the YAML default
    # (e.g. with_satiation: true) even though the run actually trained without satiation.
    if args.no_satiation: config.set('body.with_satiation', False)
    if args.no_overeating_death: config.set('body.overeating_death', False)
    if args.checkpoint_frequency is not None: config.set('training.checkpoint_frequency', args.checkpoint_frequency)

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

    if algorithm == "DreamerV3":
        raise ValueError(
            "The in-house DreamerV3 (NNX) stack was archived on 2026-07-10 "
            "(development stopped 2026-05-11; superseded by the sheeprl-parity port). "
            "Use src/algorithms/dreamer_srl/ (entry: src/algorithms/dreamer_srl/"
            "dreamer_srl_main.py) instead. Archived code: src/models/archive/"
            "dreamer_v3_nnx/ — see its README and docs/develop/active/diagnosis/"
            "dreamer_sheeprl_parity_2026-07-06/archive_plan_dreamer_v3_nnx.md.")

    # --- Experiment eval during training: gate + mandatory-key validation at
    # STARTUP, not at the first checkpoint (nit fix) -- a bad/incomplete config raises
    # here, in the first second of the run, instead of hours in after training has
    # already made progress. `experiment.during_training.enabled` lives in the
    # evaluation config layer (configs/evaluation/default.yaml, default false; merged for
    # every algorithm at L~504-514 above), so every algorithm has it; the other keys are
    # only required when the gate is true. Opt in via `--eval-config
    # configs/evaluation/experiment_on.yaml` (or any preset with during_training.enabled: true).
    experiment_eval_enabled = config.get_mandatory('experiment.during_training.enabled')
    if experiment_eval_enabled and algorithm != "RecurrentPPO":
        raise ValueError(
            f"experiment.during_training.enabled=true is only supported for "
            f"RecurrentPPO (see docs/develop/active/behavior/"
            f"EXPERIMENT_EVAL_DURING_TRAINING.md), got algorithm={algorithm!r}.")
    experiment_eval_cfg = None
    if experiment_eval_enabled:
        _log_conditions_raw = config.get_mandatory('experiment.log_conditions')
        _log_conditions = (None if _log_conditions_raw == "all" else
                            tuple(c.strip() for c in str(_log_conditions_raw).split(",") if c.strip()))
        experiment_eval_cfg = {
            "every_n_checkpoints": config.get_mandatory('experiment.during_training.every_n_checkpoints'),
            "conditions": config.get_mandatory('experiment.conditions'),
            "episodes": config.get_mandatory('experiment.episodes'),
            "on_node": config.get_mandatory('experiment.during_training.on_node'),
            "log_measures": tuple(config.get_mandatory('experiment.log_measures')),
            "log_conditions": _log_conditions,
        }
        if experiment_eval_cfg["on_node"] != "self":
            raise ValueError(
                f"experiment.during_training.on_node={experiment_eval_cfg['on_node']!r} is not "
                f"implemented (only 'self' — on-node Popen — is supported today).")

    if args.profile:
        profile_trace_dir = os.path.join(profile_parent, f"{algorithm}_trace")
        os.makedirs(profile_trace_dir, exist_ok=True)
        print(f"[PROFILE] trace dir: {profile_trace_dir}")
        print(f"[PROFILE] warm-up: {PROFILE_WARMUP_ITERS} iters, "
              f"trace window: {PROFILE_TRACE_ITERS} iters")

    # Strictly Resolve Parameters (No Safe Defaults)
    if schedule is not None:
        if algorithm not in ("RecurrentPPO",):
            raise ValueError(
                f"Continual learning (--configs-dir) is only supported for "
                f"RecurrentPPO, got algorithm='{algorithm}'. "
                "Use Option A: restrict to supported algorithms at startup.")
        if args.episodes is not None:
            raise ValueError("--episodes is incompatible with --configs-dir; "
                             "episode budget is set by the schedule's last boundary.")
        episodes = schedule.episode_boundaries[-1]
    else:
        episodes = args.episodes if args.episodes is not None else config.get_mandatory('episodes')
        # Fix (Finding G3/L4): persist into `config` (dumped to models/config.yaml below) so
        # re-evaluating the run reads back what actually trained, not the pre-override YAML
        # default. Continual (--configs-dir) runs are excluded: episode budget there comes
        # from the schedule, not this key, and is already dumped separately (schedule.yaml).
        config.set('episodes', episodes)
    env_max_steps = config.get_mandatory('environment.max_steps')
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')
    from src.utils.rolling_logging import resolve_logging_cfg
    # Two-level logging (docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md).
    # `logging_cfg is None` -> LEGACY log_interval path (unchanged behavior for configs
    # that predate this block; the basic04 sweep depends on this).
    logging_cfg = resolve_logging_cfg(
        config.get,
        defaults={'smoothing_episodes': 5000, 'interval_episodes': 4000,
                  'smoothing_iters': 100, 'interval_iters': 50},
    )
    if logging_cfg is not None and args.log_interval is not None:
        print("[WARN] --log-interval is IGNORED: this config uses the two-level `logging:` "
              "block. Set logging.episode.interval_episodes / logging.step.interval_iters "
              "in the config instead.", flush=True)
    log_interval = args.log_interval or config.get('training.log_interval', 1)
    log_accumulate = args.log_accumulate if args.log_accumulate is not None else config.get('training.log_accumulate', True)
    if logging_cfg is None:
        print("[DEPRECATION] training.log_interval is deprecated — it conflates smoothing "
              "with interval and its unit (iterations) differs 128x between rPPO and Dreamer. "
              "Migrate to the `logging:` block: "
              "docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md", flush=True)
    config.set('training.num_envs', num_envs)
    config.set('training.log_interval', log_interval)
    config.set('training.log_accumulate', log_accumulate)
    if logging_cfg is not None:
        # Persist resolved values so the dumped models/config.yaml records what actually ran.
        config.set('logging.episode.smoothing_episodes', logging_cfg['smoothing_episodes'])
        config.set('logging.episode.interval_episodes',  logging_cfg['interval_episodes'])
        config.set('logging.step.smoothing_iters',       logging_cfg['smoothing_iters'])
        config.set('logging.step.interval_iters',        logging_cfg['interval_iters'])

    # Budget scales with parallelization: episodes * steps per episode * num environments
    total_timesteps = args.total_timesteps or (episodes * env_max_steps * num_envs)

    if algorithm in ["RecurrentPPO", "PPO"]:
        if algorithm == "RecurrentPPO":
            num_steps = args.num_steps or config.get_mandatory('agent.sequence_length')
            config.set('agent.sequence_length', num_steps)
        else:
            num_steps = args.num_steps or config.get_mandatory('agent.num_steps')
            config.set('agent.num_steps', num_steps)
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr_actor')
        config.set('agent.hidden_size', hidden_size)
        config.set('agent.lr_actor', lr)
    else:
        num_steps = args.num_steps or config.get_mandatory('agent.num_steps')
        hidden_size = args.hidden_size or config.get_mandatory('agent.hidden_size')
        lr = args.lr or config.get_mandatory('agent.lr')
        config.set('agent.num_steps', num_steps)
        config.set('agent.hidden_size', hidden_size)
        config.set('agent.lr', lr)

    seed = args.seed if args.seed is not None else config.get_mandatory('seed')

    
    # Re-load EnvParams with full merged config for JAX core
    params = load_env_params(config)  # stage 0 (or single-config)

    # Apply CLI overrides to params if they exist in params (redundant now but safe)
    if args.no_satiation: params = params.replace(with_satiation=False)
    if args.no_overeating_death: params = params.replace(overeating_death=False)

    # Validate obs/action dim AND modality fingerprint consistency across all stages
    # BEFORE training starts.
    def _mask_fp(arr):
        """Per-entity flag array -> fingerprint entry.

        Returns the sentinel "none" when every entry is falsy, so the fingerprint
        does not encode entity COUNT via tuple length. Only a config that actually
        masks or blocks something contributes a count-dependent value, and such a
        config genuinely does change observation semantics.
        """
        vals = tuple(int(x) for x in arr)
        return "none" if not any(vals) else vals

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
            # directional sensors. olfactory_grid_range changes obs_dim (so
            # the dim check already catches it) -- included as defence in depth.
            # visual_blur_enabled and the per-entity visual_mask arrays change what
            # the observation MEANS at an identical dim count, which is the case
            # the dim check cannot catch and the only reason this tuple exists.
            # Masks go in as tuples of ints: appending a jnp array would silently
            # break the `!=` comparison below, which is tuple equality.
            p.olfactory_grid_range,
            p.visual_blur_enabled,
            # Collapsed to a sentinel when nothing is masked, so two curriculum
            # stages with DIFFERENT ENTITY COUNTS but no masking still match.
            # A raw tuple here would make its length entity-count-dependent and
            # forbid the historically legal food-only -> full-task pattern.
            _mask_fp(p.res_visual_mask),
            _mask_fp(p.animal_visual_mask),
            _mask_fp(p.obs_visual_mask),
            # DIRECTIONAL_SENSORS: both change observation SEMANTICS at an identical dim count.
            # The cone angle and strength are continuous and stay out, same
            # reasoning as the blur knobs.
            p.visual_value_mode,
            p.visual_occlusion_enabled,
            _mask_fp(p.res_blocks_sight),
            _mask_fp(p.animal_blocks_sight),
            _mask_fp(p.obs_blocks_sight),
            # NOT fingerprinted, deliberately: visual_blur_radial_scale /
            # _anisotropy / _sigma_floor are continuous, and fingerprinting floats
            # would forbid legitimate schedules. Same pre-existing choice applies
            # to visual_vector_size.
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

    # --- Code provenance ------------------------------------------------------------
    # Written HERE, at directory-creation time, NOT at the end of training: a run killed
    # after ten minutes must still carry its provenance, and the runs most worth
    # investigating later are exactly the ones that died.
    #
    # DELIBERATELY BEST-EFFORT — this is the one place the project's no-fallback-defaults
    # rule does NOT apply. If git is unavailable, the repo is in an odd state, or the
    # subprocess fails, the fields record the string "unknown" and training continues.
    # A run that dies at startup because `git` was missing would be strictly worse than a
    # run carrying an incomplete note. Do not "fix" this into a hard failure.
    prov_path = write_provenance(models_dir, argv=sys.argv)
    if prov_path and not args.quiet:
        print(f"Provenance saved to: {prov_path}")

    # Experiment eval during training: process/bookkeeping state, plus fix #2
    # (resume double-logging) -- pre-seed "already logged" with every result.json that
    # already exists under this results_dir (e.g. a prior session of a resumed run) so
    # this session's poll loop does not re-log old checkpoints as duplicate WandB points.
    experiment_state = {"proc": None, "ckpt_index": 0, "logged": set()}
    if experiment_eval_enabled:
        for _rj in glob.glob(os.path.join(results_dir, "experiment_eval", "*", "result.json")):
            experiment_state["logged"].add(os.path.basename(os.path.dirname(_rj)))
        if experiment_state["logged"] and not args.quiet:
            print(f"[experiment-eval] resume: pre-seeded {len(experiment_state['logged'])} "
                  f"already-logged checkpoint(s) from a prior session.")

    # Async checkpoint-video render (docs/develop/active/refactors/
    # ASYNC_CHECKPOINT_VIDEO_RENDER.md): master switch + dispatch/poll/drain
    # state. true → the checkpoint video pass Popen-dispatches the MP4 render
    # (evaluate_jax_checkpoint(async_render_state=...)) and this loop polls/
    # uploads; false → legacy blocking subprocess.run render (kill-switch).
    async_video_render = config.get_mandatory('training.async_video_render')
    async_render_state = new_render_state()

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
        dump_config_yaml(config.to_dict(), f)
    if not args.quiet:
        print(f"Config saved to: {config_save_path}")

    # In continual mode: also dump each stage config and the schedule for auditability
    if schedule is not None:
        for i, (name, cfg) in enumerate(zip(schedule.stage_names, schedule.stage_configs)):
            out = os.path.join(models_dir, f"stage_{i:02d}_{name}.yaml")
            with open(out, "w") as f:
                dump_config_yaml(cfg.to_dict(), f)
        sched_dump = {
            "continual": {
                "episode_boundaries": schedule.episode_boundaries,
                "checkpoint_frequencies": schedule.checkpoint_frequencies,
                "stage_names": schedule.stage_names,
            }
        }
        with open(os.path.join(models_dir, "schedule.yaml"), "w") as f:
            dump_config_yaml(sched_dump, f)
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
            "job_type": config.get_mandatory('wandb.job_type'),
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
        # Fix: WandB define_metric patterns are case-sensitive.  Actual logged keys use
        # uppercase prefixes (Loss/*, Modulator/*, Behavior/*, WorldModel/*) so lowercase
        # patterns silently fall through to the "*" catch-all.  Ported from train.py:L630-L632.
        wandb.define_metric("Loss/*",       step_metric="iteration")   # was "loss/*" — uppercase fix
        wandb.define_metric("Modulator/*",  step_metric="iteration")   # was "modulator/*" — uppercase fix
        wandb.define_metric("Behavior/*",   step_metric="iteration")   # was "behavior/*" (lowercase + timesteps)
        wandb.define_metric("WorldModel/*", step_metric="iteration")   # new — no pattern existed before
        wandb.define_metric("stage/index",      step_metric="Episode/Number")
        wandb.define_metric("stage/transition", step_metric="Episode/Number")
        # Experiment eval during training: distinct namespace + its OWN step-metric
        # (NOT "Behavior/*" above, which is already bound to "iteration" -- a step-metric
        # pattern can only carry one binding). checkpoint_episode is set explicitly in
        # every Experiment/* log call so a late-arriving async result still plots at the
        # CORRECT x-position no matter how far training has advanced since dispatch.
        wandb.define_metric("Experiment/*", step_metric="Episode/Number")  # episode-level x-axis

        # --- Source snapshot for reproducibility -------------------------------
        # DO NOT use wandb.run.log_code(".") here. It walks the ENTIRE repo:
        # wandb's filtered_dir() does `for dirpath, _, files in os.walk(root)` and
        # DISCARDS dirnames, so it never prunes directories -- it stats every file
        # under results/ (~370k recordings on the NAS) before the first training
        # step. include_fn/exclude_fn do NOT help: they run AFTER the walk has
        # visited each file, so they cut uploads, not traversal. Measured cost had
        # grown to >1.5 h of dead time per launch, and it worsens with every eval
        # sweep (see docs/diary/2026-08-16; first noted 2026-07-03).
        #
        # Instead: build the same artifact from ANCHORED globs (122 files, ~1 s).
        # WARNING: keep every pattern ANCHORED to a code directory. A bare
        # "**/*.py" is UNANCHORED and silently restores the full-repo walk --
        # results/ holds no .py files, so it scans ~370k paths yielding nothing.
        _code_files = (
            glob.glob("*.py")
            + glob.glob("src/**/*.py", recursive=True)
            + glob.glob("scripts/**/*.py", recursive=True)
        )
        _code_name = wandb.util.make_artifact_name_safe(f"source-{wandb.run.project}")
        try:
            # Same class log_code() uses, so the WandB UI "Code" tab renders it
            # (the public API rejects type="code" as reserved).
            from wandb.sdk.artifacts._internal_artifact import InternalArtifact
            _code_art = InternalArtifact(_code_name, "code")
        except Exception:  # wandb moved the private module -> public fallback
            _code_art = wandb.Artifact(_code_name, type="source")
        for _f in _code_files:
            _code_art.add_file(_f, name=_f)
        _logged_art = wandb.run.log_artifact(_code_art)
        try:
            # log_code() sets this itself; do it manually so the UI links the snapshot.
            wandb.run.config.update(
                {"_wandb": {"code_path": _logged_art.name}}, allow_val_change=True
            )
        except Exception as _e:  # cosmetic UI linkage only -- never block training
            print(f"[wandb] could not set code_path (cosmetic): {_e}")

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

    # Buffer for episodes that finish across iterations (Stage 3) — LEGACY path only.
    iteration_episodes = []
    # Two-level logging: Buffer B (episode stream) + Buffer S (iteration stream).
    ep_window = step_window = None
    if logging_cfg is not None:
        from src.utils.rolling_logging import RollingWindow
        ep_window = RollingWindow(logging_cfg['smoothing_episodes'],
                                  logging_cfg['interval_episodes'], name="episode")
        step_window = RollingWindow(logging_cfg['smoothing_iters'],
                                    logging_cfg['interval_iters'], name="step")

    # --- Checkpoint Restoration (Continual Learning / Transfer) ---
    # H1 fix (diag_fable5_20260704/01 Finding 1): restore-to-target, FATAL on
    # any failure. The old strict-match block never matched a single leaf and
    # a blanket except silently trained from random weights.
    if args.load_checkpoint:
        if args.debug: print(f"[DEBUG] Phase 6.5: Restoring Checkpoint from {args.load_checkpoint}...", flush=True)
        if not args.quiet:
            print(f"Restoring checkpoint from {args.load_checkpoint}...")

        if algorithm == "RecurrentPPO":
            restored_meta = restore_rppo_training_state(
                args.load_checkpoint, model, optimizer, h_state, key,
                quiet=args.quiet)
            h_state = restored_meta['h_state']
            key = restored_meta['key']
            global_step = int(restored_meta['step'])
            iteration = int(restored_meta['iteration'])
            total_episodes_completed = int(restored_meta['episode'])
            if schedule is not None:
                current_stage = int(restored_meta['stage'])
        else:
            raise ValueError(
                f"--load-checkpoint is not supported for algorithm {algorithm!r} "
                f"(only RecurrentPPO saves checkpoints).")

    # --- H2 fix (diag_fable5_20260704/01 Finding 2): continual resume must
    # rebuild the env for the restored stage. The env above was built from
    # stage 0 BEFORE restore, and restoring current_stage makes the in-loop
    # transition check compare equal — so without this block a stage-N resume
    # trains stage-N counters on a stage-0 world.
    if args.load_checkpoint and schedule is not None:
        resumed_stage = schedule.stage_for_episode(total_episodes_completed)
        if resumed_stage != current_stage and not args.quiet:
            print(f"[RESUME] Checkpoint 'stage' field ({current_stage}) != "
                  f"schedule-derived stage ({resumed_stage}) for "
                  f"ep={total_episodes_completed}; trusting the schedule.")
        current_stage = resumed_stage
        # Unconditional rebuild (idempotent for stage 0; resume is rare).
        params = load_env_params(schedule.stage_configs[current_stage])
        env = ParallelEnv(params)
        key, reset_key = jax.random.split(key)
        env_state, obs = env.reset(reset_key, num_envs)
        # Fresh env episodes -> fresh recurrent state (mirrors the in-loop
        # stage-transition block at the 'Fix 4' comment).
        if algorithm == "RecurrentPPO":
            h_state = model.initial_state(num_envs)
        if not args.quiet:
            print(f"[RESUME] Stage {current_stage}:"
                  f"{schedule.stage_names[current_stage]} environment rebuilt "
                  f"at ep={total_episodes_completed}.")

    start_time = datetime.now()
    if args.debug: print(f"[DEBUG] Loop start time: {start_time.strftime('%H:%M:%S')}", flush=True)

    def _stage_tag() -> dict:
        """Return stage WandB tag dict; empty in single-config mode."""
        if schedule is None:
            return {}
        return {"stage/index": current_stage,
                "stage/name": schedule.stage_names[current_stage]}

    def _emit_episode_row(eps, total_eps):
        """Emit one WandB row aggregated over `eps` (a window of episode dicts).
        Shared by the two-level path (rolling window) and the legacy path
        (cleared-per-interval list) so key coverage can never diverge."""
        from src.utils.rolling_logging import spread
        # _window_n: sample count behind this row. RollingWindow emits PARTIAL
        # windows on the interval (2026-07-28 curriculum-audit fix), so this
        # makes a 300-sample mean distinguishable from a 5000-sample one.
        ep_log = {"Episode/Number": total_eps,
                  "Episode/_window_n": len(eps),
                  **_stage_tag()}
        # Spread on the two headline metrics: a mean survival of 133 that is secretly
        # bimodal (~400 no-predator vs ~20 with-predator) looks fine while hiding a
        # mixture — the std exposes it. (Reward_Min/Max already existed; keep names.)
        spread([ep['r'] for ep in eps], "Episode/Reward", ep_log)
        spread([ep['l'] for ep in eps], "Episode/Steps",  ep_log)
        if 'ate_food' in eps[0]:
            ep_log.update({
                "Episode/FoodEaten": np.mean([ep['ate_food'] for ep in eps]),
                "Episode/PredatorHits": np.mean([ep['hit_predator'] for ep in eps]),
                # Note: WandB labels like 'Episode/DangerHits' are kept for dashboard-history continuity
                "Episode/DangerHits": np.mean([ep['hit_hiding_predator'] for ep in eps]),
                "Episode/RestCount": np.mean([ep['rested'] for ep in eps]),
                "Episode/Collisions": np.mean([ep['event_collided'] for ep in eps]),
                "Episode/TotalDamage": np.mean([ep['damage'] for ep in eps]),
                "Episode/DamagePredator": np.mean([ep['damage_predator'] for ep in eps]),
                "Episode/DamageDanger": np.mean([ep['damage_hiding_predator'] for ep in eps]),
                "Episode/DamageObstacle": np.mean([ep['damage_obstacle'] for ep in eps]),
                "Episode/MeanDistFood": np.mean([ep['dist_to_food'] for ep in eps]),
                "Episode/MeanDistPredator": np.mean([ep['dist_to_pred'] for ep in eps]),
                "Episode/MeanDistRabbit": np.mean([ep['dist_to_neutral'] for ep in eps]),
                "Episode/MeanDistHidingPredator": np.mean([ep['dist_to_hiding_predator'] for ep in eps]),
                "Episode/RabbitHits": np.mean([ep['hit_neutral'] for ep in eps]),
                "Episode/HidingPredatorHits": np.mean([ep['hit_hiding_predator'] for ep in eps]),
            })
            # Termination reason distribution (fraction of episodes ending each way)
            term_reasons = [ep['termination_reason'] for ep in eps]
            for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                ep_log[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
            # Per-tag fan-out
            _append_per_tag_means(ep_log, eps, neutral_tags,
                                  'mean_dist_rabbit',   'Episode/MeanDistRabbit')
            _append_per_tag_means(ep_log, eps, predator_tags,
                                  'mean_dist_predator', 'Episode/MeanDistPredator')
            # Behavior-measure toolkit v1: WandB fan-out (Site 1)
            if bm_enabled:
                _bm_log_wandb(ep_log, eps)
        wandb.log(ep_log)

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
                        # Two-level path: Buffer B can hold up to smoothing_episodes
                        # (production default 5000) completed episodes, spanning far
                        # longer than a single log_interval window — long enough to
                        # bridge a stage transition and average pre/post-transition
                        # episodes into the same row. Not in the original plan's File
                        # Changes for train.py (only dreamer_srl_main.py's stage-swap
                        # was specified there) — added here for consistency with that
                        # same fix, flagged in the Implementation Report. Reset the
                        # counter too, so the warm-up gate re-arms post-swap.
                        if ep_window is not None:
                            ep_window.buf.clear()
                            ep_window.count = 0
                        # Behavior-measure toolkit v1: stage-transition wipe.
                        # Use the canonical per-env reset helper (_bm_reset_env wraps
                        # bm_reset_env from src.behavior.accumulators) so this site
                        # stays in sync with every other BMState-reset call in this file
                        # (episode-end at ~train.py:1409 and 1683).
                        if bm_enabled:
                            for _i in range(num_envs):
                                _bm_reset_env(_i)   # full per-env BMState reset (M1/M2/M5 per-class + per-tag + K-buffer + age/seen state)

                        # --- Fix 4: reset agent recurrent state at stage transition.
                        # Symmetric with replay-buffer clearing — the env is fresh, so the
                        # agent's memory of the old env should not contaminate new-stage rollouts.
                        if algorithm == "RecurrentPPO":
                            # Same init as training startup (train.py:717)
                            h_state = model.initial_state(num_envs)

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

                # Reset behavior depends on accumulation mode.
                # Two-level path: Buffer B EVICTS (deque maxlen), never clears — the
                # clear-after-emit is exactly what welds window to interval today.
                if logging_cfg is None:
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
                                if logging_cfg is None:
                                    # LEGACY: Buffer A drains into a cleared-per-window list.
                                    iteration_episodes.append(ep_data)
                                else:
                                    # Two-level: push directly into Buffer B (rolling,
                                    # maxlen=smoothing_episodes) as each episode is
                                    # discovered — no intermediate Buffer A list, so
                                    # nothing is ever capped at num_envs. This matters
                                    # for rPPO specifically: one iteration is a whole
                                    # jitted rollout (num_steps=128 x num_envs), and
                                    # episodes are extracted post-hoc by this
                                    # `for t in range(num_steps): ... for i in
                                    # completed_indices:` loop, so a single iteration can
                                    # yield up to num_steps*num_envs finishes (far more
                                    # than num_envs) — capping at num_envs would silently
                                    # drop episodes. Pushing one-at-a-time in t-ascending
                                    # order (order within a t is irrelevant — simultaneous
                                    # finishes are exchangeable) also means emission can
                                    # fire mid-batch, exactly when the counter crosses a
                                    # multiple of interval_episodes, not deferred to the
                                    # end of the rollout.
                                    if ep_window.push(ep_data) and wandb_enabled:
                                        _emit_episode_row(list(ep_window.buf),
                                                          total_episodes_completed)

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

                    # LEGACY path only — two-level path emits at the push site above.
                    if (logging_cfg is None and wandb_enabled and iteration_episodes
                            and iteration % log_interval == 0):
                        _emit_episode_row(iteration_episodes, total_episodes_completed)

                    # Update progress bar based on total episodes completed
                    pbar.n = min(total_episodes_completed, episodes) if episodes > 0 else 0
                    pbar.refresh()

                    avg_policy_loss = jnp.mean(jnp.array([l[1][0] for l in losses]))
                    avg_value_loss = jnp.mean(jnp.array([l[1][1] for l in losses]))
                    avg_ent_loss = jnp.mean(jnp.array([l[1][2] for l in losses]))
                    avg_grad_norm = jnp.mean(jnp.array([l[1][3] for l in losses]))
                    avg_mod_grad_norm = jnp.mean(jnp.array([l[1][4] for l in losses]))
                    total_loss = jnp.mean(jnp.array([l[0] for l in losses]))
                    
                    # Two-level logging: only the five loss/* scalars are windowed
                    # (pushed every iteration) and go through spread() at emission.
                    # Values are kept as JAX scalars (NOT float()-converted here) so
                    # pushing every iteration does not force a host sync every
                    # iteration — spread()'s np.asarray() does one batched sync at
                    # emission time only, mirroring today's async-dispatch pattern.
                    # modulator/* is NOT windowed — computing jnp.mean/std over
                    # mod_info every iteration (whether or not we're about to log)
                    # would be a hot-path regression vs. today (those reductions were
                    # only ever computed inside the log gate). Instead modulator/*
                    # keeps its current single-iteration semantics by being computed
                    # fresh, from THIS iteration's mod_info, only at emission time —
                    # i.e. it is always the most-recent sample, never spread.
                    _loss_sample = {
                        "loss/total":     total_loss,
                        "loss/policy":    avg_policy_loss,
                        "loss/value":     avg_value_loss,
                        "loss/entropy":   avg_ent_loss,
                        "loss/grad_norm": avg_grad_norm,
                    }
                    if logging_cfg is None:
                        _do_step_log = wandb_enabled and iteration % log_interval == 0
                        _loss_vals = [_loss_sample]
                    else:
                        _emit = step_window.push(_loss_sample)
                        _do_step_log = wandb_enabled and _emit
                        _loss_vals = list(step_window.buf)
                    if _do_step_log:
                        from src.utils.rolling_logging import spread
                        wandb_logs = {}
                        for k in _loss_sample:
                            spread([s[k] for s in _loss_vals if k in s], k, wandb_logs)

                        # Add Modulator metrics if enabled (single-iteration semantics —
                        # not part of the rolling window; see note above).
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
                            # Commit 7: Time/sps_env for rPPO branch. Mirrors dreamer_srl_main.py:L590
                            "Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9),
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

                # Experiment eval: cheap once-per-iteration poll for finished async
                # eval results (fix #1: fully failure-isolated at BOTH this call site and
                # inside the helper -- see _poll_and_log_experiment_results docstring -- so a
                # malformed result.json, a NAS glob hiccup, or a wandb.log raise can never
                # stall or crash the training loop).
                if experiment_eval_enabled:
                    try:
                        _poll_and_log_experiment_results(experiment_state, results_dir, iteration,
                                                     global_step, wandb_enabled, args.quiet,
                                                     experiment_eval_cfg["log_measures"],
                                                     experiment_eval_cfg["log_conditions"])
                    except Exception as e:
                        print(f"[experiment-eval] WARNING: poll skipped (non-fatal): {e}")

                # Async checkpoint-video render: cheap once-per-iteration poll for a
                # finished render child → MP4 upload from THIS parent process (sole
                # WandB writer), stamped step=checkpoint_pct exactly as the legacy
                # blocking path did. Failure-isolated like the experiment-eval poll.
                if async_video_render:
                    try:
                        poll_render(async_render_state, wandb_enabled=wandb_enabled,
                                    step=global_step, upload_step_mode='checkpoint_pct',
                                    quiet=args.quiet)
                    except Exception as e:
                        print(f"[render] WARNING: poll skipped (non-fatal): {e}")

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

                    if ckpt_data:
                        pbar.write(f"[CHECKPOINT] Saving model at episode {total_episodes_completed} (Iteration {iteration})...")
                        checkpointer.save(total_episodes_completed, args=ocp.args.StandardSave(ckpt_data))
                        checkpointer.wait_until_finished()  # Ensure sync for stability

                        # Async experiment eval (non-blocking, on-node CPU subprocess).
                        # Failure-isolated: any error here is swallowed so training never
                        # stalls (fix #1 -- see also the two internal try/excepts inside
                        # _maybe_dispatch_experiment_eval for the Popen-construction case).
                        if algorithm == "RecurrentPPO" and experiment_eval_enabled:
                            try:
                                _maybe_dispatch_experiment_eval(
                                    experiment_eval_cfg, experiment_state, results_dir, models_dir,
                                    total_episodes_completed, global_step, iteration,
                                    args.quiet,
                                )
                            except Exception as e:
                                pbar.write(f"[experiment-eval] dispatch skipped (non-fatal): {e}")

                        # Trigger evaluation after checkpoint
                        vis_flag = config.get_mandatory('visualization.enabled')
                        eval_v_flag = config.get_mandatory('training.video_during_training')
                        eval_s_flag = config.get_mandatory('training.stats_during_training')

                        if eval_v_flag or eval_s_flag:
                            if args.debug:
                                print(f"  [DEBUG] Starting evaluation... Video={eval_v_flag}, Stats={eval_s_flag}", flush=True)
                            try:
                                from src.utils.evaluation_core import evaluate_jax_checkpoint
                                # Eval seed comes from configs/evaluation/default.yaml
                                # `testing.seed` -- NOT the training seed. Re-using the
                                # training seed made eval scenarios follow it, so a run
                                # launched with --seed 0 evaluated on a predator-free
                                # draw. Same key the standalone evaluator uses
                                # (evaluation.py:250) and the dreamer-srl driver.
                                
                                # Pass 1: Video
                                if eval_v_flag:
                                    video_eps = config.get_mandatory('training.eval_video_episodes')
                                    if args.debug:
                                        print(f"  [EVAL] Pass 1: Video (eps={video_eps})")
                                    evaluate_jax_checkpoint(
                                        model=model,
                                        params=params, config=config, num_episodes=video_eps,
                                        seed=config.get_mandatory('testing.seed'),
                                        results_dir=results_dir, checkpoint_pct=total_episodes_completed,
                                        render_video=True, record_stats=False, wandb_enabled=wandb_enabled, debug=args.debug,
                                        quiet=not args.debug, num_envs=1, device=jax.config.values['jax_default_device'],
                                        # Async render (non-blocking Popen dispatch) only when
                                        # training.async_video_render; None keeps the legacy
                                        # blocking render inside evaluate_jax_checkpoint.
                                        async_render_state=async_render_state if async_video_render else None,
                                    )
                                
                                # Pass 2: Stats
                                eval_results = None
                                if eval_s_flag:
                                    stats_eps = config.get_mandatory('training.eval_stats_episodes')
                                    stats_envs = config.get_mandatory('training.eval_stats_num_envs')
                                    if args.debug:
                                        print(f"  [EVAL] Pass 2: Stats (eps={stats_eps}, envs={stats_envs})")
                                    eval_results = evaluate_jax_checkpoint(
                                        model=model,
                                        params=params, config=config, num_episodes=stats_eps,
                                        seed=config.get_mandatory('testing.seed'),
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
                                    # (`subprocess` and `sys` are already imported at
                                    # module level; a redundant local import here made
                                    # `sys` a LOCAL of main() for its whole body, so any
                                    # earlier use of `sys` in main() raised
                                    # UnboundLocalError.)
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
            training_interrupted = True
        else:
            training_interrupted = False

    # Final cleanup
    # Experiment eval: bounded drain of any in-flight subprocess so its series reach
    # WandB before the run closes (fix #4). Sharply shortened on a Ctrl-C exit -- the user
    # asked training to stop NOW, not in up to 300s (see _drain_experiment_results docstring).
    if experiment_eval_enabled:
        try:
            _drain_experiment_results(experiment_state, results_dir, iteration, global_step,
                                  wandb_enabled, args.quiet, interrupted=training_interrupted,
                                  focus_measures=experiment_eval_cfg["log_measures"],
                                  focus_conds=experiment_eval_cfg["log_conditions"])
        except Exception as e:
            print(f"[experiment-eval] WARNING: final drain skipped (non-fatal): {e}")

    # Async checkpoint-video render: bounded drain (300s normal / 10s Ctrl-C) that
    # ENDS WITH A POLL, so an in-flight render finishing within the window still
    # gets its MP4 uploaded before wandb.finish() (plan-reviewer finding 1). On
    # timeout the child is deliberately left running (finite CPU-only work; only
    # the upload is missed — the MP4 still lands on disk).
    if async_video_render:
        try:
            drain_render(async_render_state, wandb_enabled=wandb_enabled,
                         step=global_step, upload_step_mode='checkpoint_pct',
                         interrupted=training_interrupted, quiet=args.quiet)
        except Exception as e:
            print(f"[render] WARNING: final drain skipped (non-fatal): {e}")

    print(f"Training complete. Results saved to {results_dir}")
    if wandb_enabled:
        wandb.finish()
    checkpointer.close()

if __name__ == "__main__":
    main()
