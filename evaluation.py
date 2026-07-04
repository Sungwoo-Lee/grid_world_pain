"""
JAX Evaluation Script for GridWorld RL Agents.

This script mirrors evaluation.py but uses JAX-native components:
1. Loads configuration saved during training.
2. Loads JAX model checkpoint.
3. Runs deterministic evaluation episodes.
4. Generates evaluation statistics.

Arguments:
- `--results_dir <path>`: (Required) Path to results directory of the run.
- `--episodes <int>`: Number of evaluation episodes.
- `--seed <int>`: Override testing seed.
- `--checkpoint <str>`: Specific checkpoint name or path.
- `--wandb-run-path <str>`: WandB run path for uploads.
- `--render / --no-render`: Toggle video rendering (BooleanOptionalAction).
- `--device <str>`: Device to use: 'cpu', 'gpu', or specific ID like 'cuda:1', 'gpu:0'.

Usage:
    python evaluation.py --results_dir results/JAX_RecurrentPPO/my_run --episodes 10
    python evaluation.py --results_dir results/JAX_RecurrentPPO/my_run --no-render --device cuda:1
"""
import os
import argparse

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

import glob
import re
import yaml
import jax
import jax.numpy as jnp
from flax import nnx
from src.utils.config import Config
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.environment.config_loader import load_env_params
from src.models.recurrent_ppo_network import ActorCriticRNN
from src.environment.wrapper import ParallelEnv
from src.environment.core import jax_step, jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown
from src.utils.evaluation_core import evaluate_jax_checkpoint

def peel_nnx_state(st):
    """
    Recursively removes 'value' keys from restored NNX states.
    Orbax often saves Param objects as {'value': array}.
    NNX expects the nested dict structure without the 'value' leaf if updating from a dict.
    """
    if isinstance(st, dict):
        if 'value' in st:
            # If it's a leaf Param-like dict, return the value
            return st['value']
        return {k: peel_nnx_state(v) for k, v in st.items()}
    return st


def _merge_restored_into_module_state(module_state, restored_state, _path="", _missing=None):
    """
    Recursively copy leaves from restored_state into the structure of module_state.
    Handles Orbax restoring with string keys ('0','1') where module has int keys (0,1).
    Returns a new dict with module_state structure but values from restored_state.

    Finding L3 (docs/reviews/diag_v3_pipeline_jax.md): any param key/subtree present
    in module_state but ABSENT from restored_state is left at its freshly-initialised
    (random) value -- previously with no warning or error. This now records the
    dotted path of every such uncovered leaf into `_missing` (a list, mutated in
    place) so callers can assert full checkpoint coverage via `_assert_full_restore`
    instead of silently proceeding with a half-random model.
    """
    if _missing is None:
        _missing = []
    if isinstance(module_state, dict):
        out = {}
        for k in module_state.keys():
            sub_path = f"{_path}.{k}" if _path else str(k)
            # Restored may have str(k) when module has int k (e.g. Sequential indices)
            rkey = k if k in restored_state else (str(k) if str(k) in restored_state else None)
            if rkey is not None:
                out[k] = _merge_restored_into_module_state(module_state[k], restored_state[rkey], sub_path, _missing)
            else:
                out[k] = module_state[k]  # keep original if not in restored
                _collect_uncovered_leaf_paths(module_state[k], sub_path, _missing)
        return out
    # Leaf (array or other): use restored value if we have a matching leaf
    return restored_state


def _collect_uncovered_leaf_paths(node, path, missing):
    """Recursively append the dotted path of every leaf under `node` to `missing`."""
    if isinstance(node, dict):
        for k, v in node.items():
            _collect_uncovered_leaf_paths(v, f"{path}.{k}" if path else str(k), missing)
    else:
        missing.append(path)


def _assert_full_restore(missing_paths, label):
    """Fail loudly (Finding L3) if any param leaf was left at its randomly
    initialised value instead of being restored from the checkpoint. Without
    this, a train/eval structural mismatch silently loads a half-random model
    and reports meaningless survival numbers with no signal anything is wrong."""
    if missing_paths:
        raise ValueError(
            f"[{label}] Incomplete checkpoint restore: {len(missing_paths)} param "
            f"leaf(ves) left at randomly-initialised values (absent from checkpoint):\n  "
            + "\n  ".join(sorted(missing_paths))
        )


def _load_eval_config(config_path, eval_default_path, vis_config_path):
    """Load an environment/training config YAML and apply the same
    evaluation-defaults + visualization-defaults merge as the top-level config
    load in main(), so a per-checkpoint stage config (Finding L2, below) is
    prepared identically to the run's own stage-0 `models/config.yaml`."""
    with open(config_path, 'r') as f:
        cfg = Config(yaml.safe_load(f))
    if os.path.exists(eval_default_path):
        cfg.merge(Config.load_yaml(eval_default_path))
    if os.path.exists(vis_config_path):
        cfg.merge(Config.load_yaml(vis_config_path))
    return cfg


def _resolve_checkpoint_stage_config(models_dir, restored, quiet=False):
    """Resolve a checkpoint's OWN curriculum-stage environment config for a
    continual (multi-stage) run (Finding L2, docs/develop/active/diagnosis/
    v3_pipeline_correctness_diagnosis.md), instead of always evaluating every
    checkpoint against the shared stage-0 `models/config.yaml` loaded once for
    the whole `--all` batch.

    Mirrors scripts/eval/eval_rollout.py's `_resolve_continual_stage_config`
    (Finding E, commit a3ab4cc), adapted for evaluation.py's `--all` loop where
    a SINGLE invocation evaluates MANY checkpoints that may each belong to a
    different curriculum stage. Detection is conservative: this only fires
    when `models/schedule.yaml` exists (the continual-run signature); a
    non-continual run has no such file and this returns None (unchanged,
    single-config behavior).

    The checkpoint's stage is read from its OWN saved `restored['stage']`
    field -- already present in the full checkpoint payload dict returned by
    `checkpointer.restore(iteration)` for both RecurrentPPO and DreamerV3
    (train.py ~line 2436/2447), so no extra partial-restore call is needed --
    rather than recomputed from episode_boundaries: train.py's per-iteration
    stage-transition check means a checkpoint's recorded stage can lag one
    behind what a boundary-only recompute would give (see Finding E), so the
    checkpoint's own field is ground truth.
    """
    schedule_path = os.path.join(models_dir, "schedule.yaml")
    if not os.path.exists(schedule_path):
        return None  # not a continual run; unchanged behavior

    if 'stage' not in restored:
        raise ValueError(
            f"Continual run detected ({schedule_path}) but the checkpoint "
            f"payload has no 'stage' field; cannot resolve its curriculum-"
            f"stage config."
        )

    sched = Config.load_yaml(schedule_path)
    stage_names = sched.get_mandatory("continual.stage_names")
    stage_idx = int(restored['stage'])

    if not (0 <= stage_idx < len(stage_names)):
        raise ValueError(
            f"Checkpoint reports stage index {stage_idx}, out of range for "
            f"{len(stage_names)} stages in {schedule_path}."
        )

    stage_cfg_path = os.path.join(
        models_dir, f"stage_{stage_idx:02d}_{stage_names[stage_idx]}.yaml"
    )
    if not os.path.exists(stage_cfg_path):
        raise ValueError(
            f"Resolved stage config {stage_cfg_path} does not exist "
            f"(stage {stage_idx}: {stage_names[stage_idx]})."
        )

    if not quiet:
        print(f"  [Stage] Continual run; checkpoint belongs to stage "
              f"{stage_idx} ('{stage_names[stage_idx]}'). Using stage config: "
              f"{stage_cfg_path}")

    return stage_cfg_path


def main():
    parser = argparse.ArgumentParser(description="JAX GridWorld Evaluation")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory (Required)")
    parser.add_argument("--episodes", type=int, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, help="Override testing seed")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint name or path to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints found in the directory")
    parser.add_argument("--wandb-run-path", type=str, help="WandB run path (e.g. 'entity/project/run_id') for uploads")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, help="Toggle video recording (adds --no-render)")
    parser.add_argument("--num_envs", type=int, help="Number of parallel envs (default from config testing.num_envs or 1). Effective parallelism is min(episodes, num_envs).")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints (e.g. episode-ticket lifecycle in parallel eval).")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use for evaluation (e.g. cpu, gpu, cuda:0, gpu:1)")
    args = parser.parse_args()

    results_dir = args.results_dir
    
    # --- Finalize Device Selection ---
    try:
        device_str = args.device.lower()
        if ":" in device_str or device_str in ["gpu", "cuda"]:
            # If we set CUDA_VISIBLE_DEVICES, JAX sees the target GPU as index 0
            jax.config.update("jax_default_device", jax.devices("cuda")[0])
        elif device_str == "cpu":
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
    except Exception as e:
        print(f"Warning: Device configuration for '{args.device}' failed ({e}). Using JAX default: {jax.devices()[0]}")

    models_dir = os.path.join(results_dir, "models")
    config_path = os.path.join(models_dir, "config.yaml")

    # 1. Load saved configuration
    if not os.path.exists(config_path):
        print(f"Error: Training configuration file not found at {config_path}")
        return

    print(f"Loading training configuration from {config_path}...")
    # 1.1 Merge evaluation defaults (Strictly) + visualization config (icons, layout)
    # so evaluation video matches training/tuningEnv.
    eval_default_path = "configs/evaluation/default.yaml"
    vis_config_path = "configs/visualization/default.yaml"
    config = _load_eval_config(config_path, eval_default_path, vis_config_path)

    # 2. Resolve Parameters (No Safe Defaults)
    seed = args.seed or config.get_mandatory('testing.seed')
    num_episodes = args.episodes or config.get_mandatory('testing.evaluation_episodes')
    algorithm = config.get_mandatory('agent.algorithm')
    
    # Reconstruct JAX EnvParams from saved configuration
    # Note: load_env_params handles the mapping from YAML structure to JAX arrays
    params = load_env_params(config)

    # Determine if video rendering should be enabled
    if args.render is not None:
        render_video = args.render
    else:
        render_video = config.get('testing.render_video', False)

    num_envs = args.num_envs if args.num_envs is not None else config.get('testing.num_envs', 1)

    # 3. Print Summary
    print(f"\n{'='*50}")
    print(f"JAX Evaluation: {algorithm}")
    print(f"{'='*50}")
    print(f"Grid: {params.height}x{params.width}")
    print(f"Episodes: {num_episodes}")
    print(f"Num envs: {num_envs} (effective: {min(num_episodes, num_envs)})")
    print(f"Seed: {seed}")
    # Show default device if set, else first available
    actual_device = jax.config.values.get("jax_default_device") or jax.devices()[0]
    print(f"Device: {jax.default_backend()} ({actual_device})")
    print(f"Video Rendering: {'Enabled' if render_video else 'Disabled'}")
    print(f"{'='*50}\n")

    # 4. Find checkpoints
    # Note: Modern Orbax just uses iteration numbers as folder names.
    checkpoints = []
    
    if args.checkpoint:
        # Explicit path or numeric iteration
        ckpt_path = os.path.join(models_dir, args.checkpoint)
        if os.path.isdir(ckpt_path):
            checkpoints.append(ckpt_path)
        else:
            print(f"Error: Checkpoint '{args.checkpoint}' not found at {ckpt_path}")
            return
    elif args.all:
        # Find all iteration subdirectories
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()]
        checkpoints = [os.path.join(models_dir, d) for d in sorted(subdirs, key=int)]
    else:
        # Latest numeric subdirectory
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()]
        if subdirs:
            latest = max(subdirs, key=int)
            checkpoints = [os.path.join(models_dir, latest)]

    if not checkpoints:
        print(f"No valid checkpoints found in {models_dir}")
        return

    # 5. Initialize WandB if requested
    if args.wandb_run_path and WANDB_AVAILABLE:
        try:
            wandb_login(quiet=True)
            path_parts = args.wandb_run_path.strip().split('/')
            if len(path_parts) == 3:
                wandb.init(entity=path_parts[0], project=path_parts[1], id=path_parts[2], resume="must", job_type="evaluation")
            else:
                wandb.init(id=args.wandb_run_path, resume="must", job_type="evaluation")
        except Exception as e:
            print(f"WandB init failed: {e}")

    # 6. Evaluate each checkpoint
    import orbax.checkpoint as ocp
    
    # Matching CheckpointManager setup
    checkpointer = ocp.CheckpointManager(
        os.path.abspath(models_dir),
        checkpointers=ocp.StandardCheckpointer()
    )
    
    for ckpt_path in checkpoints:
        iteration_str = os.path.basename(ckpt_path)
        iteration = int(iteration_str)
        print(f"\nEvaluating Iteration: {iteration}")

        # Restore the full checkpoint payload FIRST (before building env params /
        # model): for a continual run this dict already carries the checkpoint's
        # own 'stage' field (train.py ~line 2436/2447), which the Finding L2 fix
        # below uses to resolve this checkpoint's OWN curriculum-stage config
        # instead of always using the shared stage-0 config.yaml loaded above.
        restored = checkpointer.restore(iteration)

        # Finding L2 (docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md):
        # evaluate each checkpoint against ITS OWN stage ENVIRONMENT, not always
        # stage 0. Non-continual runs (no schedule.yaml) fall through to the
        # shared `params` loaded once above -- unchanged behavior. Mirrors
        # eval_rollout.py's Finding E split: only the environment config swaps
        # per checkpoint -- agent hyperparameters (`agent.*`) are read from the
        # stage-0 `config` throughout (below), matching train.py, where the
        # SAME policy is trained continuously across stages and only the
        # environment changes; some historical continual runs' per-stage
        # config dumps do not even carry an `agent:` section (see stage_XX
        # dumps predating the schedule-builder's later self-containment fix),
        # so agent hyperparams must not be read from the per-stage dump.
        stage_cfg_path = _resolve_checkpoint_stage_config(models_dir, restored, quiet=False)
        if stage_cfg_path is not None:
            with open(stage_cfg_path, 'r') as f:
                ckpt_env_config = Config(yaml.safe_load(f))
            ckpt_params = load_env_params(ckpt_env_config)
        else:
            ckpt_params = params

        # Reconstruct Model based on algorithm
        test_state = jax_reset(ckpt_params, jax.random.PRNGKey(seed))
        obs = get_observation(test_state, ckpt_params)
        input_dim = obs.shape[0]

        # Dynamic action dimension (matching train_jax.py)
        rest_enabled = ckpt_params.rest_action_enabled
        eat_enabled = ckpt_params.eat_action_enabled
        action_dim = 4 + int(rest_enabled) + int(eat_enabled)

        print(f"  [Model] Input Dim: {input_dim}, Action Dim: {action_dim}")

        rngs = nnx.Rngs(jax.random.PRNGKey(seed))

        if algorithm == "RecurrentPPO":
            rnn_type = config.get_mandatory('agent.rnn_type')
            activation = config.get_mandatory('agent.activation')

            # Read neuromodulation config exactly as train.py does (train.py:745-747):
            # the WHOLE dict, not a hand-picked whitelist, so eval and train share one
            # source of truth and no future key (e.g. memory_clip) can silently drift
            # out of sync (Finding A #1 / #4, docs/reviews/diag_v3_pipeline_jax.md).
            modulation_config = config.get('agent.modulation')
            if modulation_config is not None and modulation_config.get('type') is None:
                modulation_config = None

            model = ActorCriticRNN(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=config.get_mandatory('agent.hidden_size'),
                rngs=rngs,
                rnn_type=rnn_type,
                activation=activation,
                modulation_config=modulation_config,
                observation_breakdown=get_observation_breakdown(ckpt_params),
                encoding_config=config.to_dict().get('agent', {})
            )

            if 'model' in restored:
                from flax.nnx.statelib import to_pure_dict
                model_state = restored['model']
                peeled_state = peel_nnx_state(model_state)

                # Robustly merge restored state into current model structure
                current_struct = to_pure_dict(nnx.state(model, nnx.Param))
                missing = []
                merged_state = _merge_restored_into_module_state(current_struct, peeled_state, _missing=missing)
                # Finding L3: fail loudly on a train/eval structural mismatch instead of
                # silently loading a half-random model.
                _assert_full_restore(missing, "RecurrentPPO model")
                nnx.update(model, merged_state)
                print(f"  [Success] Restored RecurrentPPO model weights from iteration {iteration}")
            else:
                print(f"  [Warning] 'model' key not found in restored checkpoint. Keys: {list(restored.keys())}")

        elif algorithm == "DreamerV3":
            from src.models.dreamer_v3_trainer import DreamerTrainer
            # Build the trainer exactly as train.py does (train.py:815-817): pass the
            # full Config object (DreamerTrainer.__init__ calls config.get_mandatory(...),
            # which a plain dict does not support) plus obs_breakdown and
            # modulation_config, so the rebuilt world model matches the shape train.py
            # saved (Finding A #2, docs/reviews/diag_v3_pipeline_jax.md).
            dreamer_mod_config = config.get('agent.modulation')
            if dreamer_mod_config is not None and dreamer_mod_config.get('type') is None:
                dreamer_mod_config = None
            trainer = DreamerTrainer(
                input_dim, action_dim, config, rngs=rngs,
                obs_breakdown=get_observation_breakdown(ckpt_params),
                modulation_config=dreamer_mod_config,
            )

            from flax.nnx.statelib import to_pure_dict
            # Peel Orbax 'value' wrappers
            wm_restored = peel_nnx_state(restored['wm'])
            actor_restored = peel_nnx_state(restored['actor'])
            critic_restored = peel_nnx_state(restored['critic'])
            # Merge restored state into module state structure (handles str vs int keys for Sequential)
            wm_struct = to_pure_dict(nnx.state(trainer.agent.wm, nnx.Param))
            actor_struct = to_pure_dict(nnx.state(trainer.agent.ac.actor, nnx.Param))
            critic_struct = to_pure_dict(nnx.state(trainer.agent.ac.critic, nnx.Param))
            missing = []
            wm_state = _merge_restored_into_module_state(wm_struct, wm_restored, "wm", missing)
            actor_state = _merge_restored_into_module_state(actor_struct, actor_restored, "actor", missing)
            critic_state = _merge_restored_into_module_state(critic_struct, critic_restored, "critic", missing)
            # Finding L3: fail loudly on a train/eval structural mismatch instead of
            # silently loading a half-random model.
            _assert_full_restore(missing, "DreamerV3 model")
            nnx.update(trainer.agent.wm, wm_state)
            nnx.update(trainer.agent.ac.actor, actor_state)
            nnx.update(trainer.agent.ac.critic, critic_state)
            model = trainer.agent
        else:
            raise ValueError(f"Unsupported algorithm for JAX evaluation: {algorithm}")
            
        evaluate_jax_checkpoint(
            model, ckpt_params, config, num_episodes, seed, results_dir, iteration,
            render_video=render_video, quiet=False, num_envs=num_envs, debug=args.debug
        )

    checkpointer.close()

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
