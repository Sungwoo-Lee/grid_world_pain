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
from src.environment.sensor import get_observation
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


def _merge_restored_into_module_state(module_state, restored_state):
    """
    Recursively copy leaves from restored_state into the structure of module_state.
    Handles Orbax restoring with string keys ('0','1') where module has int keys (0,1).
    Returns a new dict with module_state structure but values from restored_state.
    """
    if isinstance(module_state, dict):
        out = {}
        for k in module_state.keys():
            # Restored may have str(k) when module has int k (e.g. Sequential indices)
            rkey = k if k in restored_state else (str(k) if str(k) in restored_state else None)
            if rkey is not None:
                out[k] = _merge_restored_into_module_state(module_state[k], restored_state[rkey])
            else:
                out[k] = module_state[k]  # keep original if not in restored
        return out
    # Leaf (array or other): use restored value if we have a matching leaf
    return restored_state

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
    with open(config_path, 'r') as f:
        saved_config_dict = yaml.safe_load(f)
        config = Config(saved_config_dict)

    # 1.1 Merge evaluation defaults (Strictly)
    eval_default_path = "configs/evaluation/default.yaml"
    if os.path.exists(eval_default_path):
        eval_defaults = Config.load_yaml(eval_default_path)
        config.merge(eval_defaults)
    # Merge visualization config (icons, layout) so evaluation video matches training/tuningEnv
    vis_config_path = "configs/visualization/visualization.yaml"
    if os.path.exists(vis_config_path):
        vis_defaults = Config.load_yaml(vis_config_path)
        config.merge(vis_defaults)

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
        
        # Reconstruct Model based on algorithm
        test_state = jax_reset(params, jax.random.PRNGKey(seed))
        obs = get_observation(test_state, params)
        input_dim = obs.shape[0]
        
        # Dynamic action dimension (matching train_jax.py)
        rest_enabled = params.rest_action_enabled
        eat_enabled = params.eat_action_enabled
        action_dim = 4 + int(rest_enabled) + int(eat_enabled)
        
        print(f"  [Model] Input Dim: {input_dim}, Action Dim: {action_dim}")
        
        rngs = nnx.Rngs(jax.random.PRNGKey(seed))
        
        if algorithm == "RecurrentPPO":
            rnn_type = config.get_mandatory('agent.rnn_type')
            activation = config.get_mandatory('agent.activation')
            
            # Read modulation config (type can be null = disabled baseline)
            mod_type = config.get('agent.modulation.type')
            if mod_type is not None:
                modulation_config = {
                    'type': mod_type,
                    'mod_hidden_size': config.get_mandatory('agent.modulation.mod_hidden_size'),
                    'grouping_size': config.get_mandatory('agent.modulation.grouping_size'),
                    'percept_bias_init': config.get_mandatory('agent.modulation.percept_bias_init'),
                    'memory_bias_init': config.get_mandatory('agent.modulation.memory_bias_init'),
                    'temp_clip': config.get_mandatory('agent.modulation.temp_clip'),
                }
            else:
                modulation_config = None
            
            model = ActorCriticRNN(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=config.get_mandatory('agent.hidden_size'),
                rngs=rngs,
                rnn_type=rnn_type,
                activation=activation,
                modulation_config=modulation_config
            )
            
            # Restore via manager (returns the raw dict)
            restored = checkpointer.restore(iteration)
            
            # Use the 'model' key as saved in train_jax.py
            if 'model' in restored:
                model_state = restored['model']
                peeled_state = peel_nnx_state(model_state)
                nnx.update(model, peeled_state)
                print(f"  [Success] Restored RecurrentPPO model weights from iteration {iteration}")
            else:
                print(f"  [Warning] 'model' key not found in restored checkpoint. Keys: {list(restored.keys())}")
            
        elif algorithm == "DreamerV3":
            from src.models.dreamer_v3_trainer import DreamerTrainer
            dreamer_config = {
                'model_lr': config.get_mandatory('agent.model_lr'),
                'actor_lr': config.get_mandatory('agent.actor_lr'),
                'value_lr': config.get_mandatory('agent.value_lr'),
                'batch_size': config.get_mandatory('agent.batch_size'),
                'sequence_length': config.get_mandatory('agent.sequence_length'),
            }
            trainer = DreamerTrainer(input_dim, action_dim, dreamer_config, rngs=rngs)
            
            restored = checkpointer.restore(iteration)
            from flax.nnx.statelib import to_pure_dict
            # Peel Orbax 'value' wrappers
            wm_restored = peel_nnx_state(restored['wm'])
            actor_restored = peel_nnx_state(restored['actor'])
            critic_restored = peel_nnx_state(restored['critic'])
            # Merge restored state into module state structure (handles str vs int keys for Sequential)
            wm_struct = to_pure_dict(nnx.state(trainer.agent.wm, nnx.Param))
            actor_struct = to_pure_dict(nnx.state(trainer.agent.ac.actor, nnx.Param))
            critic_struct = to_pure_dict(nnx.state(trainer.agent.ac.critic, nnx.Param))
            wm_state = _merge_restored_into_module_state(wm_struct, wm_restored)
            actor_state = _merge_restored_into_module_state(actor_struct, actor_restored)
            critic_state = _merge_restored_into_module_state(critic_struct, critic_restored)
            nnx.update(trainer.agent.wm, wm_state)
            nnx.update(trainer.agent.ac.actor, actor_state)
            nnx.update(trainer.agent.ac.critic, critic_state)
            model = trainer.agent 
        else:
            raise ValueError(f"Unsupported algorithm for JAX evaluation: {algorithm}")
            
        evaluate_jax_checkpoint(
            model, params, config, num_episodes, seed, results_dir, iteration,
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
