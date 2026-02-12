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
- `--checkpoint <str>`: Specific checkpoint to evaluate.
- `--wandb-run-path <str>`: WandB run path for uploads.

Usage:
    python evaluation_jax.py --results_dir results/JAX_RecurrentPPO/my_run --episodes 10
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
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

def main():
    parser = argparse.ArgumentParser(description="JAX GridWorld Evaluation")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory (Required)")
    parser.add_argument("--episodes", type=int, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, help="Override testing seed")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint name or path to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints found in the directory")
    parser.add_argument("--wandb-run-path", type=str, help="WandB run path (e.g. 'entity/project/run_id') for uploads")
    parser.add_argument("--render-video", action="store_true", help="Enable video recording")
    args = parser.parse_args()

    results_dir = args.results_dir
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
    else:
        # If missing, we must have them via CLI or saved config
        pass

    # 2. Resolve Parameters (No Safe Defaults)
    seed = args.seed or config.get_mandatory('testing.seed')
    num_episodes = args.episodes or config.get_mandatory('testing.evaluation_episodes')
    algorithm = config.get_mandatory('agent.algorithm')
    
    # Reconstruct JAX EnvParams from saved configuration
    # Note: load_env_params handles the mapping from YAML structure to JAX arrays
    params = load_env_params(config)

    # Determine if video rendering should be enabled (CLI flag or config default)
    render_video = args.render_video or config.get('testing.render_video', False)

    # 3. Print Summary
    print(f"\n{'='*50}")
    print(f"JAX Evaluation: {algorithm}")
    print(f"{'='*50}")
    print(f"Grid: {params.height}x{params.width}")
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
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
            model = ActorCriticRNN(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=config.get_mandatory('agent.hidden_size'),
                rngs=rngs
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
                'batch_length': config.get_mandatory('agent.batch_length'),
            }
            trainer = DreamerTrainer(input_dim, action_dim, dreamer_config, rngs=rngs)
            
            restored = checkpointer.restore(iteration)
            
            # Peel and update each component
            nnx.update(trainer.agent.wm, peel_nnx_state(restored['wm']))
            nnx.update(trainer.agent.ac.actor, peel_nnx_state(restored['actor']))
            nnx.update(trainer.agent.ac.critic, peel_nnx_state(restored['critic']))
            model = trainer.agent 
        else:
            raise ValueError(f"Unsupported algorithm for JAX evaluation: {algorithm}")
            
        evaluate_jax_checkpoint(
            model, params, config, num_episodes, seed, results_dir, iteration, 
            render_video=render_video, quiet=False
        )

    checkpointer.close()

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
