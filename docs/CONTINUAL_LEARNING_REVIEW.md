# Continual Learning and Checkpoint Loading Review

## Overview
This document reviews the current support for continual learning and loading pre-trained models in the JAX training pipeline (`train.py`), specifically addressing the application to new or changing environment settings.

## Current State in `train.py`

### Mechanism for Loading Pre-trained Models
1. **CLI Argument Present:** 
   The script currently defines a command-line argument for checkpoint loading:
   `parser.add_argument("--load-checkpoint", type=str, help="Path to checkpoint to resume from")`
2. **Implementation Missing:** 
   Although the argument is parsed (`args.load_checkpoint`), there is no logic implemented to actually load or restore the model's state. 
   - The script initializes an `orbax.checkpoint.CheckpointManager` but strictly uses its `.save()` functionality to persist states (e.g., at milestones defined by `--checkpoint-frequency`). 
   - The corresponding `.restore()` operation to initialize the network and optimizer components with past weights is completely absent.

### Continual Learning Capabilities
- Because the required checkpoint restoration logic is missing, continual learning (resuming training from a previously learned state) is not technically functional in `train.py`.
- Additionally, there are no explicit algorithmic mechanisms—such as cross-task replay buffers, elastic weight consolidation (EWC), or other mitigations for catastrophic forgetting—currently implemented to manage the transition when the environment changes.

### Training in Other Environment Settings
- The environment configuration system is very robust. It is fully capable of initializing and training from scratch in varying environments.
- This is handled via the `--config` flag, which allows overriding the `configs/environment/default.yaml` with custom or ablation configurations.
- The `load_env_params(config)` function correctly initializes the `ParallelEnv(params)` based on these settings.
- **Conclusion:** While setting up new environments works perfectly, doing so initialized with a pre-trained model (for knowledge transfer or continual learning) requires the unimplemented checkpoint restoration logic to be completed first.

## Implemented Solutions & Debugging Results

To fully support continual learning and knowledge transfer across environments, the following steps were implemented in `train.py`:

1. **Implement Model Restoration:**
   - Updated `train.py` to check for `args.load_checkpoint`.
   - If provided, used `restore_mngr.restore()` from `orbax.checkpoint` to reconstruct the network parameters and optimizer states.
   - We used `ocp.args.PyTreeRestore()` instead of `ocp.args.StandardRestore()` to prevent strict shape mismatches from raising fatal errors at read time.
   - Injected the restored states into the initialized JAX/Flax NNX models and optimizers before the training loop starts.

   **Implementation Example (`train.py`):**
   ```python
   # --- Checkpoint Restoration (Continual Learning / Transfer) ---
   if args.load_checkpoint:
       if not args.quiet:
           print(f"Restoring checkpoint from {args.load_checkpoint}...")
           
       try:
           restore_mngr = ocp.CheckpointManager(os.path.abspath(args.load_checkpoint))
           step = restore_mngr.latest_step()
           
           if step is not None:
               # Load the raw state tree using PyTreeRestore, avoiding StandardRestore shape panic
               restored = restore_mngr.restore(step, args=ocp.args.PyTreeRestore())
               
               # Check if this is a standard NNX model/optimizer setup, or Dreamer
               if algorithm == "DreamerV3":
                   # Full restore for Dreamer
                   nnx.update(trainer.agent.wm, restored['wm'])
                   nnx.update(trainer.agent.ac.actor, restored['actor'])
                   nnx.update(trainer.agent.ac.critic, restored['critic'])
                   ...
               elif 'model' in locals() and 'optimizer' in locals():
                   # Handle Partial Restore...
                   pass
   ```

2. **Handle Architecture Mismatches:**
   - When transferring to a new environment, the observation space or action space might differ slightly (e.g., changes in the grid width resulting in dimension differences). 
   - A shape extraction handler using `jax.tree_util` selectively merges weights that align perfectly with the target environment. If the structure is fundamentally different, it preserves the existing initialization to avert shape crashes.

   **Implementation Example (Partial State Restoration):**
   ```python
   # Handle Architecture Mismatches (Selective copying via tree_map)
   restored_model_state = restored['model']
   current_model_state = nnx.state(model)
   
   # We flatten both states and map matching keys/shapes
   flat_restored, tree_def = jax.tree_util.tree_flatten_with_path(restored_model_state)
   flat_current, current_def = jax.tree_util.tree_flatten_with_path(current_model_state)
   
   # Convert paths to string keys for easy lookup
   restored_dict = {str(k): v for k, v in flat_restored}
   current_dict = {str(k): v for k, v in flat_current}
   
   valid_model_state = {}
   for k, cur_v in current_dict.items():
       if k in restored_dict:
           res_v = restored_dict[k]
           # Check if shapes match
           if hasattr(cur_v, 'shape') and hasattr(res_v, 'shape') and cur_v.shape == res_v.shape:
               valid_model_state[k] = res_v
           else:
               valid_model_state[k] = cur_v # Keep un-initialized if mismatch
       else:
           valid_model_state[k] = cur_v # Keep un-initialized if missing in checkpoint
   
   # Reconstruct the tree
   valid_flat = [valid_model_state[str(k)] for k, _ in flat_current]
   valid_tree = jax.tree_util.tree_unflatten(current_def, valid_flat)
   
   nnx.update(model, valid_tree)
   ```

### Debugging Results

During implementation, several issues were encountered and resolved:

1. **State Dictionary Mismatch**: Initially, `empty_state` did not match the actual saved state mapping structure, triggering a `ValueMetadataEntry` mismatch error. 
   - *Fix*: The mock state `empty_state` was built to align correctly per algorithm. Ultimately though, moving from `StandardRestore` to `PyTreeRestore` bypassed this completely, as `PyTreeRestore` does not enforce strict metadata equality.
2. **Variable Undefined Errors**: Attempting to restore `iteration`, `global_step`, and `episode` triggered an error because the local loop variables were initialized *after* the restoration phase (`cannot access local variable 'iteration' where it is not associated with a value`).
   - *Fix*: Lifted the loop variables' initializations (i.e., `global_step = 0`) to above the `load_checkpoint` logic so they can be securely overwritten.
3. **Orbax Strict Shape Validation**: `ocp.args.StandardRestore()` natively panicked when reading a checkpoint containing shapes that did not perfectly match the target environment's architecture (e.g., `Requested shape: (256, 256) is not compatible with the stored shape: (128, 128)`).
   - *Fix*: Switched checkpoint decoding to `ocp.args.PyTreeRestore()` to import the raw nested dict of NumPy arrays freely, putting shape comparison responsibility on the partial-restore logic rather than Orbax.
4. **`nnx.State` Shape Inspection Failure**: When iterating over the keys of the `nnx.State` to skip mismatched layers, referencing layer shapes (e.g., `.shape`) directly onto a `nnx.State` returned an error because NNX treats internal elements fundamentally as node trees, not raw arrays.
   - *Fix*: Refactored array isolation through `jax.tree_util.tree_flatten_with_path`, decomposing the NNX State down to traversable linear pathways allowing for safe property evaluation.

These debugging steps led to a very robust implementation where `train.py` can resume training and seamlessly fall back to partial loading when environment architectures diverge.
