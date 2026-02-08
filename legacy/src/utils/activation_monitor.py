
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import h5py


class ActivationMonitor:
    def __init__(self, model, tracked_layers=None):
        """
        Monitors and stores activations of a PyTorch model.

        Args:
            model (nn.Module): The model to monitor.
            tracked_layers (list of type, optional): List of layer types to track. 
                                                     Defaults to [nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU].
        """
        self.model = model
        self.activations = {}
        # Register hooks
        self.hooks = []
        
        # 1. Register Input Pre-hook (to capture 'Input' first)
        self.hooks.append(model.register_forward_pre_hook(self._hook_input))
        
        # 2. Register Layer hooks
        if tracked_layers is None:
             tracked_types = (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU, nn.RNN)
        else:
             tracked_types = tuple(tracked_layers)
             
        self._register_hooks_recursive(model, tracked_types)
        
        # Storage for the current step
        self.current_activations = {}
        
        # Storage for history (list of dicts)
        self.history = []

    def _hook_input(self, module, args):
        """
        Pre-hook to capture model input.
        """
        try:
            # args is a tuple of input arguments
            if args:
                inp = args[0] # Assume first arg is main input
                if isinstance(inp, torch.Tensor):
                     val = inp.detach().cpu().numpy()
                     self.current_activations["Input"] = val
        except Exception as e:
            print(f"Error capturing input: {e}")

    def _register_hooks_recursive(self, module, tracked_types, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, tracked_types):
                self._hook_layer(child, full_name)
            
            # Recurse
            self._register_hooks_recursive(child, tracked_types, full_name)

    def _hook_layer(self, layer, name):
        def hook(module, input, output):
            # Handle different output types (e.g., RNNs return tuple)
            if isinstance(output, tuple):
                # For RNNs, output[0] is usually the sequence of hidden states
                data = output[0]
            else:
                data = output
            
            # We only care about the last batch/step for visualization usually
            # But let's detach and cpu
            if isinstance(data, torch.Tensor):
                self.current_activations[name] = data.detach().cpu().numpy()
        
        handle = layer.register_forward_hook(hook)
        self.hooks.append(handle)
        
    def get_current_activations(self):
        """Returns activations from the most recent forward pass."""
        return self.current_activations
    
    def record_step(self):
        """Saves current activations to history and clears current buffer for safety (optional)."""
        # Deep copy the current activations to history
        step_data = {k: v.copy() for k, v in self.current_activations.items()}
        self.history.append(step_data)
        
    def clear_history(self):
        self.history = []
        self.current_activations = {}

    def save_history(self, filepath):
        """
        Saves the history to an HDF5 file.
        Structure: 
          - Root Group
             - Dataset "Input": (Time, ...)
             - Dataset "LayerName": (Time, ...)
        """
        if not self.history:
            return
            
        # Consolidate history: list of dicts -> dict of lists
        keys = self.history[0].keys()
        consolidated = {k: [] for k in keys}
        
        for step_data in self.history:
            for k in keys:
                 if k in step_data:
                     consolidated[k].append(step_data[k])
        
        # Convert to numpy arrays
        final_data = {k: np.array(v) for k, v in consolidated.items()}
        
        # Save to HDF5
        try:
            with h5py.File(filepath, 'w') as f:
                for k, v in final_data.items():
                    f.create_dataset(k, data=v, compression="gzip")
            print(f"Saved activation history to {filepath}")
        except Exception as e:
            print(f"Failed to save history to HDF5: {e}")


    def close(self):
        """Removes all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
