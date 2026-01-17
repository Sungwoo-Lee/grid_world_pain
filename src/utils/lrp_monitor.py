
import torch
import torch.nn as nn
from captum.attr import LayerLRP, LRP
import numpy as np

class LRPMonitor:
    """
    Monitors and computes Layerwise Relevance Propagation (LRP) for a PyTorch model.
    """
    def __init__(self, model, tracked_layers=None):
        """
        Args:
            model (nn.Module): The PyTorch model to explain.
            tracked_layers (list): List of layer types to track (e.g. [nn.Linear, nn.Conv2d]).
        """
        self.model = model
        self.tracked_layers = tracked_layers if tracked_layers else [nn.Linear, nn.Conv2d]
        self.target_layers = self._find_target_layers(model)
        self.lrp = LRP(model)
        # We might need per-layer LRP if global LRP doesn't give intermediate scores easily
        # Captum's LRP returns input attribution.
        # To get layer-wise, we use LayerLRP.
        # However, LayerLRP is for a specific layer. We want ALL relevant layers.
        # Creating multiple LayerLRP objects might be expensive or redundant.
        # Actually, standard LRP creates a graph. We might be able to hook into it.
        # But for simplicity and reliability with Captum, we will instantiate LayerLRP for each target layer.
        self.layer_lrps = {name: LayerLRP(model, layer) for name, layer in self.target_layers.items()}

    def _find_target_layers(self, model):
        """
        recursively find layers to track.
        """
        layers = {}
        for name, module in model.named_modules():
            # Check if module is an instance of any tracked type
            if any(isinstance(module, t) for t in self.tracked_layers):
                # storage key: simple name
                layers[name] = module
        return layers

    def compute_relevance(self, input_tensor, target_action):
        """
        Computes relevance scores for all tracked layers for a given input and target action.

        Args:
            input_tensor (torch.Tensor): Input to the model.
            target_action (int): The target class index to explain.

        Returns:
            dict: {layer_name: relevance_numpy_array}
        """
        attributions = {}
        
        # Ensure input requires grad (Captum usually handles this, but good practice)
        if not input_tensor.requires_grad:
             input_tensor.requires_grad = True

        # For each layer, compute attribution
        # This might be slow if we do it sequentially for many layers.
        # But for debugging/viz it's acceptable.
        for name, layer_lrp in self.layer_lrps.items():
            try:
                # attribute returns a Tensor or tuple of Tensors
                attr = layer_lrp.attribute(input_tensor, target=target_action)
                # Detach and convert to numpy
                if isinstance(attr, torch.Tensor):
                    attributions[name] = attr.detach().cpu().numpy()
                elif isinstance(attr, tuple):
                     attributions[name] = attr[0].detach().cpu().numpy()
            except Exception as e:
                # Some layers (e.g. RNNs) might be tricky with standard LRP rules in Captum without modifications
                # or if the graph structure is dynamic.
                # Fallback or skip
                # print(f"Warning: LRP failed for layer {name}: {e}")
                pass
        
        return attributions
