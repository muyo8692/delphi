from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class MLPTopKCoder:
    """
    Treats MLP as a transcoder for interpretation purposes.

    Instead of separating up_proj and down_proj as encoder/decoder,
    this approach treats the entire MLP as a transcoder unit and
    focuses on the hidden activations directly.
    """

    def __init__(self, mlp_module: nn.Module, device: str = "cuda"):
        """
        Initialize with an MLP module from a transformer.

        Args:
            mlp_module: The MLP module from the transformer
            device: Device to use for processing
        """
        self.mlp_module = mlp_module
        self.device = device
        self.hidden_size = None  # Will be set during first forward pass

    def __call__(self, x: Tensor) -> Tensor:
        """
        Extract hidden activations from the MLP.

        Args:
            x: Input tensor from residual stream

        Returns:
            Hidden activations after up_proj and activation function
        """
        # Get the up_proj weights and compute pre-activations
        up_proj = self.mlp_module.up_proj
        pre_activations = up_proj(x)

        # Apply activation function (SiLU, GELU, etc.)
        if hasattr(self.mlp_module, "act_fn"):
            activation_fn = self.mlp_module.act_fn
        elif hasattr(self.mlp_module, "activation_fn"):
            activation_fn = self.mlp_module.activation_fn
        else:
            # Default to GELU if no activation function is found
            activation_fn = torch.nn.functional.gelu

        hidden_activations = activation_fn(pre_activations)

        # Set hidden size if not already set
        if self.hidden_size is None:
            self.hidden_size = hidden_activations.shape[-1]

        return hidden_activations


def extract_mlp_module(model, hookpoint: str) -> nn.Module:
    """
    Extract the MLP module from the model given a hookpoint.

    Args:
        model: The transformer model
        hookpoint: String path to the MLP module

    Returns:
        The MLP module
    """
    parts = hookpoint.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            raise ValueError(f"Could not find {part} in module {module}")
    return module


def process_mlp_activations(
    activations: Tensor,
    sparsity_ratio: float = 0.01,
    top_k: Optional[int] = None,
    threshold: float = 0.3,
) -> Tensor:
    """
    Process raw MLP activations to make them sparse for interpretation.

    Args:
        activations: Raw MLP hidden activations
        sparsity_ratio: Percentage of activations to keep (0.01 = 1%)
        top_k: Fixed number of activations to keep (overrides sparsity_ratio)
        threshold: Relative threshold for significant activations

    Returns:
        Sparse version of the activations
    """
    # Calculate dynamic threshold based on maximum values
    max_acts = activations.max(dim=-1, keepdim=True)[0]
    dynamic_threshold = max_acts * threshold

    # Create a mask of significant activations
    significant_mask = activations > dynamic_threshold

    # Create sparse activations
    sparse_acts = activations.clone()
    sparse_acts[~significant_mask] = 0.0

    # Determine sparsity level
    hidden_dim = activations.shape[-1]
    effective_k = (
        top_k if top_k is not None else max(1, int(hidden_dim * sparsity_ratio))
    )
    effective_k = min(effective_k, hidden_dim)

    # Keep only top-k activations
    if effective_k < hidden_dim:
        values, _ = torch.topk(activations, k=effective_k, dim=-1)
        min_values = values[:, :, -1].unsqueeze(-1)
        below_topk_mask = activations < min_values
        sparse_acts[below_topk_mask] = 0.0

    return sparse_acts


def load_topk_mlp_coder_hooks(
    model,
    hookpoints: List[str],
    sparsity_ratio: float = 0.01,
    top_k: Optional[int] = None,
    threshold: float = 0.3,
    device: str = "cuda",
) -> tuple[Dict[str, Callable], bool]:
    """
    Load MLP transcoder hooks for specified hookpoints.

    Args:
        model: The transformer model
        hookpoints: List of hookpoints to access MLP modules
        sparsity_ratio: Percentage of activations to keep
        top_k: Fixed number of activations to keep (overrides sparsity_ratio)
        threshold: Relative threshold for significant activations
        device: Device to place modules on

    Returns:
        Dictionary mapping hookpoints to hooks and transcode flag
    """
    hookpoint_to_hook = {}

    for hookpoint in hookpoints:
        # Extract the MLP module
        mlp_module = extract_mlp_module(model, hookpoint)

        # Create MLP transcoder
        transcoder = MLPTopKCoder(mlp_module, device)

        # Create the hook function
        def hook_fn(transcoder, threshold, sparsity_ratio, top_k, x):
            # Get hidden activations
            hidden_acts = transcoder(x)
            # Process into sparse form
            sparse_acts = process_mlp_activations(
                hidden_acts,
                sparsity_ratio=sparsity_ratio,
                top_k=top_k,
                threshold=threshold,
            )
            return sparse_acts

        # Create the specific hook for this MLP
        hookpoint_to_hook[hookpoint] = partial(
            hook_fn,
            transcoder=transcoder,
            threshold=threshold,
            sparsity_ratio=sparsity_ratio,
            top_k=top_k,
        )

    # MLPs as transcoders should set the transcode flag to True
    # This matches how transcoders are handled in the pipeline
    transcode = True

    return hookpoint_to_hook, transcode
