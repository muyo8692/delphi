from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


class MLPWrapper(nn.Module):
    """
    Wrapper for MLP modules to interface with the Delphi pipeline.
    Treats up_proj as encoder and down_proj as decoder to mirror SAE behavior.
    """

    def __init__(
        self,
        up_proj_weight: Tensor,
        down_proj_weight: Tensor,
        activation_fn: Optional[Callable] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.up_proj = nn.Parameter(up_proj_weight)
        self.down_proj = nn.Parameter(down_proj_weight)
        self.activation_fn = activation_fn or torch.nn.functional.gelu
        self.hidden_size = up_proj_weight.shape[1]
        self.to(device)

    def pre_acts(self, x: Tensor) -> Tensor:
        """Compute pre-activation values (equivalent to SAE pre_acts)."""
        return x @ self.up_proj

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input using up_proj and activation.
        Returns dense activation tensor compatible with Delphi pipeline.
        """
        pre_acts = self.pre_acts(x)
        acts = self.activation_fn(pre_acts)
        return acts

    def decode(self, acts: Tensor) -> Tensor:
        """Decode activations using down_proj."""
        return acts @ self.down_proj

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass through the MLP."""
        return self.decode(self.encode(x))


def mlp_dense_latents(x: Tensor, mlp_wrapper: MLPWrapper) -> Tensor:
    """
    Helper function to extract MLP hidden activations.
    Mimics the behavior of sae_dense_latents for SAEs.

    Args:
        x: Input tensor
        mlp_wrapper: MLPWrapper instance

    Returns:
        Tensor of dense hidden activations
    """
    return mlp_wrapper.encode(x)


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


def load_mlp_hooks(
    model,
    hookpoints: list[str],
    device: str = "cuda",
) -> tuple[dict[str, Callable], bool]:
    """
    Load MLP modules as hooks for the Delphi pipeline.

    Args:
        model: The transformer model
        hookpoints: List of hookpoints to access MLP modules
        device: Device to place wrappers on

    Returns:
        Dictionary mapping hookpoints to encoding functions and transcode flag
    """
    hookpoint_to_sparse_encode = {}

    for hookpoint in hookpoints:
        mlp_module = extract_mlp_module(model, hookpoint)

        # Get weights - transposing up_proj to match expected dimensions
        up_proj_weight = mlp_module.up_proj.weight.data.T
        down_proj_weight = mlp_module.down_proj.weight.data

        # Get activation function
        activation_fn = getattr(mlp_module, "act_fn", torch.nn.functional.gelu)

        # Create MLP wrapper
        mlp_wrapper = MLPWrapper(
            up_proj_weight=up_proj_weight,
            down_proj_weight=down_proj_weight,
            activation_fn=activation_fn,
            device=device,
        )

        # Create encoding function
        hookpoint_to_sparse_encode[hookpoint] = partial(
            mlp_dense_latents, mlp_wrapper=mlp_wrapper
        )

    # MLPs are not transcoders
    transcode = False

    return hookpoint_to_sparse_encode, transcode


def handle_mlp_activations(
    activations: dict[str, Tensor],
    hookpoint_to_sparse_encode: dict[str, Callable],
    threshold: float = 0.3,
    top_k: Optional[int] = None,
    sparsity_ratio: float = 0.01,  # Default to 1% sparsity
) -> dict[str, Tensor]:
    """
    Process MLP activations to make them suitable for the Delphi pipeline.

    For MLP activations, we need to handle a few cases:
    1. Activations are typically dense, not sparse like SAE features
    2. Hidden dimension can be very large

    This function:
    - Applies a threshold to focus on significant activations
    - Keeps only the top-k activations per sequence
    - Formats them to match the expected shape of SAE activations

    Args:
        activations: Dictionary of raw activations from hookpoints
        hookpoint_to_sparse_encode: Dictionary of encoder functions
        threshold: Activation threshold relative to maximum value
        top_k: Maximum number of features to keep per sequence

    Returns:
        Processed activations in the same format as SAE activations
    """
    processed_activations = {}

    for hookpoint, acts in activations.items():
        # Apply the encoder function
        encoder_fn = hookpoint_to_sparse_encode[hookpoint]
        encoded_acts = encoder_fn(acts)

        # Find significant activations
        # Calculate dynamic threshold based on the maximum value per batch element
        max_acts = encoded_acts.max(dim=-1, keepdim=True)[0]
        dynamic_threshold = max_acts * threshold

        # Create a mask of significant activations
        significant_mask = encoded_acts > dynamic_threshold

        # Zero out non-significant activations
        sparse_acts = encoded_acts.clone()
        sparse_acts[~significant_mask] = 0.0

        # Determine sparsity level - either use explicit top_k or calculate from ratio
        hidden_dim = encoded_acts.shape[-1]
        effective_k = (
            top_k if top_k is not None else max(1, int(hidden_dim * sparsity_ratio))
        )

        # Ensure k is within reasonable bounds
        effective_k = min(effective_k, hidden_dim)

        # Keep only the top k activations per sequence
        if effective_k < hidden_dim:
            # Sort activations and keep only top k
            values, _ = torch.topk(encoded_acts, k=effective_k, dim=-1)
            # Get the smallest value in the top k as a threshold
            min_values = values[:, :, -1].unsqueeze(-1)
            # Create a mask for values below this threshold
            below_topk_mask = encoded_acts < min_values
            # Zero out values not in the top k
            sparse_acts[below_topk_mask] = 0.0

        print(
            f"Using {effective_k} features out of {hidden_dim} ({effective_k / hidden_dim:.2%} density) for {hookpoint}"  # noqa: E501
        )

        processed_activations[hookpoint] = sparse_acts

    return processed_activations


def extract_mlp_feature_importances(
    model, hookpoints: list[str], top_k: int = 1000
) -> dict[str, list[int]]:
    """
    Extract feature importances from MLP modules to prioritize which
    features to interpret.

    Uses a heuristic based on:
    1. L2 norm of the feature's output projections
    2. Connectivity patterns in the MLP

    Args:
        model: The transformer model
        hookpoints: List of hookpoint paths to MLP modules
        top_k: Number of top features to return per MLP

    Returns:
        Dictionary mapping hookpoints to lists of feature indices
    """
    feature_importances = {}

    for hookpoint in hookpoints:
        # Extract the MLP module
        parts = hookpoint.split(".")
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                raise ValueError(f"Could not find {part} in module {module}")

        # Get weights
        up_proj = module.up_proj.weight.data  # [hidden_dim, input_dim]
        down_proj = module.down_proj.weight.data  # [output_dim, hidden_dim]

        # Calculate feature importance based on the L2 norm of output projections
        feature_importance = torch.norm(down_proj, dim=0)  # [hidden_dim]

        # Get indices of top-k features by importance
        _, top_indices = torch.topk(
            feature_importance, k=min(top_k, len(feature_importance))
        )

        feature_importances[hookpoint] = top_indices.tolist()

    return feature_importances
