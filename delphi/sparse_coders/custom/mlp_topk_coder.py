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

        For gated activations (as in LLaMA‑3), it computes:
            hidden_activations = act_fn(gate_proj(x)) * up_proj(x)
        Otherwise, it computes:
            hidden_activations = act_fn(up_proj(x))

        Args:
            x: Input tensor from residual stream

        Returns:
            Hidden activations after applying the activation function (and gating if present)
        """
        if hasattr(self.mlp_module, "gate_proj"):
            # Handle gated activation (e.g. in LLaMA‑3)
            gate = self.mlp_module.gate_proj(x)
            up = self.mlp_module.up_proj(x)
            if hasattr(self.mlp_module, "act_fn"):
                activation_fn = self.mlp_module.act_fn
            elif hasattr(self.mlp_module, "activation_fn"):
                activation_fn = self.mlp_module.activation_fn
            else:
                raise ValueError("No activation function found in MLP module")
            hidden_activations = activation_fn(gate) * up
        else:
            raise NotImplementedError("Non-gated MLPs are not supported yet")
            # Standard MLP case without gating
            pre_activations = self.mlp_module.up_proj(x)
            if hasattr(self.mlp_module, "act_fn"):
                activation_fn = self.mlp_module.act_fn
            elif hasattr(self.mlp_module, "activation_fn"):
                activation_fn = self.mlp_module.activation_fn
            else:
                raise ValueError("No activation function found in MLP module")
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
    debug: bool = True,
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
    max_acts = activations.max(dim=-1, keepdim=True)[0]
    # dynamic_threshold = max_acts * threshold

    # Create sparse activations - start with a copy of the original
    sparse_acts = activations.clone()

    hidden_dim = activations.shape[-1]
    effective_k = (
        top_k if top_k is not None else max(1, int(hidden_dim * sparsity_ratio))
    )
    effective_k = min(effective_k, hidden_dim)

    # Apply top-k selection directly to get guaranteed sparsity
    for batch_idx in range(activations.shape[0]):
        for seq_idx in range(activations.shape[1]):
            values, indices = torch.topk(activations[batch_idx, seq_idx], k=effective_k)
            sparse_acts[batch_idx, seq_idx, :] = 0.0
            sparse_acts[batch_idx, seq_idx, indices] = activations[
                batch_idx, seq_idx, indices
            ]

    final_sparsity = (sparse_acts > 0).float().mean().item()

    if debug and torch.rand(1).item() < 0.01:  # Log ~1% of calls
        # print(f"Dynamic threshold: {dynamic_threshold.mean().item():.6f}")
        # print(
        #     f"Threshold-based sparsity: {(activations > dynamic_threshold).float().mean().item():.6f}"
        # )
        print(f"Target sparsity: {sparsity_ratio:.4f}")
        print(f"Effective k: {effective_k} out of {hidden_dim}")
        print(f"Final sparsity: {final_sparsity:.6f}")

    return sparse_acts


def load_topk_mlp_coder_hooks(
    model,
    hookpoints: List[str],
    sparsity_ratio: float = 0.01,
    top_k: Optional[int] = None,
    threshold: float = 0.3,
    device: str = "cuda",
) -> tuple[Dict[str, Callable], bool]:
    hookpoint_to_hook = {}

    print(f"🔍 Installing MLP hooks for {len(hookpoints)} hookpoints:")

    for hookpoint in hookpoints:
        try:
            # Extract the MLP module
            mlp_module = extract_mlp_module(model, hookpoint)

            # ADDED: Print module information for verification
            print(f"  ✓ Found {hookpoint}:")
            print(f"    - Type: {type(mlp_module).__name__}")
            if hasattr(mlp_module, "gate_proj"):
                print(f"    - gate_proj: {mlp_module.gate_proj.weight.shape}")
            if hasattr(mlp_module, "up_proj"):
                print(f"    - up_proj: {mlp_module.up_proj.weight.shape}")
            else:
                print(f"    ⚠️ Warning: No up_proj found in {hookpoint}")
            if hasattr(mlp_module, "down_proj"):
                print(f"    - down_proj: {mlp_module.down_proj.weight.shape}")
            else:
                print(f"    ⚠️ Warning: No down_proj found in {hookpoint}")

            # Check activation function
            if hasattr(mlp_module, "act_fn"):
                print(f"    - Activation: {mlp_module.act_fn.__class__.__name__}")
            elif hasattr(mlp_module, "activation_fn"):
                print(
                    f"    - Activation: {mlp_module.activation_fn.__class__.__name__}"
                )
            else:
                print("    - Activation: Using fallback GELU")

            # Create MLP transcoder
            transcoder = MLPTopKCoder(mlp_module, device)

            # Create a closure to avoid the parameter naming issue
            def create_hook(_transcoder, _threshold, _sparsity_ratio, _top_k):
                def hook(x):
                    hidden_acts = _transcoder(x)
                    sparse_acts = process_mlp_activations(
                        hidden_acts,
                        sparsity_ratio=_sparsity_ratio,
                        top_k=_top_k,
                        threshold=_threshold,
                    )
                    return sparse_acts

                return hook

            hookpoint_to_hook[hookpoint] = create_hook(
                transcoder, threshold, sparsity_ratio, top_k
            )

            # MLPs as transcoders should set the transcode flag to True
            transcode = True

        except Exception as e:
            print(f"  ❌ Error setting up {hookpoint}: {e}")
            import ipdb

            ipdb.set_trace()

    return hookpoint_to_hook, transcode
