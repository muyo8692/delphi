from typing import Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from delphi.config import RunConfig

from .custom.gemmascope import load_gemma_autoencoders
from .custom.mlp_topk_coder import load_topk_mlp_coder_hooks
from .load_sparsify import load_sparsify_hooks, load_sparsify_sparse_coders


def load_hooks_sparse_coders(
    model: PreTrainedModel,
    run_cfg: RunConfig,
    compile: bool = False,
) -> tuple[dict[str, Callable], bool]:
    """
    Load sparse coders for specified hookpoints.
    Extended to support direct MLP activation interpretation.

    Args:
        model (PreTrainedModel): The model to load sparse coders for.
        run_cfg (RunConfig): The run configuration.
        compile (bool): Whether to compile the sparse coders.

    Returns:
        dict[str, Callable]: A dictionary mapping hookpoints to sparse coders.
        bool: Whether the sparse coders are transcoders.
    """

    # Check if we're using MLP mode
    if run_cfg.sparse_model_type == "mlp":
        return load_topk_mlp_coder_hooks(
            model,
            run_cfg.hookpoints,
            sparsity_ratio=run_cfg.constructor_cfg.sparsity_ratio,
            top_k=run_cfg.constructor_cfg.top_k_activations,
            threshold=run_cfg.constructor_cfg.mlp_activation_threshold,
            device=str(model.device),
        )
    # Otherwise use existing SAE/transcoder handlers
    elif "gemma" not in run_cfg.sparse_model:
        hookpoint_to_sparse_encode, transcode = load_sparsify_hooks(
            model,
            run_cfg.sparse_model,
            run_cfg.hookpoints,
            compile=compile,
        )
    else:
        # model path will always be of the form google/gemma-scope-<size>-pt-<type>/
        # where <size> is the size of the model and <type> is either res or mlp
        model_path = "google/" + run_cfg.sparse_model.split("/")[1]
        type = model_path.split("-")[-1]
        # we can use the hookpoints to determine the layer, size and l0,
        # because the module is determined by the model name
        # the hookpoint should be in the format
        # layer_<layer>/width_<sae_size>/average_l0_<l0>
        layers = []
        l0s = []
        sae_sizes = []
        for hookpoint in run_cfg.hookpoints:
            layer = int(hookpoint.split("/")[0].split("_")[1])
            sae_size = hookpoint.split("/")[1].split("_")[1]
            l0 = int(hookpoint.split("/")[2].split("_")[2])
            layers.append(layer)
            sae_sizes.append(sae_size)
            l0s.append(l0)

        hookpoint_to_sparse_encode = load_gemma_autoencoders(
            model_path=model_path,
            ae_layers=layers,
            average_l0s=l0s,
            sizes=sae_sizes,
            type=type,
            dtype=model.dtype,
            device=model.device,
        )
        transcode = False
    # throw an error if the dictionary is empty
    if not hookpoint_to_sparse_encode:
        raise ValueError("No sparse coders loaded")
    return hookpoint_to_sparse_encode, transcode


def load_sparse_coders(
    run_cfg: RunConfig,
    device: str | torch.device,
    compile: bool = False,
) -> dict[str, nn.Module]:
    """
    Load sparse coders for specified hookpoints.

    Args:
        model (PreTrainedModel): The model to load sparse coders for.
        run_cfg (RunConfig): The run configuration.

    Returns:
        dict[str, nn.Module]: A dictionary mapping hookpoints to sparse coders.
    """

    # Add SAE hooks to the model
    if "gemma" not in run_cfg.sparse_model:
        hookpoint_to_sparse_model = load_sparsify_sparse_coders(
            run_cfg.sparse_model,
            run_cfg.hookpoints,
            device,
            compile,
        )
    else:
        # model path will always be of the form google/gemma-scope-<size>-pt-<type>/
        # where <size> is the size of the model and <type> is either res or mlp
        model_path = "google/" + run_cfg.sparse_model.split("/")[1]
        type = model_path.split("-")[-1]
        # we can use the hookpoints to determine the layer, size and l0,
        # because the module is determined by the model name
        # the hookpoint should be in the format
        # layer_<layer>/width_<sae_size>/average_l0_<l0>
        layers = []
        l0s = []
        sae_sizes = []
        for hookpoint in run_cfg.hookpoints:
            layer = int(hookpoint.split("/")[0].split("_")[1])
            sae_size = hookpoint.split("/")[1].split("_")[1]
            l0 = int(hookpoint.split("/")[2].split("_")[2])
            layers.append(layer)
            sae_sizes.append(sae_size)
            l0s.append(l0)

        hookpoint_to_sparse_model = load_gemma_autoencoders(
            model_path=model_path,
            ae_layers=layers,
            average_l0s=l0s,
            sizes=sae_sizes,
            type=type,
            device=device,
        )

    return hookpoint_to_sparse_model
