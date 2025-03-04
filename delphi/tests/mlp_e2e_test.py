#!/usr/bin/env python3
"""
This script demonstrates how to use Delphi to interpret MLP activations
and compare results with SAE/transcoder interpretations.
"""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from delphi.__main__ import run
from delphi.comparison_utils import create_comparison_report
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig


async def run_mlp_interpretation():
    """
    Run MLP activation interpretation on LLaMa-3-8B.
    """
    # Configure caching
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_column="text",
        batch_size=16,
        cache_ctx_len=256,
        n_splits=5,
        n_tokens=100_000,  # Start with 100k for testing, increase for better results
    )

    # Configure example construction
    constructor_cfg = ConstructorConfig(
        min_examples=150,
        max_examples=5000,
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
        # MLP-specific configs
        mlp_activation_threshold=0.3,
        top_k_activations=5,  # Use sparsity ratio instead of fixed top-k
        sparsity_ratio=0.01,  # Keep 1% of hidden dimensions active
    )

    # Configure sampling
    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
    )

    # Main run configuration
    run_cfg = RunConfig(
        name="mlp_interpretation",
        overwrite=["cache", "scores"],
        model="meta-llama/Llama-3.2-1B",
        # Use the same model name with the special type flag for MLP interpretation
        sparse_model="meta-llama/Llama-3.2-1B",
        sparse_model_type="mlp",  # Now treating MLPs as transcoders
        # Target specific MLPs - use same format as in original pipeline
        hookpoints=[
            "layers.5.mlp",  # Lower layer
        ],
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        explainer_model_max_len=8192,
        max_latents=100,  # Limit to 100 features per MLP for testing
        seed=42,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
        verbose=True,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    # Run MLP interpretation
    start_time = time.time()
    await run(run_cfg)
    end_time = time.time()
    print(f"MLP interpretation completed in {end_time - start_time:.2f} seconds")


async def run_sae_interpretation():
    """
    Run SAE activation interpretation on LLaMa-3-8B.
    """
    # Configure caching
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_column="text",
        batch_size=16,
        cache_ctx_len=256,
        n_splits=5,
        n_tokens=100_000,  # Start with 100k for testing, increase for better results
    )

    # Configure example construction
    constructor_cfg = ConstructorConfig(
        min_examples=150,
        max_examples=5000,
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
        # MLP-specific configs
        # mlp_activation_threshold=0.3,
        # top_k_activations=None,  # Use sparsity ratio instead of fixed top-k
        # sparsity_ratio=0.01,  # Keep 1% of hidden dimensions active
    )

    # Configure sampling
    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
    )

    # Main run configuration
    run_cfg = RunConfig(
        name="sae_interpretation",
        overwrite=["cache", "scores"],
        model="meta-llama/Llama-3.2-1B",
        # Use the same model name with the special type flag for MLP interpretation
        sparse_model="EleutherAI/sae-Llama-3.2-1B-131k",
        sparse_model_type="sae",  # Now treating MLPs as transcoders
        # Target specific MLPs - use same format as in original pipeline
        hookpoints=[
            "layers.5",  # Lower layer
        ],
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        explainer_model_max_len=8192,
        max_latents=100,  # Limit to 100 features per MLP for testing
        seed=42,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
        verbose=True,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    # Run MLP interpretation
    start_time = time.time()
    await run(run_cfg)
    end_time = time.time()
    print(f"SAE interpretation completed in {end_time - start_time:.2f} seconds")


async def run_transcoder_interpretation():
    """
    Run Transcoder activation interpretation on LLaMa-3-8B.
    """
    # Configure caching
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_column="text",
        batch_size=16,
        cache_ctx_len=256,
        n_splits=5,
        n_tokens=100_000,  # Start with 100k for testing, increase for better results
    )

    # Configure example construction
    constructor_cfg = ConstructorConfig(
        min_examples=150,
        max_examples=5000,
        example_ctx_len=32,
        n_non_activating=50,
        non_activating_source="random",
        # MLP-specific configs
        # mlp_activation_threshold=0.3,
        # top_k_activations=None,  # Use sparsity ratio instead of fixed top-k
        # sparsity_ratio=0.01,  # Keep 1% of hidden dimensions active
    )

    # Configure sampling
    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
    )

    # Main run configuration
    run_cfg = RunConfig(
        name="transcoder_interpretation",
        overwrite=["cache", "scores"],
        model="meta-llama/Llama-3.2-1B",
        # Use the same model name with the special type flag for MLP interpretation
        sparse_model="EleutherAI/skip-transcoder-Llama-3.2-1B-131k",
        sparse_model_type="transcoder",  # Now treating MLPs as transcoders
        # Target specific MLPs - use same format as in original pipeline
        hookpoints=[
            "layers.5.mlp",  # Lower layer
        ],
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        explainer_model_max_len=8192,
        max_latents=100,  # Limit to 100 features per MLP for testing
        seed=42,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
        verbose=True,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    # Run MLP interpretation
    start_time = time.time()
    await run(run_cfg)
    end_time = time.time()
    print(f"Transcoder interpretation completed in {end_time - start_time:.2f} seconds")


async def run_comparison():
    """
    Compare MLP interpretation results with previous SAE results.
    """
    # Base directory with results
    base_path = Path.cwd() / "results" / str(datetime.now().strftime("%m%d%H%M"))

    # Create comparison report
    create_comparison_report(
        base_path=str(base_path),
        output_dir="comparison_results",
        model_types=["sae", "mlp"],  # Add "transcoder" if available
        score_types=["detection", "fuzz", "simulator"],
        metrics=["accuracy", "f1_score", "precision", "recall"],
    )
    print("Comparison report created in 'comparison_results' directory")


if __name__ == "__main__":
    # Run MLP interpretation
    asyncio.run(run_mlp_interpretation())

    # Run SAE interpretation
    # asyncio.run(run_sae_interpretation())

    # Run Transcoder interpretation
    # asyncio.run(run_transcoder_interpretation())

    # Compare with previous SAE results
    # asyncio.run(run_comparison())
