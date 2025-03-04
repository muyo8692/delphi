import asyncio
import torch


async def test_mlp_pipeline():
    """Test the MLP processing pipeline with diagnostics."""

    # Import your model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    # Define hookpoints to test
    hookpoints = ["model.layers.5.mlp"]  # Adjust for your model

    # Create hooks with diagnostics enabled
    from delphi.sparse_coders.custom.mlp_topk_coder import load_topk_mlp_coder_hooks

    hookpoint_to_hook, transcode = load_topk_mlp_coder_hooks(
        model,
        hookpoints,
        sparsity_ratio=0.01,
        top_k=None,
        threshold=0.3,
        device="cuda",
    )

    # Create a simple input
    inputs = tokenizer(
        "This is a test sentence to verify MLP activations.", return_tensors="pt"
    ).to(model.device)

    # Manually run hooks on output
    from delphi.latents.collect_activations import collect_activations

    # First run the model with collect_activations to get raw activations
    with collect_activations(
        model, list(hookpoint_to_hook.keys()), transcode
    ) as raw_activations:
        model(**inputs)

    # Now manually apply your hook to process the raw activations
    for hookpoint, raw_activation in raw_activations.items():
        hook_fn = hookpoint_to_hook[hookpoint]
        processed_activation = hook_fn(raw_activation)

        print(f"\nProcessed activations for {hookpoint}:")
        print(f"  - Shape: {processed_activation.shape}")
        print(f"  - Non-zero: {(processed_activation != 0).sum().item()}")
        print(f"  - Sparsity: {(processed_activation != 0).float().mean().item():.6f}")
        print(
            f"  - Min/Max: {processed_activation.min().item():.4f} / {processed_activation.max().item():.4f}"
        )

        # Check if sparsity is close to target
        target_sparsity = 0.01  # 1%
        actual_sparsity = (processed_activation != 0).float().mean().item()
        if abs(actual_sparsity - target_sparsity) < 0.001:
            print("  ✓ Sparsity matches target")
        else:
            print(
                f"  ⚠️ Sparsity {actual_sparsity:.4f} doesn't match target {target_sparsity:.4f}"
            )

    print("\nPipeline check complete")


# Run the test
asyncio.run(test_mlp_pipeline())
