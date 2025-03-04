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

    with collect_activations(
        model, list(hookpoint_to_hook.keys()), transcode
    ) as activations:
        model(**inputs)

        # Show collected activations
        for hookpoint, activation_tensor in activations.items():
            print(f"\nCollected activations for {hookpoint}:")
            print(f"  - Shape: {activation_tensor.shape}")
            print(f"  - Non-zero: {(activation_tensor != 0).sum().item()}")
            print(f"  - Sparsity: {(activation_tensor != 0).float().mean().item():.6f}")
            print(
                f"  - Min/Max: {activation_tensor.min().item():.4f} / {activation_tensor.max().item():.4f}"
            )

            # Check if sparsity is close to target
            target_sparsity = 0.01  # 1%
            actual_sparsity = (activation_tensor != 0).float().mean().item()
            if abs(actual_sparsity - target_sparsity) < 0.001:
                print("  ✓ Sparsity matches target")
            else:
                print(
                    f"  ⚠️ Sparsity {actual_sparsity:.4f} doesn't match target {target_sparsity:.4f}"
                )

    print("\nPipeline check complete")


# Run the test
asyncio.run(test_mlp_pipeline())
