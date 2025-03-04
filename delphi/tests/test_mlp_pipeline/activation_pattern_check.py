import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from delphi.sparse_coders.custom.mlp_topk_coder import (
    MLPTopKCoder,
    process_mlp_activations,
)
from delphi.latents.collect_activations import collect_activations

# Configure output directory
os.makedirs("mlp_diagnostics", exist_ok=True)

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)

# Define MLP hookpoints to analyze
hookpoints = ["model.layers.5.mlp"]  # Adjust based on your model architecture

# Configure MLP processing parameters
sparsity_configs = [
    {"name": "default", "sparsity_ratio": 0.01, "top_k": None, "threshold": 0.3},
    {"name": "high_threshold", "sparsity_ratio": 0.01, "top_k": None, "threshold": 0.5},
    {"name": "top_k_mode", "sparsity_ratio": None, "top_k": 50, "threshold": 0.3},
]

# Test sentences designed to activate different patterns
test_sentences = [
    "Artificial intelligence has transformed how we approach computing problems.",
    "The quick brown fox jumps over the lazy dog while the cat watches nearby.",
    "In mathematics, the Fibonacci sequence appears in many different contexts.",
]


# Function to extract MLP module from model
def extract_mlp_module(model, hookpoint):
    parts = hookpoint.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            raise ValueError(f"Could not find {part} in module {module}")
    return module


# Define a function to collect and process activations directly
def collect_mlp_activations(inputs, hookpoint, config=None):
    if config is None:
        config = sparsity_configs[0]  # use default config

    # Get the raw MLP activations (before any processing)
    raw_activations = {}
    with collect_activations(model, [hookpoint], transcode=False) as activations:
        # Forward pass through the model
        outputs = model(**inputs)

        # Store raw activations
        for hp, activation in activations.items():
            raw_activations[hp] = activation.clone()

    # Process the raw activations to make them sparse
    sparse_activations = {}
    for hp, raw_act in raw_activations.items():
        # Process the raw activations to create sparse activations
        sparse_act = process_mlp_activations(
            raw_act,
            sparsity_ratio=config["sparsity_ratio"],
            top_k=config["top_k"],
            threshold=config["threshold"],
        )
        sparse_activations[hp] = sparse_act

    return raw_activations, sparse_activations


# Function to plot activation distributions
def plot_activation_distributions(
    raw_acts, sparse_acts, hookpoint, config_name, sentence_idx, config=None
):
    for hp, raw_act in raw_acts.items():
        sparse_act = sparse_acts[hp]

        # Reshape to 1D arrays for histograms
        raw_flat = raw_act.flatten().detach().cpu().float().numpy()
        sparse_flat = sparse_act.flatten().detach().cpu().float().numpy()

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Raw activations histogram - only show non-zero values for readability
        nonzero_raw = raw_flat[np.abs(raw_flat) > 1e-6]
        sns.histplot(nonzero_raw, bins=100, ax=ax1)
        ax1.set_title(f"Raw Activations - {config_name}")
        ax1.set_xlabel("Activation Value")
        ax1.set_ylabel("Count")

        # Add statistics to plot
        stats_text = f"Total: {len(raw_flat)}\n"
        stats_text += f"Non-zero: {len(nonzero_raw)} ({len(nonzero_raw) / len(raw_flat) * 100:.2f}%)\n"
        stats_text += f"Mean: {np.mean(nonzero_raw):.4f}\n"
        stats_text += f"Std: {np.std(nonzero_raw):.4f}\n"
        stats_text += f"Min: {np.min(nonzero_raw):.4f}\n"
        stats_text += f"Max: {np.max(nonzero_raw):.4f}\n"
        ax1.text(
            0.05,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # Sparse activations histogram - only show non-zero values
        nonzero_sparse = sparse_flat[sparse_flat != 0]
        if len(nonzero_sparse) > 0:  # Only plot if we have non-zero values
            sns.histplot(nonzero_sparse, bins=100, ax=ax2)
        ax2.set_title(f"Sparse Activations - {config_name}")
        ax2.set_xlabel("Activation Value")
        ax2.set_ylabel("Count")

        # Add statistics to plot
        sparsity = 1.0 - (len(nonzero_sparse) / len(sparse_flat))
        stats_text = f"Total: {len(sparse_flat)}\n"
        stats_text += f"Non-zero: {len(nonzero_sparse)} ({len(nonzero_sparse) / len(sparse_flat) * 100:.2f}%)\n"
        stats_text += f"Sparsity: {sparsity * 100:.2f}%\n"
        if len(nonzero_sparse) > 0:
            stats_text += f"Mean (non-zero): {np.mean(nonzero_sparse):.4f}\n"
            stats_text += f"Std (non-zero): {np.std(nonzero_sparse):.4f}\n"
            stats_text += f"Min (non-zero): {np.min(nonzero_sparse):.4f}\n"
            stats_text += f"Max (non-zero): {np.max(nonzero_sparse):.4f}\n"
        ax2.text(
            0.05,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # Save figure
        filename = f"mlp_diagnostics/{hp.replace('.', '_')}_{config_name}_sentence{sentence_idx}_dist.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Calculate expected sparsity based on the config
        if config:
            if config.get("sparsity_ratio"):
                expected_sparsity = 1.0 - config["sparsity_ratio"]
            elif config.get("top_k"):
                expected_sparsity = 1.0 - (config["top_k"] / sparse_flat.size)
            else:
                expected_sparsity = 0.99  # Default 99% sparsity

            print(f"Saved activation distribution to {filename}")
            print(
                f"Sparsity: {sparsity * 100:.2f}% (should be close to {expected_sparsity * 100:.2f}%)"
            )
        else:
            print(f"Saved activation distribution to {filename}")
            print(f"Sparsity: {sparsity * 100:.2f}%")


# Function to find tokens with highest activations
def analyze_token_activations(
    raw_acts, sparse_acts, inputs, hookpoint, config_name, sentence_idx
):
    for hp, raw_act in raw_acts.items():
        sparse_act = sparse_acts[hp]

        input_tokens = inputs.input_ids[0].cpu().tolist()
        token_strs = tokenizer.convert_ids_to_tokens(input_tokens)

        # Find top activating dimensions (features) in sparse activations
        feature_sums = sparse_act.sum(dim=(0, 1)).cpu()
        top_features_idx = torch.argsort(feature_sums, descending=True)[
            :20
        ]  # Top 20 features

        # For each top feature, find which tokens activate it most
        feature_token_map = {}
        for feature_idx in top_features_idx:
            if feature_sums[feature_idx] == 0:
                continue  # Skip features with no activation

            # Get activations for this feature across all tokens
            feature_acts = sparse_act[0, :, feature_idx].cpu()

            # Get top activating tokens for this feature
            top_token_indices = torch.argsort(feature_acts, descending=True)
            top_tokens = []

            for token_idx in top_token_indices:
                if feature_acts[token_idx] > 0:
                    token = token_strs[token_idx]
                    act_value = feature_acts[token_idx].item()
                    top_tokens.append((token, act_value))

                    if len(top_tokens) >= 5:  # Top 5 tokens per feature
                        break

            if len(top_tokens) > 0:
                feature_token_map[feature_idx.item()] = top_tokens

        # Save to file
        filename = f"mlp_diagnostics/{hp.replace('.', '_')}_{config_name}_sentence{sentence_idx}_tokens.txt"
        with open(filename, "w") as f:
            f.write(f"Input: {tokenizer.decode(input_tokens)}\n\n")
            f.write("Top activating features and their tokens:\n")
            f.write("=" * 80 + "\n\n")

            for feature_idx, tokens in feature_token_map.items():
                f.write(
                    f"Feature {feature_idx} (sum: {feature_sums[feature_idx].item():.4f}):\n"
                )
                for token, value in tokens:
                    f.write(f"  - '{token}': {value:.4f}\n")
                f.write("\n")

        print(f"Saved token activation analysis to {filename}")


# Function to test batch consistency
def test_batch_consistency(hookpoint, config, num_runs=5):
    results = []
    inputs = tokenizer(
        "The consistency test checks if the model produces the same activations across multiple runs.",
        return_tensors="pt",
    ).to(model.device)

    for run in range(num_runs):
        print(f"Consistency test run {run + 1}/{num_runs}")

        # Use the same config for all runs
        _, sparse_activations = collect_mlp_activations(inputs, hookpoint, config)

        for hp, sparse_act in sparse_activations.items():
            # Track non-zero locations and their values
            nonzero_mask = sparse_act != 0
            nonzero_sum = sparse_act.sum().item()
            nonzero_count = nonzero_mask.sum().item()
            sparsity = 1.0 - (nonzero_count / sparse_act.numel())

            results.append(
                {
                    "run": run + 1,
                    "hookpoint": hp,
                    "nonzero_sum": nonzero_sum,
                    "nonzero_count": nonzero_count,
                    "total_elements": sparse_act.numel(),
                    "sparsity": sparsity,
                }
            )

    # Check consistency
    sparsities = [r["sparsity"] for r in results]
    expected_sparsity = 1.0 - (
        config["sparsity_ratio"] or (config["top_k"] / results[0]["total_elements"])
    )

    # Save consistency results
    filename = f"mlp_diagnostics/{hookpoint.replace('.', '_')}_{config['name']}_consistency.txt"
    with open(filename, "w") as f:
        f.write("Batch Consistency Test Results\n")
        f.write("=" * 80 + "\n\n")

        # Write summary statistics
        sums = [r["nonzero_sum"] for r in results]
        counts = [r["nonzero_count"] for r in results]

        f.write(f"Expected sparsity: {expected_sparsity * 100:.4f}%\n")
        f.write(
            f"Actual sparsity: {np.mean(sparsities) * 100:.4f}% ± {np.std(sparsities) * 100:.4f}%\n\n"
        )

        f.write(f"Non-zero sum statistics:\n")
        f.write(f"  Mean: {np.mean(sums):.4f}\n")
        f.write(f"  Std: {np.std(sums):.4f}\n")
        f.write(
            f"  Coefficient of variation: {np.std(sums) / np.mean(sums) * 100:.4f}%\n\n"
        )

        f.write(f"Non-zero count statistics:\n")
        f.write(f"  Mean: {np.mean(counts):.4f}\n")
        f.write(f"  Std: {np.std(counts):.4f}\n")
        f.write(
            f"  Coefficient of variation: {np.std(counts) / np.mean(counts) * 100:.4f}%\n\n"
        )

        # Write per-run details
        f.write("Individual run details:\n")
        for r in results:
            f.write(f"Run {r['run']}:\n")
            f.write(f"  Non-zero sum: {r['nonzero_sum']:.4f}\n")
            f.write(f"  Non-zero count: {r['nonzero_count']}\n")
            f.write(f"  Sparsity: {r['sparsity'] * 100:.4f}%\n\n")

    print(f"Saved consistency test results to {filename}")
    return results


# Main execution flow
def run_activation_pattern_checks():
    print("Starting activation pattern checks...")

    # 1. Activation Distribution Check
    for i, sentence in enumerate(test_sentences):
        print(f"\nProcessing sentence {i + 1}: '{sentence[:30]}...'")

        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        for config in sparsity_configs:
            print(f"Using config: {config['name']}")

            for hookpoint in hookpoints:
                raw_acts, sparse_acts = collect_mlp_activations(
                    inputs, hookpoint, config
                )

                # Plot activation distributions
                plot_activation_distributions(
                    raw_acts, sparse_acts, hookpoint, config["name"], i, config
                )

                # Analyze token correlations
                analyze_token_activations(
                    raw_acts, sparse_acts, inputs, hookpoint, config["name"], i
                )

    # 2. Batch Consistency Check
    print("\nRunning batch consistency checks...")

    # Use default config for consistency tests
    default_config = sparsity_configs[0]
    for hookpoint in hookpoints:
        test_batch_consistency(hookpoint, default_config)

    print(
        "\nAll activation pattern checks completed. Results saved in 'mlp_diagnostics/' directory."
    )


if __name__ == "__main__":
    run_activation_pattern_checks()
