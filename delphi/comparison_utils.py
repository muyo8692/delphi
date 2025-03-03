import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix


def load_explanation_scores(
    base_path: str,
    model_types: List[str] = ["sae", "transcoder", "mlp"],
    score_types: List[str] = ["detection", "fuzz", "simulator"],
    metrics: List[str] = ["accuracy", "f1_score", "precision", "recall"],
) -> pd.DataFrame:
    """
    Load explanation scores from multiple runs for comparison

    Args:
        base_path: Base directory containing results
        model_types: Types of models to compare
        score_types: Types of scores to compare
        metrics: Metrics to extract

    Returns:
        DataFrame with combined results
    """
    results = []

    for model_type in model_types:
        model_path = Path(base_path) / model_type
        if not model_path.exists():
            continue

        scores_path = model_path / "scores"
        if not scores_path.exists():
            continue

        for score_type in score_types:
            score_path = scores_path / score_type
            if not score_path.exists():
                continue

            # Load scores df for this model type and score type
            try:
                from delphi.log.result_analysis import build_scores_df

                # Determine hookpoints from directories
                hookpoints = [
                    d.name
                    for d in (model_path / "latents").iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]

                df = build_scores_df(scores_path, hookpoints)
                df = df[df["score_type"] == score_type].copy()

                # Add model type
                df["model_type"] = model_type

                # Add to results
                results.append(df)

            except Exception as e:
                print(f"Error loading scores for {model_type}/{score_type}: {e}")

    if not results:
        return pd.DataFrame()

    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)

    return combined_df


def plot_metric_comparisons(
    df: pd.DataFrame, metric: str = "accuracy", output_path: Optional[str] = None
) -> None:
    """
    Create comparison plots for a specific metric across model types

    Args:
        df: DataFrame with combined results
        metric: Metric to compare
        output_path: Path to save the plots
    """
    plt.figure(figsize=(15, 10))

    # Create subplots for each score type
    score_types = df["score_type"].unique()
    n_score_types = len(score_types)

    for i, score_type in enumerate(score_types):
        plt.subplot(n_score_types, 2, 2 * i + 1)

        score_df = df[df["score_type"] == score_type]

        # Boxplot comparison
        sns.boxplot(x="model_type", y=metric, data=score_df)
        plt.title(f"{score_type.capitalize()} {metric.capitalize()} Comparison")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Model Type")

        # Add mean values
        means = score_df.groupby("model_type")[metric].mean()
        for j, model_type in enumerate(score_df["model_type"].unique()):
            plt.text(
                j,
                means[model_type],
                f"{means[model_type]:.3f}",
                ha="center",
                va="bottom",
            )

        # Histogram comparison
        plt.subplot(n_score_types, 2, 2 * i + 2)
        for model_type in score_df["model_type"].unique():
            model_df = score_df[score_df["model_type"] == model_type]
            sns.histplot(model_df[metric], label=model_type, alpha=0.5, bins=20)

        plt.title(f"{score_type.capitalize()} {metric.capitalize()} Distribution")
        plt.xlabel(metric.capitalize())
        plt.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def compare_feature_correlations(
    base_path: str,
    model1_type: str,
    model2_type: str,
    latent_indices: List[int],
    hookpoint: str,
) -> Tuple[float, float]:
    """
    Compare feature correlations between different model types

    Args:
        base_path: Base directory containing results
        model1_type: First model type
        model2_type: Second model type
        latent_indices: Indices of latents to compare
        hookpoint: Hookpoint to compare

    Returns:
        Tuple of (pearson correlation, spearman correlation)
    """
    model1_path = Path(base_path) / model1_type / "latents" / hookpoint
    model2_path = Path(base_path) / model2_type / "latents" / hookpoint

    if not model1_path.exists() or not model2_path.exists():
        raise ValueError(f"Paths do not exist: {model1_path} or {model2_path}")

    # Load activations for both models
    model1_acts = []
    model2_acts = []

    for idx in latent_indices:
        # Find files that contain this latent
        model1_files = list(model1_path.glob(f"*_{idx}_*.safetensors"))
        model2_files = list(model2_path.glob(f"*_{idx}_*.safetensors"))

        if not model1_files or not model2_files:
            continue

        # Load activations
        try:
            from safetensors.torch import load_file

            model1_data = load_file(str(model1_files[0]))
            model2_data = load_file(str(model2_files[0]))

            model1_act = model1_data["activations"]
            model2_act = model2_data["activations"]

            # Append to lists
            model1_acts.append(model1_act.mean().item())
            model2_acts.append(model2_act.mean().item())

        except Exception as e:
            print(f"Error loading activations for latent {idx}: {e}")

    if not model1_acts or not model2_acts:
        return 0.0, 0.0

    # Calculate correlations
    pearson_corr, _ = pearsonr(model1_acts, model2_acts)
    spearman_corr, _ = spearmanr(model1_acts, model2_acts)

    return pearson_corr, spearman_corr


def compare_explanations(
    base_path: str,
    model_types: List[str] = ["sae", "transcoder", "mlp"],
    n_samples: int = 100,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare explanations between different model types

    Args:
        base_path: Base directory containing results
        model_types: Types of models to compare
        n_samples: Number of explanations to sample for comparison
        output_path: Path to save the comparison results

    Returns:
        DataFrame with explanation similarities
    """
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    explanations = {}

    # Load explanations for each model type
    for model_type in model_types:
        explanations[model_type] = {}

        model_path = Path(base_path) / model_type / "explanations"
        if not model_path.exists():
            continue

        # Sample explanation files
        explanation_files = list(model_path.glob("*.txt"))
        if len(explanation_files) > n_samples:
            import random

            random.shuffle(explanation_files)
            explanation_files = explanation_files[:n_samples]

        # Load explanations
        for file in explanation_files:
            latent_id = file.stem
            try:
                with open(file, "r") as f:
                    explanation = json.load(f)
                explanations[model_type][latent_id] = explanation
            except Exception as e:
                print(f"Error loading explanation {file}: {e}")

    # Compute similarities between pairs of model types
    results = []

    for i, model1 in enumerate(model_types):
        for j, model2 in enumerate(model_types):
            if j <= i:  # Only compute upper triangle
                continue

            # Find common latents
            common_latents = set(explanations[model1].keys()) & set(
                explanations[model2].keys()
            )

            if not common_latents:
                continue

            # Compute similarities
            similarities = []

            for latent in common_latents:
                text1 = explanations[model1][latent]
                text2 = explanations[model2][latent]

                # Compute TF-IDF vectors and cosine similarity
                vectorizer = TfidfVectorizer()
                try:
                    tfidf_matrix = vectorizer.fit_transform([text1, text2])
                    similarity = cosine_similarity(
                        tfidf_matrix[0:1], tfidf_matrix[1:2]
                    )[0][0]
                    similarities.append(similarity)
                except:
                    continue

            # Compute statistics
            if similarities:
                results.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "mean_similarity": np.mean(similarities),
                        "median_similarity": np.median(similarities),
                        "min_similarity": np.min(similarities),
                        "max_similarity": np.max(similarities),
                        "num_comparisons": len(similarities),
                    }
                )

    results_df = pd.DataFrame(results)

    if output_path:
        results_df.to_csv(output_path, index=False)

    return results_df


def create_comparison_report(
    base_path: str,
    output_dir: str = "comparison_results",
    model_types: List[str] = ["sae", "transcoder", "mlp"],
    score_types: List[str] = ["detection", "fuzz", "simulator"],
    metrics: List[str] = ["accuracy", "f1_score", "precision", "recall"],
) -> None:
    """
    Create a comprehensive comparison report between different model types

    Args:
        base_path: Base directory containing results
        output_dir: Directory to save the reports
        model_types: Types of models to compare
        score_types: Types of scores to compare
        metrics: Metrics to compare
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load scores
    df = load_explanation_scores(base_path, model_types, score_types, metrics)

    if df.empty:
        print("No data found for comparison")
        return

    # Generate summary statistics
    summary = df.groupby(["model_type", "score_type"])[metrics].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary.to_csv(output_path / "summary_statistics.csv")

    # Create comparison plots
    for metric in metrics:
        plot_metric_comparisons(
            df, metric, str(output_path / f"{metric}_comparison.png")
        )

    # Compare explanations
    explanation_comparison = compare_explanations(
        base_path,
        model_types,
        output_path=str(output_path / "explanation_similarities.csv"),
    )

    # Generate report
    with open(output_path / "comparison_report.md", "w") as f:
        f.write("# Model Interpretation Comparison Report\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")

        f.write("## Explanation Similarities\n\n")
        if not explanation_comparison.empty:
            f.write(explanation_comparison.to_markdown())
        else:
            f.write("No explanation similarities computed.")
        f.write("\n\n")

        f.write("## Visualization Links\n\n")
        for metric in metrics:
            f.write(f"- [{metric.capitalize()} Comparison]({metric}_comparison.png)\n")
