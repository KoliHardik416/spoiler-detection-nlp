"""
Evaluation utilities: metrics computation, plotting, and JSON export.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from src.config import RESULTS_DIR, LABEL_NAMES

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute classification metrics.

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1, and optionally roc_auc.
    """

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["roc_auc"] = float(auc(fpr, tpr))
    return metrics


def save_results(model_name: str, metrics: dict,
                 extra: dict | None = None) -> str:
    """
    Save model results to a JSON file in the results directory.
    """

    result = {"model_name": model_name, "metrics": metrics}

    if extra:
        result["details"] = extra
    path = os.path.join(RESULTS_DIR, f"model_results/{model_name}.json")

    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved → {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          labels=None, save_name=None):
    """
    Plot a styled confusion matrix heatmap.
    """

    labels = labels or LABEL_NAMES
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"shrink": 0.8}, ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_name:
        path = os.path.join(RESULTS_DIR, f"model_charts/{save_name}_cm.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {path}")
    plt.show()


def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_name=None):
    """
    Plot ROC curve with AUC.
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color="#58a6ff", linewidth=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#8b949e", linestyle="--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#58a6ff")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    if save_name:
        path = os.path.join(RESULTS_DIR, f"model_charts/{save_name}_roc.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved → {path}")
    plt.show()


def compare_models(results_dir: str = RESULTS_DIR):
    """
    Load all JSON result files and produce a comparison bar chart.

    Returns:
        dict: Combined results keyed by model name.
    """
    all_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                all_results[data["model_name"]] = data["metrics"]

    if not all_results:
        print("No result files found.")
        return {}

    models = list(all_results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    width = 0.18
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metric_names):
        vals = [all_results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=colors[i], edgecolor="#30363d", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    color="#333333")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison", fontsize=16, fontweight="bold", pad=14)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved → {path}")
    plt.show()

    return all_results


def print_classification_report(y_true, y_pred, labels=None):
    """
    Print a formatted classification report.
    """

    labels = labels or LABEL_NAMES
    print(classification_report(y_true, y_pred, target_names=labels))
