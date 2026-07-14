"""
Unified evaluation module.

Functions:
  compute_metrics()       — accuracy, precision, recall, F1
  evaluate_sklearn()      — wraps sklearn predict
  evaluate_torch()        — wraps PyTorch model on a DataLoader
  noise_robustness_test() — injects Gaussian noise and measures accuracy drop
  plot_confusion_matrix() — saves confusion matrix image
  plot_training_curves()  — saves accuracy and loss curves
  compare_models()        — prints and saves a comparison table
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_NAMES, N_CLASSES, OUTPUT_DIR, NOISE_SIGMAS, DEVICE
from src.utils import get_logger

log = get_logger("evaluate")


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Return accuracy, weighted precision, recall, F1, per-class F1, and ROC-AUC."""
    from sklearn.preprocessing import label_binarize
    n_classes = len(np.unique(y_true))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {
        CLASS_NAMES[int(k)]: round(v["f1-score"], 4)
        for k, v in report.items()
        if k.isdigit() and int(k) < len(CLASS_NAMES)
    }
    # ROC-AUC (one-vs-rest, requires probability — skip if not available)
    metrics = {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "precision":     float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":        float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1":  per_class_f1,
        "predictions":   y_pred.tolist(),
        "labels":        y_true.tolist(),
    }
    return metrics


def evaluate_sklearn(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Time and evaluate a scikit-learn classifier."""
    t0     = time.perf_counter()
    y_pred = model.predict(X_test)
    ms     = (time.perf_counter() - t0) / len(X_test) * 1000
    m      = compute_metrics(y_pred, y_test)
    m["inference_ms"] = ms
    return m


def evaluate_torch(model: torch.nn.Module,
                   X: np.ndarray, y: np.ndarray,
                   batch_size: int = 256) -> dict:
    """Time and evaluate a PyTorch model."""
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    ms = (time.perf_counter() - t0) / len(y) * 1000

    m = compute_metrics(np.array(preds), np.array(labels))
    m["inference_ms"] = ms
    return m


def noise_robustness_test(predict_fn, X_test: np.ndarray, y_test: np.ndarray,
                           sigmas: list = NOISE_SIGMAS) -> dict:
    """
    Inject Gaussian noise at different sigma levels and record accuracy.
    predict_fn should accept a numpy array and return predicted class indices.
    """
    results = {}
    rng = np.random.default_rng(42)
    for sigma in sigmas:
        X_noisy   = X_test + rng.normal(0, sigma, X_test.shape)
        y_pred    = predict_fn(X_noisy.astype(np.float32))
        acc       = float(accuracy_score(y_test, y_pred))
        results[f"sigma_{sigma}"] = acc
        log.info(f"Noise sigma={sigma:.2f}  accuracy={acc:.4f}")
    return results


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str):
    """Save a labelled confusion matrix to the outputs directory."""
    cm     = confusion_matrix(y_true, y_pred)
    thresh = cm.max() / 2

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved confusion matrix: {path}")
    return path


def plot_training_curves(train_accs: list, val_accs: list,
                         train_losses: list, val_losses: list,
                         title: str, filename: str):
    """Save accuracy and loss training curves."""
    ep = range(1, len(train_accs) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep, train_accs, label="Train", color="steelblue", lw=2)
    ax1.plot(ep, val_accs,   label="Val",   color="tomato",    lw=2)
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(max(0, min(train_accs + val_accs) - 0.05), 1.01)

    ax2.plot(ep, train_losses, label="Train", color="steelblue", lw=2)
    ax2.plot(ep, val_losses,   label="Val",   color="tomato",    lw=2)
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_noise_robustness(results_by_model: dict, filename: str = "noise_robustness.png"):
    """Bar chart comparing accuracy at each noise level for all models."""
    sigmas = [str(s) for s in NOISE_SIGMAS]
    x      = np.arange(len(sigmas))
    width  = 0.8 / max(len(results_by_model), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, res) in enumerate(results_by_model.items()):
        vals = [res.get(f"sigma_{s}", 0) for s in NOISE_SIGMAS]
        ax.bar(x + i * width, vals, width, label=name)

    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Accuracy")
    ax.set_title("Noise Robustness Comparison")
    ax.set_xticks(x + width * (len(results_by_model) - 1) / 2)
    ax.set_xticklabels([f"σ={s}" for s in NOISE_SIGMAS])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_per_class_f1(metrics: dict, model_name: str, filename: str):
    """Bar chart of per-class F1 scores with a 95% target line."""
    pcf1 = metrics.get("per_class_f1", {})
    if not pcf1:
        return None
    classes = list(pcf1.keys())
    scores  = list(pcf1.values())
    colors  = ["#0f766e" if s >= 0.95 else "#f59e0b" if s >= 0.85 else "#dc2626" for s in scores]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(classes, scores, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{model_name} — Per-Class F1")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, label="95% target")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved per-class F1: {path}")
    return path


def compare_models(all_metrics: dict, filename: str = "model_comparison.csv"):
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(m.get("accuracy",  0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall":    round(m.get("recall",    0), 4),
            "F1":        round(m.get("f1",        0), 4),
            "ms/sample": round(m.get("inference_ms", 0), 4),
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    print("\nModel Comparison")
    print(df.to_string(index=False))
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return df
