"""
Interpretability module.

Functions:
  shap_tree_explainer()     — TreeSHAP for RF / XGBoost / GBM
  shap_deep_explainer()     — DeepSHAP for PyTorch models
  lime_explainer()          — LIME local explanation for any predict_fn
  plot_attention_heatmap()  — Transformer self-attention visualisation
  permutation_importance()  — model-agnostic global feature importance
  run_all_interpretability() — calls everything and saves all plots
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ALL_FEATURES, N_CLASSES, CLASS_NAMES, OUTPUT_DIR, DEVICE
from src.utils import get_logger

log = get_logger("interpret")


def shap_tree_explainer(model, X_train: np.ndarray, X_test: np.ndarray,
                        model_name: str, max_display: int = 15):
    """
    Use TreeExplainer (exact, fast for tree models) to get global SHAP values,
    then save a summary bar plot and beeswarm plot.
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping TreeSHAP")
        return None

    log.info(f"Running TreeSHAP for {model_name}")
    explainer = shap.TreeExplainer(model)

    # Use a subsample for speed on large test sets
    idx    = np.random.default_rng(42).choice(len(X_test), size=min(200, len(X_test)), replace=False)
    X_sub  = X_test[idx]
    values = explainer.shap_values(X_sub)

    # For multi-class models shap_values is a list of arrays; average absolute values
    if isinstance(values, list):
        mean_abs = np.mean([np.abs(v) for v in values], axis=0)
    else:
        mean_abs = np.abs(values)

    importances = mean_abs.mean(axis=0)

    # Bar chart of global feature importance
    sorted_idx = np.argsort(importances)[::-1][:max_display]
    fig, ax    = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(sorted_idx)),
            importances[sorted_idx][::-1],
            color="steelblue")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in sorted_idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{model_name} — Global Feature Importance (SHAP)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_shap_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved SHAP plot: {path}")
    return importances


def shap_deep_explainer(model: torch.nn.Module,
                        X_train: np.ndarray, X_test: np.ndarray,
                        model_name: str):
    """
    DeepSHAP for PyTorch models.
    Uses a background dataset of 100 training samples.
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping DeepSHAP")
        return None

    log.info(f"Running DeepSHAP for {model_name}")
    model.eval()

    bg_idx  = np.random.default_rng(42).choice(len(X_train), 100, replace=False)
    bg      = torch.tensor(X_train[bg_idx], dtype=torch.float32)
    te_idx  = np.random.default_rng(0).choice(len(X_test), min(50, len(X_test)), replace=False)
    te      = torch.tensor(X_test[te_idx], dtype=torch.float32)

    try:
        explainer = shap.DeepExplainer(model, bg)
        shap_vals = explainer.shap_values(te)  # list of (n_samples, n_features) per class

        # Average absolute SHAP across classes
        mean_abs = np.mean([np.abs(v) for v in shap_vals], axis=0).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1][:15]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(range(len(sorted_idx)), mean_abs[sorted_idx][::-1], color="tomato")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([ALL_FEATURES[i] for i in sorted_idx[::-1]], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_name} — DeepSHAP Feature Importance")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_deep_shap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved DeepSHAP plot: {path}")
        return mean_abs
    except Exception as e:
        log.warning(f"DeepSHAP failed: {e}")
        return None


def lime_explainer(predict_fn, X_train: np.ndarray, X_test: np.ndarray,
                   sample_idx: int, model_name: str):
    """
    LIME local explanation for a single test sample.
    predict_fn must accept a 2D numpy array and return class probabilities.
    """
    try:
        import lime.lime_tabular as lt
    except ImportError:
        log.warning("lime not installed — skipping LIME")
        return None

    log.info(f"Running LIME for {model_name} on sample {sample_idx}")
    explainer = lt.LimeTabularExplainer(
        X_train,
        feature_names=ALL_FEATURES,
        class_names=CLASS_NAMES,
        mode="classification",
        random_state=42,
    )
    explanation = explainer.explain_instance(
        X_test[sample_idx], predict_fn,
        num_features=10, num_samples=500,
    )

    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"{model_name} — LIME Explanation (Sample {sample_idx})")
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_lime_sample{sample_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved LIME plot: {path}")
    return explanation


def plot_attention_heatmap(model: torch.nn.Module, x_sample: np.ndarray,
                           model_name: str = "Transformer"):
    """
    Visualise the average self-attention weight matrix across all heads and layers
    for a single input sample. Bright cells = features that attend to each other.
    """
    log.info("Generating attention heatmap")
    model.eval()
    x_t  = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)

    # Collect attention weights from all encoder layers
    all_weights = []

    def hook_fn(module, inp, out):
        # self_attn returns (output, weights) when need_weights=True
        pass  # hooks approach varies by PyTorch version; use manual extraction below

    with torch.no_grad():
        # Run each layer manually to capture attention weights
        x_tok = model.embed(x_t.unsqueeze(-1)) + model.pos
        for layer in model.encoder.layers:
            attn_out, attn_w = layer.self_attn(x_tok, x_tok, x_tok,
                                               need_weights=True,
                                               average_attn_weights=True)
            all_weights.append(attn_w.squeeze(0).cpu().numpy())
            # Run the rest of the layer manually
            x_tok = layer.norm1(x_tok + attn_out)
            ff    = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_tok))))
            x_tok = layer.norm2(x_tok + ff)

    avg_w = np.mean(all_weights, axis=0)   # (n_features, n_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_w, cmap="viridis", aspect="auto")
    ax.set_title("Transformer Self-Attention (avg across layers & heads)")
    ax.set_xlabel("Feature (key)")
    ax.set_ylabel("Feature (query)")
    ax.set_xticks(range(len(ALL_FEATURES)))
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_xticklabels(ALL_FEATURES, rotation=90, fontsize=7)
    ax.set_yticklabels(ALL_FEATURES, fontsize=7)
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_attention_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved attention heatmap: {path}")
    return avg_w


def permutation_importance(predict_fn, X_test: np.ndarray, y_test: np.ndarray,
                            model_name: str, n_repeats: int = 10):
    """
    Model-agnostic permutation importance: measure accuracy drop when each
    feature is randomly shuffled. Higher drop = more important feature.
    """
    from sklearn.metrics import accuracy_score
    log.info(f"Permutation importance for {model_name}")

    base_acc = accuracy_score(y_test, predict_fn(X_test))
    rng      = np.random.default_rng(42)
    drops    = []

    for fi in range(X_test.shape[1]):
        accs = []
        for _ in range(n_repeats):
            X_perm       = X_test.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            accs.append(accuracy_score(y_test, predict_fn(X_perm)))
        drops.append(base_acc - np.mean(accs))

    drops      = np.array(drops)
    sorted_idx = np.argsort(drops)[::-1][:15]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(sorted_idx)), drops[sorted_idx][::-1], color="mediumseagreen")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in sorted_idx[::-1]], fontsize=9)
    ax.set_xlabel("Accuracy drop when feature permuted")
    ax.set_title(f"{model_name} — Permutation Importance")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_permutation_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved permutation importance: {path}")
    return drops


def run_all_interpretability(sklearn_models: dict, torch_models: dict,
                             X_train, X_test, y_test):
    """
    Run all interpretability methods for all trained models.
    sklearn_models: {"RF": rf_model, "XGB": xgb_model, ...}
    torch_models:  {"Transformer": model, ...}
    """
    for name, model in sklearn_models.items():
        shap_tree_explainer(model, X_train, X_test, name)

        def make_predict_fn(m):
            def fn(X): return m.predict(X)
            return fn

        permutation_importance(make_predict_fn(model), X_test, y_test, name)
        lime_explainer(model.predict_proba, X_train, X_test,
                       sample_idx=0, model_name=name)

    for name, model in torch_models.items():
        model.eval()

        def make_torch_predict(m):
            def fn(X):
                with torch.no_grad():
                    return m(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
            return fn

        def make_torch_proba(m):
            def fn(X):
                with torch.no_grad():
                    return torch.softmax(
                        m(torch.tensor(X, dtype=torch.float32)), dim=1
                    ).numpy()
            return fn

        shap_deep_explainer(model, X_train, X_test, name)
        permutation_importance(make_torch_predict(model), X_test, y_test, name)
        lime_explainer(make_torch_proba(model), X_train, X_test,
                       sample_idx=0, model_name=name)

        # Attention heatmap only for Transformer
        if hasattr(model, "encoder") and hasattr(model, "embed"):
            plot_attention_heatmap(model, X_test[0], model_name=name)
