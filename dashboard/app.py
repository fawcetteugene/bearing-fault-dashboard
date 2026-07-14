"""
Production Streamlit dashboard for bearing fault diagnosis.

This app is inference-only. It loads the frozen production bundle, scores new
inputs, stores the model registry, and provides lightweight explanations.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Allow running this script directly via `streamlit run dashboard/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.artifacts import (
    build_artifact_registry,
    load_artifact_registry,
    load_preprocessing_manifest,
)
from src.config import CLASS_NAMES, ORIGINAL_FEATURES, OUTPUT_DIR
from src.inference import ProductionPredictor


st.set_page_config(
    page_title="Bearing Fault Control Room",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(135deg, #082f49 0%, #0f766e 55%, #164e63 100%);
        color: white;
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        box-shadow: 0 20px 50px rgba(15, 118, 110, 0.20);
        border: 1px solid rgba(255,255,255,0.10);
    }
    .hero h1 {margin: 0; font-size: 2rem; line-height: 1.1;}
    .hero p {margin: 0.35rem 0 0; opacity: 0.92;}
    .card {
        background: white;
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        border: 1px solid #e5eef2;
        box-shadow: 0 10px 26px rgba(2, 8, 23, 0.05);
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: #ecfeff;
        color: #0f766e;
        border: 1px solid #99f6e4;
    }
    .small-muted {color: #64748b; font-size: 0.9rem;}
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    .metric-label {color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;}
    .metric-value {font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-top: 0.1rem;}
</style>
"""


st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_registry():
    try:
        return load_artifact_registry()
    except Exception:
        return build_artifact_registry(force=True)


@st.cache_resource
def get_predictor(model_name: str | None = None):
    return ProductionPredictor(model_name=model_name)


def fmt_pct(value):
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def fmt_ms(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3f} ms"


def model_rows(registry):
    rows = []
    candidates = registry.get("candidates", [])
    prod_name = registry.get("production", {}).get("name")
    for cand in candidates:
        rows.append({
            "name": cand.get("name"),
            "kind": cand.get("kind"),
            "role": "production" if cand.get("name") == prod_name else "archived",
            "accuracy": cand.get("metrics", {}).get("accuracy", cand.get("metrics", {}).get("best_val_acc")),
            "f1": cand.get("metrics", {}).get("f1", cand.get("metrics", {}).get("average_accuracy")),
            "precision": cand.get("metrics", {}).get("precision"),
            "recall": cand.get("metrics", {}).get("recall"),
            "inference_ms": cand.get("metrics", {}).get("inference_ms"),
            "supported_for_dashboard": cand.get("supported_for_dashboard", True),
            "sha256": cand.get("sha256", "n/a"),
        })

    return pd.DataFrame(rows)


def plot_probabilities(proba: np.ndarray, class_names: list[str]):
    order = np.argsort(proba)[::-1][:5]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#0f766e" if i == order[0] else "#38bdf8" for i in order]
    ax.barh([class_names[i] for i in order[::-1]], proba[order[::-1]], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Top Class Probabilities")
    ax.grid(axis="x", alpha=0.18)
    plt.tight_layout()
    return fig


def plot_local_shap(predictor: ProductionPredictor, row_dict: dict[str, float]):
    try:
        import shap
    except Exception:
        return None, "Install `shap` to enable local explanations."

    if predictor.is_torch:
        return None, "Local SHAP is enabled for tree models only in the dashboard."

    X_scaled = predictor.transform_row(row_dict)
    model = predictor.model

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        pred_idx = int(np.argmax(model.predict_proba(X_scaled)[0]))
        if isinstance(shap_values, list):
            values = shap_values[pred_idx][0]
        else:
            values = shap_values[0]

        order = np.argsort(np.abs(values))[::-1][:10]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colors = ["#dc2626" if values[i] > 0 else "#2563eb" for i in order]
        ax.barh([predictor.feature_names[i] for i in order[::-1]], values[order[::-1]], color=colors[::-1])
        ax.axvline(0, color="#0f172a", linewidth=0.8)
        ax.set_title("Local SHAP Explanation")
        ax.set_xlabel("SHAP value")
        plt.tight_layout()
        return fig, None
    except Exception as exc:
        return None, f"SHAP could not be computed for this sample: {exc}"


registry = get_registry()
preproc_manifest = load_preprocessing_manifest()
prod_name = registry.get("production", {}).get("name", "production")
supported_models = [c["name"] for c in registry.get("candidates", []) if c.get("supported_for_dashboard")]
model_choices = [prod_name] + sorted(m for m in supported_models if m != prod_name)

st.sidebar.markdown("## Bearing Fault Control Room")
st.sidebar.markdown(
    f"<span class='badge'>Production model: {prod_name}</span>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Inference only. No retraining occurs in the dashboard.")

selected_model = st.sidebar.selectbox(
    "Frozen model",
    options=model_choices if model_choices else [prod_name],
    index=0,
)

with st.sidebar.expander("Registry snapshot", expanded=False):
    st.write(f"Candidates: {len(registry.get('candidates', []))}")
    if registry.get("production"):
        st.write(f"Production SHA256: `{registry['production']['sha256'][:12]}...`")
    if preproc_manifest:
        st.write(f"Train rows after augmentation: {preproc_manifest.get('augmentation', {}).get('final_train_rows', 'n/a')}")
        st.write(f"Validation rows: {preproc_manifest.get('split_sizes_after_cleaning', {}).get('validation', 'n/a')}")
        st.write(f"Test rows: {preproc_manifest.get('split_sizes_after_cleaning', {}).get('test', 'n/a')}")

st.markdown(
    f"""
    <div class="hero">
        <h1>Bearing Fault Diagnosis Dashboard</h1>
        <p>Frozen feature pipeline, registry-backed model selection, and explainable predictions for the CWRU bearing dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

top_cols = st.columns(4)
prod_metrics = registry.get("production", {}).get("metrics", {})
feature_count = len(registry.get("feature_order", [])) or len(ORIGINAL_FEATURES)
metric_cards = [
    ("Accuracy", fmt_pct(prod_metrics.get("accuracy", prod_metrics.get("best_val_acc")))),
    ("F1 / Task Score", fmt_pct(prod_metrics.get("f1", prod_metrics.get("average_accuracy")))),
    ("Inference", fmt_ms(prod_metrics.get("inference_ms"))),
    ("Features", str(feature_count)),
]
for col, (label, value) in zip(top_cols, metric_cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

tab_predict, tab_batch, tab_registry, tab_explain, tab_compare, tab_about = st.tabs(
    ["Predict", "Batch", "Registry", "Explain", "Compare", "About"]
)


with tab_predict:
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.subheader("Manual prediction")
        st.caption("Enter the 9 raw statistical features. Engineered features are created automatically.")
        input_values = {}
        feature_cols = st.columns(2)
        for idx, feat in enumerate(ORIGINAL_FEATURES):
            with feature_cols[idx % 2]:
                input_values[feat] = st.number_input(feat, value=0.0, format="%.6f", key=f"manual_{feat}")

        run_prediction = st.button("Run prediction", type="primary", width="stretch")

    with right:
        st.subheader("Prediction output")
        if run_prediction:
            predictor = get_predictor(None if selected_model == prod_name else selected_model)
            result = predictor.predict_row(input_values)
            color = "#0f766e" if result.confidence >= 0.95 else "#b45309"
            st.markdown(
                f"""
                <div class="card">
                    <span class="badge">Prediction complete</span>
                    <h3 style="margin: 0.8rem 0 0.25rem; color: {color};">
                        {result.label_name}
                    </h3>
                    <p class="small-muted" style="margin: 0;">
                        Confidence: <b>{result.confidence * 100:.2f}%</b> ·
                        Inference: <b>{result.inference_ms:.3f} ms</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.pyplot(plot_probabilities(result.probabilities, predictor.class_names))
            st.dataframe(
                pd.DataFrame({
                    "class": predictor.class_names,
                    "probability": result.probabilities,
                }).sort_values("probability", ascending=False),
                width="stretch",
            )
            st.download_button(
                "Download prediction JSON",
                data=json.dumps(
                    {
                        "model": predictor.model_name,
                        "prediction": result.label_name,
                        "confidence": result.confidence,
                        "probabilities": result.probabilities.tolist(),
                        "input": input_values,
                    },
                    indent=2,
                ).encode("utf-8"),
                file_name="bearing_fault_prediction.json",
                mime="application/json",
                width="stretch",
            )
        else:
            st.info("Enter values and click Run prediction.")


with tab_batch:
    st.subheader("Batch scoring")
    uploaded = st.file_uploader("Upload a CSV containing the 9 raw features", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write(f"Rows: {len(df_raw)}")
        st.dataframe(df_raw.head(10), width="stretch")
        if st.button("Run batch scoring", type="primary"):
            predictor = get_predictor(None if selected_model == prod_name else selected_model)
            start = time.perf_counter()
            try:
                result_df = predictor.predict_frame(df_raw)
                elapsed = (time.perf_counter() - start) * 1000
                st.success(f"Scored {len(result_df)} rows in {elapsed:.1f} ms")
                st.dataframe(result_df[["predicted_class", "predicted_index", "confidence"]].head(50), width="stretch")

                fig, ax = plt.subplots(figsize=(7, 4))
                counts = result_df["predicted_class"].value_counts().sort_values(ascending=False)
                ax.bar(counts.index, counts.values, color="#0f766e")
                ax.set_title("Predicted class distribution")
                ax.tick_params(axis="x", rotation=35)
                ax.grid(axis="y", alpha=0.18)
                plt.tight_layout()
                st.pyplot(fig)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored CSV",
                    data=csv_bytes,
                    file_name="bearing_fault_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )
            except Exception as exc:
                st.error(str(exc))
    else:
        st.caption("Upload a file to score multiple samples at once.")


with tab_registry:
    st.subheader("Artifact registry")
    st.caption("The dashboard reads only frozen artifacts. No training occurs here.")

    rows = model_rows(registry)
    if not rows.empty:
        display = rows[["name", "kind", "role", "accuracy", "f1", "precision", "recall", "inference_ms", "supported_for_dashboard", "sha256"]].copy()
        st.dataframe(display, width="stretch")
    else:
        st.info("No model registry entries found yet.")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Production bundle**")
        st.json(registry.get("production", {}))
    with cols[1]:
        st.markdown("**Preprocessing manifest**")
        st.json(preproc_manifest if preproc_manifest else {"status": "missing"})

    if st.button("Refresh registry", width="stretch"):
        build_artifact_registry(force=True)
        st.cache_resource.clear()
        st.rerun()


with tab_explain:
    st.subheader("Local explanation")
    st.caption("Tree-based SHAP is available for the production RF/XGB/GBM models.")
    sample_source = st.radio("Explanation sample", ["Manual input", "Uploaded batch first row"], horizontal=True)
    sample_row = input_values if "input_values" in locals() else {feat: 0.0 for feat in ORIGINAL_FEATURES}

    if sample_source == "Uploaded batch first row" and "df_raw" in locals() and not df_raw.empty:
        sample_row = df_raw.iloc[0][ORIGINAL_FEATURES].to_dict()

    explain_button = st.button("Generate explanation", type="primary")
    if explain_button:
        predictor = get_predictor(None if selected_model == prod_name else selected_model)
        fig, message = plot_local_shap(predictor, sample_row)
        if fig is not None:
            st.pyplot(fig)
        else:
            st.info(message)

        try:
            sample_result = predictor.predict_row(sample_row)
            st.write(
                pd.DataFrame(
                    {"class": predictor.class_names, "probability": sample_result.probabilities}
                ).sort_values("probability", ascending=False).head(5)
            )
        except Exception as exc:
            st.warning(str(exc))

    st.divider()
    st.markdown("### Saved interpretability outputs")
    saved = sorted([f for f in os.listdir(OUTPUT_DIR) if ("shap" in f or "attention" in f or "lime" in f) and f.endswith(".png")])
    if saved:
        for fname in saved[:12]:
            st.image(os.path.join(OUTPUT_DIR, fname), caption=fname, width="stretch")
    else:
        st.info("No saved interpretability plots were found in outputs/.")


with tab_compare:
    st.subheader("Model comparison")
    st.caption("Accuracy and F1 across all registered models.")

    rows_df = model_rows(registry)
    if not rows_df.empty:
        plot_df = rows_df.dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False)
        if not plot_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colors = ["#0f766e" if r == "production" else "#38bdf8" for r in plot_df["role"]]
            axes[0].barh(plot_df["name"][::-1], plot_df["accuracy"][::-1], color=colors[::-1])
            axes[0].set_xlim(0, 1.05)
            axes[0].axvline(0.95, color="gray", linestyle="--", linewidth=0.8, label="95% target")
            axes[0].set_title("Test Accuracy")
            axes[0].set_xlabel("Accuracy")
            axes[0].legend()
            axes[0].grid(axis="x", alpha=0.3)

            f1_df = plot_df.dropna(subset=["f1"])
            if not f1_df.empty:
                axes[1].barh(f1_df["name"][::-1], f1_df["f1"][::-1], color=colors[:len(f1_df)][::-1])
                axes[1].set_xlim(0, 1.05)
                axes[1].axvline(0.95, color="gray", linestyle="--", linewidth=0.8)
                axes[1].set_title("Weighted F1")
                axes[1].set_xlabel("F1")
                axes[1].grid(axis="x", alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.dataframe(
            rows_df[["name", "kind", "role", "accuracy", "f1", "precision", "recall", "inference_ms"]]
            .sort_values("accuracy", ascending=False, na_position="last"),
            width="stretch",
        )

        # Per-class F1 plots saved to outputs/
        st.divider()
        st.markdown("### Per-class F1 plots")
        pcf1_plots = sorted([f for f in os.listdir(OUTPUT_DIR) if "per_class_f1" in f and f.endswith(".png")])
        if pcf1_plots:
            cols = st.columns(2)
            for i, fname in enumerate(pcf1_plots):
                with cols[i % 2]:
                    st.image(os.path.join(OUTPUT_DIR, fname), caption=fname, width="stretch")
        else:
            st.info("Train models to generate per-class F1 plots.")
    else:
        st.info("No model metrics available yet.")


with tab_about:
    st.subheader("System summary")
    st.markdown(
        """
        - Dataset: CWRU feature CSV with 2,300 original rows and 10 fault classes
        - Pipeline: leakage-free split, outlier filtering, feature engineering (25 features), scaling, frozen artifact registry
        - Models: RF, XGBoost, GBM, LightGBM, Stacking Ensemble, Transformer, CNN, ResNet1D, LSTM, MAML, Meta-SGD, FBCL
        - Runtime: model selection is read from the registry and never retrained inside the dashboard
        - Output: prediction, confidence, batch scoring, local SHAP explanation, model comparison
        """
    )
    st.markdown("### Target metrics")
    st.markdown(
        """
        | Metric | Target |
        |--------|--------|
        | Test Accuracy | ≥ 95% |
        | False Positive Rate | ≤ 5% |
        | Inference Time | ≤ 150 ms |
        """
    )
