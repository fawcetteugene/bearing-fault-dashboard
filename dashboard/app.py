"""
Production Streamlit dashboard for bearing fault diagnosis.
Inference-only: loads frozen production bundle, scores inputs, shows explanations.
"""
from __future__ import annotations

import io, json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.artifacts import build_artifact_registry, load_artifact_registry, load_preprocessing_manifest
from src.config import CLASS_NAMES, ORIGINAL_FEATURES, OUTPUT_DIR
from src.inference import ProductionPredictor

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bearing Fault Control Room",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global ---- */
[data-testid="stAppViewContainer"] { background: #0b1120; }
[data-testid="stSidebar"] { background: #0f1929 !important; border-right: 1px solid #1e2d45; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 0.8rem; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1,h2,h3,h4 { color: #f1f5f9 !important; }
p, li, label, .stMarkdown { color: #cbd5e1 !important; }

/* ---- hero banner ---- */
.hero {
    background: linear-gradient(135deg, #0f2744 0%, #0f766e 60%, #134e4a 100%);
    padding: 1.5rem 2rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 24px 60px rgba(15,118,110,0.25);
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 1.9rem; color: #fff !important; }
.hero p  { margin: 0.4rem 0 0; color: rgba(255,255,255,0.82) !important; font-size: 0.95rem; }

/* ---- metric cards ---- */
.mcard {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.mcard-label { color: #64748b !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; }
.mcard-value { font-size: 1.65rem; font-weight: 800; margin-top: 0.15rem; }
.mcard-sub   { font-size: 0.75rem; margin-top: 0.2rem; }
.green  { color: #34d399 !important; }
.amber  { color: #fbbf24 !important; }
.blue   { color: #38bdf8 !important; }
.slate  { color: #94a3b8 !important; }

/* ---- prediction result card ---- */
.pred-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.5rem;
}
.pred-label { font-size: 1.5rem; font-weight: 800; }
.pred-meta  { color: #94a3b8 !important; font-size: 0.85rem; margin-top: 0.3rem; }

/* ---- badge ---- */
.badge {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    background: #134e4a;
    color: #6ee7b7 !important;
    border: 1px solid #065f46;
}
.badge-amber {
    background: #451a03;
    color: #fcd34d !important;
    border: 1px solid #92400e;
}

/* ---- section card ---- */
.scard {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}

/* ---- tabs ---- */
[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 600;
    font-size: 0.88rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #34d399 !important;
    border-bottom-color: #34d399 !important;
}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ---- divider ---- */
hr { border-color: #1e2d45 !important; }

/* ---- info / warning boxes ---- */
[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Bearing condition knowledge base ─────────────────────────────────────────
BEARING_INFO: dict[str, dict] = {
    # ---- Normal ----
    "Normal_1": {
        "status": "✅ Healthy",
        "status_color": "green",
        "location": "No fault detected",
        "severity": "None",
        "severity_color": "green",
        "what_it_means": (
            "The bearing is operating normally. No damage has been detected. "
            "Vibration levels are within expected limits."
        ),
        "urgency": "No action required",
        "urgency_color": "green",
        "recommended_action": "Continue routine scheduled maintenance.",
        "icon": "✅",
    },
    # ---- Ball fault ----
    "Ball_007_1": {
        "status": "⚠️ Ball Fault — Early Stage",
        "status_color": "amber",
        "location": "Rolling element (ball)",
        "severity": "Minor (0.007\" diameter pit)",
        "severity_color": "amber",
        "what_it_means": (
            "A small pit or spall has formed on one of the rolling balls inside the bearing. "
            "At this early stage the damage is minor and the bearing can still function, "
            "but it will worsen if left unattended."
        ),
        "urgency": "Monitor closely",
        "urgency_color": "amber",
        "recommended_action": "Schedule inspection within 2–4 weeks. Increase vibration monitoring frequency.",
        "icon": "⚠️",
    },
    "Ball_014_1": {
        "status": "🔶 Ball Fault — Moderate",
        "status_color": "amber",
        "location": "Rolling element (ball)",
        "severity": "Moderate (0.014\" diameter pit)",
        "severity_color": "amber",
        "what_it_means": (
            "The ball damage has grown to a moderate size. Vibration and noise will be noticeably higher. "
            "Risk of accelerated wear and unexpected failure is increasing."
        ),
        "urgency": "Plan replacement soon",
        "urgency_color": "amber",
        "recommended_action": "Replace bearing within 1–2 weeks. Avoid high-load operation.",
        "icon": "🔶",
    },
    "Ball_021_1": {
        "status": "🔴 Ball Fault — Severe",
        "status_color": "red",
        "location": "Rolling element (ball)",
        "severity": "Severe (0.021\" diameter pit)",
        "severity_color": "red",
        "what_it_means": (
            "The ball has significant damage. Continued operation risks catastrophic bearing failure, "
            "which can damage surrounding machinery and cause unplanned downtime."
        ),
        "urgency": "Replace immediately",
        "urgency_color": "red",
        "recommended_action": "Stop machine and replace bearing as soon as possible.",
        "icon": "🔴",
    },
    # ---- Inner race fault ----
    "IR_007_1": {
        "status": "⚠️ Inner Race Fault — Early Stage",
        "status_color": "amber",
        "location": "Inner race (the ring closest to the shaft)",
        "severity": "Minor (0.007\" diameter pit)",
        "severity_color": "amber",
        "what_it_means": (
            "A small defect has formed on the inner ring of the bearing — the part that rotates with the shaft. "
            "Each time a ball rolls over the defect it produces a small impact. At this stage the bearing is still serviceable."
        ),
        "urgency": "Monitor closely",
        "urgency_color": "amber",
        "recommended_action": "Schedule inspection within 2–4 weeks. Log vibration trend.",
        "icon": "⚠️",
    },
    "IR_014_1": {
        "status": "🔶 Inner Race Fault — Moderate",
        "status_color": "amber",
        "location": "Inner race (the ring closest to the shaft)",
        "severity": "Moderate (0.014\" diameter pit)",
        "severity_color": "amber",
        "what_it_means": (
            "The inner race defect has grown. Impacts from rolling elements are stronger and more frequent. "
            "Heat generation and wear rate are elevated."
        ),
        "urgency": "Plan replacement soon",
        "urgency_color": "amber",
        "recommended_action": "Replace bearing within 1–2 weeks. Check shaft alignment.",
        "icon": "🔶",
    },
    "IR_021_1": {
        "status": "🔴 Inner Race Fault — Severe",
        "status_color": "red",
        "location": "Inner race (the ring closest to the shaft)",
        "severity": "Severe (0.021\" diameter pit)",
        "severity_color": "red",
        "what_it_means": (
            "The inner race has severe damage. The bearing is at high risk of sudden failure. "
            "Continued operation may cause shaft damage and secondary failures in connected equipment."
        ),
        "urgency": "Replace immediately",
        "urgency_color": "red",
        "recommended_action": "Stop machine and replace bearing immediately. Inspect shaft for scoring.",
        "icon": "🔴",
    },
    # ---- Outer race fault ----
    "OR_007_6_1": {
        "status": "⚠️ Outer Race Fault — Early Stage",
        "status_color": "amber",
        "location": "Outer race (the stationary outer ring)",
        "severity": "Minor (0.007\" diameter pit)",
        "severity_color": "amber",
        "what_it_means": (
            "A small defect has formed on the outer ring of the bearing — the stationary part fixed to the housing. "
            "Outer race faults produce a characteristic repetitive impact pattern. Early detection is good."
        ),
        "urgency": "Monitor closely",
        "urgency_color": "amber",
        "recommended_action": "Schedule inspection within 2–4 weeks. Check lubrication.",
        "icon": "⚠️",
    },
    "OR_014_6_1": {
        "status": "🔶 Outer Race Fault — Moderate",
        "status_color": "amber",
        "location": "Outer race (the stationary outer ring)",
        "severity": "Moderate (0.014\" diameter pit)",
        "severity_color": "amber",
        "what_it_means": (
            "The outer race defect has grown to a moderate size. Vibration amplitude is noticeably higher. "
            "The bearing housing may also be experiencing increased stress."
        ),
        "urgency": "Plan replacement soon",
        "urgency_color": "amber",
        "recommended_action": "Replace bearing within 1–2 weeks. Inspect housing bore for wear.",
        "icon": "🔶",
    },
    "OR_021_6_1": {
        "status": "🔴 Outer Race Fault — Severe",
        "status_color": "red",
        "location": "Outer race (the stationary outer ring)",
        "severity": "Severe (0.021\" diameter pit)",
        "severity_color": "red",
        "what_it_means": (
            "The outer race has severe damage. The bearing is close to end-of-life. "
            "Risk of sudden seizure or fragmentation is high, which can cause serious equipment damage."
        ),
        "urgency": "Replace immediately",
        "urgency_color": "red",
        "recommended_action": "Stop machine and replace bearing immediately. Inspect housing and lubrication system.",
        "icon": "🔴",
    },
}

_URGENCY_BG = {"green": "#052e16", "amber": "#451a03", "red": "#450a0a"}
_URGENCY_BORDER = {"green": "#166534", "amber": "#92400e", "red": "#991b1b"}
_URGENCY_TEXT = {"green": "#86efac", "amber": "#fcd34d", "red": "#fca5a5"}

def render_bearing_condition(label_name: str, confidence: float) -> None:
    info = BEARING_INFO.get(label_name)
    if not info:
        return
    uc = info["urgency_color"]
    sc = info["severity_color"]
    low_conf_note = (
        f"<p style='color:#fbbf24;font-size:0.8rem;margin:0.4rem 0 0'>⚠️ Confidence is {confidence*100:.1f}% — "
        "consider re-checking sensor readings before acting.</p>"
        if confidence < 0.80 else ""
    )
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d45;border-radius:18px;padding:1.2rem 1.4rem;margin-top:1rem">
      <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:0.8rem">
        🔎 Bearing Condition Report
      </div>

      <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:1rem">
        <span style="background:{_URGENCY_BG[uc]};color:{_URGENCY_TEXT[uc]};border:1px solid {_URGENCY_BORDER[uc]};
              padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:700">
          {info['urgency']}
        </span>
        <span style="background:#0f172a;color:#94a3b8;border:1px solid #1e2d45;
              padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem">
          📍 {info['location']}
        </span>
        <span style="background:{_URGENCY_BG[sc]};color:{_URGENCY_TEXT[sc]};border:1px solid {_URGENCY_BORDER[sc]};
              padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem">
          📏 {info['severity']}
        </span>
      </div>

      <p style="color:#cbd5e1;font-size:0.92rem;line-height:1.6;margin:0 0 0.9rem">
        {info['what_it_means']}
      </p>

      <div style="background:{_URGENCY_BG[uc]};border:1px solid {_URGENCY_BORDER[uc]};
                  border-radius:12px;padding:0.75rem 1rem">
        <div style="color:{_URGENCY_TEXT[uc]};font-size:0.78rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.3rem">
          Recommended Action
        </div>
        <div style="color:#f1f5f9;font-size:0.9rem">{info['recommended_action']}</div>
        {low_conf_note}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def get_registry():
    try:
        return load_artifact_registry()
    except Exception:
        return build_artifact_registry(force=True)

@st.cache_resource
def get_predictor(model_name: str | None = None):
    return ProductionPredictor(model_name=model_name)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_pct(v):
    return f"{float(v)*100:.2f}%" if v is not None else "n/a"

def fmt_ms(v):
    return f"{float(v):.1f} ms" if v is not None else "n/a"

def status_color(v, target=0.95):
    if v is None: return "slate"
    return "green" if float(v) >= target else "amber"

def model_rows(registry) -> pd.DataFrame:
    prod_name = registry.get("production", {}).get("name")
    rows = []
    for c in registry.get("candidates", []):
        m = c.get("metrics", {})
        rows.append({
            "name": c.get("name"),
            "kind": c.get("kind"),
            "role": "🟢 production" if c.get("name") == prod_name else "archived",
            "accuracy": m.get("accuracy", m.get("best_val_acc")),
            "f1": m.get("f1", m.get("average_accuracy")),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "inference_ms": m.get("inference_ms"),
            "supported": c.get("supported_for_dashboard", True),
            "sha256": (c.get("sha256") or "n/a")[:12] + "…",
        })
    return pd.DataFrame(rows)

def plot_probabilities(proba: np.ndarray, class_names: list[str]):
    order = np.argsort(proba)[::-1][:6]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    colors = ["#34d399" if i == order[0] else "#1e3a5f" for i in order]
    bars = ax.barh([class_names[i] for i in order[::-1]], proba[order[::-1]], color=colors[::-1], height=0.6)
    for bar, val in zip(bars, proba[order[::-1]]):
        ax.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", color="#cbd5e1", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Probability", color="#64748b")
    ax.set_title("Top Predicted Classes", color="#f1f5f9", fontsize=11, pad=10)
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#1e2d45")
    ax.grid(axis="x", color="#1e2d45", linewidth=0.6)
    plt.tight_layout()
    return fig

def plot_comparison(rows_df: pd.DataFrame):
    plot_df = rows_df.dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False)
    if plot_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(plot_df)*0.55)))
    fig.patch.set_facecolor("#111827")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#1e2d45")
        ax.grid(axis="x", color="#1e2d45", linewidth=0.6)

    colors = ["#34d399" if "production" in str(r) else "#38bdf8" for r in plot_df["role"]]
    axes[0].barh(plot_df["name"][::-1], plot_df["accuracy"][::-1], color=colors[::-1], height=0.6)
    axes[0].axvline(0.95, color="#fbbf24", linestyle="--", linewidth=1, label="95% target")
    axes[0].set_xlim(0, 1.08)
    axes[0].set_title("Test Accuracy", color="#f1f5f9")
    axes[0].set_xlabel("Accuracy", color="#64748b")
    axes[0].legend(facecolor="#111827", labelcolor="#fbbf24", edgecolor="#1e2d45")

    f1_df = plot_df.dropna(subset=["f1"])
    if not f1_df.empty:
        axes[1].barh(f1_df["name"][::-1], f1_df["f1"][::-1], color=colors[:len(f1_df)][::-1], height=0.6)
        axes[1].axvline(0.95, color="#fbbf24", linestyle="--", linewidth=1)
        axes[1].set_xlim(0, 1.08)
        axes[1].set_title("Weighted F1", color="#f1f5f9")
        axes[1].set_xlabel("F1", color="#64748b")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    return fig

def plot_local_shap(predictor, row_dict):
    try:
        import shap
    except Exception:
        return None, "Install `shap` to enable local explanations."
    if predictor.is_torch:
        return None, "Local SHAP is available for tree models only."
    X_scaled = predictor.transform_row(row_dict)
    try:
        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer.shap_values(X_scaled)
        pred_idx = int(np.argmax(predictor.model.predict_proba(X_scaled)[0]))
        values = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0]
        order = np.argsort(np.abs(values))[::-1][:10]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        colors = ["#f87171" if values[i] > 0 else "#60a5fa" for i in order]
        ax.barh([predictor.feature_names[i] for i in order[::-1]], values[order[::-1]], color=colors[::-1], height=0.6)
        ax.axvline(0, color="#94a3b8", linewidth=0.8)
        ax.set_title("Local SHAP Explanation", color="#f1f5f9")
        ax.set_xlabel("SHAP value", color="#64748b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#1e2d45")
        ax.grid(axis="x", color="#1e2d45", linewidth=0.6)
        plt.tight_layout()
        return fig, None
    except Exception as exc:
        return None, f"SHAP could not be computed: {exc}"

# ── Load data ─────────────────────────────────────────────────────────────────
registry        = get_registry()
preproc_manifest = load_preprocessing_manifest()
prod_name       = registry.get("production", {}).get("name", "production")
supported_models = [c["name"] for c in registry.get("candidates", []) if c.get("supported_for_dashboard")]
model_choices   = [prod_name] + sorted(m for m in supported_models if m != prod_name)
prod_metrics    = registry.get("production", {}).get("metrics", {})
feature_count   = len(registry.get("feature_order", [])) or len(ORIGINAL_FEATURES)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Control Room")
st.sidebar.markdown(
    f"<span class='badge'>Production: {prod_name}</span>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Inference only — no retraining in the dashboard.")
st.sidebar.divider()

selected_model = st.sidebar.selectbox(
    "Active model",
    options=model_choices if model_choices else [prod_name],
    index=0,
)

acc_val = prod_metrics.get("accuracy", prod_metrics.get("best_val_acc"))
inf_val = prod_metrics.get("inference_ms")
st.sidebar.markdown("**Production health**")
acc_ok = acc_val is not None and float(acc_val) >= 0.95
inf_ok = inf_val is None or float(inf_val) <= 150
st.sidebar.markdown(
    f"{'✅' if acc_ok else '⚠️'} Accuracy: **{fmt_pct(acc_val)}**  \n"
    f"{'✅' if inf_ok else '⚠️'} Inference: **{fmt_ms(inf_val)}**",
)

with st.sidebar.expander("Registry snapshot", expanded=False):
    st.write(f"Candidates: {len(registry.get('candidates', []))}")
    if registry.get("production"):
        sha = registry["production"].get("sha256", "")
        st.write(f"SHA256: `{sha[:14]}…`")
    if preproc_manifest:
        aug = preproc_manifest.get("augmentation", {})
        split = preproc_manifest.get("split_sizes_after_cleaning", {})
        st.write(f"Train rows: {aug.get('final_train_rows', 'n/a')}")
        st.write(f"Val rows: {split.get('validation', 'n/a')}")
        st.write(f"Test rows: {split.get('test', 'n/a')}")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚙️ Bearing Fault Diagnosis</h1>
  <p>Frozen feature pipeline · Registry-backed model selection · Explainable predictions · CWRU dataset</p>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
kpi_cols = st.columns(4)
kpis = [
    ("Accuracy",    fmt_pct(acc_val),  status_color(acc_val),  "Target ≥ 95%"),
    ("F1 Score",    fmt_pct(prod_metrics.get("f1", prod_metrics.get("average_accuracy"))),
                    status_color(prod_metrics.get("f1", prod_metrics.get("average_accuracy"))), "Weighted"),
    ("Inference",   fmt_ms(inf_val),   "green" if inf_ok else "amber", "Target ≤ 150 ms"),
    ("Features",    str(feature_count), "blue", "Engineered"),
]
for col, (label, value, color, sub) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="mcard">
          <div class="mcard-label">{label}</div>
          <div class="mcard-value {color}">{value}</div>
          <div class="mcard-sub slate">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.write("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_registry, tab_explain, tab_compare, tab_about = st.tabs([
    "🔍 Predict", "📂 Batch", "🗂 Registry", "💡 Explain", "📊 Compare", "ℹ️ About"
])

# ── Tab: Predict ──────────────────────────────────────────────────────────────
with tab_predict:
    left, right = st.columns([1.1, 1.0], gap="large")

    with left:
        st.markdown("#### Enter raw features")
        st.caption("9 statistical features — engineered features are derived automatically.")
        input_values: dict[str, float] = {}
        fc = st.columns(2)
        for idx, feat in enumerate(ORIGINAL_FEATURES):
            with fc[idx % 2]:
                input_values[feat] = st.number_input(feat, value=0.0, format="%.6f", key=f"m_{feat}")
        run_btn = st.button("▶ Run Prediction", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Result")
        if run_btn:
            predictor = get_predictor(None if selected_model == prod_name else selected_model)
            with st.spinner("Running inference…"):
                result = predictor.predict_row(input_values)
            conf_color = "green" if result.confidence >= 0.95 else "amber"
            badge_cls  = "badge" if result.confidence >= 0.95 else "badge-amber"
            st.markdown(f"""
            <div class="pred-card">
              <span class="{badge_cls}">{'HIGH CONFIDENCE' if result.confidence >= 0.95 else 'LOW CONFIDENCE'}</span>
              <div class="pred-label {conf_color}" style="margin-top:0.6rem">{result.label_name}</div>
              <div class="pred-meta">
                Confidence: <b>{result.confidence*100:.2f}%</b> &nbsp;·&nbsp;
                Inference: <b>{result.inference_ms:.2f} ms</b> &nbsp;·&nbsp;
                Model: <b>{predictor.model_name}</b>
              </div>
            </div>""", unsafe_allow_html=True)

            render_bearing_condition(result.label_name, result.confidence)

            st.pyplot(plot_probabilities(result.probabilities, predictor.class_names))
            plt.close("all")

            with st.expander("Full probability table"):
                st.dataframe(
                    pd.DataFrame({"class": predictor.class_names, "probability": result.probabilities})
                    .sort_values("probability", ascending=False)
                    .style.format({"probability": "{:.4f}"}),
                    use_container_width=True,
                )
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps({
                    "model": predictor.model_name,
                    "prediction": result.label_name,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities.tolist(),
                    "input": input_values,
                }, indent=2).encode(),
                file_name="bearing_prediction.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Enter feature values and click **Run Prediction**.")


# ── Tab: Batch ────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Batch scoring")
    st.caption("Upload a CSV with the 9 raw feature columns.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")

    df_raw: pd.DataFrame | None = None
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df_raw))
        c2.metric("Columns", len(df_raw.columns))
        missing = [f for f in ORIGINAL_FEATURES if f not in df_raw.columns]
        c3.metric("Missing features", len(missing))
        if missing:
            st.warning(f"Missing columns: {missing}")
        st.dataframe(df_raw.head(8), use_container_width=True)

        if st.button("▶ Run Batch Scoring", type="primary", use_container_width=True):
            predictor = get_predictor(None if selected_model == prod_name else selected_model)
            with st.spinner(f"Scoring {len(df_raw)} rows…"):
                t0 = time.perf_counter()
                try:
                    result_df = predictor.predict_frame(df_raw)
                    elapsed = (time.perf_counter() - t0) * 1000
                    st.success(f"✅ Scored **{len(result_df)}** rows in **{elapsed:.1f} ms** ({elapsed/len(result_df):.2f} ms/row)")

                    col_a, col_b = st.columns([1.2, 1])
                    with col_a:
                        st.dataframe(
                            result_df[["predicted_class", "predicted_index", "confidence"]].head(50),
                            use_container_width=True,
                        )
                    with col_b:
                        counts = result_df["predicted_class"].value_counts().sort_values(ascending=False)
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        fig.patch.set_facecolor("#111827")
                        ax.set_facecolor("#111827")
                        ax.bar(counts.index, counts.values, color="#38bdf8")
                        ax.set_title("Predicted class distribution", color="#f1f5f9")
                        ax.tick_params(axis="x", rotation=40, colors="#94a3b8")
                        ax.tick_params(axis="y", colors="#94a3b8")
                        ax.spines[:].set_color("#1e2d45")
                        ax.grid(axis="y", color="#1e2d45", linewidth=0.6)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    st.download_button(
                        "⬇ Download scored CSV",
                        data=result_df.to_csv(index=False).encode(),
                        file_name="bearing_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("Upload a CSV file to score multiple samples at once.")


# ── Tab: Registry ─────────────────────────────────────────────────────────────
with tab_registry:
    st.markdown("#### Artifact registry")
    st.caption("Read-only view of frozen model artifacts. No training occurs here.")

    rows_df = model_rows(registry)
    if not rows_df.empty:
        st.dataframe(
            rows_df.style.format({
                "accuracy": lambda v: fmt_pct(v) if v is not None else "n/a",
                "f1":       lambda v: fmt_pct(v) if v is not None else "n/a",
                "precision":lambda v: fmt_pct(v) if v is not None else "n/a",
                "recall":   lambda v: fmt_pct(v) if v is not None else "n/a",
                "inference_ms": lambda v: fmt_ms(v) if v is not None else "n/a",
            }),
            use_container_width=True,
        )
    else:
        st.info("No registry entries found.")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Production bundle**")
        st.json(registry.get("production", {}))
    with r2:
        st.markdown("**Preprocessing manifest**")
        st.json(preproc_manifest if preproc_manifest else {"status": "missing"})

    if st.button("🔄 Refresh registry", use_container_width=True):
        build_artifact_registry(force=True)
        st.cache_resource.clear()
        st.rerun()


# ── Tab: Explain ──────────────────────────────────────────────────────────────
with tab_explain:
    st.markdown("#### Local explanation")
    st.caption("Tree-based SHAP for RF / XGBoost / GBM production models.")

    sample_source = st.radio("Sample source", ["Manual input", "Batch first row"], horizontal=True)
    sample_row = input_values if "input_values" in locals() else {f: 0.0 for f in ORIGINAL_FEATURES}
    if sample_source == "Batch first row" and df_raw is not None and not df_raw.empty:
        sample_row = df_raw.iloc[0][ORIGINAL_FEATURES].to_dict()

    if st.button("💡 Generate Explanation", type="primary", use_container_width=True):
        predictor = get_predictor(None if selected_model == prod_name else selected_model)
        with st.spinner("Computing SHAP values…"):
            fig, msg = plot_local_shap(predictor, sample_row)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(msg)
        try:
            r = predictor.predict_row(sample_row)
            st.dataframe(
                pd.DataFrame({"class": predictor.class_names, "probability": r.probabilities})
                .sort_values("probability", ascending=False).head(5)
                .style.format({"probability": "{:.4f}"}),
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(str(exc))

    st.divider()
    st.markdown("**Saved interpretability outputs**")
    saved = sorted([f for f in os.listdir(OUTPUT_DIR)
                    if any(k in f for k in ("shap", "attention", "lime")) and f.endswith(".png")])
    if saved:
        img_cols = st.columns(2)
        for i, fname in enumerate(saved[:12]):
            with img_cols[i % 2]:
                st.image(os.path.join(OUTPUT_DIR, fname), caption=fname, use_container_width=True)
    else:
        st.info("No saved interpretability plots found in outputs/.")


# ── Tab: Compare ──────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("#### Model comparison")
    rows_df = model_rows(registry)
    if not rows_df.empty:
        fig = plot_comparison(rows_df)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

        st.dataframe(
            rows_df[["name", "kind", "role", "accuracy", "f1", "precision", "recall", "inference_ms"]]
            .sort_values("accuracy", ascending=False, na_position="last")
            .style.format({
                "accuracy": lambda v: fmt_pct(v) if v is not None else "n/a",
                "f1":       lambda v: fmt_pct(v) if v is not None else "n/a",
                "precision":lambda v: fmt_pct(v) if v is not None else "n/a",
                "recall":   lambda v: fmt_pct(v) if v is not None else "n/a",
                "inference_ms": lambda v: fmt_ms(v) if v is not None else "n/a",
            }),
            use_container_width=True,
        )

        st.divider()
        st.markdown("**Per-class F1 plots**")
        pcf1 = sorted([f for f in os.listdir(OUTPUT_DIR) if "per_class_f1" in f and f.endswith(".png")])
        if pcf1:
            pc_cols = st.columns(2)
            for i, fname in enumerate(pcf1):
                with pc_cols[i % 2]:
                    st.image(os.path.join(OUTPUT_DIR, fname), caption=fname, use_container_width=True)
        else:
            st.info("Train models to generate per-class F1 plots.")
    else:
        st.info("No model metrics available yet.")


# ── Tab: About ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("#### System summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Dataset**
- CWRU feature CSV — 2,300 samples, 10 fault classes
- 9 raw statistical features → 25 engineered features

**Pipeline**
- Leakage-free train/val/test split
- Outlier filtering, feature engineering, scaling
- Frozen artifact registry (no drift in dashboard)
        """)
    with col2:
        st.markdown("""
**Models**
- Baselines: RF, XGBoost, GBM, LightGBM, Stacking Ensemble
- Deep: Transformer, CNN, ResNet1D, LSTM
- Meta: MAML, Meta-SGD
- Continual: FBCL

**Runtime**
- Inference-only dashboard
- Model selected from frozen registry
        """)

    st.divider()
    st.markdown("**Target metrics**")
    st.dataframe(
        pd.DataFrame({
            "Metric":  ["Test Accuracy", "False Positive Rate", "Inference Time"],
            "Target":  ["≥ 95%", "≤ 5%", "≤ 150 ms"],
            "Status":  [
                "✅" if acc_ok else "⚠️",
                "n/a",
                "✅" if inf_ok else "⚠️",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )
