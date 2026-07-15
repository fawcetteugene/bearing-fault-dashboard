"""
Bearing Fault Diagnosis — Professional Dashboard v2
"""
from __future__ import annotations
import io, json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_root = os.path.dirname(_here)
# On Streamlit Cloud cwd is the repo root — prefer it
if os.path.isdir(os.path.join(os.getcwd(), "src")):
    _root = os.getcwd()
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.artifacts import build_artifact_registry, load_artifact_registry, load_preprocessing_manifest
from src.config import CLASS_NAMES, ORIGINAL_FEATURES, OUTPUT_DIR
from src.inference import ProductionPredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BearingIQ — Fault Diagnosis",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #060d1a;
    background-image: radial-gradient(ellipse at 20% 0%, rgba(14,165,233,0.06) 0%, transparent 60%),
                      radial-gradient(ellipse at 80% 100%, rgba(16,185,129,0.05) 0%, transparent 60%);
}
[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid #0f2040;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1400px; }
h1,h2,h3,h4 { color: #f1f5f9 !important; }
p, li, label, .stMarkdown { color: #cbd5e1 !important; }

/* ── Sidebar brand ── */
.sb-brand {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.8rem 0 1rem;
    border-bottom: 1px solid #0f2040;
    margin-bottom: 1rem;
}
.sb-brand-icon { font-size: 1.6rem; }
.sb-brand-name { font-size: 1.1rem; font-weight: 800; color: #f1f5f9 !important; }
.sb-brand-sub  { font-size: 0.7rem; color: #475569 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0c1f3f 0%, #0a3d2e 50%, #0c1f3f 100%);
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.hero::before {
    content: "";
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-size: 2rem; font-weight: 900; color: #f1f5f9 !important; margin: 0; letter-spacing: -0.02em; }
.hero-sub   { color: rgba(203,213,225,0.75) !important; font-size: 0.9rem; margin: 0.4rem 0 0; }
.hero-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.9rem; }
.hero-pill  {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 0.2rem 0.7rem;
    font-size: 0.72rem; color: rgba(203,213,225,0.8) !important;
}

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
.kpi {
    background: #0a1628;
    border: 1px solid #0f2040;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi:hover { border-color: #1e3a5f; }
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 16px 16px 0 0; }
.kpi-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: #475569 !important; }
.kpi-value { font-size: 1.8rem; font-weight: 900; line-height: 1.1; margin: 0.3rem 0 0.1rem; }
.kpi-sub   { font-size: 0.72rem; color: #475569 !important; }
.c-green  { color: #34d399 !important; }
.c-amber  { color: #fbbf24 !important; }
.c-blue   { color: #38bdf8 !important; }
.c-purple { color: #a78bfa !important; }
.c-slate  { color: #64748b !important; }
.c-red    { color: #f87171 !important; }
.bg-green  { background: #34d399; }
.bg-amber  { background: #fbbf24; }
.bg-blue   { background: #38bdf8; }
.bg-purple { background: #a78bfa; }

/* ── Section card ── */
.scard {
    background: #0a1628;
    border: 1px solid #0f2040;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.scard-title { font-size: 0.85rem; font-weight: 700; color: #94a3b8 !important;
               text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.8rem; }

/* ── Prediction result ── */
.pred-wrap {
    background: #0a1628;
    border: 1px solid #0f2040;
    border-radius: 18px;
    padding: 1.4rem;
    margin-top: 0.4rem;
}
.pred-class { font-size: 1.6rem; font-weight: 900; letter-spacing: -0.02em; }
.pred-meta  { font-size: 0.82rem; color: #64748b !important; margin-top: 0.4rem; }

/* ── Status badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.7rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
}
.badge-ok     { background: #052e16; color: #34d399 !important; border: 1px solid #166534; }
.badge-warn   { background: #451a03; color: #fcd34d !important; border: 1px solid #92400e; }
.badge-danger { background: #450a0a; color: #fca5a5 !important; border: 1px solid #991b1b; }
.badge-info   { background: #0c1f3f; color: #7dd3fc !important; border: 1px solid #1e3a5f; }

/* ── Condition report ── */
.cond-card {
    background: #0a1628;
    border: 1px solid #0f2040;
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    margin-top: 1rem;
}
.cond-title { font-size: 0.78rem; font-weight: 700; color: #475569 !important;
              text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.9rem; }
.cond-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.cond-desc  { font-size: 0.9rem; color: #cbd5e1 !important; line-height: 1.65; margin-bottom: 1rem; }
.action-box {
    border-radius: 12px; padding: 0.8rem 1rem;
}
.action-title { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.action-text  { font-size: 0.88rem; color: #f1f5f9 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #475569 !important; font-weight: 600; font-size: 0.85rem;
    padding: 0.5rem 1rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input {
    background: #0a1628 !important;
    border: 1px solid #0f2040 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #10b981) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important;
    transform: translateY(-1px);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #0f2040 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Sidebar health indicator ── */
.health-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.35rem 0; border-bottom: 1px solid #0f2040;
    font-size: 0.8rem;
}
.health-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 0.4rem;
}
.dot-green  { background: #34d399; box-shadow: 0 0 6px #34d399; }
.dot-amber  { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }
.dot-red    { background: #f87171; box-shadow: 0 0 6px #f87171; }

/* ── Metric comparison table ── */
.metric-table { width: 100%; border-collapse: collapse; }
.metric-table th {
    background: #0f2040; color: #64748b !important;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.5rem 0.8rem; text-align: left;
}
.metric-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #0f2040; font-size: 0.85rem; }
.metric-table tr:hover td { background: #0f2040; }
</style>
""", unsafe_allow_html=True)

# ── Bearing knowledge base ────────────────────────────────────────────────────
BEARING_INFO: dict[str, dict] = {
    "Normal_1": {
        "status": "Healthy", "urgency": "No action required",
        "urgency_color": "ok", "severity": "None", "severity_color": "ok",
        "location": "No fault detected",
        "what_it_means": "The bearing is operating normally. Vibration levels are within expected limits. No damage has been detected.",
        "recommended_action": "Continue routine scheduled maintenance.",
        "icon": "✅",
    },
    "Ball_007_1": {
        "status": "Ball Fault — Early", "urgency": "Monitor closely",
        "urgency_color": "warn", "severity": "Minor (0.007\" pit)", "severity_color": "warn",
        "location": "Rolling element (ball)",
        "what_it_means": "A small pit has formed on one of the rolling balls. At this early stage the bearing is still serviceable, but damage will worsen if left unattended.",
        "recommended_action": "Schedule inspection within 2–4 weeks. Increase vibration monitoring frequency.",
        "icon": "⚠️",
    },
    "Ball_014_1": {
        "status": "Ball Fault — Moderate", "urgency": "Plan replacement soon",
        "urgency_color": "warn", "severity": "Moderate (0.014\" pit)", "severity_color": "warn",
        "location": "Rolling element (ball)",
        "what_it_means": "Ball damage has grown to a moderate size. Vibration and noise are noticeably higher. Risk of accelerated wear is increasing.",
        "recommended_action": "Replace bearing within 1–2 weeks. Avoid high-load operation.",
        "icon": "🔶",
    },
    "Ball_021_1": {
        "status": "Ball Fault — Severe", "urgency": "Replace immediately",
        "urgency_color": "danger", "severity": "Severe (0.021\" pit)", "severity_color": "danger",
        "location": "Rolling element (ball)",
        "what_it_means": "The ball has significant damage. Continued operation risks catastrophic bearing failure and damage to surrounding machinery.",
        "recommended_action": "Stop machine and replace bearing as soon as possible.",
        "icon": "🔴",
    },
    "IR_007_1": {
        "status": "Inner Race — Early", "urgency": "Monitor closely",
        "urgency_color": "warn", "severity": "Minor (0.007\" pit)", "severity_color": "warn",
        "location": "Inner race (rotates with shaft)",
        "what_it_means": "A small defect on the inner ring. Each ball rolling over it produces a small impact. The bearing is still serviceable at this stage.",
        "recommended_action": "Schedule inspection within 2–4 weeks. Log vibration trend.",
        "icon": "⚠️",
    },
    "IR_014_1": {
        "status": "Inner Race — Moderate", "urgency": "Plan replacement soon",
        "urgency_color": "warn", "severity": "Moderate (0.014\" pit)", "severity_color": "warn",
        "location": "Inner race (rotates with shaft)",
        "what_it_means": "The inner race defect has grown. Impacts from rolling elements are stronger. Heat generation and wear rate are elevated.",
        "recommended_action": "Replace bearing within 1–2 weeks. Check shaft alignment.",
        "icon": "🔶",
    },
    "IR_021_1": {
        "status": "Inner Race — Severe", "urgency": "Replace immediately",
        "urgency_color": "danger", "severity": "Severe (0.021\" pit)", "severity_color": "danger",
        "location": "Inner race (rotates with shaft)",
        "what_it_means": "Severe inner race damage. High risk of sudden failure. Continued operation may cause shaft damage and secondary failures.",
        "recommended_action": "Stop machine and replace bearing immediately. Inspect shaft for scoring.",
        "icon": "🔴",
    },
    "OR_007_6_1": {
        "status": "Outer Race — Early", "urgency": "Monitor closely",
        "urgency_color": "warn", "severity": "Minor (0.007\" pit)", "severity_color": "warn",
        "location": "Outer race (stationary ring)",
        "what_it_means": "A small defect on the stationary outer ring. Outer race faults produce a characteristic repetitive impact pattern. Early detection is good.",
        "recommended_action": "Schedule inspection within 2–4 weeks. Check lubrication.",
        "icon": "⚠️",
    },
    "OR_014_6_1": {
        "status": "Outer Race — Moderate", "urgency": "Plan replacement soon",
        "urgency_color": "warn", "severity": "Moderate (0.014\" pit)", "severity_color": "warn",
        "location": "Outer race (stationary ring)",
        "what_it_means": "The outer race defect has grown. Vibration amplitude is noticeably higher. The bearing housing may be experiencing increased stress.",
        "recommended_action": "Replace bearing within 1–2 weeks. Inspect housing bore for wear.",
        "icon": "🔶",
    },
    "OR_021_6_1": {
        "status": "Outer Race — Severe", "urgency": "Replace immediately",
        "urgency_color": "danger", "severity": "Severe (0.021\" pit)", "severity_color": "danger",
        "location": "Outer race (stationary ring)",
        "what_it_means": "Severe outer race damage. The bearing is near end-of-life. Risk of sudden seizure or fragmentation is high.",
        "recommended_action": "Stop machine and replace bearing immediately. Inspect housing and lubrication system.",
        "icon": "🔴",
    },
}

_BADGE_MAP = {
    "ok":     "badge-ok",
    "warn":   "badge-warn",
    "danger": "badge-danger",
    "info":   "badge-info",
}
_ACTION_BG = {
    "ok":     ("background:#052e16;border:1px solid #166534", "#34d399"),
    "warn":   ("background:#451a03;border:1px solid #92400e", "#fcd34d"),
    "danger": ("background:#450a0a;border:1px solid #991b1b", "#fca5a5"),
}

# ── Helper functions ──────────────────────────────────────────────────────────
def fmt_pct(v):
    return f"{float(v)*100:.2f}%" if v is not None else "—"

def fmt_ms(v):
    return f"{float(v):.1f} ms" if v is not None else "—"

def kpi_color(v, target=0.95):
    if v is None: return "c-slate", "bg-blue"
    return ("c-green", "bg-green") if float(v) >= target else ("c-amber", "bg-amber")

def render_condition(label_name: str, confidence: float) -> None:
    info = BEARING_INFO.get(label_name)
    if not info:
        return
    uc = info["urgency_color"]
    sc = info["severity_color"]
    ab, ac = _ACTION_BG.get(uc, _ACTION_BG["warn"])
    low_conf = (
        f"<p style='color:#fbbf24;font-size:0.8rem;margin:0.5rem 0 0'>"
        f"⚠️ Confidence is {confidence*100:.1f}% — verify sensor readings before acting.</p>"
        if confidence < 0.80 else ""
    )
    st.markdown(f"""
    <div class="cond-card">
      <div class="cond-title">🔎 Bearing Condition Report</div>
      <div class="cond-pills">
        <span class="badge {_BADGE_MAP[uc]}">{info['icon']} {info['urgency']}</span>
        <span class="badge badge-info">📍 {info['location']}</span>
        <span class="badge {_BADGE_MAP[sc]}">📏 {info['severity']}</span>
      </div>
      <div class="cond-desc">{info['what_it_means']}</div>
      <div class="action-box" style="{ab};border-radius:12px;padding:0.8rem 1rem">
        <div class="action-title" style="color:{ac}"">Recommended Action</div>
        <div class="action-text">{info['recommended_action']}</div>
        {low_conf}
      </div>
    </div>
    """, unsafe_allow_html=True)

def plot_probabilities(proba: np.ndarray, class_names: list[str]):
    order = np.argsort(proba)[::-1][:6]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0a1628")
    colors = ["#10b981" if i == order[0] else "#1e3a5f" for i in order]
    bars = ax.barh([class_names[i] for i in order[::-1]], proba[order[::-1]],
                   color=colors[::-1], height=0.55, edgecolor="none")
    for bar, val in zip(bars, proba[order[::-1]]):
        ax.text(min(val + 0.012, 0.97), bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", color="#94a3b8", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", color="#475569", fontsize=9)
    ax.set_title("Class Probability Distribution", color="#f1f5f9", fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#0f2040")
    ax.grid(axis="x", color="#0f2040", linewidth=0.7)
    plt.tight_layout()
    return fig

def plot_comparison(rows_df: pd.DataFrame):
    plot_df = rows_df.dropna(subset=["Accuracy"]).sort_values("Accuracy", ascending=False)
    if plot_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(plot_df) * 0.55)))
    fig.patch.set_facecolor("#0a1628")
    for ax in axes:
        ax.set_facecolor("#0a1628")
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#0f2040")
        ax.grid(axis="x", color="#0f2040", linewidth=0.7)

    colors = ["#10b981" if "Production" in str(r) else "#0ea5e9" for r in plot_df["Role"]]
    axes[0].barh(plot_df["Model"][::-1], plot_df["Accuracy"][::-1],
                 color=colors[::-1], height=0.55, edgecolor="none")
    axes[0].axvline(0.95, color="#fbbf24", linestyle="--", linewidth=1.2, label="95% target")
    axes[0].set_xlim(0, 1.1)
    axes[0].set_title("Test Accuracy", color="#f1f5f9", fontweight="bold")
    axes[0].set_xlabel("Accuracy", color="#475569", fontsize=9)
    axes[0].legend(facecolor="#0a1628", labelcolor="#fbbf24", edgecolor="#0f2040", fontsize=8)

    f1_df = plot_df.dropna(subset=["F1"])
    if not f1_df.empty:
        axes[1].barh(f1_df["Model"][::-1], f1_df["F1"][::-1],
                     color=colors[:len(f1_df)][::-1], height=0.55, edgecolor="none")
        axes[1].axvline(0.95, color="#fbbf24", linestyle="--", linewidth=1.2)
        axes[1].set_xlim(0, 1.1)
        axes[1].set_title("Weighted F1 Score", color="#f1f5f9", fontweight="bold")
        axes[1].set_xlabel("F1", color="#475569", fontsize=9)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    return fig

def plot_per_class_f1(metrics: dict):
    pcf1 = metrics.get("per_class_f1", {})
    if not pcf1:
        return None
    names = list(pcf1.keys())
    vals  = [pcf1[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0a1628")
    colors = ["#10b981" if v >= 0.95 else "#fbbf24" if v >= 0.85 else "#f87171" for v in vals]
    ax.bar(names, vals, color=colors, edgecolor="none", width=0.6)
    ax.axhline(0.95, color="#fbbf24", linestyle="--", linewidth=1.2, label="95% target")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score", color="#f1f5f9", fontweight="bold")
    ax.set_ylabel("F1", color="#475569", fontsize=9)
    ax.tick_params(axis="x", rotation=35, colors="#64748b", labelsize=8)
    ax.tick_params(axis="y", colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#0f2040")
    ax.grid(axis="y", color="#0f2040", linewidth=0.7)
    ax.legend(facecolor="#0a1628", labelcolor="#fbbf24", edgecolor="#0f2040", fontsize=8)
    plt.tight_layout()
    return fig

def plot_local_shap(predictor, row_dict):
    try:
        import shap
    except Exception:
        return None, "Install `shap` to enable local explanations."
    if predictor.is_torch:
        return None, "Local SHAP is available for tree-based models only."
    X_scaled = predictor.transform_row(row_dict)
    try:
        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer.shap_values(X_scaled)
        pred_idx = int(np.argmax(predictor.model.predict_proba(X_scaled)[0]))
        values = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0]
        order = np.argsort(np.abs(values))[::-1][:10]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        fig.patch.set_facecolor("#0a1628")
        ax.set_facecolor("#0a1628")
        colors = ["#f87171" if values[i] > 0 else "#38bdf8" for i in order]
        ax.barh([predictor.feature_names[i] for i in order[::-1]], values[order[::-1]],
                color=colors[::-1], height=0.55, edgecolor="none")
        ax.axvline(0, color="#475569", linewidth=0.8)
        ax.set_title("Local SHAP Feature Importance", color="#f1f5f9", fontweight="bold")
        ax.set_xlabel("SHAP value (red = pushes toward prediction, blue = against)", color="#475569", fontsize=8)
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#0f2040")
        ax.grid(axis="x", color="#0f2040", linewidth=0.7)
        plt.tight_layout()
        return fig, None
    except Exception as exc:
        return None, f"SHAP could not be computed: {exc}"

def model_rows(registry) -> pd.DataFrame:
    prod_name = registry.get("production", {}).get("name")
    rows = []
    for c in registry.get("candidates", []):
        m = c.get("metrics", {})
        rows.append({
            "Model": c.get("name"),
            "Type": c.get("kind"),
            "Role": "🟢 Production" if c.get("name") == prod_name else "Archived",
            "Accuracy": m.get("accuracy", m.get("best_val_acc")),
            "F1": m.get("f1", m.get("average_accuracy")),
            "Precision": m.get("precision"),
            "Recall": m.get("recall"),
            "Inference (ms)": m.get("inference_ms"),
            "SHA256": (c.get("sha256") or "n/a")[:12] + "…",
        })
    return pd.DataFrame(rows)

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

# ── Load data ─────────────────────────────────────────────────────────────────
registry         = get_registry()
preproc_manifest = load_preprocessing_manifest()
prod_name        = registry.get("production", {}).get("name", "production")
prod_metrics     = registry.get("production", {}).get("metrics", {})

# =============================================================================
# 🔽 REMOVE BASELINE MODELS FROM UI
# -----------------------------------------------------------------------------
BASELINE_NAMES = {"rf", "xgb", "gbm", "lgbm", "stack"}   # models to hide

# Filter supported models to exclude baselines
all_supported = [c["name"] for c in registry.get("candidates", []) if c.get("supported_for_dashboard")]
supported_models = [m for m in all_supported if m not in BASELINE_NAMES]

# If no non-baseline models, show warning and use fallback (all models)
if not supported_models:
    st.warning("No non‑baseline models found. Showing all available models.")
    supported_models = all_supported

# We no longer include "Production" in the dropdown because it's a baseline (xgb).
# Instead, we list only non-baseline models, with the first one as default.
model_choices = sorted(supported_models)   # alphabetical order
# If there is a non-baseline production (unlikely, but we could set default to that)
# For now, just take the first in alphabetical order.
default_index = 0
# Optionally, if a non-baseline model is marked as production, set it as default.
prod_non_baseline = None
if prod_name not in BASELINE_NAMES:
    prod_non_baseline = prod_name
    # Move it to top (optional)
    if prod_non_baseline in model_choices:
        model_choices.remove(prod_non_baseline)
        model_choices.insert(0, prod_non_baseline)
        default_index = 0

# =============================================================================

feature_count    = len(registry.get("feature_order", [])) or len(ORIGINAL_FEATURES)

acc_val = prod_metrics.get("accuracy", prod_metrics.get("best_val_acc"))
f1_val  = prod_metrics.get("f1", prod_metrics.get("average_accuracy"))
inf_val = prod_metrics.get("inference_ms")
acc_ok  = acc_val is not None and float(acc_val) >= 0.95
inf_ok  = inf_val is None or float(inf_val) <= 150

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-icon">🔩</div>
      <div>
        <div class="sb-brand-name">BearingIQ</div>
        <div class="sb-brand-sub">Fault Diagnosis System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Active Model**")
    # Now model_choices contains only non-baseline models; no "Production" option
    selected_model = st.selectbox(
        "Active model",
        options=model_choices if model_choices else ["No models"],
        index=default_index if model_choices else 0,
        label_visibility="collapsed",
    )
    if model_choices and prod_non_baseline:
        st.caption(f"Recommended: `{prod_non_baseline}`")
    elif model_choices:
        st.caption(f"Available models: {len(model_choices)}")
    else:
        st.caption("No models available.")
    st.divider()

    st.markdown("**System Health**")
    dot_acc = "dot-green" if acc_ok else "dot-amber"
    dot_inf = "dot-green" if inf_ok else "dot-amber"
    st.markdown(f"""
    <div class="health-row">
      <span><span class="health-dot {dot_acc}"></span>Accuracy</span>
      <span style="color:{'#34d399' if acc_ok else '#fbbf24'}">{fmt_pct(acc_val)}</span>
    </div>
    <div class="health-row">
      <span><span class="health-dot {dot_inf}"></span>Inference</span>
      <span style="color:{'#34d399' if inf_ok else '#fbbf24'}">{fmt_ms(inf_val)}</span>
    </div>
    <div class="health-row">
      <span><span class="health-dot dot-green"></span>Features</span>
      <span style="color:#38bdf8">{feature_count}</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    with st.expander("Registry details", expanded=False):
        st.write(f"Candidates: **{len(registry.get('candidates', []))}**")
        if registry.get("production"):
            sha = registry["production"].get("sha256", "")
            st.write(f"SHA256: `{sha[:14]}…`")
        if preproc_manifest:
            aug   = preproc_manifest.get("augmentation", {})
            split = preproc_manifest.get("split_sizes_after_cleaning", {})
            st.write(f"Train rows: **{aug.get('final_train_rows', 'n/a')}**")
            st.write(f"Val rows: **{split.get('validation', 'n/a')}**")
            st.write(f"Test rows: **{split.get('test', 'n/a')}**")

    st.divider()
    st.caption("Inference-only · No retraining in dashboard")
    st.caption("CWRU Bearing Dataset · 10 fault classes")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🔩 Bearing Fault Diagnosis</div>
  <div class="hero-sub">Real-time predictive maintenance powered by machine learning — CWRU dataset</div>
  <div class="hero-pills">
    <span class="hero-pill">⚡ Inference-only</span>
    <span class="hero-pill">🔒 Frozen artifact registry</span>
    <span class="hero-pill">🧠 10 fault classes</span>
    <span class="hero-pill">📐 25 engineered features</span>
    <span class="hero-pill">🎯 Target ≥ 95% accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
acc_c, acc_bg = kpi_color(acc_val, 0.95)
f1_c,  f1_bg  = kpi_color(f1_val,  0.95)
inf_c  = "c-green" if inf_ok else "c-amber"
inf_bg = "bg-green" if inf_ok else "bg-amber"

k1, k2, k3, k4 = st.columns(4)
for col, label, value, vc, bg, sub in [
    (k1, "Test Accuracy",  fmt_pct(acc_val), acc_c, acc_bg, "Target ≥ 95%"),
    (k2, "Weighted F1",    fmt_pct(f1_val),  f1_c,  f1_bg,  "Macro average"),
    (k3, "Inference Time", fmt_ms(inf_val),  inf_c, inf_bg, "Target ≤ 150 ms"),
    (k4, "Feature Dims",   str(feature_count), "c-purple", "bg-purple", "Engineered"),
]:
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-accent {bg}"></div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {vc}">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.write("")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_compare, tab_explain, tab_registry, tab_about = st.tabs([
    "🔍 Diagnose", "📂 Batch Score", "📊 Model Comparison",
    "💡 Explain", "🗂 Registry", "ℹ️ About",
])

# ── Tab: Diagnose ─────────────────────────────────────────────────────────────
with tab_predict:
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown('<div class="scard"><div class="scard-title">Raw Sensor Features</div>', unsafe_allow_html=True)
        st.caption("Enter the 9 statistical features. Engineered features are derived automatically.")

        # Feature input grid
        input_values: dict[str, float] = {}
        FEATURE_HINTS = {
            "max": "Maximum amplitude",
            "min": "Minimum amplitude",
            "mean": "Signal mean",
            "sd": "Standard deviation",
            "rms": "Root mean square",
            "skewness": "Distribution skewness",
            "kurtosis": "Distribution kurtosis",
            "crest": "Crest factor",
            "form": "Form factor",
        }
        fc = st.columns(3)
        for idx, feat in enumerate(ORIGINAL_FEATURES):
            with fc[idx % 3]:
                input_values[feat] = st.number_input(
                    feat,
                    value=0.0,
                    format="%.6f",
                    key=f"feat_{feat}",
                    help=FEATURE_HINTS.get(feat, feat),
                )
        st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("▶ Run Diagnosis", type="primary", width="stretch")

    with right:
        st.markdown('<div class="scard"><div class="scard-title">Diagnosis Result</div>', unsafe_allow_html=True)
        if run_btn:
            # Always pass selected_model (no None fallback)
            predictor = get_predictor(selected_model)
            with st.spinner("Running inference…"):
                result = predictor.predict_row(input_values)

            conf_pct = result.confidence * 100
            badge_cls = "badge-ok" if result.confidence >= 0.95 else "badge-warn" if result.confidence >= 0.70 else "badge-danger"
            conf_label = "HIGH CONFIDENCE" if result.confidence >= 0.95 else "MODERATE CONFIDENCE" if result.confidence >= 0.70 else "LOW CONFIDENCE"
            info = BEARING_INFO.get(result.label_name, {})
            icon = info.get("icon", "🔩")

            st.markdown(f"""
            <div class="pred-wrap">
              <span class="badge {badge_cls}">{conf_label}</span>
              <div class="pred-class c-green" style="margin-top:0.7rem">{icon} {result.label_name}</div>
              <div class="pred-meta">
                Confidence: <b style="color:#f1f5f9">{conf_pct:.2f}%</b> &nbsp;·&nbsp;
                Latency: <b style="color:#f1f5f9">{result.inference_ms:.2f} ms</b> &nbsp;·&nbsp;
                Model: <b style="color:#f1f5f9">{predictor.model_name}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            render_condition(result.label_name, result.confidence)

            st.write("")
            fig = plot_probabilities(result.probabilities, predictor.class_names)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            with st.expander("Full probability table"):
                st.dataframe(
                    pd.DataFrame({"Class": predictor.class_names, "Probability": result.probabilities})
                    .sort_values("Probability", ascending=False)
                    .style.format({"Probability": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

            st.download_button(
                "⬇ Download result (JSON)",
                data=json.dumps({
                    "model": predictor.model_name,
                    "prediction": result.label_name,
                    "confidence": result.confidence,
                    "inference_ms": result.inference_ms,
                    "probabilities": dict(zip(predictor.class_names, result.probabilities.tolist())),
                    "input_features": input_values,
                }, indent=2).encode(),
                file_name="bearing_diagnosis.json",
                mime="application/json",
                width="stretch",
            )
        else:
            st.info("Enter feature values (or use a preset) and click **Run Diagnosis**.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── Tab: Batch Score ──────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="scard"><div class="scard-title">Batch Scoring</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with the 9 raw feature columns to score multiple samples at once.")

    # Template download
    template_df = pd.DataFrame([{f: 0.0 for f in ORIGINAL_FEATURES}])
    st.download_button(
        "⬇ Download CSV template",
        data=template_df.to_csv(index=False).encode(),
        file_name="bearing_template.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
    df_raw: pd.DataFrame | None = None

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(df_raw))
        c2.metric("Columns", len(df_raw.columns))
        missing = [f for f in ORIGINAL_FEATURES if f not in df_raw.columns]
        c3.metric("Missing features", len(missing))
        c4.metric("File size", f"{uploaded.size / 1024:.1f} KB")

        if missing:
            st.warning(f"Missing required columns: {missing}")
        else:
            st.dataframe(df_raw.head(10), width="stretch")

            if st.button("▶ Run Batch Scoring", type="primary", width="stretch"):
                predictor = get_predictor(selected_model)   # pass selected model
                with st.spinner(f"Scoring {len(df_raw)} rows…"):
                    t0 = time.perf_counter()
                    try:
                        result_df = predictor.predict_frame(df_raw)
                        elapsed = (time.perf_counter() - t0) * 1000
                        st.success(f"✅ Scored **{len(result_df)}** rows in **{elapsed:.1f} ms** ({elapsed/len(result_df):.2f} ms/row)")

                        col_a, col_b = st.columns([1.3, 1])
                        with col_a:
                            st.markdown("**Predictions (first 50)**")
                            st.dataframe(
                                result_df[["predicted_class", "confidence"]].head(50)
                                .style.format({"confidence": "{:.4f}"}),
                                width="stretch",
                            )
                        with col_b:
                            st.markdown("**Class distribution**")
                            counts = result_df["predicted_class"].value_counts().sort_values(ascending=False)
                            fig, ax = plt.subplots(figsize=(5.5, 4))
                            fig.patch.set_facecolor("#0a1628")
                            ax.set_facecolor("#0a1628")
                            ax.bar(counts.index, counts.values, color="#0ea5e9", edgecolor="none", width=0.6)
                            ax.set_title("Predicted Class Distribution", color="#f1f5f9", fontweight="bold")
                            ax.tick_params(axis="x", rotation=40, colors="#64748b", labelsize=8)
                            ax.tick_params(axis="y", colors="#64748b")
                            for spine in ax.spines.values():
                                spine.set_color("#0f2040")
                            ax.grid(axis="y", color="#0f2040", linewidth=0.7)
                            plt.tight_layout()
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)

                        # Confidence histogram
                        st.markdown("**Confidence distribution**")
                        fig2, ax2 = plt.subplots(figsize=(10, 2.8))
                        fig2.patch.set_facecolor("#0a1628")
                        ax2.set_facecolor("#0a1628")
                        ax2.hist(result_df["confidence"], bins=30, color="#10b981", edgecolor="none", alpha=0.85)
                        ax2.axvline(0.95, color="#fbbf24", linestyle="--", linewidth=1.2, label="95% threshold")
                        ax2.set_xlabel("Confidence", color="#475569", fontsize=9)
                        ax2.set_ylabel("Count", color="#475569", fontsize=9)
                        ax2.set_title("Prediction Confidence Histogram", color="#f1f5f9", fontweight="bold")
                        ax2.tick_params(colors="#64748b", labelsize=9)
                        for spine in ax2.spines.values():
                            spine.set_color("#0f2040")
                        ax2.grid(color="#0f2040", linewidth=0.7)
                        ax2.legend(facecolor="#0a1628", labelcolor="#fbbf24", edgecolor="#0f2040", fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig2, width="stretch")
                        plt.close(fig2)

                        st.download_button(
                            "⬇ Download scored CSV",
                            data=result_df.to_csv(index=False).encode(),
                            file_name="bearing_batch_predictions.csv",
                            mime="text/csv",
                            width="stretch",
                        )
                    except Exception as exc:
                        st.error(str(exc))
    else:
        st.info("Upload a CSV file to begin batch scoring.")

# ── Tab: Model Comparison ─────────────────────────────────────────────────────
with tab_compare:
    rows_df = model_rows(registry)
    # Filter out baseline models from comparison chart and table
    rows_df = rows_df[~rows_df["Model"].isin(BASELINE_NAMES)]

    if not rows_df.empty:
        # Summary bar charts
        fig = plot_comparison(rows_df)
        if fig:
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.write("")
        st.markdown('<div class="scard"><div class="scard-title">All Models (Non‑Baseline)</div>', unsafe_allow_html=True)
        st.dataframe(
            rows_df.sort_values("Accuracy", ascending=False, na_position="last")
            .style.format({
                "Accuracy":      lambda v: fmt_pct(v) if v is not None else "—",
                "F1":            lambda v: fmt_pct(v) if v is not None else "—",
                "Precision":     lambda v: fmt_pct(v) if v is not None else "—",
                "Recall":        lambda v: fmt_pct(v) if v is not None else "—",
                "Inference (ms)":lambda v: fmt_ms(v)  if v is not None else "—",
            }),
            width="stretch",
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Per-class F1 ──────────────────────────────────────────────────────
        st.write("")
        st.markdown('<div class="scard"><div class="scard-title">Per-Class F1 — Best Model</div>', unsafe_allow_html=True)

        pcf1_fig = None

        # 1. If production is non-baseline and has per_class_f1, use it
        if prod_name not in BASELINE_NAMES and prod_metrics.get("per_class_f1"):
            pcf1_fig = plot_per_class_f1(prod_metrics)
        else:
            # 2. Otherwise, find the best non-baseline model by accuracy (only if at least one valid accuracy exists)
            acc_series = rows_df["Accuracy"].dropna()
            if not acc_series.empty:
                best_idx = acc_series.idxmax()
                best_non_baseline = rows_df.loc[best_idx]
                best_name = best_non_baseline["Model"]
                # Look up the candidate record to get its full metrics
                for c in registry.get("candidates", []):
                    if c["name"] == best_name:
                        pcf1_fig = plot_per_class_f1(c.get("metrics", {}))
                        break

        if pcf1_fig:
            st.pyplot(pcf1_fig, width="stretch")
            plt.close(pcf1_fig)
        else:
            st.info("No per‑class F1 data available for the selected model(s).")

        st.markdown('</div>', unsafe_allow_html=True)

        # Saved per-class plots from outputs/
        saved_plots = sorted([f for f in os.listdir(OUTPUT_DIR) if "per_class_f1" in f and f.endswith(".png")])
        if saved_plots:
            st.markdown('<div class="scard"><div class="scard-title">Saved Per-Class F1 Plots</div>', unsafe_allow_html=True)
            pc_cols = st.columns(2)
            for i, fname in enumerate(saved_plots[:6]):
                with pc_cols[i % 2]:
                    st.image(os.path.join(OUTPUT_DIR, fname), caption=fname, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No non‑baseline model metrics available. Train deep learning models first.")

# ── Tab: Explain ──────────────────────────────────────────────────────────────
with tab_explain:
    st.markdown('<div class="scard"><div class="scard-title">Local SHAP Explanation</div>', unsafe_allow_html=True)
    st.caption("Tree-based SHAP for RF / XGBoost / GBM models. Shows which features drove the prediction.")

    sample_source = st.radio(
        "Sample source",
        ["Manual input (from Diagnose tab)", "Batch first row"],
        horizontal=True,
    )
    sample_row = {f: st.session_state.get(f"feat_{f}", 0.0) for f in ORIGINAL_FEATURES}
    if sample_source == "Batch first row" and df_raw is not None and not df_raw.empty:
        sample_row = df_raw.iloc[0][ORIGINAL_FEATURES].to_dict()

    if st.button("💡 Generate SHAP Explanation", type="primary", width="stretch"):
        predictor = get_predictor(selected_model)   # pass selected model
        with st.spinner("Computing SHAP values…"):
            fig, msg = plot_local_shap(predictor, sample_row)
        if fig:
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("Red bars push the prediction toward the predicted class; blue bars push against it.")
        else:
            st.info(msg)

        try:
            r = predictor.predict_row(sample_row)
            st.markdown("**Top-5 class probabilities for this sample**")
            st.dataframe(
                pd.DataFrame({"Class": predictor.class_names, "Probability": r.probabilities})
                .sort_values("Probability", ascending=False).head(5)
                .style.format({"Probability": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.warning(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)

    # Saved interpretability outputs
    saved_interp = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if any(k in f for k in ("shap", "attention", "lime")) and f.endswith(".png")
    ])
    if saved_interp:
        st.write("")
        st.markdown('<div class="scard"><div class="scard-title">Saved Interpretability Plots</div>', unsafe_allow_html=True)
        img_cols = st.columns(2)
        for i, fname in enumerate(saved_interp[:12]):
            with img_cols[i % 2]:
                st.image(os.path.join(OUTPUT_DIR, fname), caption=fname, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# ── Tab: Registry ─────────────────────────────────────────────────────────────
with tab_registry:
    st.markdown('<div class="scard"><div class="scard-title">Artifact Registry</div>', unsafe_allow_html=True)
    st.caption("Read-only view of frozen model artifacts. No training occurs in the dashboard.")

    rows_df = model_rows(registry)
    if not rows_df.empty:
        st.dataframe(
            rows_df.style.format({
                "Accuracy":       lambda v: fmt_pct(v) if v is not None else "—",
                "F1":             lambda v: fmt_pct(v) if v is not None else "—",
                "Precision":      lambda v: fmt_pct(v) if v is not None else "—",
                "Recall":         lambda v: fmt_pct(v) if v is not None else "—",
                "Inference (ms)": lambda v: fmt_ms(v)  if v is not None else "—",
            }),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No registry entries found.")
    st.markdown('</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="scard"><div class="scard-title">Production Bundle</div>', unsafe_allow_html=True)
        st.json(registry.get("production", {}))
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="scard"><div class="scard-title">Preprocessing Manifest</div>', unsafe_allow_html=True)
        st.json(preproc_manifest if preproc_manifest else {"status": "missing"})
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh registry", width="stretch"):
        build_artifact_registry(force=True)
        st.cache_resource.clear()
        st.rerun()


# ── Tab: About ────────────────────────────────────────────────────────────────
with tab_about:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="scard"><div class="scard-title">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
- **Source:** CWRU Bearing Dataset
- **Samples:** 2,300 (balanced)
- **Classes:** 10 fault types
- **Raw features:** 9 statistical
- **Engineered features:** 25 total
- **Split:** 70% train / 15% val / 15% test
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="scard"><div class="scard-title">Models</div>', unsafe_allow_html=True)
        st.markdown("""
**Baselines**
- Random Forest, XGBoost
- GBM, LightGBM, Stacking

**Deep Learning**
- Transformer, 1D-CNN
- ResNet1D, LSTM

**Advanced**
- MAML, Meta-SGD (meta-learning)
- FBCL (continual learning)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="scard"><div class="scard-title">Target Metrics</div>', unsafe_allow_html=True)
        targets_df = pd.DataFrame({
            "Metric":  ["Test Accuracy", "False Positive Rate", "Inference Time"],
            "Target":  ["≥ 95%", "≤ 5%", "≤ 150 ms"],
            "Status":  [
                "✅ Met" if acc_ok else "⚠️ Below target",
                "—",
                "✅ Met" if inf_ok else "⚠️ Above target",
            ],
        })
        st.dataframe(targets_df, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="scard"><div class="scard-title">Fault Class Reference</div>', unsafe_allow_html=True)
    ref_rows = []
    for cls, info in BEARING_INFO.items():
        ref_rows.append({
            "Class": cls,
            "Status": info["status"],
            "Location": info["location"],
            "Severity": info["severity"],
            "Urgency": info["urgency"],
        })
    st.dataframe(pd.DataFrame(ref_rows), width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
