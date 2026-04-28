import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from dotenv import load_dotenv
load_dotenv()
from bias.model import train_model, train_mitigated_model, get_shap_values

from bias.model import train_model, train_mitigated_model, get_shap_values
from bias.metrics import compute_metrics, compute_intersectional_metrics
from explain.rag import rag_explain, generate_pdf_report


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BiasGlass — AI Fairness Auditor",
    layout="wide",
    page_icon="🔍"
)

st.markdown("""
<style>
    .bias-score-box {
        border-radius: 12px;
        padding: 20px 28px;
        text-align: center;
        margin-bottom: 16px;
    }
    .metric-card {
        background: #f8f9fb;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 24px;
        margin-bottom: 8px;
    }
    .stButton > button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #3730a3;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# CLEANING FUNCTION
# =========================
def clean_data(df, target_col):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()
    return df


# =========================
# HEADER
# =========================
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.title("🔍 BiasGlass")
    st.caption("AI Fairness Auditor — Detect, Understand, and Fix Bias in Machine Learning Models")

st.divider()


# =========================
# SIDEBAR CONFIG
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")
    scenario = st.selectbox(
        "Choose a scenario",
        ["Custom Upload", "Income Prediction", "Loan Approval", "Criminal Justice"],
    )
    st.markdown("---")
    st.markdown("**About BiasGlass**")
    st.caption(
        "BiasGlass audits ML models for demographic fairness using Demographic Parity, "
        "Equalized Odds, and the Disparate Impact Ratio (EEOC 80% rule). "
        "It also runs bias mitigation and generates compliance reports."
    )
    gemini_key = st.text_input("Gemini API Key", type="password",
                                value=os.environ.get("GEMINI_API_KEY", ""),
                                help="Required for AI explanation and PDF report")
    run_mitigation = st.checkbox("Run Bias Mitigation (Slower)", value=True)
    run_shap = st.checkbox("Run SHAP Feature Attribution", value=True)
    run_intersectional = st.checkbox("Run Intersectional Analysis", value=False)


# =========================
# LOAD DATA
# =========================
df = None
target_col = None
sensitive_col = None
extra_sensitive_cols = []

if scenario == "Custom Upload":
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        columns = df.columns.tolist()
        target_col = st.selectbox("Select Target Column", columns)
        sensitive_col = st.selectbox("Select Primary Sensitive Attribute", columns)
        if run_intersectional:
            extra_sensitive_cols = st.multiselect(
                "Select Additional Sensitive Columns (for intersectional analysis)",
                [c for c in columns if c not in [target_col, sensitive_col]]
            )

elif scenario == "Income Prediction":
    df = pd.read_csv("data/adult.csv")
    target_col = "income"
    sensitive_col = "sex" if "sex" in df.columns else "gender"
    if "gender" in df.columns and "sex" not in df.columns:
        df.rename(columns={"gender": "sex"}, inplace=True)
        sensitive_col = "sex"
    if run_intersectional:
        extra_sensitive_cols = ["race"] if "race" in df.columns else []

elif scenario == "Loan Approval":
    df = pd.read_csv("data/german_credit.csv")
    target_col = "Risk"
    sensitive_col = "Sex"

elif scenario == "Criminal Justice":
    df = pd.read_csv("data/compas.csv")
    df["risk_binary"] = df["ScoreText"].apply(
        lambda x: 1 if str(x).lower() in ["medium", "high"] else 0
    )
    target_col = "risk_binary"
    sensitive_col = "Ethnic_Code_Text"
    df = df[df["Ethnic_Code_Text"].isin(["African-American", "Caucasian"])]

if df is None:
    st.warning("Please upload a dataset or select a scenario.")
    st.stop()


# =========================
# CLEAN DATA
# =========================
df = clean_data(df, target_col)

# Target conversion
if scenario == "Income Prediction":
    df[target_col] = df[target_col].astype(str).str.replace(".", "", regex=False)
    df[target_col] = df[target_col].apply(lambda x: 1 if ">50K" in str(x) else 0)
elif scenario == "Loan Approval":
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df[target_col] = df[target_col].apply(lambda x: 1 if x in ["good", "1"] else 0)


# =========================
# DATA PREVIEW
# =========================
with st.expander("📄 Dataset Preview", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    st.dataframe(df.head(10), use_container_width=True)

st.divider()


# =========================
# RUN AUDIT
# =========================
if st.button("🚀 Run Fairness Audit", use_container_width=True):

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    # ---- TRAIN MODEL ----
    with st.spinner("Training model..."):
        y_true, y_pred, X_test, model, X_train, y_train = train_model(df, target_col)

    sensitive = df.loc[X_test.index, sensitive_col]
    if sensitive.dtype == "object":
        sensitive = sensitive.astype("category")

    sensitive_train = df.loc[X_train.index, sensitive_col]
    if sensitive_train.dtype == "object":
        sensitive_train = sensitive_train.astype("category")

    # ---- COMPUTE METRICS ----
    with st.spinner("Computing fairness metrics..."):
        metrics = compute_metrics(y_true, y_pred, sensitive)

    # ============================
    # BIAS RISK SCORE — HERO CARD
    # ============================
    score = metrics["bias_risk_score"]
    label = metrics["bias_risk_label"]
    color_map = {"Low": "#16a34a", "Moderate": "#d97706", "High": "#dc2626"}
    bg_map = {"Low": "#f0fdf4", "Moderate": "#fffbeb", "High": "#fef2f2"}
    score_color = color_map.get(label, "#000")
    score_bg = bg_map.get(label, "#fff")

    st.markdown(f"""
    <div class="bias-score-box" style="background:{score_bg}; border: 2px solid {score_color};">
        <div style="font-size:3rem; font-weight:800; color:{score_color};">{score}<span style="font-size:1.5rem">/100</span></div>
        <div style="font-size:1.3rem; font-weight:600; color:{score_color};">Bias Risk: {label}</div>
        <div style="font-size:0.9rem; color:#555; margin-top:6px;">
            {"✅ Model appears broadly fair. Monitor for edge cases." if label == "Low" else
             "⚠️ Moderate bias detected. Review before deployment." if label == "Moderate" else
             "🚨 High bias detected. Do not deploy without mitigation."}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================
    # METRICS COLUMNS
    # ============================
    st.subheader("📊 Fairness Metrics")
    m1, m2, m3, m4 = st.columns(4)

    def metric_delta_color(val, threshold, lower_better=True):
        if lower_better:
            return "normal" if abs(val) < threshold else "inverse"
        return "normal" if val > threshold else "inverse"

    m1.metric(
        "Demographic Parity Diff",
        f"{metrics['demographic_parity_difference']:.4f}",
        delta="Threshold: ±0.1",
        delta_color=metric_delta_color(metrics["demographic_parity_difference"], 0.1)
    )
    m2.metric(
        "Equalized Odds Diff",
        f"{metrics['equalized_odds_difference']:.4f}",
        delta="Threshold: ±0.1",
        delta_color=metric_delta_color(metrics["equalized_odds_difference"], 0.1)
    )
    m3.metric(
        "Disparate Impact Ratio",
        f"{metrics['disparate_impact_ratio']:.4f}",
        delta="Legal min: 0.8",
        delta_color="normal" if metrics["disparate_impact_ratio"] >= 0.8 else "inverse"
    )
    m4.metric(
        "Overall Accuracy",
        f"{metrics['overall_accuracy']:.4f}"
    )

    if metrics["disparate_impact_ratio"] < 0.8:
        st.error(
            f"⚖️ **Legal Alert**: Disparate Impact Ratio = {metrics['disparate_impact_ratio']:.3f} "
            f"is below the EEOC 80% rule threshold (0.8). "
            f"This model may expose your organization to legal liability."
        )

    # ============================
    # GROUP SELECTION RATES CHART
    # ============================
    st.subheader("👥 Group-wise Selection Rates")
    sel_rates = metrics["group_selection_rates"]["selection_rate"]
    fpr_rates = metrics["group_selection_rates"].get("false_positive_rate", {})
    fnr_rates = metrics["group_selection_rates"].get("false_negative_rate", {})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bar_color = "#4f46e5"
    highlight_color = "#dc2626"

    def plot_group_bar(ax, data, title, ylabel):
        groups_list = list(data.keys())
        values = list(data.values())
        colors_list = [highlight_color if v == min(values) else bar_color for v in values]
        bars = ax.bar(groups_list, values, color=colors_list, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, min(max(values) * 1.3, 1.0))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)

        red_patch = mpatches.Patch(color=highlight_color, label="Lowest (most disadvantaged)")
        ax.legend(handles=[red_patch], fontsize=8)

    plot_group_bar(axes[0], sel_rates, "Selection Rate by Group", "Rate")
    if fpr_rates:
        plot_group_bar(axes[1], fpr_rates, "False Positive Rate by Group", "Rate")
    else:
        axes[1].set_visible(False)
    if fnr_rates:
        plot_group_bar(axes[2], fnr_rates, "False Negative Rate by Group", "Rate")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # ============================
    # MITIGATION
    # ============================
    mitigated_metrics = None
    if run_mitigation:
        with st.spinner("Running bias mitigation (ExponentiatedGradient)..."):
            try:
                y_pred_mitigated = train_mitigated_model(
                    X_train, y_train, sensitive_train, X_test, sensitive
                )
                mitigated_metrics = compute_metrics(y_true, y_pred_mitigated, sensitive)

                st.subheader("⚖️ Before vs After Mitigation")

                before_col, after_col = st.columns(2)
                with before_col:
                    st.markdown("**Before Mitigation**")
                    st.metric("Bias Risk Score", f"{metrics['bias_risk_score']}/100", label_visibility="visible")
                    st.metric("Demographic Parity Diff", f"{metrics['demographic_parity_difference']:.4f}")
                    st.metric("Equalized Odds Diff", f"{metrics['equalized_odds_difference']:.4f}")
                    st.metric("Disparate Impact Ratio", f"{metrics['disparate_impact_ratio']:.4f}")

                with after_col:
                    st.markdown("**After Mitigation**")
                    delta_score = mitigated_metrics["bias_risk_score"] - metrics["bias_risk_score"]
                    st.metric("Bias Risk Score", f"{mitigated_metrics['bias_risk_score']}/100",
                              delta=f"{delta_score:+.1f}", delta_color="inverse")
                    delta_dp = mitigated_metrics["demographic_parity_difference"] - metrics["demographic_parity_difference"]
                    st.metric("Demographic Parity Diff",
                              f"{mitigated_metrics['demographic_parity_difference']:.4f}",
                              delta=f"{delta_dp:+.4f}", delta_color="inverse")
                    delta_eo = mitigated_metrics["equalized_odds_difference"] - metrics["equalized_odds_difference"]
                    st.metric("Equalized Odds Diff",
                              f"{mitigated_metrics['equalized_odds_difference']:.4f}",
                              delta=f"{delta_eo:+.4f}", delta_color="inverse")
                    delta_di = mitigated_metrics["disparate_impact_ratio"] - metrics["disparate_impact_ratio"]
                    st.metric("Disparate Impact Ratio",
                              f"{mitigated_metrics['disparate_impact_ratio']:.4f}",
                              delta=f"{delta_di:+.4f}", delta_color="normal")

                # Comparison bar chart
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                metric_names = ["DP Diff", "EO Diff", "1 - DI Ratio"]
                before_vals = [
                    abs(metrics["demographic_parity_difference"]),
                    abs(metrics["equalized_odds_difference"]),
                    1 - metrics["disparate_impact_ratio"]
                ]
                after_vals = [
                    abs(mitigated_metrics["demographic_parity_difference"]),
                    abs(mitigated_metrics["equalized_odds_difference"]),
                    1 - mitigated_metrics["disparate_impact_ratio"]
                ]
                x = np.arange(len(metric_names))
                width = 0.35
                ax2.bar(x - width/2, before_vals, width, label="Before", color="#dc2626", alpha=0.85)
                ax2.bar(x + width/2, after_vals, width, label="After", color="#16a34a", alpha=0.85)
                ax2.set_xticks(x)
                ax2.set_xticklabels(metric_names)
                ax2.set_ylabel("Bias Magnitude (lower = fairer)")
                ax2.set_title("Bias Reduction After Mitigation", fontweight="bold")
                ax2.legend()
                ax2.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig2)

            except Exception as e:
                st.warning(f"Mitigation failed: {e}")

    # ============================
    # SHAP FEATURE ATTRIBUTION
    # ============================
    if run_shap:
        with st.spinner("Computing SHAP feature importance..."):
            mean_shap, shap_sample, shap_vals = get_shap_values(model, X_train, X_test)

        if mean_shap is not None:
            st.subheader("🔬 SHAP Feature Attribution — What's Driving the Bias?")
            st.caption("Features with high SHAP values have the most influence on predictions. "
                       "Look for sensitive proxies like zip code, education, or marital status.")

            top_n = 15
            top_features = mean_shap.head(top_n)

            fig3, ax3 = plt.subplots(figsize=(9, 5))
            colors_shap = ["#dc2626" if i < 5 else "#4f46e5" for i in range(len(top_features))]
            ax3.barh(top_features.index[::-1], top_features.values[::-1], color=colors_shap[::-1])
            ax3.set_xlabel("Mean |SHAP Value|")
            ax3.set_title(f"Top {top_n} Most Influential Features", fontweight="bold")
            ax3.spines[["top", "right"]].set_visible(False)
            red_patch = mpatches.Patch(color="#dc2626", label="Top 5 — highest bias risk")
            blue_patch = mpatches.Patch(color="#4f46e5", label="Other influential features")
            ax3.legend(handles=[red_patch, blue_patch], fontsize=9)
            st.pyplot(fig3)
        else:
            st.info("Install `shap` package to enable SHAP analysis: `pip install shap`")

    # ============================
    # INTERSECTIONAL ANALYSIS
    # ============================
    if run_intersectional and extra_sensitive_cols:
        st.subheader("🔀 Intersectional Bias Analysis")
        st.caption(
            "Intersectional analysis checks if bias compounds across multiple attributes. "
            "E.g., 'Black women' may face more bias than 'Black people' or 'women' alone."
        )
        sensitive_df = df.loc[X_test.index, [sensitive_col] + extra_sensitive_cols]
        intersectional_rates = compute_intersectional_metrics(y_true, y_pred, sensitive_df)

        sorted_rates = dict(sorted(intersectional_rates.items(), key=lambda x: x[1]))
        fig4, ax4 = plt.subplots(figsize=(10, max(4, len(sorted_rates) * 0.5)))
        vals = list(sorted_rates.values())
        colors_int = ["#dc2626" if v == min(vals) else "#4f46e5" for v in vals]
        ax4.barh(list(sorted_rates.keys()), vals, color=colors_int)
        ax4.set_xlabel("Selection Rate")
        ax4.set_title("Selection Rate by Intersectional Group", fontweight="bold")
        ax4.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig4)

        df_int = pd.DataFrame.from_dict(
            sorted_rates, orient="index", columns=["Selection Rate"]
        ).sort_values("Selection Rate")
        st.dataframe(df_int.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

    # ============================
    # LLM EXPLANATION
    # ============================
    st.subheader("🧠 AI-Generated Audit Analysis")
    if not gemini_key and not os.environ.get("GEMINI_API_KEY"):
        st.warning("Enter your Gemini API key in the sidebar to enable AI explanations.")
    else:
        with st.spinner("Generating AI explanation..."):
            try:
                from google import genai as _genai
                _genai.Client(api_key=gemini_key or os.environ.get("GEMINI_API_KEY"))
                explanation = rag_explain(metrics, scenario=scenario,
                                          mitigated_metrics=mitigated_metrics)
                st.markdown(explanation)

                # ============================
                # PDF REPORT
                # ============================
                st.subheader("📄 Download Audit Report")
                pdf_bytes = generate_pdf_report(
                    metrics, scenario, explanation, mitigated_metrics
                )
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Download PDF Audit Report",
                        data=pdf_bytes,
                        file_name=f"biasglass_audit_{scenario.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.info("Install `reportlab` for PDF export: `pip install reportlab`")

            except Exception as e:
                st.error(f"AI explanation failed: {e}")

    # ============================
    # RAW METRICS JSON
    # ============================
    with st.expander("🗂️ Raw Metrics (JSON)", expanded=False):
        st.json(metrics)
        if mitigated_metrics:
            st.caption("After Mitigation:")
            st.json(mitigated_metrics)