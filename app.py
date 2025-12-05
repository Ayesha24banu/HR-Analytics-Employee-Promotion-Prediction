# app.py

"""
HR Analytics Promotion Prediction Dashboard
-------------------------------------------
Streamlit application for predicting employee promotions.
- Single Prediction
- Batch Prediction & Analytics
"""

import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import logging
from scripts.feature_engineering import add_features
import shap
import matplotlib.pyplot as plt

# =========================
# Logging Setup
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# =========================
# Page config + Theming
# =========================
st.set_page_config(page_title="HR Promotion Prediction", page_icon="📊", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: Inter, Segoe UI, system-ui, Arial, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.header-container { background: linear-gradient(90deg, #0B5ED7 0%, #1976D2 100%);
  color: white; border-radius: 14px; padding: 18px 20px; text-align: center;
  box-shadow: 0 4px 14px rgba(13,110,253,0.3); }
.header-title { font-size: 30px; font-weight: 800; margin: 0; }
.header-sub { font-size: 14px; opacity: 0.9; margin: 3px 0 0 0; }
.card { background: #fff; border: 1px solid #E9ECEF; border-radius: 16px;
  padding: 18px; box-shadow: 0 6px 20px rgba(13,110,253,0.05); }
.metric-card { background: #F8FAFD; border: 1px solid #E9ECEF; border-radius: 16px; padding: 16px 18px; }
.badge { border-radius: 14px; color: #fff; padding: 14px 18px; font-weight: 700;
  font-size: 18px; display:inline-flex; align-items:center; gap:10px; }
.badge-green { background:#229954; } .badge-red { background:#C0392B; }
.stButton>button { background: #0B5ED7; color:#fff; border-radius: 12px; padding: 10px 18px;
  border: 0; font-weight: 600; } .stButton>button:hover { background:#0A53BF; }
</style>
""", unsafe_allow_html=True)

# =========================
# Model + Encoder Loading
# =========================
MODEL_DIR = "../output/models"

@st.cache_resource
def load_model_and_encoders():
    if not os.path.isdir(MODEL_DIR):
        st.error(f"Model directory not found: {MODEL_DIR}")
        st.stop()

    model_file, enc_file = None, os.path.join(MODEL_DIR, "encoders.pkl")

    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl") and f != "encoders.pkl":
            model_file = os.path.join(MODEL_DIR, f)
            break

    if not model_file or not os.path.exists(enc_file):
        st.error("Model or encoders not found. Ensure both model .pkl and encoders.pkl exist.")
        st.stop()

    logging.info(f"Loaded model: {model_file} and encoders.")
    return joblib.load(model_file), joblib.load(enc_file)

model, encoders = load_model_and_encoders()

def transform_with_unknown(le, series: pd.Series) -> np.ndarray:
    """Transform with handling unseen categories."""
    series = series.astype(str)
    known_classes = set(le.classes_)
    return le.transform([x if x in known_classes else "Unknown" for x in series])

# =========================
# Header
# =========================
st.markdown("""
<div class="header-container">
  <p class="header-title">📊 HR Analytics: Employee Promotion Dashboard</p>
  <p class="header-sub">Smarter HR decisions with predictive analytics</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab_single, tab_batch = st.tabs(["✨ Single Prediction", "📂 Batch Prediction & Dashboard"])

# =========================
# SINGLE PREDICTION
# =========================
with tab_single:
    st.markdown("#### Enter Employee Details")
    with st.form("single_form"):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            department = st.selectbox("Department", ["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D", "Procurement", "HR", "Legal"])
            education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Below Secondary", "Unknown"])
            gender = st.selectbox("Gender", ["m", "f"])
            recruitment_channel = st.selectbox("Recruitment Channel", ["sourcing", "other", "referred"])
        with c2:
            region = st.text_input("Region (e.g., region_7)", "region_7")
            no_of_trainings = st.slider("No. of Trainings", 1, 10, 1)
            age = st.slider("Age", 20, 60, 30)
            avg_training_score = st.slider("Average Training Score", 0, 100, 50)
        with c3:
            previous_year_rating = st.slider("Previous Year Rating", 1, 5, 3)
            length_of_service = st.slider("Length of Service (Years)", 1, 40, 5)
            kpi_over_80 = st.selectbox("KPIs Met >80%", [0, 1])
            awards = st.selectbox("Awards Won", [0, 1])
        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        try:
            input_df = pd.DataFrame([{
                "department": department, "region": region, "education": education, "gender": gender,
                "recruitment_channel": recruitment_channel, "no_of_trainings": no_of_trainings,
                "age": age, "previous_year_rating": previous_year_rating, "length_of_service": length_of_service,
                "KPIs_met >80%": kpi_over_80, "awards_won?": awards, "avg_training_score": avg_training_score
            }])

            # Feature engineering
            input_df = add_features(input_df)

            # Apply saved encoders
            for col, le in encoders.items():
                if col in input_df.columns:
                    input_df[col] = transform_with_unknown(le, input_df[col])

            input_df = input_df.apply(pd.to_numeric, errors="coerce")

            # Prediction
            pred = int(model.predict(input_df)[0])
            prob = float(model.predict_proba(input_df)[0][1]) * 100

            if pred == 1:
                st.markdown(f'<div class="badge badge-green">✅ Likely to be Promoted — Confidence: {prob:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge badge-red">❌ Unlikely to be Promoted — Confidence: {prob:.2f}%</div>', unsafe_allow_html=True)

            st.progress(int(prob))

            # =========================
            # SHAP Explainability
            # =========================
            st.markdown("### 🔎 Why this Prediction?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

            logging.info(f"Single prediction done. Result: {pred}, Confidence: {prob:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logging.error(f"Single prediction error: {e}")

# =========================
# BATCH PREDICTION
# =========================
with tab_batch:
    st.markdown("#### Upload Employee CSV")
    template_df = pd.DataFrame({
        "employee_id": [10001, 10002], "department": ["Sales & Marketing", "Technology"],
        "region": ["region_7", "region_15"], "education": ["Bachelor's", "Master's & above"],
        "gender": ["m", "f"], "recruitment_channel": ["sourcing", "other"],
        "no_of_trainings": [1, 3], "age": [30, 41], "previous_year_rating": [3, 4],
        "length_of_service": [5, 9], "KPIs_met >80%": [1, 0], "awards_won?": [0, 1],
        "avg_training_score": [62, 78]
    })
    buf = io.BytesIO()
    template_df.to_csv(buf, index=False)
    st.download_button("📥 Download CSV Template", buf.getvalue(), file_name="hr_batch_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.dataframe(raw_df.head(), use_container_width=True)

            original_df = raw_df.copy()
            feat_df = add_features(raw_df)

            for col, le in encoders.items():
                if col in feat_df.columns:
                    feat_df[col] = transform_with_unknown(le, feat_df[col])

            feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

            X = feat_df.drop(columns=[c for c in ["employee_id"] if c in feat_df.columns])
            preds = model.predict(X).astype(int)
            probs = model.predict_proba(X)[:, 1] * 100.0

            results = pd.DataFrame({
                "employee_id": original_df["employee_id"].values,
                "predicted_promotion": preds,
                "confidence_%": np.round(probs, 2)
            })
            st.dataframe(results.head(), use_container_width=True)

            out_buf = io.BytesIO()
            results.to_csv(out_buf, index=False)
            st.download_button("💾 Download Predictions CSV", out_buf.getvalue(), file_name="promotion_predictions.csv", mime="text/csv")

            total_n = len(results)
            promoted_n = int(results["predicted_promotion"].sum())
            rate = (promoted_n / total_n * 100.0) if total_n else 0.0
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><b>Total Employees</b><h3>{total_n}</h3></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><b>Predicted Promoted</b><h3>{promoted_n}</h3></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><b>Promotion Rate</b><h3>{rate:.2f}%</h3></div>", unsafe_allow_html=True)

            # Visualizations
            merged = original_df.copy()
            merged["predicted_promotion"] = results["predicted_promotion"].values
            merged["confidence_%"] = results["confidence_%"].values

            if "department" in merged.columns:
                dept_fig = px.histogram(merged, x="department", color="predicted_promotion", barmode="group",
                                        color_discrete_map={0: "#C0392B", 1: "#229954"},
                                        labels={"predicted_promotion": "Promoted"})
                st.plotly_chart(dept_fig, use_container_width=True)

            if "KPIs_met >80%" in merged.columns:
                kpi_fig = px.pie(merged, names="KPIs_met >80%", title="KPI >80% Distribution",
                                 color="KPIs_met >80%", color_discrete_sequence=["#0B5ED7", "#39B54A"])
                st.plotly_chart(kpi_fig, use_container_width=True)

            if "education" in merged.columns:
                edu_rate = merged.groupby("education")["predicted_promotion"].mean().reset_index()
                edu_rate["promotion_rate_%"] = edu_rate["predicted_promotion"] * 100.0
                edu_fig = px.bar(edu_rate, x="education", y="promotion_rate_%",
                                 text=edu_rate["promotion_rate_%"].map(lambda x: f"{x:.1f}%"),
                                 color_discrete_sequence=["#0B5ED7"])
                st.plotly_chart(edu_fig, use_container_width=True)

            # ======================
            # Global SHAP Explainability
            # ======================
            if st.checkbox("📊 Show Feature Importance (SHAP)"):
                # Use TreeExplainer for RF/XGB models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                # Summary Plot (Global feature importance)
                st.subheader("Global Feature Importance (SHAP)")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                st.pyplot(fig)

                # Beeswarm Plot (Detailed impacts)
                st.subheader("Detailed Feature Impacts (Beeswarm)")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X, show=False)
                st.pyplot(fig2)
            
            logging.info(f"Batch prediction done. Total={total_n}, Promoted={promoted_n}, Rate={rate:.2f}%")
        except Exception as e:
            st.error(f"Error in batch prediction: {e}")
            logging.error(f"Batch prediction error: {e}")

