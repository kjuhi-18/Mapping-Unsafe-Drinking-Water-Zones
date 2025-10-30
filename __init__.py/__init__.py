# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from utils.preprocess import load_and_clean_data
from utils.model_utils import predict_risk
from utils.map_utils import generate_risk_map
import streamlit_folium as sf

st.set_page_config(
    page_title="Aurora Dashboard ✨",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- STYLES ----------
BASE_CSS = """
.header {
  display:flex;
  align-items:center;
  gap:14px;
}
.header h1 {
  margin:0;
  font-size:26px;
  letter-spacing:0.6px;
}
.big-metric {
  background: linear-gradient(135deg, rgba(58,123,213,0.12), rgba(0,210,255,0.06));
  padding:12px;
  border-radius:12px;
  box-shadow: 0 6px 18px rgba(12,20,40,0.06);
}
.card {
  background: #ffffff;
  padding:14px;
  border-radius:10px;
  box-shadow: 0 6px 18px rgba(12,20,40,0.04);
}
.small-muted { color: #666666; font-size:12px; }
"""
st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)

# ---------- LOAD REAL DATA ----------
df = load_and_clean_data("data/water_quality.csv")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="header"><img src="https://img.icons8.com/fluency/48/000000/rocket.png"/>'
                '<h1>Aurora Dashboard</h1></div>', unsafe_allow_html=True)
    st.write("Real-time risk monitoring for groundwater contamination in India.")
    st.divider()

    st.subheader("Quick Filters")
    states = df["State"].unique().tolist()
    state_filter = st.selectbox("State", states)
    min_score = st.slider("Min risk score", min_value=0, max_value=100, value=20)

    st.divider()
    st.subheader("Navigation")
    page = st.radio("Go to", ["Overview", "Predict Risk", "Map", "Settings"])

# ---------- FILTERED DATA ----------
filtered_df = df[(df["State"] == state_filter) & (df["RiskScore"] >= min_score)]

# ---------- OVERVIEW ----------
if page == "Overview":
    st.title("🌍 Water Quality Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="big-metric card">', unsafe_allow_html=True)
        st.metric("Districts Analyzed", value=f"{len(df['District'].unique())}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="big-metric card">', unsafe_allow_html=True)
        st.metric("High Risk Zones", value=f"{(df['RiskScore'] > 70).sum()}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"📊 Risk Levels in {state_filter}")
    st.dataframe(filtered_df[['District', 'Arsenic', 'Fluoride', 'Nitrate', 'RiskScore']])

# ---------- PREDICTION ----------
elif page == "Predict Risk":
    st.title("🧠 Predict Water Contamination Risk")

    arsenic = st.number_input("Arsenic level (mg/L)", min_value=0.0, step=0.01)
    fluoride = st.number_input("Fluoride level (mg/L)", min_value=0.0, step=0.01)
    nitrate = st.number_input("Nitrate level (mg/L)", min_value=0.0, step=0.01)

    if st.button("Predict"):
        features = {
            "arsenic": arsenic,
            "fluoride": fluoride,
            "nitrate": nitrate
        }
        result = predict_risk(features)
        st.success(f"Predicted Risk Category: **{result}**")

# ---------- MAP ----------
elif page == "Map":
    st.title("🗺️ Contamination Risk Map")
    st.write("Visualize groundwater contamination risk across Indian districts.")
    map_object = generate_risk_map("data/india_districts.geojson", df)
    sf.folium_static(map_object)

# ---------- SETTINGS ----------
elif page == "Settings":
    st.subheader("Settings & Debugging")
    st.write("Sidebar Inputs Summary")
    st.json({
        "State filter": state_filter,
        "Min score": min_score
    })
    if st.button("Reset"):
        st.experimental_rerun()

# ---------- FOOTER ----------
st.markdown("---")
col_a, col_b = st.columns([3,1])
with col_a:
    st.write("© Aurora · Built with Streamlit — powered by real-world water data.")
with col_b:
    st.button("Send test notification (simulated)")
