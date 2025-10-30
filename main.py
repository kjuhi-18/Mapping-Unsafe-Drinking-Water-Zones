#13

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import base64
import io
import os
import json
import folium
import streamlit_folium as sf
from st_aggrid import AgGrid, GridOptionsBuilder
from datetime import datetime
from functools import wraps
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import joblib
import pickle
from transformers import pipeline



# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Aurora Dashboard ✨",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
BASE_CSS = """
.header {
    font-size: 30px;
    font-weight: bold;
    color: #1E90FF;
    text-align: center;
    padding: 10px;
}
.subheader {
    font-size: 18px;
    color: #555;
    text-align: center;
}
.brand { font-weight:700; font-size:22px; }
.small { color: #7a7a7a; font-size:12px }
.card { padding: 18px; border-radius: 12px; box-shadow: 0 8px 20px rgba(20,20,20,0.06); background: white }
.footer { color:#999; font-size:12px; }
"""
st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)

USERS = {"admin@example.com": {"name": "Admin", "role": "admin"}, "judge@example.com": {"name": "Judge", "role": "judge"}}


# --------------------------
# LOGIN SYSTEM
# --------------------------
def user_login():
    """Sidebar login system with Name, Email, and Password"""
    st.sidebar.title("🔐 Account Login")

    # --- Authorized Users Dictionary ---
    authorized_users = {
        "Kashif": {
            "email": "kashif.kamran.btech2024@sitpune.edu.in",
            "password": "Budweiser"
        },
        "Kunal": {
            "email": "kunal.jhindal.btech2024@sitpune.edu.in",
            "password": "Kingfisher"
        },
        "Kashish": {
            "email": "kashish.chelwani.btech2024@sitpune.edu.in",
            "password": "Oaksmith"
        }
    }

    # --- Session State Setup ---
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "email" not in st.session_state:
        st.session_state.email = None

    # --- Login Form ---
    if not st.session_state.logged_in:
        with st.sidebar.form("login_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                # Check if the entered name exists in authorized users
                if name in authorized_users:
                    valid_email = authorized_users[name]["email"]
                    valid_password = authorized_users[name]["password"]

                    # Validate credentials
                    if email == valid_email and password == valid_password:
                        st.session_state.logged_in = True
                        st.session_state.username = name
                        st.session_state.email = email
                        st.success(f"✅ Welcome, {name}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password.")
                else:
                    st.error("❌ Unauthorized user. Access denied.")
    else:
        st.sidebar.success(f"✅ Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.email = None
            st.rerun()

# Run login system
user_login()

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_preprocessed_data():
    df = pd.read_csv("Preprocessed_Dataset.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_model():
    time.sleep(1)
    return {"model": "demo_model_v1"}

def df_to_download_link(df, filename='download.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

SAFE_LIMITS = {
    "pH": "6.5 – 8.5",
    "EC (μS/cm)": "≤ 3000",
    "HCO3 (mg/L)": "≤ 600",
    "Cl (mg/L)": "≤ 1000",
    "F (mg/L)": "≤ 1.5",
    "SO4 (mg/L)": "≤ 400",
    "NO3 (mg/L)": "≤ 45",
    "Total Hardness (mg/L)": "≤ 600",
    "Ca (mg/L)": "≤ 200",
    "Mg (mg/L)": "≤ 100",
    "U (ppb)": "≤ 30"
}

SAFE_LIMITS_NUMERIC = {
    "pH": (6.5, 8.5),
    "EC (μS/cm)": 3000,
    "HCO3 (mg/L)": 600,
    "Cl (mg/L)": 1000,
    "F (mg/L)": 1.5,
    "SO4 (mg/L)": 400,
    "NO3 (mg/L)": 45,
    "Total Hardness (mg/L)": 600,
    "Ca (mg/L)": 200,
    "Mg (mg/L)": 100,
    "U (ppb)": 30
}

def predict_risk(row):
    score = 0
    if "pH" in row and not (SAFE_LIMITS_NUMERIC["pH"][0] <= row["pH"] <= SAFE_LIMITS_NUMERIC["pH"][1]):
        score += 1
    if "F (mg/L)" in row and pd.notna(row.get("F (mg/L)")) and row["F (mg/L)"] > SAFE_LIMITS_NUMERIC["F (mg/L)"]:
        score += 1
    if "NO3" in row and pd.notna(row.get("NO3")) and row["NO3"] > SAFE_LIMITS_NUMERIC["NO3 (mg/L)"]:
        score += 1
    if "U (ppb)" in row and pd.notna(row.get("U (ppb)")) and row["U (ppb)"] > SAFE_LIMITS_NUMERIC["U (ppb)"]:
        score += 1
    if "SO4 (mg/L)" in row and pd.notna(row.get("SO4 (mg/L)")) and row["SO4 (mg/L)"] > SAFE_LIMITS_NUMERIC["SO4 (mg/L)"]:
        score += 1
    if "Total Hardness (mg/L)" in row and pd.notna(row.get("Total Hardness (mg/L)")) and row["Total Hardness (mg/L)"] > SAFE_LIMITS_NUMERIC["Total Hardness (mg/L)"]:
        score += 1
    if "EC (μS/cm)" in row and pd.notna(row.get("EC (μS/cm)")) and row["EC (μS/cm)"] > SAFE_LIMITS_NUMERIC["EC (μS/cm)"]:
        score += 1
    if score == 0:
        return "Safe"
    elif score == 1:
        return "Moderate"
    else:
        return "High"

def generate_risk_map(data):
    lat_col = next((c for c in data.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in data.columns if "lon" in c.lower()), None)
    if lat_col is None or lon_col is None:
        return folium.Map(location=[20.5937, 78.9629], zoom_start=5), data
    data[lat_col] = pd.to_numeric(data[lat_col], errors="coerce")
    data[lon_col] = pd.to_numeric(data[lon_col], errors="coerce")
    data = data.dropna(subset=[lat_col, lon_col])
    data["Risk Level"] = data.apply(predict_risk, axis=1)
    if data.empty:
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        return m, data
    m = folium.Map(location=[data[lat_col].mean(), data[lon_col].mean()], zoom_start=5)
    for _, row in data.iterrows():
        color = {"Safe": "green", "Moderate": "orange", "High": "red"}[row["Risk Level"]]
        popup_info = (
            f"<b>{row.get('District','-')}</b><br>"
            f"F: {row.get('F (mg/L)','-')} mg/L<br>"
            f"NO3: {row.get('NO3','-')} mg/L<br>"
            f"U: {row.get('U (ppb)','-')} ppb<br>"
            f"<b>Risk:</b> {row['Risk Level']}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=popup_info
        ).add_to(m)
    return m, data

# --------------------------
# APP FUNCTIONALITY
# --------------------------
if st.session_state.logged_in:

    # Sidebar Navigation
    st.sidebar.header("🧭 Navigation")
    page = st.sidebar.radio("Go to", [
        "Home",
        "Upload & Clean",
        "Explorer / EDA",
        "Modeling / Predictor",
        "Chatbot / Assistant",
        "Smart Forms & Widgets",
        "Export Report",
        "Admin Panel"
    ])

    # --------------------------
    # PAGE 1: HOME
    # --------------------------
    if page == "Home":
        st.subheader("🗺️ Nationwide Water Quality Risk (WHO/BIS Standards)")
        map_choice = st.sidebar.radio("Select Map View", ["Nationwide Impurity Map", "Statewise Impurity Map"])
        selected_state = None
        selected_district = None
        df = load_preprocessed_data()
        if map_choice == "Statewise Impurity Map":
            state_options = sorted(df["State"].dropna().unique())
            selected_state = st.sidebar.selectbox("Select State", state_options)
            district_options = sorted(df[df["State"] == selected_state]["District"].dropna().unique())
            selected_district = st.sidebar.selectbox("Select District", ["All Districts"] + district_options)
        with st.sidebar.expander("📘 WHO/BIS Drinking Water Standards", expanded=False):
            st.markdown("These are the official WHO/BIS limits for drinking water quality parameters:")
            st.table(pd.DataFrame(list(SAFE_LIMITS.items()), columns=["Parameter", "Safe Limit"]))
        if map_choice == "Nationwide Impurity Map":
            risk_map, risk_df = generate_risk_map(df)
            sf.folium_static(risk_map, width=1000, height=550)
        else:
            st.subheader(f"📍 {selected_state} — Statewise Water Quality Map")
            if selected_district and selected_district != "All Districts":
                filtered_df = df[(df["State"] == selected_state) & (df["District"] == selected_district)]
            else:
                filtered_df = df[df["State"] == selected_state]
            risk_map, risk_df = generate_risk_map(filtered_df)
            sf.folium_static(risk_map, width=1000, height=550)
        st.markdown("### ⚠️ High-Risk Locations (Exceeding WHO/BIS Limits)")
        high_risk_df = risk_df[risk_df["Risk Level"] == "High"] if not risk_df.empty else pd.DataFrame()
        if not high_risk_df.empty:
            st.dataframe(high_risk_df[["State", "District", "Location", "F (mg/L)", "NO3", "U (ppb)", "Risk Level"]])
            csv = high_risk_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download High-Risk Report (CSV)",
                data=csv,
                file_name=f"High_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ No high-risk locations found — all within WHO/BIS limits!")
        st.markdown("### 📘 WHO/BIS Drinking Water Standard Limits")
        st.dataframe(pd.DataFrame(list(SAFE_LIMITS.items()), columns=["Parameter", "Safe Limit"]))

    # --------------------------
    # PAGE 2: UPLOAD & CLEAN
    # --------------------------
    elif page == "Upload & Clean":
        st.header("Upload Data — CSV / Excel / JSON")
        uploaded = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls', 'json'])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    uf = pd.read_csv(uploaded)
                elif uploaded.name.endswith('.json'):
                    uf = pd.read_json(uploaded)
                else:
                    uf = pd.read_excel(uploaded)
                st.success(f"Loaded {uploaded.name} — {uf.shape[0]} rows, {uf.shape[1]} cols")
                st.subheader("Preview & Quick Cleaning")
                try:
                    gb = GridOptionsBuilder.from_dataframe(uf)
                    gb.configure_pagination()
                    gb.configure_default_column(editable=True)
                    gridOptions = gb.build()
                    AgGrid(uf, gridOptions=gridOptions, height=300)
                except Exception:
                    st.dataframe(uf.head(10))
                st.download_button("Download cleaned CSV", data=uf.to_csv(index=False).encode(), file_name='cleaned.csv')
                #with st.expander("Automatic suggestions"):
                    #st.write("Missing value counts:")
                    #st.write(uf.isna().sum())
                    #thresh = int(0.5 * len(uf))
                    #st.write([c for c in uf.columns if uf[c].isna().sum() > thresh])'''
            except Exception as e:
                st.error(f"Failed to read file: {e}")
        else:
            st.info("Upload a sample CSV to get started. Use the Home page for quick experimentation.")

    # --------------------------
    # PAGE 3: EXPLORER / EDA
    # --------------------------
    elif page == "Explorer / EDA":
        st.header("💧 Interactive Data Explorer & EDA (Water Quality Dataset)")

    # Load default dataset
        df_eda = load_preprocessed_data()
        st.success(f"Loaded default dataset with {df_eda.shape[0]} rows and {df_eda.shape[1]} columns.")

    # Allow file upload (optional)
        uploaded_file = st.file_uploader("Upload a different dataset (CSV only)", type=['csv'])
        if uploaded_file is not None:
            df_eda = pd.read_csv(uploaded_file)
            st.info(f"Loaded custom dataset: {uploaded_file.name}")

    # Sidebar filters
        with st.sidebar.expander("🔍 Filter Data", expanded=False):
            states = sorted(df_eda["State"].dropna().unique())
            selected_state = st.selectbox("Select State", ["All"] + states)
            if selected_state != "All":
                df_eda = df_eda[df_eda["State"] == selected_state]

            districts = sorted(df_eda["District"].dropna().unique())
            selected_district = st.selectbox("Select District", ["All"] + districts)
            if selected_district != "All":
                df_eda = df_eda[df_eda["District"] == selected_district]

            years = sorted(df_eda["Year"].dropna().unique())
            selected_year = st.selectbox("Select Year", ["All"] + [str(y) for y in years])
            if selected_year != "All":
                df_eda = df_eda[df_eda["Year"] == int(selected_year)]

        st.markdown("### 📊 Dataset Overview")
        st.dataframe(df_eda.head())

    # Dataset summary
        st.subheader("🔢 Basic Statistics")
        st.write(df_eda.describe())

    # Missing values
        st.subheader("🚨 Missing Values Summary")
        missing = df_eda.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["% Missing"] = (missing["Missing Values"] / len(df_eda)) * 100
        st.dataframe(missing)

    # Univariate analysis
        st.subheader("📈 Univariate Analysis")
        numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select Parameter", numeric_cols)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(df_eda, x=selected_col, nbins=30,
                                         title=f"Distribution of {selected_col}",
                                         marginal="box"),
                                use_container_width=True)
            with col2:
                st.plotly_chart(px.box(df_eda, y=selected_col,
                                   title=f"Boxplot of {selected_col}"),
                                use_container_width=True)
        else:
            st.info("No numeric columns detected in the dataset.")

    # Correlation analysis
        st.subheader("📊 Correlation Heatmap (Numeric Parameters)")
        numeric_df = df_eda.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric data available for correlation analysis.")

    # Bivariate analysis
        st.subheader("🔍 Bivariate Relationship Explorer")
        col_x = st.selectbox("X-axis Parameter", numeric_cols, index=0)
        col_y = st.selectbox("Y-axis Parameter", numeric_cols, index=min(1, len(numeric_cols)-1))
        color_by = st.selectbox("Color by", ["State", "District", "Year"], index=0)
        fig_bi = px.scatter(df_eda, x=col_x, y=col_y, color=color_by,
                        hover_data=["Location", "State", "District"],
                        title=f"{col_y} vs {col_x} colored by {color_by}")
        st.plotly_chart(fig_bi, use_container_width=True)

    # Insights on safe limits
        st.subheader("🧭 Parameter Compliance with WHO/BIS Limits")
        results = []
        for param, limit in SAFE_LIMITS_NUMERIC.items():
            if param in df_eda.columns:
                if isinstance(limit, tuple):
                    out_of_range = df_eda[(df_eda[param] < limit[0]) | (df_eda[param] > limit[1])]
                else:
                    out_of_range = df_eda[df_eda[param] > limit]
                percent_bad = len(out_of_range) / len(df_eda) * 100
                results.append({"Parameter": param, "Non-compliant (%)": round(percent_bad, 2)})
        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("No safe limit data found for this dataset.")
            
    # PAGE 5: MODELING / PREDICTOR
    elif page == "Modeling / Predictor":
        st.title("💧 Water Safety Prediction")
        st.write("This tool checks if your water is safe for drinking based on both quality parameters and an ML prediction model.")

        import pickle
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

    # =======================
    # Load Model & Scaler
    # =======================
        with open("water_safety_model.pkl", "rb") as f:
            model = pickle.load(f)

        try:
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except:
            scaler = None

    # =======================
    # Safe Limits
    # =======================
        safe_limits = {
            "pH": (6.5, 8.5),
            "EC (μS/cm)": 3000,
            "CO3 (mg/L)": None,
            "HCO3 (mg/L)": 600,
            "Cl (mg/L)": 1000,
            "F (mg/L)": 1.5,
            "SO4 (mg/L)": 400,
            "NO3 (mg/L)": 45,
            "PO4 (mg/L)": None,
            "Total Hardness (mg/L)": 600,
            "Ca (mg/L)": 200,
            "K (mg/L)": None,
            "Mg (mg/L)": 100,
            "Na (mg/L)": None,
            "U (ppb)": 30
        }

    # =======================
    # User Inputs
    # =======================
        st.subheader("🧪 Enter Measured Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
            ec = st.number_input("EC (μS/cm)", min_value=0.0)
            chloride = st.number_input("Cl (mg/L)", min_value=0.0)
            fluoride = st.number_input("F (mg/L)", min_value=0.0)
            sulfate = st.number_input("SO4 (mg/L)", min_value=0.0)
            nitrate = st.number_input("NO3 (mg/L)", min_value=0.0)
            calcium = st.number_input("Ca (mg/L)", min_value=0.0)

        with col2:
            magnesium = st.number_input("Mg (mg/L)", min_value=0.0)
            sodium = st.number_input("Na (mg/L)", min_value=0.0)
            potassium = st.number_input("K (mg/L)", min_value=0.0)
            uranium = st.number_input("U (ppb)", min_value=0.0)
            hardness = st.number_input("Total Hardness (mg/L)", min_value=0.0)
            hco3 = st.number_input("HCO3 (mg/L)", min_value=0.0)
            co3 = st.number_input("CO3 (mg/L)", min_value=0.0)

        with col3:
            year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
            latitude = st.number_input("Latitude", step=0.0001)
            longitude = st.number_input("Longitude", step=0.0001)
            state = st.text_input("State")
            district = st.text_input("District")
            location = st.text_input("Location")

    # =======================
    # Create DataFrame
    # =======================
        input_df = pd.DataFrame({
            'pH': [ph],
            'EC (μS/cm)': [ec],
            'Cl (mg/L)': [chloride],
            'F (mg/L)': [fluoride],
            'SO4 (mg/L)': [sulfate],
            'NO3 (mg/L)': [nitrate],
            'Ca (mg/L)': [calcium],
            'Mg (mg/L)': [magnesium],
            'Na (mg/L)': [sodium],
            'K (mg/L)': [potassium],
            'U (ppb)': [uranium],
            'Total Hardness (mg/L)': [hardness],
            'HCO3 (mg/L)': [hco3],
            'CO3 (mg/L)': [co3],
            'Year': [year],
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Location': [location],
            'District': [district],
            'State': [state]
        })

    # =======================
    # Check Safe Limits
    # =======================
        st.subheader("📊 Parameter Safety Check")

        for param, limit in safe_limits.items():
            if param not in input_df.columns:
                continue

            value = input_df[param].values[0]

            if limit is None:
                st.info(f"ℹ️ {param}: No official safe limit available.")
            elif isinstance(limit, tuple):  # range (like pH)
                if limit[0] <= value <= limit[1]:
                    st.success(f"✅ {param}: {value} (within safe range {limit})")
                else:
                    st.error(f"⚠️ {param}: {value} (outside safe range {limit})")
            else:  # upper limit
                if value <= limit:
                    st.success(f"✅ {param}: {value} ≤ {limit}")
                else:
                    st.error(f"⚠️ {param}: {value} exceeds safe limit {limit}")

    # =======================
    # Encode Text Columns
    # =======================
        label_cols = ["State", "District", "Location"]
        for col in label_cols:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col].astype(str))

    # =======================
    # Scale + Predict
    # =======================
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df

        if st.button("🔍 Predict Overall Water Safety"):
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] * 100

            st.markdown("---")
            if prediction == 1:
                st.success(f"✅ **Model Prediction:** The water is SAFE for drinking (Confidence: {prob:.2f}%)")
            else:
                st.error(f"❌ **Model Prediction:** The water is UNSAFE for drinking (Confidence: {prob:.2f}%)")




    # --------------------------
    # OTHER PLACEHOLDERS
    # --------------------------
    
    elif page == "Export Report":
        st.header("📊 Reports & Export")

    # -------------------------------
    # STEP 1: Upload or Load Dataset
    # -------------------------------
        uploaded_file = st.file_uploader("📁 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

        if uploaded_file is not None:
        # Read the uploaded dataset
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ Dataset loaded successfully!")
            st.write("### Preview of your dataset:")
            st.dataframe(df.head())

        # ---------------------------------
        # STEP 2: Download as CSV
        # ---------------------------------
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name="water_quality_dataset.csv",
                mime="text/csv"
            )

        # ---------------------------------
        # STEP 3: Download as Excel
        # ---------------------------------
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
            excel_data = excel_buffer.getvalue()

            st.download_button(
                label="⬇️ Download as Excel",
                data=excel_data,
                file_name="water_quality_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # ---------------------------------
        # STEP 4: Generate a simple PDF Report
        # ---------------------------------
            def generate_pdf(dataframe):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(200, 760, "Water Quality Report")
                c.setFont("Helvetica", 10)

            # Basic info
                c.drawString(50, 730, f"Total Rows: {dataframe.shape[0]}")
                c.drawString(50, 715, f"Total Columns: {dataframe.shape[1]}")
                c.drawString(50, 700, f"Columns: {', '.join(dataframe.columns[:5])}...")

                c.drawString(50, 680, "Summary Statistics:")
                text_object = c.beginText(50, 660)
                text_object.setFont("Helvetica", 9)
                summary = dataframe.describe().to_string()
                for line in summary.split("\n"):
                    text_object.textLine(line)
                c.drawText(text_object)

                c.showPage()
                c.save()
                buffer.seek(0)
                return buffer

            if st.button("🧾 Generate PDF Report"):
                pdf = generate_pdf(df)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf,
                    file_name="water_quality_report.pdf",
                    mime="application/pdf"
                )

        else:
            st.info("📤 Please upload your dataset above to enable export options.")
    
    
    elif page == "Smart Forms & Widgets":
        st.title("🤖 Smart Widgets Dashboard")
        st.write("Interactively explore and visualize your **Water Quality Dataset** using smart filters and widgets.")

    # --------------------------
    # Load Dataset
    # --------------------------
        try:
            df = pd.read_csv("Preprocessed_Dataset.csv")
            st.success("✅ Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Could not find 'Preprocessed_Dataset.csv'. Please place it in the same folder as this script.")
            st.stop()

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

    # --------------------------
    # Smart Filters
    # --------------------------
        st.divider()
        st.header("🎛️ Smart Filters & Interactive Controls")

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        with st.expander("🔍 Filter Your Data"):
            col1, col2, col3 = st.columns(3)

            with col1:
                selected_state = None
                if "State" in df.columns:
                    selected_state = st.selectbox("Select State", ["All"] + sorted(df["State"].dropna().unique().tolist()))

            with col2:
                selected_district = None
                if "District" in df.columns:
                    selected_district = st.selectbox("Select District", ["All"] + sorted(df["District"].dropna().unique().tolist()))

            with col3:
                selected_range_col = st.selectbox("Select Numeric Column to Filter", numeric_cols)
                min_val, max_val = df[selected_range_col].min(), df[selected_range_col].max()
                value_range = st.slider(f"Filter {selected_range_col}", float(min_val), float(max_val), (float(min_val), float(max_val)))

        # Apply Filters
            filtered_df = df.copy()
            if selected_state and selected_state != "All":
                filtered_df = filtered_df[filtered_df["State"] == selected_state]
            if selected_district and selected_district != "All":
                filtered_df = filtered_df[filtered_df["District"] == selected_district]
            if selected_range_col:
                filtered_df = filtered_df[
                    (filtered_df[selected_range_col] >= value_range[0]) & 
                    (filtered_df[selected_range_col] <= value_range[1])
                ]

            st.success(f"📊 Showing {len(filtered_df)} rows after filtering")
            st.dataframe(filtered_df.head(10))

    # --------------------------
    # Summary Statistics
    # --------------------------
        st.divider()
        st.header("📈 Summary Insights")

        with st.expander("📊 Basic Statistics", expanded=True):
            st.write(filtered_df.describe())

        with st.expander("📉 Correlation Heatmap"):
            import seaborn as sns
            import matplotlib.pyplot as plt

            corr = filtered_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, cmap="coolwarm", annot=False)
            st.pyplot(fig)

   # --------------------------
        # 📊 Enhanced Visualizations
# --------------------------
        st.divider()
        st.header("📊 Data Visualization & Insights")

        import plotly.express as px

        # ---- Scatter Plot ----
        st.subheader("🔹 Interactive Scatter Plot")
        x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Select Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        color_col = st.selectbox("Color by (optional)", ["None"] + numeric_cols + categorical_cols)

        if color_col != "None":
            fig_scatter = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color=color_col,
                title=f"{y_axis} vs {x_axis} colored by {color_col}",
                template="plotly_white",
                opacity=0.7,
                height=500
            )
        else:
            fig_scatter = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} vs {x_axis}",
                template="plotly_white",
                opacity=0.7,
                height=500
            )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --------------------------
    # Export Filtered Data
    # --------------------------
        st.divider()
        st.header("📤 Export Data")

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")    
    # --------------------------
    
    elif page in ["Chatbot / Assistant"]:
        st.title("🤖 AI Chatbot Assistant")

    # Load Hugging Face GPT-2 model (once)
        @st.cache_resource
        def load_chat_model():
            return pipeline("text-generation", model="gpt2")

        generator = load_chat_model()

    # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    # Display previous messages
        for chat in st.session_state.messages:
            if chat["role"] == "user":
                st.markdown(f"🧑‍🌾 **You:** {chat['content']}")
            else:
                st.markdown(f"🤖 **Assistant:** {chat['content']}")

    # Chat input box
        user_input = st.chat_input("Ask me anything...")

        if user_input:
        # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate AI response
            with st.spinner("Thinking..."):
                response = generator(
                    user_input,
                    max_length=150,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )[0]["generated_text"]

        # Clean up the response (optional)
            bot_reply = response[len(user_input):].strip()

        # Add bot message
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        # Refresh conversation
            st.rerun()


    # --------------------------
    # ADMIN PANEL
    # --------------------------
    
    elif page == "Admin Panel":
        st.title("🛠️ Admin Panel — Team Information & Vision")
        st.write("Welcome to the heart of our project — meet the minds behind it!")

    # Team Information Section
        st.subheader("👩‍💻 Our Team Members")
        st.markdown("""
    | Name |Email |Connection |
    |------|----  |--------|
    | **Kashif Kamran** | kashif.kamran.btech2024@sitpune.edu.in | https://www.linkedin.com/in/kashif-kamran-350133380/ |
    | **Kunal Jhindal** | kunal.jhindal.btech2024@sitpune.edu.in | https://www.linkedin.com/in/kunal-jhindal/ |
    | **Kashish Chelwani** | kashish.chelwani.btech2024@sitpune.edu.in | https://www.linkedin.com/in/kashish-chelwani08/ |
    """)
    
        st.divider()

    # Project Vision Section
        st.subheader("🌾 Project Vision")
        st.info("""
    Our goal is to empower communities with technology that ensures safe and sustainable drinking water.
    Through machine learning and environmental analytics, we aim to make clean water accessible to all.
    """)

        st.divider()

    # Words of Wisdom
        st.subheader("💡 Words of Wisdom")
        st.success("""
    “Technology is best when it brings people together.”  
    — Matt Mullenweg  
    """)
        st.write("""
    We believe that innovation, when combined with empathy and awareness,  
    has the power to transform lives and build a healthier, more sustainable future.
    """)

        st.divider()

    # Optional Aesthetic Touch
        st.caption("© 2025 Team AquaSafe")


# --------------------------
# IF NOT LOGGED IN
# --------------------------
else:
    st.warning("🔒 Please log in from the sidebar to access the Aurora Water Dashboard.")