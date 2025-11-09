# 🌊 Aurora Water Dashboard ✨  
### *Mapping Unsafe Drinking Water Zones using AI, ML & Data Visualization*

---

> 💡 **Did you know?**  
> Over **80% of India’s rural drinking water** comes from groundwater sources —  
> yet **nearly 60%** of these are contaminated with unsafe levels of fluoride, nitrate, and arsenic  
> *(Source: UNICEF & Central Ground Water Board, India)*  

---

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Powered_by-Machine_Learning-orange" />
</p>

---

## 🧭 Overview

**Aurora Water Dashboard** is an interactive web application designed to analyze, visualize, and predict **drinking water safety** across India.  
By combining **data science**, **machine learning**, and **geo-mapping**, it identifies regions with unsafe groundwater and helps visualize contamination patterns at state and district levels.

💧 Our mission:  
> *To empower communities with transparent, data-driven insights about the quality of the water they drink every day.*

---

## 🚀 Key Features

| 🌐 Category | 🧩 Description |
|--------------|----------------|
| 🗺️ **Nationwide Impurity Maps** | Visualize impurity hotspots using interactive **Folium maps**. |
| 📊 **EDA Dashboard** | Explore datasets through histograms, heatmaps, and correlation analysis. |
| 🤖 **Water Safety Predictor** | Predict if your water sample is *safe* or *unsafe* using a trained ML model. |
| 💬 **AI Chatbot Assistant** | Ask questions or get guidance using a built-in GPT-powered chatbot. |
| 🧾 **Report Export** | Download analysis reports as **PDF, CSV, or Excel**. |
| 🧠 **Smart Forms & Filters** | Filter data by state, district, and water parameters easily. |
| 🔐 **Secure Login System** | Role-based login for Admins, Judges, and Users. |

---


## 📂 Repository Structure

```
Mapping-Unsafe-Drinking-Water-Zones/
│
├── __init__.py
├── .gitattributes
│
├── 📁 unsafe/                      # Unsafe parameter datasets
│   ├── Ca_(mg_L)_unsafe.csv
│   ├── Cl_(mg_L)_unsafe.csv
│   ├── EC_(μS_cm)_unsafe.csv
│   ├── F_(mg_L)_unsafe.csv
│   ├── HCO3_unsafe.csv
│   ├── Mg_(mg_L)_unsafe.csv
│   ├── NO3_unsafe.csv
│   ├── PH_Unsafe.csv
│   ├── SO4_unsafe.csv
│   ├── Total_Hardness_unsafe.csv
│   └── U_(ppb)_unsafe.csv
│
├── EDA with model training.ipynb    # Exploratory Data Analysis + Model building
├── preprocessing.ipynb              # Data cleaning, transformation
├── scrapping.ipynb                  # Web scraping for raw datasets
│
├── water_safety_model.pkl           # Trained ML model
├── Preprocessed_Dataset.csv         # Main cleaned dataset
│
├── main.py                          # Streamlit app file
├── requirements.txt                 # Dependency list
│
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kjuhi-18/Mapping-Unsafe-Drinking-Water-Zones.git
cd Mapping-Unsafe-Drinking-Water-Zones
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run main.py
```

### 4️⃣ Demo Login Credentials

| Name | Email | Password |
|------|--------|-----------|
| Kashif | kashif.kamran.btech2024@sitpune.edu.in | Budweiser |
| Kunal | kunal.jhindal.btech2024@sitpune.edu.in | Kingfisher |
| Kashish | kashish.chelwani.btech2024@sitpune.edu.in | Oaksmith |

---

## 🧠 Machine Learning Model

| Step | Description |
|------|--------------|
| **Algorithm** | Logistic Regression / Random Forest |
| **Input Parameters** | pH, EC, Cl, F, SO4, NO3, Ca, Mg, U, etc. |
| **Output** | Binary classification — Safe (1) / Unsafe (0) |
| **Scaler** | StandardScaler normalization |
| **Accuracy** | 90%+ on test dataset |

The trained model (`water_safety_model.pkl`) is used within the Streamlit dashboard for real-time predictions.

---

## 📚 WHO/BIS Drinking Water Limits

| Parameter | Safe Range / Limit |
|------------|--------------------|
| pH | 6.5 – 8.5 |
| EC (μS/cm) | ≤ 3000 |
| HCO3 (mg/L) | ≤ 600 |
| Cl (mg/L) | ≤ 1000 |
| F (mg/L) | ≤ 1.5 |
| SO4 (mg/L) | ≤ 400 |
| NO3 (mg/L) | ≤ 45 |
| Total Hardness (mg/L) | ≤ 600 |
| Ca (mg/L) | ≤ 200 |
| Mg (mg/L) | ≤ 100 |
| U (ppb) | ≤ 30 |

---

## 🧰 Tech Stack

| Category | Tools Used |
|-----------|------------|
| **Frontend** | Streamlit, Plotly, Folium, PyDeck |
| **Backend / Logic** | Python, Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly Express, Seaborn, Matplotlib |
| **Model Handling** | Joblib, Pickle |
| **AI Chatbot** | HuggingFace Transformers (GPT-2) |
| **Reporting** | ReportLab (PDF generation) |

---

## 🌍 Impact & Vision

> 💧 *“Clean water is not a privilege — it’s a right.”*  
> Aurora Water Dashboard strives to make **data about water quality accessible** and **actionable**.  
> With real-time visualization and predictive AI, it helps identify unsafe zones and drive preventive action.

Our long-term vision:
- 📡 Integrate **IoT sensors** for live groundwater data  
- ☁️ Deploy on **Streamlit Cloud / Hugging Face Spaces**  
- 📈 Enable **time-series tracking** for long-term pollution trends  
- 🧩 Provide open APIs for environmental agencies and researchers  

---

## 👨‍💻 Team AquaSafe

| Member |
|---------|
| **Kashif Kamran** |
| **Kunal Jhindal** |
| **Kashish Chelwani** |

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use, share, and modify with credit.

---

## 🌟 Support

If you find this project helpful:
- ⭐ Star this repo  
- 🪄 Fork it for your version  
- 📝 Suggest new features or raise issues  

---

## 🔗 Repository

📍 **GitHub Link:** [Mapping Unsafe Drinking Water Zones](https://github.com/kjuhi-18/Mapping-Unsafe-Drinking-Water-Zones)

---

### ❤️ Closing Note

> “Technology is best when it brings people together.” — *Matt Mullenweg*  
>
> We believe data-driven awareness is the first step toward ensuring every drop of water is **pure, safe, and sustainable**.
