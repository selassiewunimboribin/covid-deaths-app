import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ---------------- Load trained pipeline ----------------
@st.cache_resource
def load_pipeline():
    return joblib.load("covid_deaths_pipeline.joblib")

model = load_pipeline()

# ---------------- Page header ----------------
st.set_page_config(page_title="COVID-19 Death Predictor", layout="centered")
st.title("🧮 COVID-19 Cumulative Death Predictor")
st.markdown("""
This app uses a machine learning model to predict the **total COVID-19 deaths**
in a country or region based on current outbreak data.
""")

# ---------------- User inputs ----------------
continent = st.selectbox(
    "Continent",
    ["Africa", "Asia", "Europe", "North America", "South America", "Oceania", "missing"]
)
location = st.text_input("Country / Region", "United States")
the_date = st.date_input("Date", date(2021, 10, 1))

col1, col2 = st.columns(2)
with col1:
    population = st.number_input("Population", 0, value=331_000_000)
    total_cases = st.number_input("Total COVID Cases", 0, value=40_000_000)
    new_cases = st.number_input("New Cases Today", 0, value=100_000)
    total_tests = st.number_input("Total Tests", 0, value=550_000_000)
with col2:
    new_deaths = st.number_input("New Deaths Today", 0, value=1500)
    total_cases_pm = st.number_input("Total Cases per Million", 0.0, value=120_000.0)
    new_cases_pm = st.number_input("New Cases per Million", 0.0, value=300.0)
    total_deaths_pm = st.number_input("Total Deaths per Million", 0.0, value=2000.0)
    new_deaths_pm = st.number_input("New Deaths per Million", 0.0, value=5.0)

reproduction_rate = st.number_input("Reproduction Rate (R)", 0.0, value=1.1)
weekly_hosp_adm = st.number_input("Weekly Hospital Admissions", 0.0)
weekly_hosp_adm_per_million = st.number_input("Weekly Hospital Admissions (per million)", 0.0)

# ---------------- Predict button ----------------
if st.button("Predict"):
    row = pd.DataFrame([{
        "continent": continent,
        "location": location,
        "date": the_date.strftime("%Y-%m-%d"),
        "population": float(population),
        "total_cases": float(total_cases),
        "new_cases": float(new_cases),
        "new_deaths": float(new_deaths),
        "total_cases_per_million": float(total_cases_pm),
        "new_cases_per_million": float(new_cases_pm),
        "total_deaths_per_million": float(total_deaths_pm),
        "new_deaths_per_million": float(new_deaths_pm),
        "reproduction_rate": float(reproduction_rate),
        "weekly_hosp_admissions": float(weekly_hosp_adm),
        "weekly_hosp_admissions_per_million": float(weekly_hosp_adm_per_million),
        "total_tests": float(total_tests)
    }])

    try:
        pred = model.predict(row)[0]
        st.success(f"Predicted cumulative deaths: **{pred:,.0f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with LOVE by Selassie Wunimboribin | Powered by Streamlit")


