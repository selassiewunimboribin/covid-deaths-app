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
st.set_page_config(page_title="COVID-19 Total-Deaths Predictor")
st.title("🦠 COVID-19 Total Deaths Predictor")
st.markdown(
    "Enter the current situation for a country/region and predict the "
    "**cumulative total deaths** estimated by the model."
)

# ---------------- User inputs ----------------
iso_code = st.text_input("ISO Code (e.g. USA, NGA, IND)", "USA")
continent = st.selectbox(
    "Continent",
    ["Africa", "Asia", "Europe", "North America", "South America", "Oceania", "missing"]
)
location  = st.text_input("Country / Region", "United States")
the_date  = st.date_input("Date", date(2021, 10, 1))

col1, col2 = st.columns(2)
with col1:
    population   = st.number_input("Population",                0, value=331_000_000)
    total_cases  = st.number_input("Total Cases",               0, value=40_000_000)
    new_cases    = st.number_input("New Cases (that day)",      0, value=100_000)
    total_tests  = st.number_input("Total Tests",               0, value=550_000_000)
with col2:
    new_deaths        = st.number_input("New Deaths (that day)",        0, value=1500)
    total_cases_pm    = st.number_input("Total Cases / Million",        0.0, value=120_000.0)
    new_cases_pm      = st.number_input("New Cases  / Million",         0.0, value=300.0)
    total_deaths_pm   = st.number_input("Total Deaths / Million",       0.0, value=2000.0)
    new_deaths_pm     = st.number_input("New Deaths / Million",         0.0, value=5.0)

reproduction_rate           = st.number_input("Reproduction Rate (R)",      0.0, value=1.1)
weekly_hosp_adm             = st.number_input("Weekly Hosp. Admissions",    0.0)
weekly_hosp_adm_per_million = st.number_input("Weekly Hosp. Adm. / Million",0.0)

# ---------------- Predict button ----------------
if st.button("Predict"):
    row = pd.DataFrame([{
        "iso_code": iso_code,
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


