🧮 COVID-19 Total Deaths Predictor

This is a Streamlit web application that uses a machine learning model to predict the total cumulative COVID-19 deaths in a country or region based on outbreak data such as cases, tests, and hospital admissions.

📌 Features

Easy-to-use web interface built with Streamlit

Predicts total deaths based on current stats

Uses a trained Random Forest Regressor pipeline

Can be deployed locally or on Streamlit Cloud

🚀 Try it Online

You can use the app here:
👉 Streamlit Cloud App Link 

📂 Project Structure

covid-deaths-app/
├── app.py                      # Streamlit app UI
├── pipeline.py                # Code to build and train pipeline
├── covid_deaths_pipeline.joblib  # Trained pipeline model
├── covid_death.csv            # COVID dataset (optional for training)
├── requirements.txt           # Python dependencies
└── README.md                  # Project info

🧪 How to Run Locally

Clone the repo

git clone https://github.com/selassiewunimboribin/covid-deaths-app.git
cd covid-deaths-app

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run app.py

🛠️ Requirements

Python 3.8+

Streamlit

Pandas

Scikit-learn

joblib

Install everything via:

pip install -r requirements.txt

📊 Dataset

The model was trained using a COVID-19 dataset with features such as:

Total cases

New cases

Total deaths per million

Reproduction rate

Hospital admissions

Note: The training file covid_death.csv is required only if retraining the model.

📫 Contact

For questions or collaboration:

Author: Selassie Wunimboribin

GitHub: @selassiewunimboribin

🌍 License

This project is open-source under the MIT License.