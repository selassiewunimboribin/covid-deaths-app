# train_model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

def build_and_fit_pipeline(csv_path):
    # 1. Load data
    covid_deaths = pd.read_csv(csv_path)

    # 2. Drop irrelevant columns
    columns_to_drop = [
        'new_cases_smoothed', 'icu_patients', 'icu_patients_per_million',
        'hosp_patients', 'hosp_patients_per_million',
        'weekly_icu_admissions', 'weekly_icu_admissions_per_million',
        'new_deaths_smoothed', 'new_cases_smoothed_per_million',
        'new_deaths_smoothed_per_million'
    ]
    covid_deaths = covid_deaths.drop(columns=columns_to_drop, errors='ignore')

    # 3. Drop rows with missing target and shuffle
    covid_deaths = covid_deaths.dropna(subset=['total_deaths']).sample(frac=1).reset_index(drop=True)

    # 4. Split into features and target
    x = covid_deaths.drop('total_deaths', axis=1).head(30000)
    y = covid_deaths['total_deaths'].head(30000)

    # 5. Define feature groups
    cat_features = ['iso_code', 'continent', 'location', 'date']
    cons_features = ['population']
    num_features = [
        'total_cases', 'new_cases', 'new_deaths',
        'total_cases_per_million', 'new_cases_per_million',
        'total_deaths_per_million', 'new_deaths_per_million',
        'reproduction_rate', 'weekly_hosp_admissions',
        'weekly_hosp_admissions_per_million', 'total_tests'
    ]

    # 6. Define transformers
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    cons_imputer = SimpleImputer(strategy='constant', fill_value=405285)
    num_imputer  = SimpleImputer(strategy='mean')

    # 7. ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer([
        ('cat',  cat_pipeline,  cat_features),
        ('cons', cons_imputer, cons_features),
        ('num',  num_imputer,  num_features),
    ])

    # 8. Full pipeline with model
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 9. Fit the pipeline
    pipeline.fit(x, y)
    print("✅ Pipeline fitted! Train R²:", pipeline.score(x, y))
    return pipeline

if __name__ == "__main__":
    # 📥 CSV file location
    csv_path = r"C:\Users\DELL\Downloads\datasets\covid_death.csv"
    pipeline = build_and_fit_pipeline(csv_path)

    # 💾 Save pipeline to file
    output_path = r"C:\Users\DELL\Desktop\New folder\projects\covid_deaths_pipeline.joblib"
    joblib.dump(pipeline, output_path)
    print(f"💾 Pipeline saved to: {output_path}")
