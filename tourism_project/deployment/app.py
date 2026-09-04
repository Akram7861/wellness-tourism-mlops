from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "tourism_project" / "deployment" / "model.joblib"
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="🌴"
)

st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to estimate the probability "
    "of purchasing the wellness tourism package."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()
features = list(model.feature_names_in_)

values = {}

with st.form("prediction_form"):
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(series):
                default = float(series.median())

                if pd.api.types.is_integer_dtype(df[col].dropna()):
                    values[col] = st.number_input(
                        col,
                        value=int(default),
                        step=1
                    )
                else:
                    values[col] = st.number_input(
                        col,
                        value=default
                    )
            else:
                values[col] = st.number_input(col, value=0.0)
        else:
            options = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )
            values[col] = st.selectbox(col, options)

    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    input_df = pd.DataFrame([values], columns=features)

    probability = float(
        model.predict_proba(input_df)[:, 1][0]
    )

    prediction = probability >= 0.5

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )

    if prediction:
        st.success(
            "Customer is likely to purchase the package."
        )
    else:
        st.info(
            "Customer is unlikely to purchase the package."
        )
