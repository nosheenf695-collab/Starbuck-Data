import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "starbucks.csv"
MODEL_PATH = BASE_DIR / "starbucks_calories_classifier.pkl"


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("starbucks.csv was not found. Put it in the same folder as app.py.")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Repeat the cleaning steps used in the original notebook.
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
            try:
                df[col] = pd.to_numeric(cleaned)
            except (ValueError, TypeError):
                pass
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "starbucks_calories_classifier.pkl was not found. Put it in the same folder as app.py."
        )
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


st.set_page_config(
    page_title="Starbucks Calorie Classifier",
    page_icon="☕",
    layout="centered",
)

st.title("☕ Starbucks Calorie Classifier")
st.write(
    "Choose a Starbucks drink and its preparation. The trained Decision Tree "
    "Classifier will predict whether it belongs to the **Low Calorie** or "
    "**High Calorie** group."
)

try:
    df = load_dataset()
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

required_columns = {"Calories", "Beverage_category", "Beverage", "Beverage_prep"}
if not required_columns.issubset(df.columns):
    st.error("The CSV does not contain the columns expected by this app.")
    st.stop()

# Prepare the feature table in the same way as the notebook did.
X = pd.get_dummies(df.drop(columns=["Calories"]), drop_first=True)
expected_features = list(model.feature_names_in_)
X = X.reindex(columns=expected_features, fill_value=0)
X = X.fillna(X.median())

st.subheader("Select your drink")
categories = sorted(df["Beverage_category"].dropna().unique().tolist())
category = st.selectbox("Beverage category", categories)

category_df = df[df["Beverage_category"] == category]
beverages = sorted(category_df["Beverage"].dropna().unique().tolist())
beverage = st.selectbox("Beverage", beverages)

drink_df = category_df[category_df["Beverage"] == beverage]
preparations = sorted(drink_df["Beverage_prep"].dropna().unique().tolist())
preparation = st.selectbox("Preparation / size", preparations)

if st.button("Predict calorie group", type="primary", use_container_width=True):
    matching_rows = drink_df[drink_df["Beverage_prep"] == preparation]
    if matching_rows.empty:
        st.warning("No matching drink was found in the dataset.")
    else:
        row_index = matching_rows.index[0]
        input_features = X.loc[[row_index]]
        prediction = int(model.predict(input_features)[0])

        st.divider()
        if prediction == 1:
            st.error("Prediction: High Calorie")
        else:
            st.success("Prediction: Low Calorie")

        actual_calories = df.loc[row_index, "Calories"]
        st.metric("Calories in the dataset", f"{actual_calories:g} kcal")
        st.caption(
            "The calorie number is taken from the selected row in the supplied dataset. "
            "The model predicts the calorie group, not an exact calorie count."
        )

with st.expander("About this project"):
    st.write(
        "This project uses the Starbucks nutrition dataset and the Decision Tree "
        "Classifier saved from the supplied Jupyter Notebook. The notebook defines "
        "High Calorie as a drink with calories above the dataset median; otherwise "
        "the drink is Low Calorie."
    )
    st.write(f"Dataset rows: {len(df)}")
    st.write(f"Model input features: {len(expected_features)}")

st.caption("For learning and demonstration only. This is not dietary or medical advice.")
