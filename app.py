import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load files
# -------------------------

model = joblib.load("models/trained_model.pkl")
scaler = joblib.load("models/scaler.pkl")
results = joblib.load("models/results.pkl")

df = pd.read_csv("data/Titanic-Dataset.csv")

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")
st.write("A simple Machine Learning project made using Streamlit.")

# -------------------------
# Dataset
# -------------------------

st.header("Dataset Overview")

st.dataframe(df.head())

# -------------------------
# Statistics
# -------------------------

st.header("Statistical Summary")

st.write(df.describe())

# -------------------------
# Correlation Heatmap
# -------------------------

st.header("Correlation Heatmap")

numeric = df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

# -------------------------
# Prediction
# -------------------------

st.header("Predict Survival")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1,2,3])

    sex = st.selectbox("Sex", ["Female","Male"])

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )

    sibsp = st.number_input(
        "Siblings/Spouses",
        0,
        8,
        0
    )

with col2:

    parch = st.number_input(
        "Parents/Children",
        0,
        6,
        0
    )

    fare = st.number_input(
        "Fare",
        0.0,
        600.0,
        50.0
    )

    embarked = st.selectbox(
        "Embarked",
        ["C","Q","S"]
    )

family = sibsp + parch + 1

sex = 0 if sex=="Female" else 1

embarked = {
    "C":0,
    "Q":1,
    "S":2
}[embarked]

features = pd.DataFrame([[
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    family
]], columns=[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize"
])

scaled = scaler.transform(features)

if st.button("Predict"):

    pred = model.predict(scaled)[0]

    if pred == 1:
        st.success("Passenger is likely to Survive ✅")
    else:
        st.error("Passenger is unlikely to Survive ❌")

# -------------------------
# Performance
# -------------------------

st.header("Model Performance")

performance = pd.DataFrame(results).T

st.dataframe(performance)

st.bar_chart(performance["accuracy"])
