import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

# ---------- Load data + trained artifacts ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/Titanic-Dataset.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_sex = joblib.load("models/le_sex.pkl")
    le_emb = joblib.load("models/le_emb.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    results = joblib.load("models/results.pkl")
    return model, scaler, le_sex, le_emb, feature_columns, results

df = load_data()
model, scaler, le_sex, le_emb, feature_columns, results = load_model()

st.title("🚢 Titanic Survival Prediction System")
st.write(
    "A small ML app built for the Week 3 internship task. It trains a couple of "
    "classifiers on the Titanic dataset and lets you plug in a passenger's details "
    "to see whether the model thinks they would have survived."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset Overview", "📈 Model Performance", "🔮 Make a Prediction", "🧮 Visualizations"]
)

# ---------------- Tab 1: Dataset overview ----------------
with tab1:
    st.subheader("A first look at the data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Survived", int(df["Survived"].sum()))
    col4.metric("Did not survive", int((df["Survived"] == 0).sum()))

    st.markdown("**Sample rows**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Statistical summary**")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("**Missing values per column**")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        st.bar_chart(missing)
    else:
        st.write("No missing values.")

# ---------------- Tab 2: Model performance ----------------
with tab2:
    st.subheader("How the models performed on the held-out test set")

    results_df = pd.DataFrame(results).T
    results_df = results_df.rename(
        columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-Score"}
    )
    st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)

    best_model_name = results_df["Accuracy"].idxmax()
    st.success(f"Best performing model: **{best_model_name}** "
               f"(accuracy: {results_df.loc[best_model_name, 'Accuracy']:.3f}) — this is the one used for predictions below.")

    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy / Precision / Recall / F1 by model")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# ---------------- Tab 3: Prediction ----------------
with tab3:
    st.subheader("Enter passenger details")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                                   help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 80, 28)

        with c2:
            sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0)
            parch = st.number_input("Parents / Children aboard", min_value=0, max_value=6, value=0)
            fare = st.number_input("Fare paid ($)", min_value=0.0, max_value=550.0, value=32.0, step=1.0)

        with c3:
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                                     help="S = Southampton, C = Cherbourg, Q = Queenstown")

        submitted = st.form_submit_button("Predict")

    if submitted:
        family_size = sibsp + parch + 1

        input_dict = {
            "Pclass": pclass,
            "Sex": le_sex.transform([sex])[0],
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": le_emb.transform([embarked])[0],
            "FamilySize": family_size,
        }
        input_df = pd.DataFrame([input_dict])[feature_columns]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

        st.divider()
        if prediction == 1:
            st.success("### 🟢 Prediction: This passenger would likely have **SURVIVED**")
        else:
            st.error("### 🔴 Prediction: This passenger would likely **NOT have survived**")

        if proba is not None:
            st.write(f"Confidence — Survived: **{proba[1]*100:.1f}%**, Did not survive: **{proba[0]*100:.1f}%**")
            st.progress(float(proba[1]))

# ---------------- Tab 4: Visualizations ----------------
with tab4:
    st.subheader("A few exploratory visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Survival rate by passenger class**")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="Pclass", y="Survived", ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Survival rate by sex**")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="Sex", y="Survived", ax=ax)
        st.pyplot(fig)

    st.markdown("**Correlation heatmap**")
    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.divider()
st.caption("Week 3 ML internship task — Titanic dataset, scikit-learn, Streamlit.")
