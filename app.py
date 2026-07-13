import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("🏢 Bankruptcy Prediction System")
st.write("Select the company details below and click **Predict**.")

# -----------------------------
# Dropdown Options
# -----------------------------
options = {
    "Low (0)": 0,
    "Medium (0.5)": 0.5,
    "High (1)": 1
}

col1, col2 = st.columns(2)

with col1:
    industrial_risk = options[
        st.selectbox("Industrial Risk", list(options.keys()))
    ]

    financial_flexibility = options[
        st.selectbox("Financial Flexibility", list(options.keys()), index=2)
    ]

    competitiveness = options[
        st.selectbox("Competitiveness", list(options.keys()), index=2)
    ]

with col2:
    management_risk = options[
        st.selectbox("Management Risk", list(options.keys()))
    ]

    credibility = options[
        st.selectbox("Credibility", list(options.keys()), index=2)
    ]

    operating_risk = options[
        st.selectbox("Operating Risk", list(options.keys()))
    ]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Bankruptcy"):

    input_df = pd.DataFrame({
        "industrial_risk":[industrial_risk],
        "management_risk":[management_risk],
        "financial_flexibility":[financial_flexibility],
        "credibility":[credibility],
        "competitiveness":[competitiveness],
        "operating_risk":[operating_risk]
    })

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ **Prediction: Bankrupt Company**")
    else:
        st.success("✅ **Prediction: Non-Bankrupt Company**")

    # -----------------------------
    # Probability
    # -----------------------------
    st.subheader("Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Non-Bankrupt",
            f"{probability[0]*100:.2f}%"
        )

    with col2:
        st.metric(
            "Bankrupt",
            f"{probability[1]*100:.2f}%"
        )

    # -----------------------------
    # Pie Chart
    # -----------------------------
    fig, ax = plt.subplots(figsize=(5,5))

    labels = ["Non-Bankrupt", "Bankrupt"]
    colors = ["green", "red"]
    explode = (0.05,0.05)

    ax.pie(
        probability,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90
    )

    ax.axis("equal")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)

    # -----------------------------
    # Input Summary
    # -----------------------------
    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)
