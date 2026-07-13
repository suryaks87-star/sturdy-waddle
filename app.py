import streamlit as st
import pandas as pd
import joblib

model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Bankruptcy Prediction")

industrial_risk = st.selectbox("Industrial Risk",[0,0.5,1],index=0)
management_risk = st.selectbox("Management Risk",[0,0.5,1],index=0)
financial_flexibility = st.selectbox("Financial Flexibility",[0,0.5,1],index=2)
credibility = st.selectbox("Credibility",[0,0.5,1],index=2)
competitiveness = st.selectbox("Competitiveness",[0,0.5,1],index=2)
operating_risk = st.selectbox("Operating Risk",[0,0.5,1],index=0)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "industrial_risk":[industrial_risk],
        "management_risk":[management_risk],
        "financial_flexibility":[financial_flexibility],
        "credibility":[credibility],
        "competitiveness":[competitiveness],
        "operating_risk":[operating_risk]
    })

    scaled = scaler.transform(input_df)

    pred = model.predict(scaled)[0]

    prob = model.predict_proba(scaled)[0]

    if pred==1:
        st.error("⚠️ Bankrupt")
    else:
        st.success("✅ Not Bankrupt")

    st.write("### Probability")

    st.write(f"Not Bankrupt : {prob[0]*100:.2f}%")
    st.write(f"Bankrupt : {prob[1]*100:.2f}%")
