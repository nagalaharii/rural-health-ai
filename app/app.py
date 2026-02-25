import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import json
from datetime import datetime

# ================= BASE DIR =================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ================= LOAD =================
model = joblib.load(os.path.join(BASE_DIR,"model","model.pkl"))
le = joblib.load(os.path.join(BASE_DIR,"model","label_encoder.pkl"))

train = pd.read_csv(os.path.join(BASE_DIR,"data","Training.csv"))
severity_df = pd.read_csv(os.path.join(BASE_DIR,"data","Symptom-severity.csv"))
desc = pd.read_csv(os.path.join(BASE_DIR,"data","symptom_Description.csv"))
prec = pd.read_csv(os.path.join(BASE_DIR,"data","symptom_precaution.csv"))

train = train.loc[:,~train.columns.str.contains("^Unnamed")]
symptoms = train.drop("prognosis",axis=1).columns.tolist()

# ================= UI =================
st.title("🩺 CDSS — Rural Patient Triage (ASHA Tool)")

st.warning("⚠️ Clinical decision support only. Not a doctor replacement.")

consent = st.checkbox("Patient consent obtained")

st.divider()

# ================= INPUT =================
st.subheader("Patient Symptoms")
selected = st.multiselect("Select Symptoms", symptoms)

st.subheader("Demographics & Risk Factors")
col1,col2 = st.columns(2)

with col1:
    age = st.slider("Age",1,100)
    bmi = st.slider("BMI",10.0,50.0)
    gender = st.selectbox("Gender",["Male","Female"])

with col2:
    htn = st.checkbox("Hypertension")
    diabetes = st.checkbox("Diabetes")
    smoking = st.checkbox("Smoking")

st.subheader("Vitals (optional)")
temp = st.slider("Temperature",95.0,105.0)
pulse = st.slider("Pulse",40,140)
bp = st.slider("Systolic BP",80,200)

# ================= INPUT VECTOR =================
x = [0]*len(symptoms)
for s in selected:
    x[symptoms.index(s)] = 1

# ================= ANALYZE =================
if st.button("Analyze Patient") and consent:

    # ---- Disease prediction ----
    probs = model.predict_proba([x])[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(le.inverse_transform([i])[0], probs[i]) for i in top3_idx]

    st.subheader("Top Predicted Diseases")
    for d,c in top3:
        st.write(f"• {d} — {round(c*100,1)}%")

    best_disease,confidence = top3[0]
    confidence = confidence*100

    # ---- Severity score ----
    sev = 0
    for s in selected:
        row = severity_df[severity_df["Symptom"]==s]
        if not row.empty:
            sev += int(row["weight"].values[0])

    sev_norm = min(100, sev*5)

    # ---- Risk score ----
    risk = 0
    if age>60: risk+=10
    if bmi>30: risk+=5
    if htn: risk+=7
    if diabetes: risk+=7
    if smoking: risk+=5

    risk = min(100,risk*2)

    # ---- Priority score ----
    priority = sev_norm*0.4 + risk*0.3 + confidence*0.3

    # ---- TRIAGE ----
    if priority>=75:
        triage="🔴 RED — Immediate referral required"
    elif priority>=40:
        triage="🟡 YELLOW — Visit clinic soon"
    else:
        triage="🟢 GREEN — Home care possible"

    # ---- DASHBOARD ----
    st.header("Patient Summary")

    colA,colB,colC = st.columns(3)
    colA.metric("Severity", round(sev_norm,1))
    colB.metric("Risk", round(risk,1))
    colC.metric("Priority", round(priority,1))

    st.subheader(triage)

    st.write(f"Predicted Disease: **{best_disease}** ({round(confidence,1)}%)")

    # ---- Explanation ----
    drow = desc[desc["Disease"]==best_disease]
    if not drow.empty:
        st.subheader("Explanation")
        st.write(drow["Description"].values[0])

    # ---- Precautions ----
    prow = prec[prec["Disease"]==best_disease]
    if not prow.empty:
        st.subheader("Recommended Precautions")
        for i in range(1,5):
            st.write("-",prow.iloc[0][f"Precaution_{i}"])

    # ---- Save patient history (BONUS FEATURE) ----
    history_file = os.path.join(BASE_DIR,"history.json")

    record = {
        "time":str(datetime.now()),
        "age":age,
        "gender":gender,
        "disease":best_disease,
        "priority":round(priority,1),
        "triage":triage
    }

    with open(history_file,"a") as f:
        f.write(json.dumps(record)+"\n")

    st.success("Patient record saved locally")