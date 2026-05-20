import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("📊 Employee Attrition Prediction")
st.markdown("Predict whether an employee is likely to leave the company.")

# =========================
# INPUT FIELDS
# =========================

age = st.slider("Age", 18, 60, 30)

income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=200000,
    value=5000
)

job_level = st.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5]
)

job_satisfaction = st.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4],
    help="1 = Low, 4 = Very High"
)

work_life_balance = st.selectbox(
    "Work Life Balance",
    [1, 2, 3, 4],
    help="1 = Bad, 4 = Excellent"
)

years_at_company = st.number_input(
    "Years At Company",
    min_value=0,
    max_value=40,
    value=1
)

overtime_option = st.selectbox(
    "OverTime",
    ["No", "Yes"]
)

# Convert Overtime
overtime = 1 if overtime_option == "Yes" else 0

# =========================
# FEATURE ORDER
# =========================
selected_features = [
    'Age',
    'MonthlyIncome',
    'JobLevel',
    'JobSatisfaction',
    'WorkLifeBalance',
    'YearsAtCompany',
    'OverTime'
]

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Attrition"):

    # Create DataFrame
    input_data = pd.DataFrame(
        [[
            age,
            income,
            job_level,
            job_satisfaction,
            work_life_balance,
            years_at_company,
            overtime
        ]],
        columns=selected_features
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("Prediction Result")

    # =========================
    # RESULT DISPLAY
    # =========================
    if prediction == 1:

        st.error(
            f"⚠️ Employee is likely to leave the company.\n\n"
            f"Confidence Level: {probability:.2f}%"
        )

        st.progress(int(probability))

        st.metric(
            label="Attrition Risk",
            value=f"{probability:.2f}%"
        )

    else:

        stay_probability = 100 - probability

        st.success(
            f"✅ Employee is likely to stay in the company.\n\n"
            f"Confidence Level: {stay_probability:.2f}%"
        )

        st.progress(int(stay_probability))

        st.metric(
            label="Retention Confidence",
            value=f"{stay_probability:.2f}%"
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")
