import streamlit as st
import pandas as pd
import joblib

diabetes_model = joblib.load('rf_model_diabetes-robust.pkl')  
heart_disease_model = joblib.load('xgb_model_heart-robust2.pkl') 
scaler_diabetes = joblib.load('robust-scaler-diabetes.pkl')
scaler_heart = joblib.load('robust-scaler-heart2.pkl')

st.title('Health Prediction App')

def user_input_features_shared():
    st.subheader("Basic Information")

    age_real = st.slider('Age (years)', 1, 100, 50)
    age_group = min(max(int(age_real // 7), 1), 13)

    gender = st.selectbox('Gender', ['Female', 'Male'])
    gender_encoded = 1 if gender == 'Male' else 0

    hypertension = st.selectbox('Hypertension / High Blood Pressure', ['No', 'Yes'])
    hypertension_encoded = 1 if hypertension == 'Yes' else 0

    bmi = st.slider('BMI', 10.0, 99.0, 25.0)

    smoking_history = st.selectbox('Smoking History', ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
    smoker_encoded = 1 if smoking_history in ['current', 'former', 'ever'] else 0

    return {
        'age_real': age_real,
        'age_group': age_group,
        'gender_encoded': gender_encoded,
        'hypertension_encoded': hypertension_encoded,
        'bmi': bmi,
        'smoker_encoded': smoker_encoded,
    }

def build_diabetes_input(shared_input):
    HbA1c_level = st.slider('HbA1c Level', 3.5, 9.0, 5.7)
    blood_glucose_level = st.slider('Blood Glucose Level', 80, 300, 140)

    heart_disease_status = st.selectbox("Heart disease", ['No', 'Yes'])
    heart_disease = {'No': 0, 'Yes': 1}[heart_disease_status]

    df = pd.DataFrame([{
        'gender': shared_input['gender_encoded'],
        'age': shared_input['age_real'],
        'hypertension': shared_input['hypertension_encoded'],
        'heart_disease': heart_disease,
        'smoking_history': shared_input['smoker_encoded'],
        'bmi': shared_input['bmi'],
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }])
    return df

def build_heart_input(shared_input):
    high_chol_status = st.selectbox("High Cholesterol", ['No', 'Yes'])
    high_chol = {'No': 0, 'Yes': 1}[high_chol_status]

    stroke_status = st.selectbox("Stroke", ['No', 'Yes'])
    stroke = {'No': 0, 'Yes': 1}[stroke_status]

    gen_health_label = st.select_slider(
        "General Health",
        options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        value="Good"
    )
    mapping = {
        "Excellent": 5,
        "Very Good": 4,
        "Good": 3,
        "Fair": 2,
        "Poor": 1
    }
    gen_health_for_model = 6 - mapping[gen_health_label]

    men_health = st.slider("Mental Health (days)", min_value=0, max_value=30, value=0)
    phys_health = st.slider("Physical Health (days)", min_value=0, max_value=30, value=0)
    diabetes_status = st.selectbox('Do you have Diabetes?', ['No', 'Yes', 'Borderline'])
    diabetes_encoded = {'No': 0, 'Yes': 1, 'Borderline': 2}[diabetes_status]

    return pd.DataFrame([{
        'Age': shared_input['age_group'],
        'Sex': shared_input['gender_encoded'],
        'HighChol': high_chol,
        'HighBP': shared_input['hypertension_encoded'],
        'BMI': shared_input['bmi'],
        'Smoker': shared_input['smoker_encoded'],
        'Stroke': stroke,
        'Diabetes': diabetes_encoded,
        'GenHealth': gen_health_for_model,
        'MenHealth': men_health,
        'PhysHealth': phys_health
    }])

shared_input = user_input_features_shared()

st.markdown("### Diabetes")
diabetes_input = build_diabetes_input(shared_input)
diabetes_scaled = scaler_diabetes.transform(diabetes_input[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
diabetes_input_scaled = pd.concat([pd.DataFrame(diabetes_scaled, columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']), diabetes_input.drop(columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])], axis=1)
diabetes_pred = diabetes_model.predict(diabetes_input_scaled)[0]


st.markdown("### Heart Disease")
heart_input = build_heart_input(shared_input)
heart_scaled = scaler_heart.transform(heart_input[['BMI']])
heart_input_scaled = pd.concat([pd.DataFrame(heart_scaled, columns=['BMI']), heart_input.drop(columns=['BMI'])], axis=1)
heart_pred = heart_disease_model.predict(heart_input_scaled)[0]

if st.button("Predict"):
    prediction = diabetes_model.predict(diabetes_input_scaled)[0]
    proba = diabetes_model.predict_proba(diabetes_input_scaled)[0][1]

    if proba <= 0.25:
        diabetes_risk = "Low Risk"
    elif proba <= 0.7:
        diabetes_risk = "Medium Risk"
    else:
        diabetes_risk = "High Risk"

    st.header("Diabetes Prediction") 
    st.success(f"Diabetes Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    st.info(f"Risk Level: **{diabetes_risk}**")

    st.markdown("---")

    heart_scaled = scaler_heart.transform(heart_input[['BMI']])
    heart_input_scaled = pd.concat([pd.DataFrame(heart_scaled, columns=['BMI']), heart_input.drop(columns=['BMI'])], axis=1)
    y_proba = heart_disease_model.predict_proba(heart_input_scaled)[0][1]
    threshold = 0.6
    prediction = int(y_proba > threshold)

    if y_proba <= 0.25:
        heart_risk = "Low Risk"
    elif y_proba <= 0.6:
        heart_risk = "Medium Risk"
    else:
        heart_risk = "High Risk"

    st.header("Heart Disease Prediction") 
    st.success(f"Heart Disease Prediction: {'Has Heart Disease' if prediction == 1 else 'No Heart Disease'}")
    st.info(f"Risk Level: **{heart_risk}**")
