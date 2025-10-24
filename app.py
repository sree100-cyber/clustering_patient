import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model_artifacts()


st.title("Heart Disease Prediction App")

st.write("""
This app predicts the **likelihood of heart disease** based on patient data.
Enter the details below and click **Predict**.
""")


age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["male", "female"])
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking = st.selectbox("Smoking", ["yes", "no"])
diabetes = st.selectbox("Diabetes", ["yes", "no"])
chest_pain_type = st.selectbox("Chest Pain Type", ["typical", "atypical", "non-anginal", "asymptomatic"])


input_dict = {
    "age": [age],
    "sex": [sex],
    "cholesterol": [cholesterol],
    "blood_pressure": [blood_pressure],
    "bmi": [bmi],
    "smoking": [smoking],
    "diabetes": [diabetes],
    "chest_pain_type": [chest_pain_type],
}

input_df = pd.DataFrame(input_dict)


input_encoded = pd.get_dummies(input_df)


input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)


try:
    numeric_cols = scaler.feature_names_in_
except AttributeError:
    numeric_cols = [col for col in feature_columns if any(
        word in col.lower() for word in ['age', 'bmi', 'chol', 'pressure']
    )]

cols_to_scale = [col for col in numeric_cols if col in input_encoded.columns]

if len(cols_to_scale) > 0:
    input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])


if st.button("🔍 Predict"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {probability:.2f})")


st.markdown("---")
st.caption("Developed using Streamlit and scikit-learn.")
