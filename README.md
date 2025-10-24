##  Heart Disease Prediction Web App
link : https://clusteringpatient-gvzfcioappppjtxnlgpvmupx.streamlit.app/
##  Overview

This project is a Machine Learning web application built using Streamlit that predicts whether a person is likely to have Heart Disease based on medical and lifestyle factors.

A Random Forest Classifier model is trained on the dataset, scaled using StandardScaler, and saved using pickle for reuse in the app.
The app allows users to input new patient data and get an instant prediction with probability scores.

## Project Workflow

Data Preprocessing

Handle missing values (mean/mode imputation)

Encode categorical variables (LabelEncoder)

Scale numeric features (StandardScaler)

Remove or cap outliers using the IQR method

Model Training (train_model.py)

Trains a Random Forest Classifier

Splits data into train/test sets

Saves the trained model (model.pkl) and scaler (scaler.pkl)

Web Application (streamlit_app.py)

Loads the trained model and scaler

Collects user input interactively

Displays prediction and confidence probability

## 📂 Folder Structure
heart-disease-prediction/
│
├── your_dataset.csv         # Dataset used for training
├── train_model.py           # Model training and saving script
├── streamlit_app.py         # Streamlit web app
├── model.pkl                # Saved trained model
├── scaler.pkl               # Saved StandardScaler
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

## 🧾 Dataset Information

The dataset contains medical and demographic information of patients.
Each row represents an individual, and the target variable is heart_disease (0 = No, 1 = Yes).

## Feature	Description
age	Age of the patient
gender	Male / Female
resting_bp	Resting blood pressure (mm Hg)
cholesterol	Serum cholesterol (mg/dl)
max_heart_rate	Maximum heart rate achieved
oldpeak	ST depression induced by exercise relative to rest
exercise_angina	Exercise-induced angina (Yes/No)
hypertension	Whether the patient has high blood pressure
residence_type	Urban / Rural
chest_pain_type	Type of chest pain (typical/atypical/asymptomatic)
smoking_status	Current smoker / Former / Never
heart_disease

## Usage

Enter numeric values such as age, resting blood pressure, cholesterol, max heart rate, oldpeak, etc.

For categorical features like gender or exercise_angina, select the appropriate option.

The app outputs:

Prediction: Likely or Unlikely to have Heart Disease

Probability: Confidence score of the prediction
