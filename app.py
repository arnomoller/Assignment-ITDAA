import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved SVM model
model_path = "best_model.pkl"
model = joblib.load(model_path)

# Function to predict heart disease
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = {
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    }
    input_df = pd.DataFrame(input_data)

    # Preprocess input data
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Convert categorical variables to one-hot encoding
    input_df[categorical_features] = input_df[categorical_features].astype('category')
    input_df = pd.get_dummies(input_df)

    # Ensure all categorical columns present after one-hot encoding
    expected_columns = set(X_train.columns)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Heart Disease Prediction App')
    st.write('Enter patient details to predict if the patient has heart disease.')

    # Input fields
    age = st.slider('Age', 18, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.slider('Number of Major Vessels Colored by Flouroscopy', 0, 4, 0)
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Convert categorical inputs to numerical
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0
T
    if cp == 'Typical Angina':
        cp = 0
    elif cp == 'Atypical Angina':
        cp = 1
    elif cp == 'Non-anginal Pain':
        cp = 2
    else:
        cp = 3

    if restecg == 'Normal':
        restecg = 0
    elif restecg == 'ST-T wave abnormality':
        restecg = 1
    else:
        restecg = 2

    if slope == 'Upsloping':
        slope = 0
    elif slope == 'Flat':
        slope = 1
    else:
        slope = 2

    if thal == 'Normal':
        thal = 0
    elif thal == 'Fixed Defect':
        thal = 1
    else:
        thal = 2

    # Predict button
    if st.button('Predict'):
        result = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        if result == 1:
            st.error('Patient likely has heart disease. Recommend further evaluation.')
        else:
            st.success('Patient is likely healthy.')

if __name__ == '__main__':
    main()
