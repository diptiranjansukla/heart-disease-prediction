import streamlit as st
import pickle
import numpy as np

with open('standardscaler.pkl', 'rb') as fl:
    scale = pickle.load(fl)

with open('RandomForest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

st.title('Heartdisease prediction ')
def main():
    age = st.number_input('Age',0,100)
    
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex_to_binary = {'Male': 1, 'Female': 0}
    mapped_sex = sex_to_binary[sex]

    currentSmoker= st.selectbox('Currentsmoker', ['yes', 'no'])
    smoke_to_binary = {'yes': 1, 'no': 0}
    mapped_smoker = smoke_to_binary[currentSmoker]

    cigsPerDay	= st.number_input('cigarates per day',0,100)

    BPMeds = st.selectbox('BP medications',['yes','no'])
    BPMeds_to_binary = {'yes': 1.0, 'no': 0.0}
    mapped_BPMeds = BPMeds_to_binary[BPMeds]

    prevalentStroke = st.selectbox('prevalent stroke',['yes','no'])
    stroke_to_binary = {'yes': 1, 'no': 0}
    mapped_stroke = stroke_to_binary[prevalentStroke]

    prevalentHyp = st.selectbox('prevalent hypertension stroke',['yes','no'])
    hypertension_to_binary = {'yes': 1, 'no': 0}
    mapped_hyp = hypertension_to_binary[prevalentHyp]

    diabetes = st.selectbox('diabetes',['yes','no'])
    diabetes_to_binary = {'yes': 1, 'no': 0}
    mapped_diabetes = diabetes_to_binary[diabetes]

    totChol = st.number_input('total colestrole',0,300)

    sysBP = st.number_input('systolic BP',0 ,300)

    diaBP = st.number_input('diastolic BP',0,300)

    BMI = st.number_input('Body mass index',0,300)

    heartRate = st.number_input('Heart rate',0,300)

    glucose = st.number_input('glucose level',0,300)

    test = np.array([ mapped_sex, age, mapped_smoker, cigsPerDay, mapped_BPMeds, mapped_stroke, mapped_hyp, mapped_diabetes, totChol, sysBP,diaBP, BMI, heartRate, glucose])
    test = test.reshape(1,14)
    test = scale.transform(test)
    # st.button('predict')
    if st.button('predict',key="prediction_button"):
        pred = loaded_model.predict(test)
        if pred == 1:
            st.success('Heart disease detected')
            
        else:
            st.success('No heart disease detected')
            st.balloons()

if __name__ == '__main__':
    main()