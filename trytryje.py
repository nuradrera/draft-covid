import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.write("""
# Simple COVID-19 Prediction App
This app predicts whether you are infected with Covid-19 or not based on your symptoms!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    F1 = st.sidebar.selectbox('Do you have Breathing Problem?',['Yes','No'])
    F2 = st.sidebar.selectbox('Do you have Fever?',['Yes','No'])
    F3 = st.sidebar.selectbox('Do you have Dry Cough?',['Yes','No'])
    F4 = st.sidebar.selectbox('Do you have Sore throat?',['Yes','No'])
    F5 = st.sidebar.selectbox('Do you have Running Nose?',['Yes','No'])
    F6 = st.sidebar.selectbox('Do you have Asthma?',['Yes','No'])
    F7 = st.sidebar.selectbox('Do you have Chronic Lung Disease?',['Yes','No'])
    F8 = st.sidebar.selectbox('Do you have Headache?',['Yes','No'])
    F9 = st.sidebar.selectbox('Do you have Heart Disease?',['Yes','No'])
    F10 = st.sidebar.selectbox('Do you have Diabetes?',['Yes','No'])
    F11 = st.sidebar.selectbox('Do you have Hyper Tension?',['Yes','No'])
    F12 = st.sidebar.selectbox('Do you have Fatigue?',['Yes','No'])
    F13 = st.sidebar.selectbox('Do you have Gastrointestinal?',['Yes','No'])
    F14 = st.sidebar.selectbox('Do you have Abroad travel?',['Yes','No'])
    F15 = st.sidebar.selectbox('Do you have Contact with COVID Patient?',['Yes','No'])
    F16 = st.sidebar.selectbox('Do you have Attended Large Gathering?',['Yes','No'])
    F17 = st.sidebar.selectbox('Do you have Visited Public Exposed Places?',['Yes','No'])
    F18 = st.sidebar.selectbox('Do you have Family working in Public Exposed Places?',['Yes','No'])
    F19 = st.sidebar.selectbox('Do you Wearing Masks?',['Yes','No'])
    F20 = st.sidebar.selectbox('Do you Sanitization from Market?',['Yes','No'])
    data = {'Breathing Problem': F1,
        'Fever': F2,
        'Dry Cough': F3,
        'Sore throat': F4,
        'Running Nose': F5,
        'Asthma': F6,
        'Chronic Lung Disease': F7,
        'Headache': F8,
        'Heart Disease': F9,
        'Diabetes': F10,
        'Hyper Tension': F11,
        'Fatigue': F12,
        'Gastrointestinal': F13,
        'Abroad travel': F14,
        'Contact with COVID Patient': F15,
        'Attended Large Gathering': F16,
        'Visited Public Exposed Places': F17,
        'Family working in Public Exposed Places': F18,
        'Wearing Masks': F19,
        'Sanitization from Market': F20}
    features = pd.DataFrame(data, index=[0])
    return features

data = pd.read_csv('https://raw.githubusercontent.com/nuradrera/covid-19/main/Covid%20Dataset.csv')

df = user_input_features()

st.subheader('User Input parameters')
st.write(df.T)

X = data.drop('COVID-19', axis=1)
Y = data['COVID-19']

X = X.apply(LabelEncoder().fit_transform)

clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(df)

st.subheader('Prediction')
st.write(prediction)

# prediction(F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20)
