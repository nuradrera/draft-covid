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
    BP = st.sidebar.selectbox('Do you have Breathing Problem?',['Yes','No'])
    FVR = st.sidebar.selectbox('Do you have Fever?',['Yes','No'])
    DC = st.sidebar.selectbox('Do you have Dry Cough?',['Yes','No'])
    ST = st.sidebar.selectbox('Do you have Sore throat?',['Yes','No'])
    RN = st.sidebar.selectbox('Do you have Running Nose?',['Yes','No'])
    A = st.sidebar.selectbox('Do you have Asthma?',['Yes','No'])
    CLD = st.sidebar.selectbox('Do you have Chronic Lung Disease?',['Yes','No'])
    H = st.sidebar.selectbox('Do you have Headache?',['Yes','No'])
    HD = st.sidebar.selectbox('Do you have Heart Disease?',['Yes','No'])
    D = st.sidebar.selectbox('Do you have Diabetes?',['Yes','No'])
    HT = st.sidebar.selectbox('Do you have Hyper Tension?',['Yes','No'])
    FTG = st.sidebar.selectbox('Do you have Fatigue?',['Yes','No'])
    G = st.sidebar.selectbox('Do you have Gastrointestinal?',['Yes','No'])
    AT = st.sidebar.selectbox('Do you have Abroad travel?',['Yes','No'])
    CWCP = st.sidebar.selectbox('Do you have Contact with COVID Patient?',['Yes','No'])
    ALG = st.sidebar.selectbox('Do you have Attended Large Gathering?',['Yes','No'])
    VPEP = st.sidebar.selectbox('Do you have Visited Public Exposed Places?',['Yes','No'])
    FWIPEP = st.sidebar.selectbox('Do you have Family working in Public Exposed Places?',['Yes','No'])
    WM = st.sidebar.selectbox('Do you Wearing Masks?',['Yes','No'])
    SFM = st.sidebar.selectbox('Do you Sanitization from Market?',['Yes','No'])
    data = {'Breathing Problem': BP,
        'Fever': FVR,
        'Dry Cough': DC,
        'Sore throat': ST,
        'Running Nose': RN,
        'Asthma': A,
        'Chronic Lung Disease': CLD,
        'Headache': H,
        'Heart Disease': HD,
        'Diabetes': D,
        'Hyper Tension': HT,
        'Fatigue': FTG,
        'Gastrointestinal': G,
        'Abroad travel': AT,
        'Contact with COVID Patient': CWCP,
        'Attended Large Gathering': ALG,
        'Visited Public Exposed Places': VPEP,
        'Family working in Public Exposed Places': FWIPEP,
        'Wearing Masks': WM,
        'Sanitization from Market': SFM}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df.T)

data = pd.read_csv('https://raw.githubusercontent.com/nuradrera/covid-19/main/Covid%20Dataset.csv')
X = data.drop('COVID-19', axis=1)
Y = data['COVID-19']

X = X.apply(LabelEncoder().fit_transform)

clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(df)

st.subheader('Prediction')
# st.write(iris.target_names[prediction])
st.write(prediction)
