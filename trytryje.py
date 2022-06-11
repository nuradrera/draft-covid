import streamlit as st 
import numpy as np 
import pandas as pd

#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

st.title('Simple COVID-19 Prediction App')
st.write('DISCLAIMER: Please do not refer this as your primary indicator to predict heart disease due to very low precision rate (only 40%).')
st.write('Please refer to your qualified physician for more information. Use this at your own risk.')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

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

data = pd.read_csv('https://raw.githubusercontent.com/nuradrera/covid-19/main/Covid%20Dataset.csv')

X = data.drop('COVID-19', axis=1)
y = data['COVID-19']

X = X.apply(LabelEncoder().fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1234,test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
ypred = nb.predict(X_test)

def prediction(BreathingProblem, Fever, DryCough, SoreThroat, RunningNose, Asthma, ChronicLungDisease, Headache, HeartDisease, Diabetes, HyperTension, Fatigue, Gastrointestinal, AbroadTravel, ContactWithCOVIDPatient, AttendedLargeGathering, VisitedPublicExposedPlaces, FamilyWorkingInPublicExposedPlaces, WearingMasks, SanitizationFromMarket):
    data2 = pd.DataFrame(columns = ['Breathing Problem','Fever,Dry Cough','Sore throat','Running Nose','Asthma','Chronic Lung Disease','Headache','Heart Disease','Diabetes','Hyper Tension','Fatigue','Gastrointestinal','Abroad travel','Contact with COVID Patient','Attended Large Gathering','Visited Public Exposed Places','Family working in Public Exposed Places','Wearing Masks','Sanitization from Market'])
    data2 = data2.append({'BreathingProblem': F1,
        'Fever': F2,
        'DryCough': F3,
        'SoreThroat': F4,
        'RunningNose': F5,
        'Asthma': F6,
        'ChronicLungDisease': F7,
        'Headache': F8,
        'HeartDisease': F9,
        'Diabetes': F10,
        'HyperTension': F11,
        'Fatigue': F12,
        'Gastrointestinal': F13,
        'AbroadTravel': F14,
        'ContactWithCOVIDPatient': F15,
        'AttendedLargeGathering': F16,
        'VisitedPublicExposedPlaces': F17,
        'FamilyWorkingInPublicExposedPlaces': F18,
        'WearingMasks': F19,
        'SanitizationFromMarket': F20}, ignore_index = True)
    ypred = nb.predict(data2)
    st.write('Your prediction to have Covid-19 is:')
  
prediction(F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20)
