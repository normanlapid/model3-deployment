import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler


model_filename = 'model_3.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

scaler_filename = 'robustscaler_3.sav'
scaler = pickle.load(open(scaler_filename, 'rb'))

explainer_filename = 'shap_explainer_gb.sav'
explainer_gb = pickle.load(open(explainer_filename, 'rb'))

st.header("Model 3: Predicting Final GWA for Business Students")

st.subheader("Please input relevant features for the student:")

st.text("Student Grades:")

input_eng1 = st.number_input("ENG 1 (PC)", min_value=1.00, max_value=5.00, step=0.01)
input_math1 = st.number_input("MATH 1 (MMW)", min_value=1.00, max_value=5.00, step=0.01)
input_cw1 = st.number_input("CW 1", min_value=1.00, max_value=5.00, step=0.01)
input_ba1 = st.number_input("BA 1 (HRM)", min_value=1.00, max_value=5.00, step=0.01)
input_cbmec1 = st.number_input("CBMEC 1", min_value=1.00, max_value=5.00, step=0.01)
input_cbmec2 = st.number_input("CBMEC 2", min_value=1.00, max_value=5.00, step=0.01)
input_eco1 = st.number_input("ECO 1 (BM)", min_value=1.00, max_value=5.00, step=0.01)
input_ba2 = st.number_input("BA 2 (MA)", min_value=1.00, max_value=5.00, step=0.01)


input_course = st.selectbox("What is the student's course?",
                            ('BSBA FM (NEW)', 'BSBA MM (NEW)', 'BSBA MM', 
                             'BSBA HRM (NEW)', 'BSBA HRM', 'BSOA'))

course_BSBA_HRM = 0
course_BSBA_HRM_new = 0
course_BSBA_MM = 0
course_BSBA_MM_new = 0
course_BSOA = 0

if input_course == 'BSBA MM (NEW)':
    course_BSBA_MM_new = 1
elif input_course == 'BSBA MM':
    course_BSBA_MM = 1
elif input_course == 'BSBA HRM (NEW)':
    course_BSBA_HRM_new = 1
elif input_course == 'BSBA HRM':
    course_BSBA_HRM = 1
elif input_course == 'BSOA':
    course_BSOA = 1

input_scholar = st.selectbox("What is the student's scholarship?",
                            ('None', 'Listahanan', 'Tulong-Dunong', 'ESGPPA'))

scholar_Listahanan = 0
scholar_Tulong_Dunong = 0

if input_scholar == 'Listahanan':
    scholar_Listahanan = 1
elif input_scholar == 'Tulong-Dunong':
    scholar_Tulong_Dunong = 1

d = {'ENG 1 (PC)': input_eng1,
     'MATH 1 (MMW)': input_math1, 
     'CW 1': input_cw1,
     'BA 1 (HRM)': input_ba1,
     'CBMEC 1': input_cbmec1,
     'CBMEC 2': input_cbmec2,
     'ECO 1 (BM)': input_eco1,
     'BA 2 (MA)': input_ba2,
     'course_BSBA HRM': course_BSBA_HRM,
     'course_BSBA HRM (NEW)': course_BSBA_HRM_new,
     'course_BSBA MM': course_BSBA_MM,
     'course_BSBA MM (NEW)': course_BSBA_MM_new,
     'course_BSOA': course_BSOA,
     'scholar_Listahanan': scholar_Listahanan,
     'scholar_Tulong Dunong': scholar_Tulong_Dunong}

X_test = pd.DataFrame(data=d, index=[0])
X_scaled = scaler.transform(X_test)

if st.button('Make Prediction'):
    prediction = loaded_model.predict(X_test)
    st.subheader(np.round(prediction[0], 3))
    feature_names = list(X_test.columns)
    shap_values_gb = explainer_gb.shap_values(X_scaled, check_additivity=False)
    display(shap.force_plot(explainer_gb.expected_value, 
                            shap_values_gb[0], feature_names, matplotlib=True))