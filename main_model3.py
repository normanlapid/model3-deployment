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


