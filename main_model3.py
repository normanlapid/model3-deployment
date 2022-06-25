import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import shap
import numpy as np
import pickle


model_filename = 'model_3.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

scaler_filename = 'robustscaler_3.sav'
scaler = pickle.load(open(scaler_filename, 'rb'))

st.header("Model 3: Predicting Final GWA for Business Students")
