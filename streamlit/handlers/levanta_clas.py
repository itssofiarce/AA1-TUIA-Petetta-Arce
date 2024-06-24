# Puesta en producción del modelo 
import streamlit as st
import numpy as np
from joblib import load



@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load('streamlit/handlers/model/logisticmodel.joblib')

    return pipe