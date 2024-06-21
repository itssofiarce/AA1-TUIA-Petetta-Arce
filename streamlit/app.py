# Puesta en producción del modelo de Regresión Lógistica con HiperParametros optimizados
import streamlit as st
import numpy as np
from joblib import load



@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load('model/model.joblib')

    return pipe

