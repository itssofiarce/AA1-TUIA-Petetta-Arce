import streamlit as st 
import numpy as np
import joblib as load
import pipeline

@st.cache_resource(show_spinner="Cargando el modelo...")
def load_model():
    pipe = load('streamlit/model.joblib')

    return pipe

if __name__ == "__main__":
    st.title("Modelo de regresión lineal para predecir la cantidad de lluvia del  día siguiente")
    
    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)

        with col1:
