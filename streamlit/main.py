import streamlit as st
import pandas as pd
import handlers.clean_igual as clean_igual
import joblib
from handlers.clean_igual import preprocessor
import streamlit as st
import numpy as np
from joblib import load


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load("streamlit/handlers/model/logisticmodel.joblib")

    return pipe


import os

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("""Prediccion de lluvia en Australia""")

    path = "./weatherAUS.csv"
    dataframe = pd.read_csv(path, usecols=range(1, 25))
    df_limpio = preprocessor.fit_transform(dataframe)

    #######################################################
    current_dir = os.path.dirname(os.path.abspath(__file__))

    PATH_CLAS = "/home/jester/Desktop/tpaa/AA1-TUIA-Petetta-Arce/streamlit/handlers/model/logisticmodel.joblib"
    # REG_PATH = 'vA REGRESION ACA'

    # pipeline_reg = joblib.load(REG_PATH)
    pipeline_clas = joblib.load(PATH_CLAS)
    feature_names = pipeline_clas.named_steps["imputer"].get_feature_names_out()
##################################################################3


with model_training:
    columnas_numericas = list(df_limpio.columns[:-1])
    st.header("Ajusta los sliders y vas a recibir una prediccion")
    features = [
        st.slider(
            columna,
            df_limpio[columna].min(),
            df_limpio[columna].max(),
            round(df_limpio[columna].mean(), 2),
        )
        for columna in columnas_numericas
    ]
    raintoday_option_mapping = {"Sí": 1, "No": 0}
    raintoday_option = st.selectbox(
        "¿Hoy llovió?", list(raintoday_option_mapping.keys())
    )

all_features = features + [raintoday_option_mapping[raintoday_option]]

datos = pd.DataFrame([all_features], columns=feature_names)

pred_clasifiacion = pipeline_clas.predict(datos)


# Hacemos las predicciones
# pred_regresion = pipeline_reg.predict(datos)
pred_clasifiacion = pipeline_clas.predict(datos)

# Mostramos las predicciones en la app

resultado_clas = "SI" if pred_clasifiacion else "NO"
# resultado_reg  = round(float(pred_reg[0][0]), 2)

st.markdown(
    f"Mañana probablemente {resultado_clas} llueva, en caso de que llueva caeran aprox __resultado_reg__ mm/h de lluvia."
)
