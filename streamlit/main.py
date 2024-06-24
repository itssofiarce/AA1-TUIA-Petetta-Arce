import streamlit as st
import pandas as pd
import handlers.clean_igual as clean_igual
import joblib
from handlers.clean_igual import preprocessor
import streamlit as st
import numpy as np


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load('streamlit/handlers/model/logisticmodel.joblib')

    return pipe
import os

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("""Prediccion de lluvia en Australia""")
    path = "weatherAUS.csv"
    dataframe = pd.read_csv(path, usecols=range(1, 25))
    df_limpio = preprocessor.fit_transform(dataframe)


    #######################################################
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the joblib file
    #PATH_CLAS = '/home/jester/Desktop/tpaa/AA1-TUIA-Petetta-Arce/streamlit/handlers/model/logisticmodel.joblib'


    PATH_CLAS = 'handlers/model/logisticmodel.joblib'
    pipeline_clas = joblib.load(PATH_CLAS)
    feature_names = pipeline_clas.named_steps['imputer'].get_feature_names_out()
##################################################################3



with model_training:
    columnas_numericas = list(
        df_limpio.columns[:-1]
    )  # acomodar esto asi no uso raintomorrow
    st.header("Ajusta los parametros para que el modelo prediga")
    features = [
        st.slider(
            columna,
            df_limpio[columna].min(),  # esto o queda asi o se ajusta
            df_limpio[columna].max(),  # asi no muestra valores normalizados
            round(df_limpio[columna].mean(), 2),
        )  # medio raro tener humedad negativa
        for columna in columnas_numericas
    ]
    raintoday_option_mapping = {'Sí': 1, 'No': 0}
    raintoday_option = st.selectbox('¿Hoy llovió?',
                                list(raintoday_option_mapping.keys()))

all_features = features + [raintoday_option_mapping[raintoday_option]]
datos = pd.DataFrame([all_features],
                                  columns=feature_names)

pred_clas = pipeline_clas.predict(datos)

resultado_clasificacion = 'Si' if pred_clas else 'NO'
#resultado_regresion

st.markdown(f'Mañana probablemente {resultado_clasificacion} llueva')

### ----- Modelo de Regresión Lineal ----- ###

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load('streamlit/handlers/model/linealmodel.joblib')

    return pipe

header_lineal = st.container()
dataset_lineal = st.container()
features_lineal = st.container()
model_training_lineal = st.container()

with header_lineal:
    st.title("""Prediccion cantidad de lluvia para el día siguiente en Australia""")

    #######################################################
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the joblib file
    #PATH_REGR = '/Users/Lenovo/Documents/Sofia/Sofia/Programacion/AI/SEGUNDO AÑO/AA 1/AA1-TUIA-Petetta-Arce/streamlit/handlers/model/linealmodel.joblib'

    PATH_REGR = './handlers/model/linealmodel.joblib'
    pipeline_reg = joblib.load(PATH_REGR)
    feature_names = pipeline_reg.named_steps['imputer'].get_feature_names_out()
##################################################################3

with model_training_lineal:
    columnas_numericas = list(
        df_limpio.columns[:-1]
    )  # acomodar esto asi no uso raintomorrow
    st.header("Ajusta los parametros para que el modelo prediga")
    features = [
        st.slider(
            f"{columna}_{index}",
            df_limpio[columna].min(),  # esto o queda asi o se ajusta
            df_limpio[columna].max(),  # asi no muestra valores normalizados
            round(df_limpio[columna].mean(), 2),
        )  # medio raro tener humedad negativa
        for index, columna in enumerate(columnas_numericas)
    ]
    raintoday_option_mapping = {'Sí': 1, 'No': 0}
    raintoday_option = st.selectbox('¿Hoy llovió?',
                                list(raintoday_option_mapping.keys()), key="Llovió")

all_features = features + [raintoday_option_mapping[raintoday_option]]
datos = pd.DataFrame([all_features],
                                  columns=feature_names)

pred_clas = pipeline_reg.predict(datos)
#resultado_regresion
if pred_clas > 0:
    st.markdown(f'Mañana probablemente llueva {pred_clas[0]} cm3')
else :
    st.markdown(f'Mañana probablemente llueva 0 cm3')