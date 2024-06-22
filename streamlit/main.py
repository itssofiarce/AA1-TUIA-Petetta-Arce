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
    pipe = load('streamlit/handlers/model/logisticmodel.joblib')

    return pipe
import os

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("""I met a strange lady, she made me nervous""")
    st.subheader("She took me in and gave me breakfast")
    path = "./weatherAUS.csv"
    dataframe = pd.read_csv(path, usecols=range(1, 25))
    df_limpio = preprocessor.fit_transform(dataframe)


    #######################################################
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the joblib file
    PATH_CLAS = '/home/jester/Desktop/tpaa/AA1-TUIA-Petetta-Arce/streamlit/handlers/model/logisticmodel.joblib'


    #PATH_CLAS = '/handlers/joblib/rain_pred_clasificacion.joblib'
    pipeline_clas = joblib.load(PATH_CLAS)
    feature_names = pipeline_clas.named_steps['imputer'].get_feature_names_out()
##################################################################3



with model_training:
    columnas_numericas = list(
        df_limpio.columns[:-1]
    )  # acomodar esto asi no uso raintomorrow
    st.header("He just smiled and gave me a Vegemite sandwich")
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