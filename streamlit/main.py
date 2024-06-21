import streamlit as st
import pandas as pd
import handlers.clean_igual as clean_igual
from handlers.clean_igual import preprocessor

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("""I met a strange lady, she made me nervous""")
    st.subheader("She took me in and gave me breakfast")
    st.text(
        "Do you come from a land down under Where women glow and men plunder? Can't you hear, can't you hear the thunder? You better run, you better take cover"
    )
    path = "./weatherAUS.csv"
    dataframe = pd.read_csv(path, usecols=range(1, 25))

    st.write("### dataset original")
    st.write(dataframe.head())

    df_limpio = preprocessor.fit_transform(dataframe)

    st.write("### dataset Limpio Y normalizado")
    st.write(df_limpio.head())

    distribuciones_llueve_o_no = pd.DataFrame(df_limpio["RainTomorrow"].value_counts())
    st.subheader("Distribucion - ¿Llovio o no?")
    st.bar_chart(distribuciones_llueve_o_no)


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
