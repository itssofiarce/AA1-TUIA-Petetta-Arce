import streamlit as st
import pandas as pd
import clean_igual
from clean_igual import preprocessor

header = st.container()
dataset = st.container()
features = st.container()

with header:
    st.title("""I met a strange lady, she made me nervous""")
    st.subheader("She took me in and gave me breakfast")
    st.text(
        "Do you come from a land down under Where women glow and men plunder? Can't you hear, can't you hear the thunder? You better run, you better take cover"
    )
    path = "./weatherAUS.csv"
    dataframe = pd.read_csv(path, usecols=range(1, 25))
    # df = pd.read_csv(file_path, sep=",", engine="python")
    # wheater_data = pd.read_csv("weatherAUS.csv")
    st.write("### dataset original")
    st.write(dataframe.head())

    df_limpio = preprocessor.fit_transform(dataframe)

    st.write("### dataset Limpio Y normalizado")
    st.write(df_limpio.head())


with dataset:
    st.header("He just smiled and gave me a Vegemite sandwich")
