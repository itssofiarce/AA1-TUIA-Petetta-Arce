import streamlit as st
import pandas as pd
import joblib
import os
from handlers.clean_igual import preprocessor

# modelos
LOGISTIC_MODEL_PATH = "handlers/model/logisticmodel.joblib"
LINEAR_MODEL_PATH = "handlers/model/linealmodel.joblib"


# Clasificacion
@st.cache_resource(show_spinner="Loading model...")
def load_classification_model():
    return joblib.load(LOGISTIC_MODEL_PATH)


# Regresion
@st.cache_resource(show_spinner="Loading model...")
def load_regression_model():
    return joblib.load(LINEAR_MODEL_PATH)


# Cargar y limpiar
def load_and_preprocess_data():
    path = "weatherAUS.csv"
    dataframe = pd.read_csv(path, usecols=range(1, 25))
    col = [
        "Date",
        "Location",
        "WindGustDir",
        "WindDir9am",
        "WindDir3pm",
        "RainTomorrow",
        "RainfallTomorrow",
    ]
    print(dataframe.columns)
    dataframe_display = dataframe.drop(col, axis=1)
    dataframe_display = dataframe_display.apply(
        lambda x: pd.to_numeric(x, errors="coerce")
    )
    df_limpio = preprocessor.fit_transform(dataframe)
    return (
        dataframe_display,
        df_limpio,
    )


# FRONT
header = st.container()
model_training = st.container()
header_lineal = st.container()
model_training_lineal = st.container()

with header:
    st.title("Predicción de lluvia en Australia")
    original_df, df_limpio = load_and_preprocess_data()

    # CLas
    pipeline_clas = load_classification_model()
    feature_names = pipeline_clas.named_steps["imputer"].get_feature_names_out()

    # Input class
    columnas_numericas = list(original_df.columns[:-1])
    st.header("Ajusta los parámetros para que el modelo prediga")
    features = [
        st.slider(
            columna,
            float(original_df[columna].min()),  # usar data original para los sliders
            float(original_df[columna].max()),
            float(original_df[columna].mean()),
        )
        for columna in columnas_numericas
    ]
    raintoday_option_mapping = {"Sí": 1, "No": 0}
    raintoday_option = st.selectbox(
        "¿Hoy llovió?", list(raintoday_option_mapping.keys())
    )

    # pjrediccion class
    all_features = features + [raintoday_option_mapping[raintoday_option]]
    datos = pd.DataFrame([all_features], columns=feature_names)
    pred_clas = pipeline_clas.predict(datos)
    resultado_clasificacion = "Si" if pred_clas else "NO"
    st.markdown(f"Mañana probablemente {resultado_clasificacion} llueva")

with header_lineal:
    st.title("Predicción cantidad de lluvia para el día siguiente en Australia")

    # Reg
    pipeline_reg = load_regression_model()
    feature_names = pipeline_reg.named_steps["imputer"].get_feature_names_out()

    # Input reg
    columnas_numericas = list(original_df.columns[:-1])
    st.header("Ajusta los parámetros para que el modelo prediga")
    features = [
        st.slider(
            f"{columna}_{index}",
            float(original_df[columna].min()),  # usar data original para los sliders
            float(original_df[columna].max()),
            float(original_df[columna].mean()),
        )
        for index, columna in enumerate(columnas_numericas)
    ]
    raintoday_option_mapping = {"Sí": 1, "No": 0}
    raintoday_option = st.selectbox(
        "¿Hoy llovió?", list(raintoday_option_mapping.keys()), key="Llovió"
    )

    # Reg pred
    all_features = features + [raintoday_option_mapping[raintoday_option]]
    datos = pd.DataFrame([all_features], columns=feature_names)
    pred_clas = pipeline_reg.predict(datos)
    if pred_clas > 0:
        st.markdown(f"Mañana probablemente llueva {pred_clas[0]} cm³")
    else:
        st.markdown("Mañana probablemente llueva 0 cm³")
