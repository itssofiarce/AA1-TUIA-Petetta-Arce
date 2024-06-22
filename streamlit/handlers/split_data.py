import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from clean_igual import preprocessor

df = pd.read_csv("streamlit/handlers/weatherAUS.csv")


def splitter(dataframe, preprocessor):
    """Esta funcion separa el dataset en Features y variables a predecir para regresion y clasificacion."""

    # Separar las características y las variables objetivo
    X = df.drop(['RainTomorrow', 'RainfallTomorrow'], axis=1).copy()
    y_clas = df[['RainTomorrow']].copy()
    y_reg = df[['RainfallTomorrow']].copy()

        # Split into training and testing sets for regression
    X_train, X_test, y_train_regresion, y_test_regresion = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Split into training and testing sets for classification
    X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(X, y_clas, test_size=0.2, random_state=42)

    # Fit and transform with preprocessor for regression
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Crear dataframes de entrenamiento y prueba para clasificación
    df_train_regresion = pd.DataFrame(X_train, columns=X.columns)
    df_train_regresion['RainfallTomorrow'] = y_reg['RainfallTomorrow']

    df_test_regresion = pd.DataFrame(X_test, columns=X.columns)
    df_test_regresion['RainfallTomorrow'] = y_reg['RainfallTomorrow']

    df_train_clasificacion = pd.DataFrame(X_train, columns=X.columns)
    df_train_clasificacion['RainTomorrow'] = y_clas['RainTomorrow']

    df_test_clasificacion = pd.DataFrame(X_test, columns=X.columns)
    df_test_clasificacion['RainTomorrow'] = y_clas['RainTomorrow']

    # Preprocesar los conjuntos de datos de entrenamiento y prueba para clasificación
    df_train_regresion = preprocessor.fit_transform(df_train_regresion)
    df_test_regresion = preprocessor.fit_transform(df_test_regresion)
    df_train_clasificacion = preprocessor.fit_transform(df_train_clasificacion)
    df_test_clasificacion = preprocessor.fit_transform(df_test_clasificacion)

    # Separar nuevamente en características y etiquetas para clasificación
    X_train = df_train_clasificacion.drop(['RainfallTomorrow', 'RainTomorrow'], axis=1)
    X_test = df_train_clasificacion.drop(['RainTomorrow', 'RainfallTomorrow'], axis=1).copy()
    
    y_train_clasificacion = df_train_clasificacion['RainTomorrow'].copy()
    
    y_test_clasificacion = df_test_clasificacion['RainTomorrow'].copy()


    y_train_regresion = df_train_regresion['RainfallTomorrow']

    y_test_regresion = df_test_regresion['RainfallTomorrow']

    return (
        X_train,
        X_test,
        y_train_regresion,
        y_test_regresion,
        y_train_clasificacion,
        y_test_clasificacion,
    )


# Llamar a la función splitter
(
    X_train,
    X_test,
    y_train_regresion,
    y_test_regresion,
    y_train_clasificacion,
    y_test_clasificacion,
) = splitter(df, preprocessor)
