import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def splitter(dataframe):
    """Esta funcion separa el dataset en Features y variables a predecir para regresion y clasificacion"""
    features = dataframe.drop(["RainfallTomorrow", "RainTomorrow"], axis=1)
    pred_regresion = dataframe["RainfallTomorrow"]
    pred_clasificacion = dataframe["RainTomorrow"]

    (
        X_train,
        X_test,
        y_train_regresion,
        y_test_regresion,
        y_train_clasificacion,
        y_test_clasificacion,
    ) = train_test_split(
        features, pred_regresion, pred_clasificacion, test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train_regresion,
        y_test_regresion,
        y_train_clasificacion,
        y_test_clasificacion,
    )


# Example data
X = np.random.rand(100, 5)  # 100 samples, 5 features
Y1 = np.random.rand(100, 1)  # 100 samples, 1 target variable
Y2 = np.random.rand(100, 1)  # 100 samples, 1 target variable
data = np.hstack((X, Y1, Y2))
df = pd.DataFrame(data)
# Splitting the data
X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = train_test_split(
    X, Y1, Y2, test_size=0.2, random_state=42
)

#ej de uso

X_train, X_test, y_train_regresion, y_test_regresion, y_train_clasificacion, y_test_clasificacion = splitter(df)