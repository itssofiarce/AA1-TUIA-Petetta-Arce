from handlers.pipeline_clas import *
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from split_data import X_train, X_test, y_train_regresion, y_test_regresion


# Ciclo de entrenamiento
for epoch in range(1000):  # Ajusta según el número real de épocas
    # train
    model_sgd.partial_fit(X_train, y_train)

# Predicciones en el conjunto de prueba (Test)
y_pred_sgd = model_sgd.predict(X_test)

# Métricas del modelo
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)
