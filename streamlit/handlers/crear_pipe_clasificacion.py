import joblib
from sklearn.pipeline import Pipeline
from clean_igual import preprocessor
from clasificacion_pipe import clasificacion_pipeline
from split_data import X_train, X_test, y_train_clasificacion, y_test_clasificacion

# instanciar las clases de limpieza y regresion
limpiar_datos = preprocessor()
clasificacion_pipeline = clasificacion_pipeline()

# ipeline
clasificacion_pipeline = Pipeline(
    [
        ("Preprocesado de datos", limpiar_datos),
        ("Red neuronal para regresion", clasificacion_pipeline),
    ]
)
clasificacion_pipeline


# Entrenar el modelo

clasificacion_pipeline.fit(X_train, y_train_clasificacion)

# Predecir
predictions = clasificacion_pipeline.predict(X_test)

# Metricas
r2 = clasificacion_pipeline[1].score(X_test, y_test_clasificacion, metric="r2")
rmse = clasificacion_pipeline[1].score(X_test, y_test_clasificacion, metric="rmse")


joblib.dump(clasificacion_pipeline, 'lluvia_regresion.pkl')
