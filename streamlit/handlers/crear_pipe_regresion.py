import joblib
from sklearn.pipeline import Pipeline
from clean_igual import preprocessor
from red_regresion import regresion_pipeline
from split_data import X_train, X_test, y_train_regresion, y_test_regresion

# instanciar las clases de limpieza y regresion
limpiar_datos = preprocessor()
regresion_pipeline = regresion_pipeline()

# ipeline
regresion_pipeline = Pipeline(
    [
        ("Preprocesado de datos", limpiar_datos),
        ("Red neuronal para regresion", regresion_pipeline),
    ]
)
regresion_pipeline


# Entrenar el modelo

regresion_pipeline.fit(X_train, y_train_regresion)

# Predecir
predictions = regresion_pipeline.predict(X_test)

# Metricas
r2 = regresion_pipeline[1].score(X_test, y_test_regresion, metric="r2")
rmse = regresion_pipeline[1].score(X_test, y_test_regresion, metric="rmse")


joblib.dump(regresion_pipeline, 'lluvia_regresion.pkl')