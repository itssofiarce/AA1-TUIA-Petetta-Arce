import joblib
from sklearn.pipeline import Pipeline
from streamlit.handlers.clean_igual import preprocessor
from reglogistica import RegLogistica
from split_data import X_train, X_test, y_train_clasificacion, y_test_clasificacion

# instanciar las clases de limpieza y regresion


reg_logistica = RegLogistica()
# ipeline
clasificacion_pipeline = Pipeline(
    [
        ("Clasificacion con reg Logistica", reg_logistica),
    ]
)
print(clasificacion_pipeline)

# Entrenar el modelo
clasificacion_pipeline.fit(X_train, y_train_clasificacion)

# Predecir
predictions = clasificacion_pipeline.predict(X_test)


accuracy = RegLogistica.score(X_test, y_test_clasificacion, metric="accuracy")
print(f"Accuracy: {accuracy}")

# Guardar el modelo
joblib.dump(clasificacion_pipeline, 'lluvia_regresion.pkl')