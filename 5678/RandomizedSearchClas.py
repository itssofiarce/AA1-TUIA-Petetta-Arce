clasificacion_parametros = {
    "max_iter": [10, 100, 300, 500],
    "C": [0.001, 0.01, 0.1, 1, 10],
    "Penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
}

optimizar_clasificacion = RandomizedSearchCV(
    LogisticRegression(random_state=23), clasificacion_parametros
)

optimizar_clasificacion.fit(x_train_clasificacion, y_train_clasificacion)


modelo_optimizado_clasificacion = optimizar_clasificacion.best_estimator_

y_pred_clasificacion_optimizado = modelo_optimizado_clasificacion.predict(
    x_test_clasificacion
)

print("Logistica Modelo optimizado\n")
print(classification_report(x_test_clasificacion, y_pred_clasificacion_optimizado))


ConfusionMatrixDisplay(
    confusion_matrix(y_test_clasificacion, y_pred_clasificacion_optimizado)
).plot()
