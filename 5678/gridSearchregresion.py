regresion_parametros = {
    "fit-intercept": [True, False],
    "positive": [True, False],
    "copy_X": [True, False],
    "n_jobs": [None, 1, 2, 3, 4],
}

optimizar_regresion = GridSearchCV(LinearRegression(), regresion_parametros)

optimizar_regresion.fit(x_train_regresion, y_train_regresion)


modelo_optimizado_regresion = optimizar_regresion.best_estimator_

y_pred_regresion_optimizado = modelo_optimizado_regresion.predict(x_test_regresion)

errores(y_test_regresion, y_pred_regresion_optimizado)
