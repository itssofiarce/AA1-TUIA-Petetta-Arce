X_base_regresion = dataframe_regresion[["RainToday", "Sunshine"]]
Y_base_regresion = dataframe_regresion["RainfallTomorrow"]

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train_base, X_test_base, Y_train_base, Y_test_base = train_test_split(
    X_base_regresion, Y_base_regresion, test_size=0.20, random_state=42
)

mod_base_regresion = LinearRegresion()

mod_base_regresion.fit(X_train_base, Y_train_base)

y_pred_base = mod_base_regresion.predict(X_test_base)


def errores(y_test, y_pred):
    print("R2:   %.2f" % r2_score(y_test, y_pred))
    print("MSE:  %.2f" % mean_squared_error(y_test, y_pred))
    print("RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:  %.2f" % mean_absolute_error(y_test, y_pred))
    print("MAPE: %.2f" % mean_absolute_percentage_error(y_test, y_pred))


errores(Y_test_base, y_pred_base)
