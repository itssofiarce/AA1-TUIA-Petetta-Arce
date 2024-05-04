X_base_clasificacion = dataframe_clasificacion[["RainfallToday", "Sunshine", "MinTemp"]]
Y_base_clasificacion = dataframe_clasificacion["RainTomorrow"]

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train_base_clas, X_test_base_clas, Y_train_base_clas, Y_test_base_clas = (
    train_test_split(
        X_base_clasificacion, Y_base_clasificacion, test_size=0.20, random_state=42
    )
)

mod_base_clasificacion = LogisticRegression()

mod_base_clasificacion.fit(X_train_base_clas, Y_train_base_clas)

y_pred_base = mod_base_clasificacion.predict(X_test_base_clas)


print("Logistica Modelo base\n")
print(classification_report(X_test_base_clas, y_pred_base))
ConfusionMatrixDisplay(confusion_matrix(Y_test_base_clas, y_pred_base)).plot()
