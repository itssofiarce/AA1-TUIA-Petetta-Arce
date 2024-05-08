explicar_clasificacion = shap.LinearExplainer(
    reg_log,
    x_train_clasificacion.astype("float64").copy(),
    features_names=x_train_clasificacion.columns,
)


shap_valores_clasificacion = explicar_clasificacion(
    x_train_clasificacion.astype("float64")
)

shap.summary_plot(shap_valores_clasificacion, x_train_clasificacion)


explicabilidad_global_clasificacion = shap.Explanation(
    shap_valores_clasificacion,
    base_values=explicar_clasificacion.expected_value,
    features_names=x_train_clasificacion.columns,
    data=x_test_clasificacion,
)


shap.plots.bar(explicabilidad_global_clasificacion)

################################

index = 50

reg_log.predict(x_test_clasificacion)[index]

shap_valores_clasificacion = explicar_clasificacion(x_test_clasificacion)

explicacion = shap.Explanation(
    values=shap_valores_clasificacion[index],
    base_values=explicar_clasificacion.expected_value,
    feature_names=x_train_clasificacion.columns,
)


shap.plots.waterfall(explicacion)
