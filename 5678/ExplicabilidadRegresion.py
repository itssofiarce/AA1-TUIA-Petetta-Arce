explicar_regresion = shap.LinearExplainer(
    reg_lin,
    x_train_regresion.astype("float64").copy(),
    features_names=x_train_regresion.columns.copy(),
)


shap_valores_regresion = explicar_regresion(x_train_regresion.astype("float64").copy())

shap.summary_plot(shap_valores_regresion, x_train_regresion)


explicabilidad_global_regresion = shap.Explanation(
    shap_valores_regresion,
    base_values=explicar_regresion.expected_value,
    features_names=x_train_regresion.columns,
    data=x_test_regresion,
)


shap.plots.bar(explanation_global_reg)


index = 50

reg_lin.predict(x_test_regresion)[index]

shap_valores_regresion = explicar_regresion(x_test_regresion)

explicacion = shap.Explanation(
    values=shap_valores_regresion[index],
    base_values=explicar_regresion.expected_value,
    feature_names=x_train_regresion.columns,
)
