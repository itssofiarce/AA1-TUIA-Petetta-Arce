from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(random_state = 0)


classifier.fit(train_clasificacion_X, y_train_clasificacion)





valor_y_predecir = classifier.predict(X_test_clasificacion)

#metricas

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(valor_y_test, valor_y_predecir)


as = accuracy_score(valor_y_test, valor_y_predecir)


# Curva ROC

probabilidades_y = classifier.predict_proba(X_test_clasificacion)[:,-1]

fpr, tpr, thresholds = roc_curve(y_test_clasificacion, probabilidades_y)

roc_auc = auc(fpr,tpr)

LR_JaccardIndex = jaccard_score(y_test_clasificacion, valor_y_predecir, average='weighted')

# Grafico ROC

plt.figure(figsize(5,5))
plt.plot(fpr, tpr, color="red", lw=2, label='CURVA ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1], color="blue", lw=2, linestyle="--")
plt.xlim([0.0, 1.10])
plt.ylim([0.0, 1.10])
plt.xlabel("Tasa de Falsos positivos(FPR)")
plt.ylabel("Tasa de Verdaderos positivos(TPR)")
plt.title("ROC")
plt.legent(loc="lower right")
plt.show()
