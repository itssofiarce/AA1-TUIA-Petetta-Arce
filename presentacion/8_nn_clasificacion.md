En la red de clasificacion hacemos los mismos pasos, agregando unos label encoders para poder ver los resultados, se maximiza accuracy o precision

Los resultados se convierten a etiquetas binarias, en caso de tener varias dimensiones con argmax se pasa a una etiqueta de clase, tenemos la matriz de confusion y roc auc

Accuracy: 0.85
recall 0.52
precision 0.73
Parecido al resto de modelos anteriores

**Clasificación:** Al comparar los valores de Precision, para el problema de clasificación, el mejor modelo es el de Regresión Logística con los hiperparametros optimizados con una Precision del _78_% seguido por las Redes Neuronales con los hiperparámetros optimizados que tiene un 73%
