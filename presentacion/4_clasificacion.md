Para clasificacion si bien usamos siempre regresion logistica lo que vamos cambiando es usar primero el dataset como esta, despues lo balanceamos, ya vimos desde el principio que la variable a predecir RainTomorrow es mayoritariamente No/0 100k a 30k antes de limpiar el dataset

Despues balanceando con Undersampling, smote y smotetomek.
Nos quedamos con smote porque: [ampliar explicacion]

Undersampling: reducir la clase mayoritaria para que quede igual que la minoritaria

- mas rapido pero pierde informacion

Smote: genera datos para la clase minoritaria interpolando entre estos

- evita overfit, no pierde informacion
- crea ruido o outliers, quizas los datos sinteticos no reflejan la realidad

SMOTETomek: mezcla las 2 tecnicas, es mas intenso y su uso es mas especifico.

Nos quedamos con smote, no perdemos informacion, ayuda a representar la clase minoritaria, sin perjudicar la mayoritaria

Primer reglog sin balancear:

- Precision: 0.87/0.71 ---> predichos correctos (positivos) sobre todos los (positivos)
  usar cuando el costo de falsos positivos es alto
- Recall: 0.93/0.53 ---> predichos correctos (positivos) sobre todos los (valores de la clase)
  usar cuando los falsos negativos es alto
- F1: 0.90/0.61 ---> media armonica de precision y recall
  usar cuando necesites un balance
- Accuracy 0.84 (de todos los valores los predichos correctamente)
  util cuando el dataset esta balanceado

ROC: Reciever operating characteristic
es uun grafico que ilustre la habilidad de un clasificador binario
TPR: es lo mismo que recall
FPR: ratio de positivos predichos incorrectamente al total de negativos Falsoneg/Trueneg

AUC: area under the curve
da una medida de performance a lo largo de todos los posibles umbrales de clasificacion
rango de 0 a 1
AUC = 0.5: El modelo funciona igual que una eleccion al azar
AUC = 1.0: el modelo predice perfectamente
AUC > 0.5 el modelo predice con cierta capacidad

Valor de umbral:
El valor sobre cual un modelo clasifica como positiva una instancia y debajo como negativo
bajar el umbral sube el recall

# Modelos:

- Sin balancear AUC = 0.87
- Balanceado con undersampling = 0.86
- Balanceado con SMOTE AUC = 0.868
- Balanceado con SMOTETomek AUC = 0.868

reglog con smote:

- Precision: 0.92/0.53 ---> predichos correctos (positivos) sobre todos los (positivos)
  usar cuando el costo de falsos positivos es alto
- Recall: 0.80/0.76 ---> predichos correctos (positivos) sobre todos los (valores de la clase)
  usar cuando los falsos negativos es alto
- F1: 0.85/0.62 ---> media armonica de precision y recall
  usar cuando necesites un balance
- Accuracy 0.79 (de todos los valores los predichos correctamente)
  util cuando el dataset esta balanceado

El haber balanceado el dataset no cambio tanto los resultados, podemos ver la variabilidad de los datos a lo largo de las columnas los valores son bastante similares

## 1. Accuracy

**Formula**:
\[ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \]

- \(TP\): True Positives
- \(TN\): True Negatives
- \(FP\): False Positives
- \(FN\): False Negatives

## 2. Precision

**Formula**:
\[ \text{Precision} = \frac{TP}{TP + FP} \]

## 3. Recall (Sensitivity or True Positive Rate)

**Formula**:
\[ \text{Recall} = \frac{TP}{TP + FN} \]

## 4. F1 Score

**Formula**:
\[ \text{F1 \, Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]
