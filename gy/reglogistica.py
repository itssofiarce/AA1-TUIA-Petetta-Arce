# aca va un modelo de regresion logistica
# tambien con la pipeline de joblip para exportar a main
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from split_data import X_train, X_test, y_train_clasificacion, y_test_clasificacion
class RegLogistica:
    def __init__(self):
        self.parametros = {
            "C": 10,
            "class_weight": "balanced",
            "max_iter": 500,
            "solver": "newton-cg",
        }
        self.pipeline = None
        self.modelo = None
        self.y_pred_clasificacion = None

    def fit(self, X, y):
        if self.parametros:
            self.modelo = LogisticRegression(**self.parametros)
            self.modelo.fit(X_train, y_train_clasificacion)
        else:
            raise ValueError("Se necesita pasarle parametros al modelo")

        return self

    def predict(self, X):
        if self.model:
            y_pred_clasificacion = self.modelo.predict(X_train)
            return y_pred_clasificacion
        else:
            raise ValueError("hubo un error entrenando el modelo")

    def metrics(self, y_test_clasificacion):
        if y_test_clasificacion is None:
            raise ValueError("Se necesita el set de prueba")
        
        if self.y_pred_clasificacion is None:
            raise ValueError("Se necesita predecir")

        accuracy = accuracy_score(y_test_clasificacion, self.y_pred_clasificacion)
        precision = precision_score(y_test_clasificacion, self.y_pred_clasificacion)
        recall = recall_score(y_test_clasificacion, self.y_pred_clasificacion)
        f1 = f1_score(y_test_clasificacion, self.y_pred_clasificacion)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }

        return metrics
