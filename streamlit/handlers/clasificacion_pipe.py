#aca va un modelo de regresion logistica
# tambien con la pipeline de joblip para exportar a main


class ClassificationPipeline:
    def __init__(self):
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)
        self.classifier = classifier
        self.pipeline = None

    def create_pipeline(self):
        self.pipeline = ImbPipeline(steps=[
            ('scaler', self.scaler),
            ('smote', self.smote),
            ('classifier', self.classifier)
        ])

    def fit(self, X_train, y_train):
        if self.pipeline is None:
            self.create_pipeline()
        self.pipeline.fit(X_train, y_train)

    def score(self, X_test, y_test):
        if self.pipeline is not None:
            return self.pipeline.score(X_test, y_test)
        else:
            raise ValueError("The pipeline has not been created or fitted yet.")

    def predict(self, X_test):
        if self.pipeline is not None:
            return self.pipeline.predict(X_test)
        else:
            raise ValueError("The pipeline has not been created or fitted yet.")

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred)