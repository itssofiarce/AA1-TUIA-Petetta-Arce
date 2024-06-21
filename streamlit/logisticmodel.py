import pipeline

preprocessor = Pipeline([
     ('drop_null_val_rl', RLValDropper()),
     ('drop_not_needed_features', ColDropper()),
     ('drop_nor_needed_locations',LocDropper()),
     ('yes_no_dropper_encoder', BoolYNDropperEncoder()),
     ('fill_null_cat', CatFiller()),
     ('fill_num_cat', NumFiller()),
     ('encode_loc', LocEncoder()),
     ('encode_wind_dir', CoordRecat()),
     ('reset_index',ResetIndex()),
     ('treat_outliers',OutliersTreater()),
     ('standariza_values', Standarizer())
])

# Cargo dataset
path = 'weatherAUS.csv'
df = pd.read_csv(path, usecols=range(1,25))
df.head()


# Dropeo valores nulos de 'RainfallTomorrow y Raintomorrow' de mi dataframe original
df.dropna(subset=['RainfallTomorrow', 'RainTomorrow'], inplace=True)

# Separación de variables explicativas y variables objetivo --> Para este modelo de regresión lineal
X = df.drop(['RainfallTomorrow'], axis=1).copy()
y = df[['RainfallTomorrow']].copy()

# Spliteo mi dataset en train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Pipeline de modelo
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", SGDRegressor(max_iter=10000, random_state=42))
    ]
)

# Entreno el modelo
# Listas para almacenar las pérdidas
training_losses = []
validation_losses = []

# Ciclo de entrenamiento
for epoch in range(1000):  # Ajusta según el número real de épocas
    # train
    model_sgd.partial_fit(X_train, y_train)
    y_train_pred = model_sgd.predict(X_train)
    training_loss = mean_squared_error(y_train, y_train_pred)
    training_losses.append(training_loss)

    # Validación
    y_val_pred = model_sgd.predict(X_test)
    validation_loss = mean_squared_error(y_test, y_val_pred)
    validation_losses.append(validation_loss)

# Predicciones en el conjunto de prueba (Test)
y_pred_sgd = model_sgd.predict(X_test)

# Métricas del modelo
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

# Guardo el modelo
dump(model, 'model/logisticmodel.joblib')
