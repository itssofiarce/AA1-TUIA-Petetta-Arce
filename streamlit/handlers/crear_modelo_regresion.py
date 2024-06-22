import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from script import *
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

path= 'streamlit/weatherAUS.csv'
df = pd.read_csv(path)

# Separación de variables explicativas y variables objetivo
X = df.drop(['RainfallTomorrow'], axis=1).copy()
y = df[['RainfallTomorrow']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creo un Dataframe de TRAIN
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train['RainfallTomorrow'] = y['RainfallTomorrow']

# Creo un Dataframe de TEST
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test['RainfallTomorrow'] = y['RainfallTomorrow']


# Preproceso mi df de test y mi df de train
df_train = preprocessor.fit_transform(df_train)
df_test = preprocessor.fit_transform(df_test)


# Splitteo en base a mi df preprocesado
X_train = df_train.drop(['RainfallTomorrow','RainTomorrow'], axis=1)
y_train = df_train['RainfallTomorrow']

X_test = df_test.drop(['RainfallTomorrow','RainTomorrow'], axis=1)
y_test = df_test['RainfallTomorrow']


# instanciar las clases regresion
model_sgd = SGDRegressor(max_iter=10000, random_state=42)

# Ciclo de entrenamiento
for epoch in range(1000):  # Ajusta según el número real de épocas
    # train
    model_sgd.partial_fit(X_train, y_train)

# Predicciones en el conjunto de prueba (Test)
y_pred_sgd = model_sgd.predict(X_test)

# Métricas del modelo
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

joblib.dump(model_sgd, 'streamlit/models/lluvia_regresion.joblib')