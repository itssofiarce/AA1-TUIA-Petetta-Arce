from pipeline import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDRegressor
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


# DESCARTAR VARIABLES NO NUMERICAS Y ACOMODAR EL DATASET PARA ML OPS
# SOLAMENTE ML-OPS
cols = ['costa_este','WindGustDir_sin',	'WindGustDir_cos','WindDir9am_sin',	'WindDir9am_cos','WindDir3pm_sin','WindDir3pm_cos']
class DescartarNoUsarMlOPS(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None):
        return self
    
    def transform(self,X):
        X = X.drop(cols, axis=1)
        return X


preprocessor = Pipeline(
    [
        ("drop_null_val_rl", RLValDropper()),
        ("drop_not_needed_features", ColDropper()),
        ("drop_nor_needed_locations", LocDropper()),
        ("yes_no_dropper_encoder", BoolYNDropperEncoder()),
        ("fill_null_cat", CatFiller()),
        ("fill_num_cat", NumFiller()),
        ("encode_loc", LocEncoder()),
        ("encode_wind_dir", CoordRecat()),
        ("reset_index", ResetIndex()),
        ("treat_outliers", OutliersTreater()),
        ("standariza_values", Standarizer()),
        ("Preparar_MLOPS", DescartarNoUsarMlOPS())
    ]
)

# Cargo dataset########################################################

path = "streamlit/handlers/weatherAUS.csv"
df = pd.read_csv(path, usecols=range(1,25))
df.head()


# # Dropeo valores nulos de 'RainfallTomorrow y Raintomorrow' de mi dataframe original
# df.dropna(subset=['RainfallTomorrow', 'RainTomorrow'], inplace=True)

# Separación de variables explicativas y variables objetivo
X = df.drop(['RainfallTomorrow', ], axis=1).copy()
y = df[['RainfallTomorrow']].copy()

# Spliteo mi dataset en train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Creo un Dataframe de TRAIN
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train['RainfallTomorrow'] = y['RainfallTomorrow']



# Creo un Dataframe de TEST
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test['RainfallTomorrow'] = y['RainfallTomorrow']

#Preproceso mi df de test y mi df de train
df_train = preprocessor.fit_transform(df_train)
df_test = preprocessor.fit_transform(df_test)


X_train = df_train.drop(['RainTomorrow', 'RainfallTomorrow'], axis=1).copy()
y_train = df_train['RainTomorrow'].copy()

X_test = df_test.drop(['RainTomorrow','RainfallTomorrow'], axis=1).copy()
y_test = df_test['RainTomorrow'].copy()


# Pipeline de modelo
model = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ("regressor", SGDRegressor(max_iter=10000, random_state=42))
    ]
)

# Ciclo de entrenamiento
for epoch in range(1000):  # Ajusta según el número real de épocas
    # train
    model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred_sgd_train = model.predict(X_train)

# Predicciones en el conjunto de prueba (Test)
y_pred_sgd = model.predict(X_test)

# Métricas del modelo
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

# Guardo el modelo
joblib.dump(model, 'streamlit/handlers/model/linealmodel.joblib')
