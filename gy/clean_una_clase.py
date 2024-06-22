import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def drop_unnecessary_columns(self, X):
        return X.drop(["Unnamed: 0", "Date", "RainTomorrow"], axis=1)

    def filter_locations(self, X):
        aus_loc = [
            "Adelaide",
            "Canberra",
            "Cobar",
            "Dartmoor",
            "Melbourne",
            "MelbourneAirport",
            "MountGambier",
            "Sydney",
            "SydneyAirport",
        ]
        return X[X["Location"].isin(aus_loc)]

    def fill_categorical_na(self, X):
        X["WindGustDir"] = X.groupby("Location")["WindGustDir"].transform(
            lambda x: x.fillna(x.mode()[0])
        )
        X["WindDir9am"] = X.groupby("Location")["WindDir9am"].transform(
            lambda x: x.fillna(x.mode()[0])
        )
        X["WindDir3pm"] = X.groupby("Location")["WindDir3pm"].transform(
            lambda x: x.fillna(x.mode()[0])
        )
        return X

    def fill_numerical_na(self, X):
        remanining_vnul_columns = X.columns[X.isna().any()].tolist()
        for col in remanining_vnul_columns:
            X[col] = X[col].fillna(X[col].mean())
        return X

    def encode_wind_direction(self, X):
        coord = {
            "N": 0,
            "NNE": 22.5,
            "NE": 45,
            "ENE": 67.5,
            "E": 90,
            "ESE": 112.5,
            "SE": 135,
            "SSE": 157.5,
            "S": 180,
            "SSW": 202.5,
            "SW": 225,
            "WSW": 247.5,
            "W": 270,
            "WNW": 292.5,
            "NW": 315,
            "NNW": 337.5,
        }

        for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
            X[col] = X[col].map(coord)
            X[f"{col}_rad"] = np.deg2rad(X[col])
            X[f"{col}_sin"] = np.sin(X[f"{col}_rad"]).round(5)
            X[f"{col}_cos"] = np.cos(X[f"{col}_rad"]).round(5)

        columns_to_drop = [
            f"{col}_rad" for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]
        ] + ["WindGustDir", "WindDir9am", "WindDir3pm"]
        X = X.drop(columns=columns_to_drop, axis=1)
        return X

    def encode_location(self, X):
        dummies = pd.get_dummies(X["Location"], dtype=int)
        X = pd.concat([X, dummies], axis=1)
        X.drop("Location", axis=1, inplace=True)
        return X

    def reset_index(self, X):
        return X.reset_index(drop=True)

    def encode_bool_yn(self, X):
        X.dropna(subset=["RainToday"], inplace=True)
        X["RainTomorrow"] = X["RainTomorrow"].map({"No": 0, "Yes": 1}).astype(float)
        X["RainToday"] = X["RainToday"].map({"No": 0, "Yes": 1}).astype(float)
        return X

    def standardize_values(self, X):
        exc_c = ["RainToday", "RainTomorrow"]
        df_sub = X[[col for col in X.columns if col not in exc_c]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sub)
        X_scaled = pd.DataFrame(X_scaled, columns=df_sub.columns)
        for col in exc_c:
            X_scaled[f"{col}"] = X[col]
        return X_scaled

    def treat_outliers(self, X):
        cols_with_outliers = [
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "Evaporation",
            "Sunshine",
            "WindGustSpeed",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp9am",
            "Temp3pm",
        ]

        for col in cols_with_outliers:
            IQR = X[col].quantile(0.75) - X[col].quantile(0.25)
            lower_bridge = X[col].quantile(0.25) - (IQR * 1.5)
            upper_bridge = X[col].quantile(0.75) + (IQR * 1.5)

            X.loc[X[col] >= round(upper_bridge, 2), col] = round(upper_bridge, 2)
            X.loc[X[col] <= round(lower_bridge, 2), col] = round(lower_bridge, 2)
        return X

    def drop_rain_tomorrow_null(self, X):
        X.dropna(subset=["RainTomorrow"], inplace=True)
        return X

    def transform(self, X):
        X = self.drop_rain_tomorrow_null(X)
        X = self.drop_unnecessary_columns(X)
        X = self.filter_locations(X)
        X = self.encode_bool_yn(X)
        X = self.fill_categorical_na(X)
        X = self.fill_numerical_na(X)
        X = self.encode_location(X)
        X = self.encode_wind_direction(X)
        X = self.reset_index(X)
        X = self.treat_outliers(X)
        X = self.standardize_values(X)
        return X


# Create an instance of the pipeline with the single class
preprocessor = Pipeline(DataCleaner())
