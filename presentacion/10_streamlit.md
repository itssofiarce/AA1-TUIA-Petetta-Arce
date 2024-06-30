Overview
The application is structured into several sections using Streamlit containers: header, dataset, features, and model training. It loads and processes a dataset, accepts user input for feature values, and uses pre-trained machine learning models to make predictions.

Loading Libraries and Models
Libraries:

streamlit for creating the web app.
pandas for data manipulation.
joblib for loading pre-trained models.
os for file path handling.
Model Loading:

@st.cache_resource(show_spinner="Loading model...") decorator caches the loaded models to optimize performance.
Two models are loaded: one for classification (logisticmodel.joblib) and one for regression (linealmodel.joblib).
Data Preprocessing
The dataset is loaded from a CSV file (weatherAUS.csv) and certain columns are selected for use.
preprocessor.fit_transform(dataframe) cleans and preprocesses the data.
User Interface
Classification Model
Header:

Displays the title "Prediccion de lluvia en Australia".
Feature Input:

A series of sliders is generated for the user to input feature values. These sliders correspond to various numerical features in the dataset (excluding the target variable).
A select box is used to input whether it rained today (¿Hoy llovió?), with options 'Sí' and 'No'.
Prediction:

The user inputs are combined into a DataFrame (datos) with the necessary feature names.
The classification model (pipeline_clas) predicts whether it will rain tomorrow.
The result is displayed: 'Si' (Yes) if rain is predicted, otherwise 'NO' (No).
Regression Model
Header:

Displays the title "Prediccion cantidad de lluvia para el día siguiente en Australia".
Feature Input:

Similar to the classification section, sliders are used for inputting feature values.
An additional select box for whether it rained today is included.
Prediction:

The regression model (pipeline_reg) predicts the amount of rain for the next day.
The result is displayed, showing the predicted amount of rain in cm³. If the prediction is less than or equal to zero, it states that it will probably not rain.
Summary
This Streamlit application allows users to interactively input feature values and obtain predictions for both the likelihood of rain and the amount of rain for the next day in Australia. The app demonstrates the practical use of machine learning models for real-world prediction tasks, providing a user-friendly interface for exploring weather data and predictions.
