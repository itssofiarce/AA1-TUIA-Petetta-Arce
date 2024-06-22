# Puesta en producción del modelo de Regresión Lógistica con HiperParametros optimizados
import streamlit as st
import numpy as np
import datetime
from crear_modelo_regresion import preprocessor

if "pred" not in st.session_state:
    st.session_state["pred"] = None

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load('models/lluvia_regresion.joblib')

    return pipe

def make_prediction(pipe):

    MinTemp = st.session_state["MinTemp"]
    MaxTemp = st.session_state["MaxTemp"]
    Rainfall = st.session_state["Rainfall"]
    Evaporation = st.session_state["Evaporation"]
    Sunshine = st.session_state["Sunshine"]
    WindGustSpeed = st.session_state["WindGustSpeed"]
    WindSpeed9am = st.session_state["WindSpeed9am"]
    WindSpeed3pm = st.session_state["WindSpeed3pm"]
    Humidity9am = st.session_state["Humidity9am"]
    Humidity3pm = st.session_state["Humidity3pm"]
    Pressure9am = st.session_state["Pressure9am"]
    Pressure3pm = st.session_state["Pressure3pm"]
    Cloud9am = st.session_state["Cloud9am"]
    Cloud3pm = st.session_state["Cloud3pm"]
    Temp9am = st.session_state["Temp9am"]
    Temp3pm = st.session_state["Temp3pm"]
    #Date = st.session_state["date"].strftime('%m/%d/%Y')
    Location = st.session_state["Location"]
    RainToday = st.session_state["RainToday"]
    WindGustDir = st.session_state["WindGustDir"]
    WindDir9am = st.session_state["WindDir9am"]
    WindDir3pm = st.session_state["WindDir3pm"]

    X_pred = np.array([MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
    WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, 
    Pressure9am, Pressure3pm, Cloud9am, Cloud3pm,Temp9am, Temp3pm, Date, Location,RainToday,
    WindGustDir,WindDir9am,WindDir3pm]).reshape(1,-1)

    X_pred = preproccesor.fit_transform(X_pred)

    pred = pipe.predict(X_pred)
    pred = round(pred[0], 2)

    st.session_state["pred"] = pred


if __name__ == "__main__":
    st.title("Predicción de cantidad de lluvia para mañana")

    pipe = load_model()

    with st.form(key="form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.date_input("Date", key='date')
            st.number_input("Sunshine", value=1.5, min_value=0.0, step=0.1, key='Sunshine')
            st.number_input("Max Temperature", value=25.0, min_value=-23.0, step=0.1, key='MaxTemp')
            st.number_input("Wind Speed at 3pm", value=1.5, min_value=0.0, step=0.1, key='WindSpeed3pm')
            st.number_input("Pressure at 3pm", value=1.5, min_value=0.0, step=0.1, key='Pressure3pm')
            st.number_input("Clouds at 3pm", value=1.5, min_value=0.0, step=0.1, key='Cloud3pm')
            
        with col2:
            st.selectbox("Location", index=0, options=["Adelaide","Canberra",
            "Cobar","Dartmoor","Melbourne","MelbourneAirport","MountGambier","Sydney","SydneyAirport"], key='Location')
            st.number_input("Wind Direction", value=0.0, step=0.1, key='WindGustDir')
            st.number_input("Min Temperature", value=0.0, min_value=-100.0, step=0.1, key='MinTemp')
            st.number_input("Humidity at 9am", value=1.5, min_value=0.0, step=0.1, key='Humidity9am')
            st.number_input("Temperature at 9am", value=25.0, min_value=-23.0, step=0.1, key='Temp9am')
            st.selectbox("Did it rain today?", index=0, options=["YES","NO"], key='RainToday')

        with col3:
            st.number_input("Amount of rain fallen", value=0.0, min_value=0.0, step=0.1, key='Rainfall')
            st.number_input("Wind Direction at 3pm", value=0.0, step=0.1, key='WindDir3pm')
            st.number_input("Wind Speed", value=1.5, min_value=0.0, step=0.1, key='WindGustSpeed') 
            st.number_input("Humidity at 3pm", value=1.5, min_value=0.0, step=0.1, key='Humidity3pm')
            st.number_input("Temperature at 3pm", value=25.0, min_value=-23.0, step=0.1, key='Temp3pm')
            
        with col4: 
            st.number_input("Evaporation", value=0.0, step=0.1, key='Evaporation')
            st.number_input("Wind Direction at 9am", value=0.0, step=0.1, key='WindDir9am')
            st.number_input("Wind Speed at 9am", value=1.5, min_value=0.0, step=0.1, key='WindSpeed9am')
            st.number_input("Pressure at 9am", value=1.5, min_value=0.0, step=0.1, key='Pressure9am')
            st.number_input("Clouds at 9am", value=1.5, min_value=0.0, step=0.1, key='Cloud9am')
        
        st.form_submit_button("Predecir", type="primary", on_click=make_prediction, kwargs=dict(pipe=pipe))

    if st.session_state["pred"] is not None:
        st.subheader(f"La lluvia estimada para mañanaes de: {st.session_state.pred}$")
    else:
        st.write("Agregá la informacion y hace click en Predecir para tener una estimación de la cantidad de lluvia para mañana")
    
    st.write(st.session_state)