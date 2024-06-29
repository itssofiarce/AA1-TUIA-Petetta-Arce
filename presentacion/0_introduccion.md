

Date fecha de la observacion

MinTemp Temperatura minima. En grados celsius

MaxTemp Temperatura maxima. En grados celsius

Rainfall cantidad de lluvia registrada en el dia. En mm

Evaporation evaporacion (mm) de 00 a 09am.

Sunshine Numero de horas de luz solar durante el dia.

WindGustDir direccion de la rafaga de viento mas fuerte en las 24 horas

WindGustSpeed velocidad de la rafaga de viento mas fuerte en km/h

WindDir9am direccion del viento a las 9 am

WindDir3pm direccion del viento a las 3 pm

WindSpeed9am velocidad del viento en km/h, a las 9 am

WindSpeed3pm velocidad del viento en km/h, a las 3 pm

Humidity9am humedad en porcentaje a las 9 am

Humidity3pm humedad en porcentaje a las 3 pm

Pressure9am presion atmosferica en (hpa) al nivel del mar a las 9 am

Pressure9am presion atmosferica en (hpa) al nivel del mar a las 3 pm

Cloud9am Fraccion del cielo oscurecida por nubes medida en fracciones de 8 (0 indica sin nubes, 8 totalmente nublado) a las 9 am

Cloud9am Fraccion del cielo oscurecida por nubes medida en fracciones de 8 (0 indica sin nubes, 8 totalmente nublado) a las 3pm

Temp9am temperatura en grados celsius a las 9 am

Temp3pm temperatura en grados celsius a las 3 pm

RainToday valor booleano si llovio o no durante el dia (1 si pasa 1 mm)

RainTomorrow Cantidad de lluvia al dia siguiente en mm

RainfallTomorrow cantidad de lluvia al dia siguiente en mm




Como no usamos la variable fecha la descartamos

La variable localizacion se procedio de la siguiente forma:
    . Elegir las filas con nombre que ibamos a usar,
    . Segun la consigna podian considerarse como una sola localizacion
    . Cambiamos el nombre por costa este y la descartamos (reemplazamos por todos 0)


Observamos que no hay datos duplicados, muchos nulos

Pasar rapido las observaciones del heatmap porque son bastante obvias
- si esta nublado hace menos calor
- si llovio ayer, capaz llueve mañana
- si esta nublado hay menos horas de sol (no way)
- Las variables de temperatura estan relacionadas entre si (no way)
- Presion y humedad son importantes, pero se va a ver en la parte de shap


- Proximo grafico, mas humedad, probablemente llueva al otro dia

- Proximo grafico, menos presion, probablemente llueva al otro dia

- El grafico de Max y Min temp puede interpretarse como que los dias mas frios es probable que llueva (frios en comparacion con los otros)

- De vuelta se puede poner en evidencia que a mayor humedad a la tarde mayor cantidad de lluvia al otro dia, esto se puede contrastar con el grafico de humedad anterior

- Vemos en el grafico de densidad de rainfall tomorrow un comportamiento normal (la mayoria de dias no llueve)

- El QQ-plot tendria que presentar una forma de /, no se como explicarlo, para que los graficos tengan una distribucion normal
