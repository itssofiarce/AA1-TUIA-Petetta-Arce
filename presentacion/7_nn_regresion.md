dar un pantallazo del modelo
dar un pantallazo del trial de optuna
explicar por que nos quedamos con SGDR en regresion y no este (train y test r2 es distinto, implica que el modelo esta overfitteando)

explicar el bswarm plot de explicabilidad

La primera parte tira los hiperparametros para realizar un estudio de optuna, para que se entrene y evalue con un set de prueba, intentando minimizar el mse como el valor a optimizar

despues hicimos un modelo de una red neuronal a mano, porque en la entrega de este ejercicio habiamos entregado solo la optimizada sin alguna de referencia, el optimizador a usar es ADAM

una vez terminado de encontrar la mejor combinacion de hiperparametros se pasan a un nuevo modelo, donde se evalua de vuelta con su MSE y su r2 para ver la performance del modelo, tanto como la capacidad de predecir

No se que tanto decir del Bswarm plot de shap, se explica demasiado por si mismo

Para las redes neuronales, las variables mas explicativas son La humedad de las 3pm, La presion de las 3pm y la presion de las 9am. Tanto la presión como la humedad, a mayor valor mayor es la cantidad de lluvia que predice mi modelo. En cambio para la presion de las 3pm a mayor valor, en el modelo resta la cantidad de lluvia. Esto lo concluimos con los resultado de la primera vez que creamos la red.
