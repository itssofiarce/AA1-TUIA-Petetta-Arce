# Regresion Lineal
con las variables que muestren algun comportamiento mas o menos linear (correlacion >= 0.1)


Primero regresion lineal pelado
Da un r2 de 0.17, esta bien porque este modelo nunca va a dar bien para modelos lineales.
Por lo menos da 0.17 en train y test, da mal pero no overfitea

Grafico de residuos

Sugiere que el comportamiento de las variables no es linear
No esta distribuido al azar por el grafico, hay valores con residuos muy altos y muy bajos, esta prediciendo mal


Si haces regresion lineal con solo humedad 3pm te da un r2 de 0.1, vamos a seguir sin captar la esencia del dataset con modelos tan sencillos

Pasamos a un modelo mas complejo, SGDRegressor 

Si bien mejora el r2 a 0.34, el doble de regresion lineal, sigue siendo un modelo no muy bueno



Aplicamos regresion con regularizacion

Lasso R2 de 0.11 da peores resultados
Primer grafico ---> daltonismo wins
Segundo grafico, a mayor fuerza de regularizacion bajan las features


Ridge
El alpha de ridge es mas alto
Usa mas features
Tiene un R2 de 0.34

Elastic net tiene r2 menor que ridge


Desempate entre SGDR y Ridge por su MSE, si bien no es muy alta la diferencia nos quedamos con SGDR
SGDR = 0.66
Ridge = 0.69



¿Qué puede concluir de cada uno de los modelos entrenados?
 ¿Cuál es el que mejor preforma?

- ¿Por qué se calculan las métricas de train y test?

- ¿Hubo algún buen fitting?