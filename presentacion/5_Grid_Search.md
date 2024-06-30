En optimizacion de hiperparametros aclarar que usamos
randomizedSearchCV, mas o menos como funciona, que deberia devolver

Setear hiperparametros es util para mejorar la perfomance del modelo, los hiperparametros son parametros que no aprende el modelo si no que se setean anteriormente.

por que no grid search, no evalua todas las configuraciones, es mas rapido, mas eficiente y encuentra buenos hiperparametros de igual manera

Con la optimización de hiperparámetros usando gridsearchcv, nos muestra que los mejores hiperparametros para nuestro modelo de regresión logística con buen score de precision y recall son los siguentes:

No hace falta que el data set esté balanceado, converge en 200 iteraciones y con un valor de regularización bajo|.

Por default en LogisticRegression de scikit-learn por default usa un valor
C= 1.0
ClassWeight=None, que no es optimizado
max_iter=100
y seteamos el solver para que solo pruebe con newton-cg pero por default usa 'lbfgs'
