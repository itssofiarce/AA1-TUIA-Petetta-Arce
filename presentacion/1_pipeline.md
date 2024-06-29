Como funciona el pipeline?
Tenemos 11 clases, si bien los nombres son bastante explicativos, hacemos un pantallazo rapido

- ColDropper, elimina ciertas columnas basadas en nombre, principalmente las que no vamos a usar, tambien las de variables que no deberian estar en el conjunto X (rainTomorrow)
por que no dropea rainfall tomorrow tambien?
porque lo hacemos despues a mano

- LocDropper solo toma las localizaciones que fueron solicitadas

- Cat Filler, llena los valores faltantes por su moda

- Num filler, llena los valores numericos faltantes por su media

- Coord recat transforma las direcciones del viento a 2 nuevas columnas una en su componente senoidal y la otra en su coseno

- Loc Encoder pasa a 0's las localizaciones

- ResetIndex, reinicia el index (dije que algunas eran obvias en el nombre)

- Sandarizer: normaliza los datos numericos con Z-score

- OutlierTreatment: Elimina los valores atipicos usando la regla del rango intercuartilico (los limita al rango Q1−1.5×IQR y Q3+1.5×IQR )

- ColDropper elimina la fecha
# nos van a decir algo de que este pisa al coldropper anterior

- Loc dropper igual, hace lo mismo que el anterior nada mas que este las cambia todas a costa este tambien

- Cat filler y num filler hacen lo mismo
- Coord recat hace lo mismo

Esto pasa cuando pedis un solo notebook de entrega 

- BoolYNDropperEncoder elimina valores nulos de RainToday, tambien pasa las variables booleanas de yes no a 1 y 0

- RLValDropper elimina RainTomorrow



En cuanto a la división de datos en los set de entrenamiento y testeo, ¿se tomó algún criterio? Al existir una relación temporal, se recomienda respetarla y no dividir al azar

- No aporta informacion al modelo
- No tuvimos en cuenta la fecha


¿Puede decir algo respecto de los valores atípicos?