### Respuestas práctico 4

> ¿Qué significan estas métricas?

- MAE (Mean Absolute Error): Promedio de los errores absolutos, sin importar si son positivos o negativos.

- MSE (Mean Squared Error): Promedio de los errores al cuadrado, penaliza más los errores grandes.

- RMSE: Raíz cuadrada del MSE, vuelve a las unidades originales del problema.

- R²: Indica qué porcentaje de la varianza es explicada por el modelo (0-1, donde 1 es perfecto).

- MAPE: Error porcentual promedio, útil para comparar modelos con diferentes escalas.

> Completa las definiciones

- Accuracy: Porcentaje de predicciones ccorrectas sobre el total.

- Precision: De todas las predicciones positivas, ¿cuántas fueron realmente correctas?

- Recall (Sensibilidad): De todos los casos positivos reales, ¿cuántos detectamos?

- F1-Score: Promedio ponderado entre precision y recall.

- Matriz de Confusión: Tabla que muestra clases de predicción vs clases reales.

> Responde

> ¿Cuál es la diferencia principal entre regresión lineal y logística?

La regresión lineal predice un valor continuo lineal, mientras que la logística
predice un sigmoide útil para clasificasión binaria.

> ¿Por qué dividimos los datos en entrenamiento y prueba?

Para que el resultado final no esté sesgado por overfitting.

> ¿Qué significa una exactitud del 95%?

En el 95% de los casos, la predicción fue correcta.

> ¿Cuál es más peligroso: predecir "benigno" cuando es "maligno", o al revés?

En este caso, los falsos negativos son más peligrosos, ya que el resultado es
un paciente con cáncer que no se trata.

> Completa la tabla

| Aspecto           | Regresión lineal                                  | Regresión logística                                                          |
| ----------------- | ------------------------------------------------- | ---------------------------------------------------------------------------- |
| Qué predice       | Números continuos                                 | Categorías                                                                   |
| Ejemplo de uso    | Predecir edad en base a gustos de entretenimiento | Predecir resultado de partido de fútbol (quién gana) en base al historial de los cuadros. |
| Rango de salida   | R                                                 | 0 a 1                                                                        |
| Métrica principal | R²                                                | F1-Score                                                                     |
