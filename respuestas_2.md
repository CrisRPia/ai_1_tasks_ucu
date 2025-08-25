## Tarea 2 Unidad 1

[Código fuente](./src/ex_2.py)

### Matriz de confusión: ¿En qué casos se equivoca más el modelo: cuando predice que una persona sobrevivió y no lo hizo, o al revés?

El modelo tiene más falsos negativos; predice más frecuentemente que una persona
no sobrevivirá cuando no es el caso.

### Clases atendidas: ¿El modelo acierta más con los que sobrevivieron o con los que no sobrevivieron?

El modelo acierta más en los que no sobrevivieron (se ve en el recall).

```
  precision    recall  f1-score   support

0       0.82      0.89      0.86       110
1       0.80      0.70      0.74        69
```

### Comparación con baseline: ¿La Regresión Logística obtiene más aciertos que el modelo que siempre predice la clase más común?

Sí:

```
Baseline acc: 0.6145251396648045
LogReg acc  : 0.8156424581005587
```

### Errores más importantes: ¿Cuál de los dos tipos de error creés que es más grave para este problema?

Predecir que alguien sobrevivió cuando no lo hizo da motivo es el peor de los
dos problemas, ya que puede crear esfuerzos desperdiciados en situaciones de
rescate.

### Observaciones generales: Mirando las gráficas y números, ¿qué patrones interesantes encontraste sobre la supervivencia?

La precision para los supervivientes es menor, lo que tiene sentido porque el
modelo tiene menos supervivientes para probar.

### Mejoras simples: ¿Qué nueva columna (feature) se te ocurre que podría ayudar a que el modelo acierte más?

Añadir tamaño familiar; probablemente las personas solas y las familias
muy grandes tengan peores probabilidades.
