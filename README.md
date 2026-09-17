# Portafolio Análisis - Random Forest

## Descripción

Este proyecto corresponde al análisis del desempeño de un modelo **Random Forest** aplicado a un problema de clasificación de patinaje artístico.

El objetivo principal fue evaluar el comportamiento del modelo utilizando conjuntos de **entrenamiento, validación y prueba**, así como analizar su nivel de **bias, varianza y ajuste**.

## Metodología

El dataset fue dividido aproximadamente en:

- 60% entrenamiento
- 20% validación
- 20% prueba

Se evaluó un modelo inicial de Random Forest y posteriormente se analizó el efecto del parámetro `max_depth` para identificar posibles problemas de sobreajuste.

A partir de este análisis se aplicó regularización limitando la profundidad de los árboles.

## Resultados principales

El modelo inicial presentó un ligero sobreajuste, con una diferencia entre entrenamiento y validación.

Después de aplicar regularización con:

```python
max_depth = 2
