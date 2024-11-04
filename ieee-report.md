# Clasificación de Patrones de Movimiento usando Perceptrón Multicapa

## Resumen

Este trabajo presenta la implementación de un perceptrón multicapa para clasificar patrones de movimiento en tres categorías: lineal, circular y aleatorio. Se desarrolló un sistema que genera datos sintéticos de movimiento y los utiliza para entrenar una red neuronal. Los resultados muestran una precisión del 93.33% en el conjunto de prueba, demostrando la efectividad del enfoque propuesto.

## I. Introducción

El reconocimiento de patrones de movimiento es fundamental en diversas aplicaciones, desde la robótica hasta el análisis de comportamiento. Este proyecto implementa un clasificador neuronal capaz de distinguir entre tres tipos básicos de movimiento: lineal, circular y aleatorio.

## II. Metodología

### A. Generación de Datos

Se implementaron tres generadores de datos:

1. Movimiento Lineal: Trayectorias rectas con dirección aleatoria
2. Movimiento Circular: Trayectorias circulares con radio y centro variables
3. Movimiento Aleatorio: Trayectorias basadas en random walk

Cada trayectoria consiste en 10 puntos (x,y) secuenciales, resultando en vectores de entrada de 20 dimensiones.

### B. Arquitectura del Perceptrón Multicapa

La red neuronal implementada tiene:

- Capa de entrada: 20 neuronas (10 pares x,y)
- Capa oculta: Variable (se experimentó con diferentes tamaños)
- Capa de salida: 3 neuronas (clasificación one-hot)
- Funciones de activación:
  - Sigmoide en la capa oculta
  - Softmax en la capa de salida

### C. Entrenamiento

Se probaron cuatro configuraciones diferentes:

1. 5 neuronas ocultas, tasa de aprendizaje 0.05, 0.2, 0.5, 0.8
2. 10 neuronas ocultas, tasa de aprendizaje 0.05, 0.2, 0.5, 0.8
3. 15 neuronas ocultas, tasa de aprendizaje 0.05, 0.2, 0.5, 0.8
4. 18 neuronas ocultas, tasa de aprendizaje 0.05, 0.2, 0.5, 0.8

Datos utilizados:

- Entrenamiento: 300 ejemplos (100 por clase)
- Prueba: 90 ejemplos (30 por clase)
- Épocas de entrenamiento: 6000

## III. Resultados

### A. Comparación de Configuraciones

Los mejores resultados de precisión obtenidos fueron:

1. Configuración 1: 5 neuronas, tasa aprendizaje = 0.8:
   - Precisión entrenamiento: 100%
   - Precisión prueba: 96,6%

### B. Análisis de Errores

En el conjunto de prueba final:

- Total predicciones correctas: 87/90 (96.67%)
- Total predicciones incorrectas: 3/90 (3.33%)

Los errores más comunes fueron:

- Confundir movimiento circular con aleatorio
- Confundir movimiento lineal con aleatorio

## IV. Conclusiones

1. El sistema muestra una alta capacidad para distinguir entre los tres tipos de movimiento, con una precisión superior al 93% en datos no vistos.

2. La mayoría de los errores ocurren al clasificar movimientos aleatorios, lo cual es comprensible dada su naturaleza más irregular.

## V. Recomendaciones

1. Para mejorar el rendimiento se podría:
   - Aumentar el tamaño del conjunto de datos
   - Experimentar con arquitecturas más profundas
   - Implementar técnicas de regularización

2. El sistema podría extenderse para clasificar patrones de movimiento más complejos.

## Referencias

[1] Documentación de NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)  
[2] Documentación de Matplotlib: [https://matplotlib.org/](https://matplotlib.org/)
