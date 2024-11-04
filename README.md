# Clasificación de Patrones de Movimiento usando Perceptrón Multicapa

## Descripción

Este proyecto implementa un perceptrón multicapa para clasificar patrones de movimiento en tres categorías: lineal, circular y aleatorio. Se generan datos sintéticos de movimiento y se utilizan para entrenar una red neuronal. Los resultados muestran una alta precisión en la clasificación de los patrones de movimiento.

## Estructura del Proyecto

- `perceptron_multicapa.py`: Implementación del perceptrón multicapa.
- `datos_movimiento.py`: Generación de datos de movimiento lineal, circular y aleatorio.
- `main_script.py`: Script principal para entrenar y evaluar el modelo.
- `grafico.py`: Script para visualizar la precisión de prueba por configuración.
- `grafico_prom.py`: Script para visualizar la precisión promedio de prueba por tamaño de capa oculta.
- `datos_entrenamiento.txt`: Datos de entrenamiento generados automáticamente.
- `datos_prueba.txt`: Datos de prueba generados automáticamente.
- `ieee-report.md`: Reporte del proyecto.

## Requisitos

- Python 3.x
- NumPy
- Matplotlib

## Instalación

1. Clona el repositorio:

    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```

2. Instala las dependencias:

    ```bash
    pip install numpy matplotlib
    ```

## Uso

1. Genera los datos de entrenamiento y prueba, y entrena el modelo:

    ```bash
    python main_script.py
    ```

2. Visualiza los gráficos de precisión de datos de ejemplo:

    ```bash
    python grafico.py
    python grafico2.py
    ```

## Resultados

Los mejores resultados de precisión obtenidos fueron:

- Precisión en entrenamiento: 100%
- Precisión en prueba: 96.67%

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cualquier cambio.
