import matplotlib.pyplot as plt
import numpy as np

# Datos originales
hidden_sizes = [5, 10, 15, 18]
learning_rates = [0.05, 0.20, 0.50, 0.80]
precision_prueba = {
    (5, 0.05): [81.11, 74.44, 87.78, 83.33],
    (5, 0.20): [91.11, 88.89, 91.11, 88.89],
    (5, 0.60): [82.22, 91.11, 66.67, 91.11],
    (5, 0.80): [92.22, 92.22, 94.44, 64.44],
    (10, 0.05): [83.33, 66.67, 62.22, 80.00],
    (10, 0.20): [88.89, 88.89, 87.78, 86.67],
    (10, 0.50): [86.67, 73.33, 88.89, 86.67],
    (10, 0.80): [91.11, 83.33, 70.00, 71.11],
    (15, 0.05): [83.33, 82.22, 84.44, 74.44],
    (15, 0.20): [88.89, 86.67, 87.78, 86.67],
    (15, 0.50): [81.11, 86.67, 77.78, 88.78],
    (15, 0.80): [87.78, 84.44, 80.00, 91.11],
    (18, 0.05): [80.00, 78.89, 83.33, 77.78],
    (18, 0.20): [90.00, 82.22, 90.00, 90.00],
    (18, 0.50): [70.00, 77.78, 87.78, 87.78],
    (18, 0.80): [92.22, 84.44, 82.22, 83.33]
}

# Cálculo del promedio de precisión por tamaño de capa oculta
promedios_por_hidden_size = {hs: [] for hs in hidden_sizes}

# Agregar las precisiones promedio de prueba para cada hidden_size
for (hs, lr), precisiones in precision_prueba.items():
    promedio = np.mean(precisiones)
    promedios_por_hidden_size[hs].append(promedio)

# Calcular el promedio final para cada tamaño de capa oculta
avg_precisions_by_hidden_size = {hs: np.mean(valores) for hs, valores in promedios_por_hidden_size.items()}

# Crear gráfico de barras para precisión promedio por tamaño de capa oculta
plt.figure(figsize=(10, 6))
x_positions = range(1, len(avg_precisions_by_hidden_size) + 1)  # Posiciones fijas en el eje x
plt.bar(x_positions, avg_precisions_by_hidden_size.values(), color="black", width=0.5)  # Ajustar el grosor de las barras
plt.xlabel('Tamaño de la Capa Oculta')
plt.ylabel('Precisión Promedio de Prueba (%)')
plt.title('Precisión Promedio de Prueba por Tamaño de Capa Oculta')
plt.ylim(80, 86)
plt.xticks(x_positions, avg_precisions_by_hidden_size.keys())  # Etiquetas originales en las posiciones fijas

plt.show()
