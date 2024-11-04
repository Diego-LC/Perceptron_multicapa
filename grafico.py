import matplotlib.pyplot as plt
import numpy as np

# Datos
hidden_sizes = [5, 10, 15, 18]
learning_rates = [0.05, 0.20, 0.50, 0.80]
precision_prueba = {
    (5, 0.05): [81.11, 74.44, 87.78, 83.33],
    (5, 0.20): [91.11, 88.89, 91.11, 88.89],
    (5, 0.50): [82.22, 91.11, 66.67, 91.11],
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

# Configuración del gráfico
bar_width = 0.2
index = np.arange(4)  # Número de ejecuciones

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.ravel()

# Crear un gráfico separado para cada tamaño de capa oculta
for i, hidden_size in enumerate(hidden_sizes):
    ax = axs[i]
    index = np.arange(4)  # Número de ejecuciones

    # Crear barras para cada tasa de aprendizaje en el gráfico correspondiente
    for j, lr in enumerate(learning_rates):
        precisiones = precision_prueba[(hidden_size, lr)]
        ax.bar(index + j * bar_width, precisiones, bar_width, label=f'LR: {lr}')

    # Configuraciones del gráfico
    ax.set_xlabel('Ejecuciones')
    ax.set_ylabel('Precisión de Prueba (%)')
    ax.set_title(f'Precisión de prueba para tamaño de capa oculta = {hidden_size}')
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels([f'Ejecución {i+1}' for i in range(4)])
    ax.legend(title="Tasa de Aprendizaje", loc='upper right')
    ax.set_ylim(60, 100)  # Establecer el valor mínimo del eje vertical a 40

plt.tight_layout()
plt.show()

