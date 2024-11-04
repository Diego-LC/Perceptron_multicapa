import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def generar_movimiento_lineal(n_ejemplos: int, n_pasos: int = 10) -> List[np.ndarray]:
    """
    Genera trayectorias de movimiento lineal.
    
    Args:
        n_ejemplos: Número de trayectorias a generar
        n_pasos: Número de pasos de tiempo por trayectoria
    Returns:
        Lista de trayectorias, cada una como matriz numpy de shape (n_pasos, 2)
    """
    trayectorias = []
    
    for _ in range(n_ejemplos):
        # Generar punto inicial aleatorio
        x0 = np.random.uniform(-5, 5)
        y0 = np.random.uniform(-5, 5)
        
        # Generar dirección aleatoria
        angulo = np.random.uniform(0, 2*np.pi)
        dx = np.cos(angulo)
        dy = np.sin(angulo)
        
        # Generar trayectoria
        t = np.linspace(0, np.random.uniform(0.6,1.2), n_pasos)
        x = x0 + dx * t * 5  # multiplicar por 5 para hacer el movimiento más notorio
        y = y0 + dy * t * 5
        
        trayectoria = np.column_stack((x, y))
        trayectorias.append(trayectoria)
    
    return trayectorias

def generar_movimiento_circular(n_ejemplos: int, n_pasos: int = 10) -> List[np.ndarray]:
    """
    Genera trayectorias de movimiento circular.
    """
    trayectorias = []
    
    for _ in range(n_ejemplos):
        # Centro del círculo
        x0 = np.random.uniform(-3, 3)
        y0 = np.random.uniform(-3, 3)
        
        # Radio aleatorio
        radio = np.random.uniform(1, 3)
        
        # Ángulo inicial aleatorio
        angulo_inicial = np.random.uniform(1, 2*np.pi)
        
        # Generar puntos en círculo
        angulos = np.linspace(angulo_inicial, angulo_inicial + 2*np.pi-np.pi/12, n_pasos)
        x = x0 + radio * np.cos(angulos)
        y = y0 + radio * np.sin(angulos)
        
        trayectoria = np.column_stack((x, y))
        trayectorias.append(trayectoria)
    
    return trayectorias

def generar_movimiento_aleatorio(n_ejemplos: int, n_pasos: int = 10) -> List[np.ndarray]:
    """
    Genera trayectorias de movimiento aleatorio (random walk).
    """
    trayectorias = []
    
    for _ in range(n_ejemplos):
        # Punto inicial
        x = [np.random.uniform(-6.5, 6.5)]
        y = [np.random.uniform(-6.5, 6.5)]
        
        # Generar random walk
        for _ in range(n_pasos - 1):
            dx = np.random.normal(0, 1.2)
            dy = np.random.normal(0, 1.2)
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
        
        trayectoria = np.column_stack((x, y))
        trayectorias.append(trayectoria)
    
    return trayectorias

def visualizar_trayectorias(trayectorias_list: List[List[np.ndarray]], 
                            labels: List[str], n_ejemplos: int = 5):
    """
    Visualiza ejemplos de trayectorias de cada tipo de movimiento.
    """
    fig, axes = plt.subplots(1, len(trayectorias_list), figsize=(15, 5))
    
    for i, (trayectorias, label) in enumerate(zip(trayectorias_list, labels)):
        for j in range(min(n_ejemplos, len(trayectorias))):
            trayectoria = trayectorias[j]
            axes[i].plot(trayectoria[:, 0], trayectoria[:, 1], '-o', 
                        label=f'Ejemplo {j+1}', alpha=0.7)
            # Marcar punto inicial
            axes[i].plot(trayectoria[0, 0], trayectoria[0, 1], 'go', 
                        markersize=10, label='Inicio' if j == 0 else "")
        
        axes[i].set_title(f'Movimiento {label}')
        axes[i].grid(True)
        axes[i].legend()
    
    for ax in axes:
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
    plt.tight_layout()
    plt.show()
    plt.pause(2)
    plt.close(fig)

def generar_datos(n_ejemplos_por_clase: int = 100, n_pasos: int = 10
                    ) -> Tuple[List[np.ndarray], List[int]]:
    """
    Genera el conjunto completo de datos de entrenamiento.
    
    Returns:
        Tuple con (datos, etiquetas)
    """
    # Generar trayectorias
    trayectorias_lineales = generar_movimiento_lineal(n_ejemplos_por_clase, n_pasos)
    trayectorias_circulares = generar_movimiento_circular(n_ejemplos_por_clase, n_pasos)
    trayectorias_aleatorias = generar_movimiento_aleatorio(n_ejemplos_por_clase, n_pasos)
    
    # Combinar datos
    X = trayectorias_lineales + trayectorias_circulares + trayectorias_aleatorias
    
    # Generar etiquetas (one-hot encoding)
    y = ([[1,0,0]] * n_ejemplos_por_clase + 
         [[0,1,0]] * n_ejemplos_por_clase + 
         [[0,0,1]] * n_ejemplos_por_clase)
    
    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    """# Mezclar datos
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices] """
    
    return X, y

""" x, y = generar_datos(100)
visualizar_trayectorias([x[y[:,0] == 1], x[y[:,1] == 1], x[y[:,2] == 1]], 
                        ['Lineal', 'Circular', 'Aleatorio']) """

def guardar_datos(X: List[np.ndarray], y: List[int], ruta: str):
    """
    Guarda los datos en un archivo de texto legible.
    """
    with open(ruta, 'w') as f:
        for i in range(len(X)):
            f.write(f"Trayectoria {i+1}\n")
            f.write(f"Datos: {X[i]}\n")
            f.write(f"Etiqueta: {y[i]}\n\n")