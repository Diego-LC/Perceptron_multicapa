import numpy as np
from datos_movimiento import (generar_datos, visualizar_trayectorias,
                            generar_movimiento_lineal, generar_movimiento_circular,
                            generar_movimiento_aleatorio, guardar_datos)
from perceptron_multicapa import PerceptronMulticapa
import matplotlib.pyplot as plt

def preparar_datos(trayectorias: np.ndarray) -> np.ndarray:
    """
    Prepara los datos para el perceptrón multicapa.
    Aplana cada trayectoria en un vector de 20 elementos (10 pares x,y).
    """
    return trayectorias.reshape(trayectorias.shape[0], -1)

def visualizar_error_entrenamiento(error_cm, config):
    plt.figure(figsize=(10, 5))
    plt.plot(error_cm)
    plt.title(f'Error cuadrático medio durante entrenamiento\n' +
                f'Tamaño capa oculta: {config["hidden_size"]}, Taza aprendizaje: {config["learning_rate"]}')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio')
    plt.grid(True)
    plt.show()
    plt.pause(3)  # Pausar por 3 segundos
    plt.close()  # Cerrar la figura

def main():
    # 1. Generar datos de entrenamiento
    print("Generando datos de entrenamiento...")
    X_train, eiquetas = generar_datos(n_ejemplos_por_clase=100)

    guardar_datos(X_train, eiquetas, "datos_entrenamiento.txt")
    
    # Visualizar algunos ejemplos
    trayectorias_lineales = generar_movimiento_lineal(5)
    trayectorias_circulares = generar_movimiento_circular(5)
    trayectorias_aleatorias = generar_movimiento_aleatorio(5)
    
    visualizar_trayectorias([trayectorias_lineales, trayectorias_circulares, trayectorias_aleatorias],
                            ['Lineal', 'Circular', 'Aleatorio'])
    
    # Preparar datos para el perceptrón
    X_train_flat = preparar_datos(X_train)
    
    # 2. Generar datos de prueba
    print("\nGenerando datos de prueba...")
    X_test, y_test = generar_datos(n_ejemplos_por_clase=30)

    guardar_datos(X_test, y_test, "datos_prueba.txt")
    X_test_flat = preparar_datos(X_test)
    
    # 3. Experimentar con diferentes configuraciones
    configuraciones = [
        {'hidden_size': 5, 'learning_rate': 0.05},
        {'hidden_size': 5, 'learning_rate': 0.2},
        {'hidden_size': 5, 'learning_rate': 0.5},
        {'hidden_size': 5, 'learning_rate': 0.8},
        {'hidden_size': 10, 'learning_rate': 0.05},
        {'hidden_size': 10, 'learning_rate': 0.2},
        {'hidden_size': 10, 'learning_rate': 0.5},
        {'hidden_size': 10, 'learning_rate': 0.8},
        {'hidden_size': 15, 'learning_rate': 0.05},
        {'hidden_size': 15, 'learning_rate': 0.2},
        {'hidden_size': 15, 'learning_rate': 0.5},
        {'hidden_size': 15, 'learning_rate': 0.8},
        {'hidden_size': 18, 'learning_rate': 0.05},
        {'hidden_size': 18, 'learning_rate': 0.2},
        {'hidden_size': 18, 'learning_rate': 0.5},
        {'hidden_size': 18, 'learning_rate': 0.8}
    ]
    
    mejores_resultados = {
        'accuracy': 0,
        'config': None,
        'model': None
    }


    for i, config in enumerate(configuraciones):
        print(f"\nProbando configuración: {config}")
        
        # Crear y entrenar modelo 
        model = PerceptronMulticapa(
            input_size=20,  # 10 pares (x,y)
            hidden_size=config['hidden_size'],
            output_size=3,  # 3 clases de movimiento
            learning_rate=config['learning_rate']
        )

        print(f"\nModelo {i+1} creado y guardado: {model}")
        model.guardar_pesos(f"modelos/modelo_sin_entrenar_{i+1}")

        # Entrenar modelo
        error_cm = model.entrenar(
            X_train_flat, 
            eiquetas,
            epochs=6000,
            #batch_size=32,
            verbose=False
        )
        
        # Evaluar modelo
        accuracy_train = model.evaluar(X_train_flat, eiquetas)
        accuracy_test = model.evaluar(X_test_flat, y_test)
        
        print(f"Accuracy entrenamiento: {accuracy_train:.4f}")
        print(f"Accuracy prueba: {accuracy_test:.4f}")
        
        # Guardar el mejor modelo entrenado
        if accuracy_test > mejores_resultados['accuracy']:
            mejores_resultados['accuracy'] = accuracy_test
            mejores_resultados['config'] = config
            mejores_resultados['model'] = model

            model.guardar_pesos("modelos/modelo_entrenado")
            print(f"\nMejor modelo entrenado guardado {i+1}: {model}")
        
        # Visualizar grafico de error durante entrenamiento
        visualizar_error_entrenamiento(error_cm, config)

    
    # 4. Mostrar mejores resultados
    print("\nMejores resultados:")
    print(f"Configuración: {mejores_resultados['config']}")
    print(f"Accuracy: {mejores_resultados['accuracy']:.4f}")
    
    # 5. Realizar algunas predicciones de ejemplo
    print(f"\nPredicciones de ejemplo (lineal, circular, aleatorio): {len(X_test)/3} c/u\n")
    indices_ejemplo = np.random.choice(len(X_test), len(X_test), replace=False)
    
    incorrectos = 0
    for idx in indices_ejemplo:
        prediccion = mejores_resultados['model'].predecir(X_test_flat[idx:idx+1])
        real = y_test[idx]
        
        tipo_movimiento_pred = ['Lineal', 'Circular', 'Aleatorio'][np.argmax(prediccion)]
        tipo_movimiento_real = ['Lineal', 'Circular', 'Aleatorio'][np.argmax(real)]
        
        if tipo_movimiento_pred == tipo_movimiento_real:
            print(f"Predicción: {tipo_movimiento_pred}, Real: {tipo_movimiento_real} - CORRECTO")
        else:
            incorrectos += 1
            print(f"Predicción: {tipo_movimiento_pred}, Real: {tipo_movimiento_real} - INCORRECTO")
    
    print("\n=============================================")
    print(f"{incorrectos} predicciones incorrectas, {len(X_test) - incorrectos} correctos")
    print(f"{incorrectos/len(X_test)*100:.2f}% incorrectas, {(len(X_test)-incorrectos)/len(X_test)*100:.2f}% correctas")
    print("=============================================")

if __name__ == "__main__":
    main()