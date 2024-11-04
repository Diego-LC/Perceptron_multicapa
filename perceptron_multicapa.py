import numpy as np
from typing import List, Tuple

class PerceptronMulticapa:
    def __init__(self, input_size: int = 20, hidden_size: int = 5, 
                    output_size: int = 3, learning_rate: float = 0.01):
        """
        Inicializa el perceptrón multicapa.
        
        Args:
            input_size: Número de características de entrada (20 para x,y en 10 pasos)
            hidden_size: Número de neuronas en la capa oculta
            output_size: Número de categorías de salida
            learning_rate: Tasa de aprendizaje
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicializar pesos y sesgos
        self.inicializar_pesos()
    
    def inicializar_pesos(self, seed: int = None):
        """Inicializa los pesos con valores aleatorios."""
        if seed is not None:
            np.random.seed(seed)
            
        # Inicialización He para mejores resultados
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2/self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2/self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función de activación sigmoide."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def derivada_sigmoid(self, x: List[List[float]]) -> List[List[float]]:
        """Derivada de la función sigmoide."""
        def sigmoid(val):
            return 1 / (1 + 2.718281828459045 ** -val)
        
        s = [[sigmoid(val) for val in row] for row in x]
        return [[val * (1 - val) for val in row] for row in s]
    
    def softmax(self, x: List[List[float]]) -> List[List[float]]:
        """Función de activación softmax."""
        def exp(val):
            return 2.718281828459045 ** val
        
        result = []
        for row in x:
            max_val = max(row)
            exp_row = [exp(val - max_val) for val in row]
            sum_exp = sum(exp_row)
            result.append([val / sum_exp for val in exp_row])
        return result
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza la propagación hacia adelante.
        
        Args:
            X: Datos de entrada de forma (n_ejemplos, input_size)
        Returns:
            Tuple con activaciones y salidas de cada capa
        """
        # Capa oculta
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        # Capa de salida
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        
        return self.Z1, self.A1, self.Z2, self.A2
    
    def backward(self, X: np.ndarray, etiquetas: np.ndarray, m: int):
        """
        Realiza la propagación hacia atrás.
        
        Args:
            X: Datos de entrada
            etiquetas: Etiquetas verdaderas
            m: Número de ejemplos
        """
        # Gradientes capa de salida
        dZ2 = self.A2 - etiquetas # derivada de la función de pérdida MSE
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Gradientes capa oculta
        dZ1 = np.dot(dZ2, self.W2.T) * self.derivada_sigmoid(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Actualizar pesos y sesgos
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def entrenar(self, X: np.ndarray, etiquetas: np.ndarray, 
                    epochs: int = 1000, verbose: bool = True) -> List[float]:
        """
        Entrena el modelo usando batch gradient descent.
        
        Returns:
            Lista con el histórico de pérdidas
        """
        m = X.shape[0]
        error_cm = []
        
        for epoch in range(epochs):
            # Forward pass
            _, _, _, A2 = self.forward(X)
            
            # Backward pass
            self.backward(X, etiquetas, m)
            
            # Calcular pérdida
            perdida_ecm = np.mean(np.sum(1/2 * (A2 - etiquetas) ** 2, axis=1))
            error_cm.append(perdida_ecm)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Época {epoch+1}/{epochs}, Pérdida: {perdida_ecm:.4f}")
        
        return error_cm
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones para nuevos datos.
        
        Returns:
            Array con las predicciones (one-hot encoding)
        """
        _, _, _, A2 = self.forward(X)
        return np.eye(self.output_size)[np.argmax(A2, axis=1)]
    
    def evaluar(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evalúa el rendimiento del modelo.
        
        Returns:
            Precisión del modelo
        """
        predicciones = self.predecir(X)
        return np.mean(np.all(predicciones == y, axis=1))
    
    def guardar_pesos(self, filename: str):
        """Guarda los pesos del modelo en un archivo."""
        with open(f"{filename}.txt", 'w+') as f:
            np.savetxt(f, self.W1, delimiter=", " ,header="Pesos entrada")
            np.savetxt(f, self.b1, header="Bias entrada")
            np.savetxt(f, self.W2, header="Pesos capa oculta")
            np.savetxt(f, self.b2, header="Bias capa oculta")
