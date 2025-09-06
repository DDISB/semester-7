import numpy as np

# Определение класса для линейной нейронной сети
class LinearNeuralNetwork:
    def __init__(self, input_size, output_size):
        # Инициализация весов случайными значениями
        self.weights = np.random.rand(input_size, output_size)
    
    def forward(self, X):
        # Прямой проход: умножение входных данных на веса
        return np.dot(X, self.weights)

# Пример использования
if __name__ == "__main__":
    # Размерность входа и выхода
    input_size = 3
    output_size = 1
    
    # Создание экземпляра модели
    model = LinearNeuralNetwork(input_size, output_size)
    
    # Входные данные (пример)
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Прямой проход
    outputs = model.forward(inputs)
    
    print("Выходные значения:", outputs)