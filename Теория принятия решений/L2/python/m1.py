import numpy as np
import pandas as pd
import os

class LinearNeuralNetwork:
    def __init__(self, input_size, output_size):
        # Инициализация весов случайными значениями
        self.weights = np.random.rand(input_size, output_size)
    
    def forward(self, X):
        # Прямой проход: умножение входных данных на веса
        return np.dot(X, self.weights)

def load_car_rating_data(file_path):
    """
    Загрузка данных из файла car_rating_data
    Поддерживает форматы: .xlsx, .dbf, .csv
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(file_path):
            # Пробуем альтернативные расширения
            base_path = file_path.rsplit('.', 1)[0]  # Убираем расширение
            alternatives = [base_path + '.xlsx', base_path + '.dbf', base_path + '.csv']
            
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    print(f"Найден файл: {file_path}")
                    break
            else:
                raise FileNotFoundError(f"Файл {file_path} не найден")
        
        # Загрузка данных в зависимости от формата
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.dbf'):
            # Для DBF файлов используем библиотеку simpledbf
            try:
                from simpledbf import Dbf5
                dbf = Dbf5(file_path)
                df = dbf.to_dataframe()
            except ImportError:
                print("Для чтения DBF файлов установите библиотеку: pip install simpledbf")
                return None
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла")
        
        print(f"Данные успешно загружены. Форма: {df.shape}")
        print("Первые 5 строк данных:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def prepare_data(df, target_column=None):
    """
    Подготовка данных для нейронной сети
    """
    # Если целевая колонка не указана, используем последнюю колонку
    if target_column is None:
        target_column = df.columns[-1]
    
    # Отделяем признаки (X) и целевую переменную (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)
    
    print(f"Признаки (X) форма: {X.shape}")
    print(f"Целевая переменная (y) форма: {y.shape}")
    
    return X, y

# Основная программа
if __name__ == "__main__":
    # Путь к файлу с данными
    file_path = "car_rating_data.dbf" 
    
    # Загрузка данных
    df = load_car_rating_data(file_path)
    
    if df is not None:
        # Подготовка данных
        X, y = prepare_data(df)
        
        # Создание экземпляра модели
        input_size = X.shape[1]  # количество признаков
        output_size = y.shape[1]  # количество выходных значений (обычно 1)

        print(f"\nСоздаем модель с {input_size} входами и {output_size} выходами")
        model = LinearNeuralNetwork(input_size, output_size)
        
        # Прямой проход через сеть
        predictions = model.forward(X)
        
        print(f"\nПервые 5 предсказаний:")
        for i in range(min(5, len(predictions))):
            print(f"Реальное: {y[i][0]:.2f}, Предсказанное: {predictions[i][0]:.2f}")
        
        # Вывод информации о весах
        print(f"\nВеса модели: {model.weights.flatten()}")
        
        # Пример предсказания для новых данных
        if input_size == 3:  # если у нас 3 признака
            new_car = np.array([[150, 8.5, 4.2]])  # пример новых данных
            prediction = model.forward(new_car)
            print(f"\nПредсказание для нового автомобиля: {prediction[0][0]:.2f}")
