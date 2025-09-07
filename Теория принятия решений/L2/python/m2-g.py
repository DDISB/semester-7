import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from simpledbf import Dbf5
import numpy as np

def load_dbf_data(dbf_file_path):
    """
    Загружает данные из DBF файла с n входными признаками, где последняя колонка - целевая переменная
    
    Args:
        dbf_file_path (str): путь к DBF файлу
    
    Returns:
        tuple: (X_tensor, y_tensor, dataset, dataloader, num_features)
    """
    # Чтение DBF файла
    dbf = Dbf5(dbf_file_path)
    df = dbf.to_dataframe()
    
    # Проверка, что в файле есть хотя бы 2 колонки
    if len(df.columns) < 2:
        raise ValueError("DBF файл должен содержать минимум 2 колонки (признаки + целевая переменная)")
    
    # Разделение на признаки (все колонки кроме последней) и целевую переменную (последняя колонка)
    feature_columns = df.columns[:-1]  # все колонки кроме последней
    target_column = df.columns[-1]     # последняя колонка
    
    # Извлечение данных
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.reshape(-1, 1).astype(np.float32)
    
    # Нормализация данных (очень важно!)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / (X_std + 1e-8)  # добавляем маленькое значение чтобы избежать деления на 0
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / (y_std + 1e-8)
    
    # Получение количества признаков
    num_features = X.shape[1]
    
    # Преобразование в тензоры PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Создание датасета и загрузчика данных
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    return X_tensor, y_tensor, dataset, dataloader, num_features, (X_mean, X_std, y_mean, y_std)


# После вашего кода обучения добавьте:


    # """
    # Прогнозирование для тестовых данных
    # """
    # # Загрузка тестовых данных
    # X_test_tensor = load_dbf_data(test_file_path)
    
    # # Проверка совпадения размерности
    # if X_test_tensor.shape[1] != num_features:
    #     raise ValueError("Несовпадение количества признаков!")
    
    # # Прогнозирование
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(X_test_tensor)
    
    # # Денормализация
    # _, _, y_mean, y_std = norm_params
    # predictions_denormalized = predictions.numpy() * y_std + y_mean
    
    # return predictions, predictions_denormalized

X_tensor, y_tensor, dataset, dataloader, num_features, norm_params = load_dbf_data("car_rating_data1000.dbf")

# Определение структуры линейной нейронной сети с динамическим количеством признаков
class LinearModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=input_features, out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)

# СОЗДАНИЕ МОДЕЛИ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model = LinearModel(num_features)
criterion = nn.MSELoss() # Функция потерь - среднеквадратичная ошибка
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # Уменьшили learning rate

# ОБУЧЕНИЕ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    
# ТЕСТИРОВАНИЕ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def predict_test_data_with_actual(model, test_dbf_file_path, norm_params, num_features):
    """
    Прогнозирует значения для тестовых данных из DBF файла с реальными значениями для сравнения
    
    Args:
        model: обученная модель PyTorch
        test_dbf_file_path (str): путь к тестовому DBF файлу
        norm_params (tuple): параметры нормализации (X_mean, X_std, y_mean, y_std)
        num_features (int): количество признаков, которое ожидает модель
    
    Returns:
        tuple: (predictions_normalized, predictions_denormalized, test_features_original, actual_values)
    """
    # Чтение тестового DBF файла
    dbf = Dbf5(test_dbf_file_path)
    df = dbf.to_dataframe()
    
    # Проверка, что в файле достаточно колонок (признаки + целевая)
    if len(df.columns) != num_features + 1:
        raise ValueError(f"Ожидается {num_features + 1} колонок (10 признаков + 1 целевая), но получено {len(df.columns)}")
    
    # Разделение на признаки (первые 10 колонок) и реальные значения (11-я колонка)
    feature_columns = df.columns[:num_features]  # первые 10 колонок - признаки
    target_column = df.columns[num_features]     # 11-я колонка - реальное значение
    
    # Извлечение тестовых данных
    X_test = df[feature_columns].values.astype(np.float32)
    y_actual = df[target_column].values.astype(np.float32)  # реальные значения
    
    # Сохраняем оригинальные значения признаков (до нормализации)
    test_features_original = X_test.copy()
    
    # Применяем нормализацию с параметрами из обучающей выборки
    X_mean, X_std, y_mean, y_std = norm_params
    X_test_normalized = (X_test - X_mean) / (X_std + 1e-8)
    
    # Преобразование в тензоры PyTorch
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    
    # Прогнозирование
    model.eval()  # Переводим модель в режим оценки
    with torch.no_grad():  # Отключаем вычисление градиентов
        predictions_normalized = model(X_test_tensor)
    
    # Денормализация предсказаний
    predictions_denormalized = predictions_normalized.numpy() * y_std + y_mean
    
    print(f"Загружено {X_test.shape[0]} тестовых векторов")
    print(f"Прогнозирование завершено успешно!")
    
    return predictions_normalized, predictions_denormalized, test_features_original, y_actual

# def print_predictions_results_with_error(predictions_denormalized, test_features_original, actual_values, norm_params):
#     """
#     Красиво выводит результаты прогнозирования с ошибкой
#     """
#     X_mean, X_std, y_mean, y_std = norm_params
    
#     print("\n" + "="*80)
#     print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ С ОШИБКОЙ")
#     print("="*80)
    
#     total_error = 0
#     absolute_errors = []
    
#     for i, (pred, features, actual) in enumerate(zip(predictions_denormalized, test_features_original, actual_values)):
#         pred_value = pred[0]  # предсказанное значение
#         error = pred_value - actual  # ошибка в рублях
#         error_percent = (error / actual) * 100  # ошибка в процентах
#         absolute_error = abs(error)  # абсолютная ошибка
        
#         total_error += error
#         absolute_errors.append(absolute_error)
        
#         # print(f"\nВектор #{i+1}:")
#         # print(f"  Реальный рейтинг: {actual:.2f}")
#         # print(f"  Прогнозируемый рейтинг: {pred_value:.2f}")
#         # print(f"  Ошибка: {error:+.2f}")
#         # print(f"  Абсолютная ошибка: {absolute_error:.2f} руб.")

# # 2. Прогнозируем на тестовых данных с реальными значениями
# test_predictions_norm, test_predictions_denorm, test_features, actual_values = predict_test_data_with_actual(
#     model=model,
#     test_dbf_file_path="car_rating_dataTEST.dbf",
#     norm_params=norm_params,
#     num_features=num_features
# )

# # 3. Выводим результаты с ошибкой
# print_predictions_results_with_error(test_predictions_denorm, test_features, actual_values, norm_params)

def print_predictions_results_with_error(predictions_denormalized, test_features_original, actual_values, norm_params):
    """
    Красиво выводит результаты прогнозирования с ошибкой
    """
    X_mean, X_std, y_mean, y_std = norm_params
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ С ОШИБКОЙ")
    print("="*80)
    
    total_error = 0
    absolute_errors = []
    max_error = 0
    max_error_idx = 0
    
    for i, (pred, features, actual) in enumerate(zip(predictions_denormalized, test_features_original, actual_values)):
        pred_value = pred[0]  # предсказанное значение
        error = pred_value - actual  # ошибка в рублях
        error_percent = (error / actual) * 100  # ошибка в процентах
        absolute_error = abs(error)  # абсолютная ошибка
        
        total_error += error
        absolute_errors.append(absolute_error)
        
        # Отслеживаем максимальную ошибку
        if absolute_error > max_error:
            max_error = absolute_error
            max_error_idx = i
        
        # print(f"\nВектор #{i+1}:")
        # print(f"  Реальный рейтинг: {actual:.2f}")
        # print(f"  Прогнозируемый рейтинг: {pred_value:.2f}")
        # print(f"  Ошибка: {error:+.2f}")
        # print(f"  Абсолютная ошибка: {absolute_error:.2f} руб.")
    
    # Вычисляем метрики ошибок
    mae = np.mean(absolute_errors)  # Средняя абсолютная ошибка (MAE)
    mse = np.mean([e**2 for e in absolute_errors])  # Среднеквадратичная ошибка
    rmse = np.sqrt(mse)  # Корень из среднеквадратичной ошибки
    
    print(f"\nСТАТИСТИКА ОШИБОК:")
    print(f"  Количество наблюдений: {len(absolute_errors)}")
    print(f"  Средняя абсолютная ошибка (MAE): {mae:.4f}")
    print(f"  Среднеквадратичная ошибка (MSE): {mse:.4f}")
    print(f"  Корень из среднеквадратичной ошибки (RMSE): {rmse:.4f}")
    print(f"  Максимальная абсолютная ошибка: {max_error:.4f}")
    print(f"  Наблюдение с максимальной ошибкой: #{max_error_idx + 1}")
    
    # Детали по наблюдению с максимальной ошибкой
    pred_max_error = predictions_denormalized[max_error_idx][0]
    actual_max_error = actual_values[max_error_idx]
    error_max = pred_max_error - actual_max_error
    
    print(f"\nДЕТАЛИ ПО МАКСИМАЛЬНОЙ ОШИБКЕ (наблюдение #{max_error_idx + 1}):")
    print(f"  Реальное значение: {actual_max_error:.4f}")
    print(f"  Предсказанное значение: {pred_max_error:.4f}")
    print(f"  Абсолютная ошибка: {abs(error_max):.4f}")
    print(f"  Относительная ошибка: {error_max:+.4f} ({error_max/actual_max_error*100:+.2f}%)")
    
    # Дополнительная статистика
    print(f"\nДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print(f"  Медианная абсолютная ошибка: {np.median(absolute_errors):.4f}")
    print(f"  Стандартное отклонение ошибок: {np.std(absolute_errors):.4f}")
    print(f"  Минимальная абсолютная ошибка: {np.min(absolute_errors):.4f}")
    print("="*80)

# 2. Прогнозируем на тестовых данных с реальными значениями
test_predictions_norm, test_predictions_denorm, test_features, actual_values = predict_test_data_with_actual(
    model=model,
    test_dbf_file_path="car_rating_dataTEST.dbf",
    norm_params=norm_params,
    num_features=num_features
)

# 3. Выводим результаты с ошибкой
print_predictions_results_with_error(test_predictions_denorm, test_features, actual_values, norm_params)