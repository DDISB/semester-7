import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from simpledbf import Dbf5

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
    X_normalized = (X - X_mean) / (X_std + 1e-8)  # добавляем маленькое значение чтобы избежать деления на 0
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    # Получение количества признаков
    num_features = X.shape[1]
    
    # Преобразование в тензоры PyTorch
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y_normalized, dtype=torch.float32)
    
    # Создание датасета и загрузчика данных
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Загружено {X.shape[0]} samples с {num_features} признаками")
    print(f"Признаки: {list(feature_columns)}")
    print(f"Целевая переменная: {target_column}")
    print(f"Диапазон признаков после нормализации: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    print(f"Диапазон целевой переменной после нормализации: [{y_normalized.min():.3f}, {y_normalized.max():.3f}]")
    
    return X_tensor, y_tensor, dataset, dataloader, num_features, (X_mean, X_std, y_mean, y_std), X, y


# Загрузка данных
X_tensor, y_tensor, dataset, dataloader, num_features, norm_params, X_original, y_original = load_dbf_data("car_rating_data.dbf")
X_mean, X_std, y_mean, y_std = norm_params

# Денормализация целевой переменной для визуализации
y_denormalized = y_tensor.numpy() * y_std + y_mean

# СОРТИРОВКА ДАННЫХ ПО РЕЙТИНГУ
# Получаем индексы для сортировки по возрастанию рейтинга
sorted_indices = torch.argsort(y_tensor, dim=0).flatten()

# Сортируем тензоры
y_sorted = y_denormalized[sorted_indices]

# Визуализация отсортированных данных (денормализованных)
plt.figure(figsize=(12, 6))
plt.plot(y_sorted, 'o-', alpha=0.7, label='Отсортированный рейтинг')
plt.xlabel('Индекс наблюдения (отсортированный по рейтингу)')
plt.ylabel('Рейтинг')
plt.title('Отсортированные значения рейтинга (денормализованные)')
plt.legend()
plt.grid(True, alpha=0.3)
# plt.show()

# Определение архитектуры нейронной сети
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ИСПРАВЛЕНО: Используем правильную размерность входа
input_dim = num_features
hidden_dim = 64  # Увеличиваем количество нейронов
output_dim = 1   # Регрессия (один выход)

model = FeedForwardNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Уменьшаем learning rate

num_epochs = 500
losses = []

# Обучение модели
for epoch in range(num_epochs):
    model.train()  # Режим обучения
    epoch_loss = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / len(dataloader))
    
    if (epoch + 1) % 50 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Средняя потеря: {losses[-1]:.4f}')

# Визуализация процесса обучения
plt.figure(figsize=(10, 5))
plt.plot(range(len(losses)), losses)
plt.xlabel("Эпоха")
plt.ylabel("Потеря (MSE)")
plt.title("График изменения потерь во время обучения")
plt.grid(True, alpha=0.3)
# plt.show()

# Предсказание на всех данных
model.eval()  # Режим оценки
with torch.no_grad():
    predicted_normalized = model(X_tensor)

# Денормализация предсказаний
predicted_denormalized = predicted_normalized.numpy() * y_std + y_mean

# СОРТИРУЕМ ПРЕДСКАЗАНИЯ в том же порядке, что и истинные значения
predicted_sorted = predicted_denormalized[sorted_indices]

# Визуализация результатов с отсортированными данными (денормализованными)
plt.figure(figsize=(14, 7))

# Создаем массив индексов для оси X
x_indices = range(len(y_sorted))

plt.plot(x_indices, y_sorted, 'o-', label='Истинные значения (отсортированные)', 
         alpha=0.7, linewidth=2, markersize=4)
plt.plot(x_indices, predicted_sorted, 's-', label='Предсказанные значения', 
         alpha=0.7, color='red', markersize=3)

plt.xlabel('Индекс наблюдения (отсортированный по рейтингу)')
plt.ylabel('Рейтинг')
plt.title('Сравнение истинных и предсказанных значений (денормализованные, отсортировано по рейтингу)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# # Дополнительная визуализация: ошибки предсказания (в денормализованных единицах)
# errors = y_sorted - predicted_sorted.flatten()
# plt.figure(figsize=(14, 6))
# plt.plot(x_indices, errors, 'o-', alpha=0.7, color='purple')
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
# plt.xlabel('Индекс наблюдения (отсортированный по рейтингу)')
# plt.ylabel('Ошибка предсказания')
# plt.title('Ошибки предсказания (Истинное - Предсказанное) в денормализованных единицах')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# # plt.show()

# Оценка качества модели на денормализованных данных
# mse_denormalized = np.mean(errors**2)
# print(f'Final MSE Loss (денормализованный): {mse_denormalized:.4f}')

# Вывод статистики ошибок в денормализованных единицах
# print(f'Средняя абсолютная ошибка (MAE): {np.mean(np.abs(errors)):.4f}')
# print(f'Максимальная ошибка: {np.max(np.abs(errors)):.4f}')
# print(f'Стандартное отклонение ошибок: {np.std(errors):.4f}')

def test_model_on_new_data(model, dbf_file_path, norm_params):
    """
    Тестирует обученную модель на новых данных из DBF файла
    
    Args:
        model: обученная модель PyTorch
        dbf_file_path (str): путь к новому DBF файлу
        norm_params: параметры нормализации (X_mean, X_std, y_mean, y_std)
    """
    # Загрузка новых данных
    dbf = Dbf5(dbf_file_path)
    df = dbf.to_dataframe()
    
    # Проверка, что в файле есть хотя бы 2 колонки
    if len(df.columns) < 2:
        raise ValueError("DBF файл должен содержать минимум 2 колонки (признаки + целевая переменная)")
    
    # Разделение на признаки и целевую переменную
    feature_columns = df.columns[:-1]
    target_column = df.columns[-1]
    
    # Извлечение данных
    X_new = df[feature_columns].values.astype(np.float32)
    y_new = df[target_column].values.reshape(-1, 1).astype(np.float32)
    
    # Проверка совместимости признаков
    if X_new.shape[1] != norm_params[0].shape[0]:
        raise ValueError(f"Несовпадение количества признаков: модель ожидает {norm_params[0].shape[0]}, а в новых данных {X_new.shape[1]}")
    
    # Нормализация новых данных с использованием параметров обучения
    X_mean, X_std, y_mean, y_std = norm_params
    X_new_normalized = (X_new - X_mean) / (X_std + 1e-8)
    
    # Преобразование в тензоры
    X_tensor_new = torch.tensor(X_new_normalized, dtype=torch.float32)
    
    # Предсказание на новых данных
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(X_tensor_new)
    
    # Денормализация предсказаний
    predicted_real = predicted_normalized.numpy() * y_std + y_mean
    
    # СОРТИРОВКА ДАННЫХ ПО РЕАЛЬНОМУ РЕЙТИНГУ
    sorted_indices_new = np.argsort(y_new.flatten())
    y_real_sorted = y_new[sorted_indices_new]
    predicted_real_sorted = predicted_real[sorted_indices_new]
    
    # Создание графиков
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # График 1: Реальный vs Предсказанный рейтинг
    x_indices = range(len(y_real_sorted))
    
    ax1.plot(x_indices, y_real_sorted, 'o-', label='Реальный рейтинг', 
             alpha=0.7, linewidth=2, markersize=4, color='blue')
    ax1.plot(x_indices, predicted_real_sorted, 's-', label='Предсказанный рейтинг', 
             alpha=0.7, color='red', markersize=3)
    
    ax1.set_xlabel('Индекс наблюдения (отсортированный по реальному рейтингу)')
    ax1.set_ylabel('Рейтинг')
    ax1.set_title('Сравнение реального и предсказанного рейтинга на новых данных')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Ошибки предсказания
    errors = y_real_sorted.flatten() - predicted_real_sorted.flatten()
    
    ax2.plot(x_indices, errors, 'o-', alpha=0.7, color='purple')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Индекс наблюдения (отсортированный по реальному рейтингу)')
    ax2.set_ylabel('Ошибка предсказания')
    ax2.set_title('Ошибки предсказания (Реальный - Предсказанный)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Оценка качества модели на новых данных
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    std_error = np.std(errors)
    
    print("=" * 50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА НОВЫХ ДАННЫХ")
    print("=" * 50)
    print(f'Количество образцов: {len(y_new)}')
    print(f'MSE (Среднеквадратичная ошибка): {mse:.4f}')
    print(f'MAE (Средняя абсолютная ошибка): {mae:.4f}')
    print(f'Максимальная ошибка: {max_error:.4f}')
    print(f'Стандартное отклонение ошибок: {std_error:.4f}')
    print(f'R² (Коэффициент детерминации): {1 - mse/np.var(y_new):.4f}')
    
    return y_new, predicted_real, errors

# Пример использования функции:
test_model_on_new_data(model, "car_rating_data20.dbf", norm_params)