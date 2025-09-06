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
    
    print(f"Загружено {X.shape[0]} samples с {num_features} признаками")
    print(f"Признаки: {list(feature_columns)}")
    print(f"Целевая переменная: {target_column}")
    print(f"Диапазон признаков после нормализации: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Диапазон целевой переменной после нормализации: [{y.min():.3f}, {y.max():.3f}]")
    
    return X_tensor, y_tensor, dataset, dataloader, num_features, (X_mean, X_std, y_mean, y_std)


X_tensor, y_tensor, dataset, dataloader, num_features, norm_params = load_dbf_data("car_rating_data.dbf")

# X = np.linspace(-10, 10, 100).reshape(-1, 1) # Входные данные
# y = X * 2 + 1 + np.random.normal(scale=0.5, size=(100, 1)) # Выходные данные с шумом
# # Преобразование данных в PyTorch-тензоры
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)
# num_features = 1
# dataset = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Определение структуры линейной нейронной сети с динамическим количеством признаков
class LinearModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=input_features, out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)

# Создание модели с правильным количеством входных признаков
model = LinearModel(num_features)
criterion = nn.MSELoss() # Функция потерь - среднеквадратичная ошибка
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # Уменьшили learning rate

# Обучение модели
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

# Проверка весов модели
print("\nВеса модели:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
    
    
# Тестирование модели - создаем тестовый ввод с правильным количеством признаков
# Берем средние значения всех признаков из обучающих данных (уже нормализованные)
mean_features = torch.mean(X_tensor, dim=0)
test_input = mean_features.unsqueeze(0)  # Добавляем dimension для batch

predicted_output = model(test_input)
print(f"\nПрогнозируемое значение (нормализованное): {predicted_output.item():.4f}")

# Денормализация предсказания
X_mean, X_std, y_mean, y_std = norm_params
print(y_mean, y_std)
predicted_denormalized = predicted_output.item() * y_std + y_mean
print(f"Прогнозируемое значение (денормализованное): {predicted_denormalized:.2f}")
print(f"Количество признаков в тестовом вводе: {test_input.shape[1]}")

# # Тестирование модели
# test_input = torch.tensor(X_mean, dtype=torch.float32)
# # predicted_output = model(X_mean)
# print(f"Прогнозируемое значение для ввода x_mean: {predicted_output.item():.2f}")