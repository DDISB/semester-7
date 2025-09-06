import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Генерация случайных данных для примера
np.random.seed(42)
X = np.linspace(-10, 10, 100).reshape(-1, 1) # Входные данные
y = X * 2 + 1 + np.random.normal(scale=0.5, size=(100, 1)) # Выходные данные с шумом
# Преобразование данных в PyTorch-тензоры
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
# print(X_tensor)
# print(y_tensor)
# Создание датасета и загрузчика данных
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Определение структуры линейной нейронной сети
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)

model = LinearModel()
criterion = nn.MSELoss() # Функция потерь - среднеквадратичная ошибка
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Оптимизатор градиентного спуска

# Обучение модели
epochs = 100

for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Тестирование модели
test_input = torch.tensor([[5.0]], dtype=torch.float32)
predicted_output = model(test_input)
print(f"Прогнозируемое значение для ввода 5: {predicted_output.item():.2f}")