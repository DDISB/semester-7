import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Случайные значения X
X = np.linspace(-10, 10, 100).reshape(-1, 1)
Y = 0.5 * X**2 + 2*X + 1 + np.random.randn(*X.shape)*2 # Добавляем небольшой шум
# Преобразовываем в PyTorch-тензоры
X_torch = torch.from_numpy(X.astype(np.float32))
Y_torch = torch.from_numpy(Y.astype(np.float32)).view(-1, 1)
plt.scatter(X, Y)
plt.title('Исходные данные')
plt.show()
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

input_dim = 1 # Размерность входа
hidden_dim = 10 # Количество скрытых нейронов
output_dim = 1 # Регрессия (один выход)
model = FeedForwardNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss() # Критерий оценки MSE
optimizer = optim.Adam(model.parameters(), lr=0.01) # Оптимизатор Adam
num_epochs = 500
losses = []
for epoch in range(num_epochs):
  optimizer.zero_grad()
  predictions = model(X_torch)
  loss = criterion(predictions, Y_torch)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  if (epoch+1)%50==0:
    print(f'Эпоха [{epoch+1}/{num_epochs}], Потеря: {loss.item():.4f}')
    
plt.plot(range(len(losses)), losses)
plt.xlabel("Эпоха")
plt.ylabel("Потеря")
plt.title("График изменения потерь")
plt.show()
with torch.no_grad():
  predicted_values = model(X_torch)
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Истинные значения', color="blue")
plt.plot(X, predicted_values.numpy(), 'r-', lw=3, label='Предсказанные значения')
plt.legend()
plt.title('Результат моделирования')
plt.show()