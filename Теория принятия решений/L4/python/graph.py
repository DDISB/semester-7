import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Создаем сетку координат
x = np.linspace(-2, 2, 100) # Диапазон от -2 до 2 по оси X
y = np.linspace(-2, 2, 100) # Диапазон от -2 до 2 по оси Y
X, Y = np.meshgrid(x, y) # Создание матриц координат

# 2. Вычисляем значение функции Z = F(X, Y) в каждой точке сетки
Z = -X * np.exp(-X**2 - Y**2)

# 3. Создаем 3D-рисунок
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. Строим поверхность
# cmap='viridis' задает цветовую карту, alpha - прозрачность
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# 5. Добавляем контурную проекцию (линии уровня) на "пол"
cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z) - 0.1, cmap='viridis', alpha=0.3)

# 6. Настройка меток осей и заголовка
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')
ax.set_title('График F(x,y) = -x * e^(-x² - y²)')

# 7. Добавляем цветовую панель
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# 8. Показываем график
plt.show()