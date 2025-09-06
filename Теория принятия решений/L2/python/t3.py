import tensorflow as tf
from tensorflow import keras
from keras import layers, models


# Загрузка датасета MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных (приведение значений пикселей к диапазону [0, 1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# Создание модели FNN
model = models.Sequential([
    # Флеттен слой для преобразования двумерных изображений в одномерный вектор
    layers.Flatten(input_shape=(28, 28)),
    
    # Полносвязный слой с 128 нейронами и функцией активации ReLU
    layers.Dense(128, activation='relu'),
    
    # Выходной слой с softmax активацией для многоклассовой классификации
    layers.Dense(10, activation='softmax')
])

# Компилирование модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Функция потерь для категориальной классификации
              metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Оценка точности модели на тестовом наборе
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность на тестовых данных: {test_acc*100:.2f}%")

# Дополнительно: вывод структуры модели
model.summary()