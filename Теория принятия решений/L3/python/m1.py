import numpy as np

# Матрица переходов для финансовой стабильности
transition_probabilities = [
    # S1    S2    S3    S4    S5    S6    S7    S8    S9    S10
    [0.70, 0.20, 0.05, 0.03, 0.01, 0.005, 0.002, 0.001, 0.001, 0.001],  # из S1 (Катастрофа)
    [0.10, 0.60, 0.20, 0.05, 0.03, 0.01, 0.005, 0.002, 0.002, 0.001],  # из S2 (Нищета)
    [0.05, 0.15, 0.55, 0.15, 0.05, 0.02, 0.01, 0.005, 0.004, 0.001],  # из S3 (Низкий доход)
    [0.03, 0.05, 0.17, 0.50, 0.15, 0.05, 0.02, 0.01, 0.01, 0.005],   # из S4 (Стаб. низкий)
    [0.02, 0.03, 0.05, 0.15, 0.45, 0.15, 0.07, 0.04, 0.03, 0.01],    # из S5 (Средний доход)
    [0.01, 0.02, 0.03, 0.05, 0.16, 0.48, 0.15, 0.05, 0.03, 0.02],    # из S6 (Стаб. средний)
    [0.005, 0.01, 0.02, 0.03, 0.05, 0.17, 0.50, 0.15, 0.05, 0.03],   # из S7 (Высокий доход)
    [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.17, 0.55, 0.15, 0.03],  # из S8 (Оч. высокий)
    [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.17, 0.60, 0.12], # из S9 (Состоятельность)
    [0.001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.15, 0.73] # из S10 (Независимость)
]

# Массив состояний
states = [
    "S1: Катастрофа (долги)",
    "S2: Нищета (нет сбережений)", 
    "S3: Низкий доход",
    "S4: Стабильный низкий доход",
    "S5: Средний доход",
    "S6: Стабильный средний доход",
    "S7: Высокий доход",
    "S8: Очень высокий доход",
    "S9: Состоятельность",
    "S10: Финансовая независимость"
]

def predict_financial_state(current_state, years=1):
    # Находим индекс текущего состояния
    try:
        current_state_index = states.index(current_state)
    except ValueError:
        raise ValueError(f"Состояние {current_state} не найдено. Используйте одно из: {states}")
    
    # Создаем вектор начальных вероятностей
    current_probabilities = np.zeros(len(states))
    current_probabilities[current_state_index] = 1
    
    # Прогнозируем на n лет вперед
    for i in range(years):
        current_probabilities = np.dot(current_probabilities, transition_probabilities)
    
    # Находим индекс наиболее вероятного состояния
    predicted_state_index = np.argmax(current_probabilities)
    
    return states[predicted_state_index]

# Примеры использования функции:

# 1. Прогноз на 1 год для человека в катастрофическом положении
print("Человек в катастрофе через 1 год:")
print(predict_financial_state("S1: Катастрофа (долги)", 1))
print()

# 2. Прогноз на 5 лет для человека со средним доходом
print("Человек со средним доходом через 5 лет:")
print(predict_financial_state("S5: Средний доход", 5))
print()

# 3. Прогноз на 10 лет для финансово независимого человека
print("Финансово независимый человек через 10 лет:")
print(predict_financial_state("S10: Финансовая независимость", 10))
print()

# Распределение вероятностей через несколько лет
def show_probability_distribution(current_state, years=1):
    """Показывает полное распределение вероятностей через заданное количество лет"""
    current_state_index = states.index(current_state)
    current_probabilities = np.zeros(len(states))
    current_probabilities[current_state_index] = 1
    
    for i in range(years):
        current_probabilities = np.dot(current_probabilities, transition_probabilities)
    
    print(f"Распределение вероятностей через {years} год(а)/лет из состояния {current_state}:")
    for i, prob in enumerate(current_probabilities):
        print(f"{states[i]}: {prob:.3f} ({prob*100:.1f}%)")
    print()

# Покажем распределение вероятностей
show_probability_distribution("S5: Средний доход", 5)