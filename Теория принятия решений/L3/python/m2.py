import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

def plot_transition_graph():
    """Визуализирует матрицу переходов в виде графа"""
    G = nx.DiGraph()
    
    # Добавляем узлы
    for i, state in enumerate(states):
        short_name = state.split(":")[0]  # Берем только S1, S2 и т.д.
        G.add_node(short_name, full_name=state)
    
    # Добавляем ребра с весами (вероятностями перехода)
    for i in range(len(states)):
        for j in range(len(states)):
            prob = transition_probabilities[i][j]
            if prob > 0.01:  # Показываем только значимые переходы (>1%)
                from_node = states[i].split(":")[0]
                to_node = states[j].split(":")[0]
                G.add_edge(from_node, to_node, weight=prob, label=f"{prob:.3f}")
    
    # Создаем график
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', 
                          alpha=0.9, edgecolors='black')
    
    # Рисуем ребра
    edge_labels = {(u, v): f"{d['weight']:.3f}" 
                   for u, v, d in G.edges(data=True) if d['weight'] > 0.05}
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          connectionstyle="arc3,rad=0.1")
    
    # Подписи узлов
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Подписи ребер (только для вероятностей > 5%)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    # Добавляем легенду с полными названиями состояний
    legend_text = "\n".join([f"{state.split(':')[0]}: {state.split(':')[1].strip()}" 
                           for state in states])
    plt.figtext(0.02, 0.02, legend_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.title("Граф переходов между финансовыми состояниями\n(показаны вероятности > 5%)", 
             fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def print_transition_matrix():
    """Выводит матрицу переходов в читаемом формате"""
    print("Матрица переходов между финансовыми состояниями:")
    print("Из\\В    ", end="")
    for state in states:
        print(f"{state.split(':')[0]:<8}", end="")
    print()
    
    for i, row in enumerate(transition_probabilities):
        print(f"{states[i].split(':')[0]:<8}", end="")
        for prob in row:
            print(f"{prob:<8.3f}", end="")
        print()

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

# Выводим матрицу переходов
print("="*80)
print_transition_matrix()
print("="*80)

# Строим граф переходов
plot_transition_graph()

# Примеры использования функции:
print("Примеры прогнозов:")
print("1. Человек в катастрофе через 1 год:")
print(predict_financial_state("S1: Катастрофа (долги)", 1))
print()

print("2. Человек со средним доходом через 5 лет:")
print(predict_financial_state("S5: Средний доход", 5))
print()

print("3. Финансово независимый человек через 10 лет:")
print(predict_financial_state("S10: Финансовая независимость", 10))
print()

# Покажем распределение вероятностей
show_probability_distribution("S5: Средний доход", 5)