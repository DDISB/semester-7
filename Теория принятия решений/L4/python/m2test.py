import random
import numpy as np

# Функция расчета стоимости маршрута
def calculate_total_distance(route, distance_matrix):
    total_dist = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) + distance_matrix[route[-1]][route[0]]
    return total_dist

# Инициализация начальной популяции
def initialize_population(population_size, num_cities):
    population = []
    cities = list(range(num_cities))
    for _ in range(population_size):
        random.shuffle(cities)
        population.append(list(cities))
    return population

# Отбор особей (элитизм + турнирный отбор)
def selection(population, fitness_scores, elite_size=2):
    sorted_indices = np.argsort(fitness_scores)
    elites = [population[idx] for idx in sorted_indices[:elite_size]]
    
    tournament_participants = random.sample(list(sorted_indices[elite_size:]), k=len(population) - elite_size)
    winners = [population[idx] for idx in tournament_participants]
    
    return elites + winners

# Кроссовер (частичный кроссовер OX)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end+1] = parent1[start:end+1]
    
    remaining_genes = [gene for gene in parent2 if gene not in child]
    
    index = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining_genes[index]
            index += 1
    
    return child

# Мутация (swap mutation)
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Основной процесс эволюции
def genetic_algorithm(distance_matrix, population_size=100, generations=100, elite_size=2, mutation_rate=0.1):
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)
    
    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 60)
    print(f"Количество городов: {num_cities}")
    print(f"Размер популяции: {population_size}")
    print(f"Количество поколений: {generations}")
    print(f"Размер элиты: {elite_size}")
    print(f"Вероятность мутации: {mutation_rate}")
    print("=" * 60)
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for generation in range(generations):
        # Расчет fitness для всей популяции
        fitness_scores = [calculate_total_distance(route, distance_matrix) for route in population]
        
        # Статистика поколения
        best_fitness = min(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = max(fitness_scores)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Вывод информации каждые 10 поколений или в последнем поколении
        if generation % 10 == 0 or generation == generations - 1:
            print(f"Поколение {generation:3d}: Лучшая длина = {best_fitness:6.1f}, "
                  f"Средняя длина = {avg_fitness:6.1f}, "
                  f"Худшая длина = {worst_fitness:6.1f}")
        
        # Отбор родителей
        selected_parents = selection(population, fitness_scores, elite_size)
        
        # Создание потомства
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child, mutation_rate)
            offspring.append(mutated_child)
        
        population = offspring
    
    # Поиск лучшего решения
    final_fitness_scores = [calculate_total_distance(route, distance_matrix) for route in population]
    best_fitness = min(final_fitness_scores)
    best_route = [route for route in population if calculate_total_distance(route, distance_matrix) == best_fitness][0]
    
    # Вывод итоговой статистики
    print("=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Лучший маршрут: {best_route}")
    print(f"Общая длина маршрута: {best_fitness}")
    
    # Дополнительная статистика
    improvement = best_fitness_history[0] - best_fitness
    improvement_percent = (improvement / best_fitness_history[0]) * 100 if best_fitness_history[0] > 0 else 0
    
    print(f"Улучшение от начального решения: {improvement:.1f} ({improvement_percent:.1f}%)")
    print(f"Лучшее решение найдено в поколении: {np.argmin(best_fitness_history)}")
    print("=" * 60)
    
    return best_route, best_fitness, best_fitness_history, avg_fitness_history

# Основная матрица расстояний
distance_matrix = [
    [0, 10, 15, 20, 25, 30],
    [10, 0, 5, 15, 20, 25],
    [15, 5, 0, 10, 15, 20],
    [20, 15, 10, 0, 5, 10],
    [25, 20, 15, 5, 0, 5],
    [30, 25, 20, 10, 5, 0]
]

# Запуск генетического алгоритма
best_route, best_fitness, best_history, avg_history = genetic_algorithm(
    distance_matrix, 
    population_size=50, 
    generations=50, 
    elite_size=3, 
    mutation_rate=0.15
)

# Дополнительный анализ результатов
print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
print(f"Начальная лучшая длина: {best_history[0]:.1f}")
print(f"Конечная лучшая длина: {best_fitness:.1f}")
print(f"Общее улучшение: {best_history[0] - best_fitness:.1f}")

# Поиск плато сходимости
plateau_start = None
for i in range(1, len(best_history)):
    if abs(best_history[i] - best_history[i-1]) < 0.1:  # Минимальное изменение
        if plateau_start is None:
            plateau_start = i
    else:
        plateau_start = None

if plateau_start is not None:
    print(f"Алгоритм достиг плато сходимости на поколении {plateau_start}")
else:
    print("Алгоритм продолжал улучшаться до конца обучения")

print("=" * 60)