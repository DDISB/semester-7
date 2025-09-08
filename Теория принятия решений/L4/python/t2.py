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
    
    for generation in range(generations):
        fitness_scores = [calculate_total_distance(route, distance_matrix) for route in population]
        selected_parents = selection(population, fitness_scores, elite_size)
        
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child, mutation_rate)
            offspring.append(mutated_child)
        
        population = offspring
    
    best_fitness = min([calculate_total_distance(route, distance_matrix) for route in population])
    best_route = [route for route in population if calculate_total_distance(route, distance_matrix) == best_fitness][0]
    
    return best_route, best_fitness

# Основная матрица расстояний (пример)
# distance_matrix = [
#     [0, 10, 15, 20],
#     [10, 0, 35, 25],
#     [15, 35, 0, 30],
#     [20, 25, 30, 0]
# ]
distance_matrix = [
    [0, 10, 15, 20, 25, 30],
    [10, 0, 5, 15, 20, 25],
    [15, 5, 0, 10, 15, 20],
    [20, 15, 10, 0, 5, 10],
    [25, 20, 15, 5, 0, 5],
    [30, 25, 20, 10, 5, 0]
]

# Запуск генетического алгоритма
best_route, best_fitness = genetic_algorithm(distance_matrix)
print("Лучший маршрут:", best_route)
print("Общая длина маршрута:", best_fitness)