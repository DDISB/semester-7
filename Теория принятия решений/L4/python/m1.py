import random
import numpy as np
import math

# Фитнес-функция (целевая функция) для двух переменных
def fitness_function(x, y):
    return -x * math.exp(-x**2 - y**2)

# Функция инициализации начальной популяции
def initialize_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        # Генерация двух хромосом: одна для x, одна для y
        chromosome_x = ''.join(random.choice('01') for _ in range(chromosome_length))
        chromosome_y = ''.join(random.choice('01') for _ in range(chromosome_length))
        population.append((chromosome_x, chromosome_y))
    return population

# Декодирование хромосомы обратно в число (для отрезка [0, 10])
def decode_chromosome(chromosome, min_value=0, max_value=10):
    decimal_value = int(chromosome, 2)
    scaled_value = min_value + (decimal_value / ((2**len(chromosome)) - 1)) * (max_value - min_value)
    return scaled_value

# Отбор особей для скрещивания (для минимизации выбираем лучших - с наименьшими значениями)
def selection(population, fitness_values, num_parents):
    # Сортируем по возрастанию fitness (лучшие для минимизации первые)
    parents = sorted(zip(population, fitness_values), key=lambda x: x[1])[:num_parents]
    return [parent[0] for parent in parents]

# Одноточечный кроссовер для пар хромосом
def crossover(parent1, parent2):
    # Кроссовер для x-хромосом
    point_x = random.randint(1, len(parent1[0]) - 1)
    child1_x = parent1[0][:point_x] + parent2[0][point_x:]
    child2_x = parent2[0][:point_x] + parent1[0][point_x:]
    
    # Кроссовер для y-хромосом
    point_y = random.randint(1, len(parent1[1]) - 1)
    child1_y = parent1[1][:point_y] + parent2[1][point_y:]
    child2_y = parent2[1][:point_y] + parent1[1][point_y:]
    
    return (child1_x, child1_y), (child2_x, child2_y)

# Мутация с определенной вероятностью для пар хромосом
def mutation(child, mutation_rate):
    mutated_x = ""
    mutated_y = ""
    
    # Мутация x-хромосомы
    for bit in child[0]:
        if random.random() < mutation_rate:
            mutated_x += '1' if bit == '0' else '0'
        else:
            mutated_x += bit
    
    # Мутация y-хромосомы
    for bit in child[1]:
        if random.random() < mutation_rate:
            mutated_y += '1' if bit == '0' else '0'
        else:
            mutated_y += bit
    
    return (mutated_x, mutated_y)

# Основной цикл генетического алгоритма для поиска минимума
def genetic_algorithm(population_size=100, generations=100, chromosome_length=16, mutation_rate=0.01):
    population = initialize_population(population_size, chromosome_length)
    
    best_fitness_history = []
    
    for generation in range(generations):
        # Декодируем и вычисляем fitness для каждой особи
        decoded_population = [(decode_chromosome(chromo_x), decode_chromosome(chromo_y)) 
                             for chromo_x, chromo_y in population]
        fitness_values = [fitness_function(x, y) for x, y in decoded_population]
        
        # Для минимизации ищем наименьшее значение fitness
        best_fitness = min(fitness_values)
        best_index = fitness_values.index(best_fitness)
        best_x, best_y = decoded_population[best_index]
        best_fitness_history.append(best_fitness)
        
        print(f"Generation {generation}: Best Fitness={best_fitness:.6f}, x={best_x:.4f}, y={best_y:.4f}")
        
        selected_parents = selection(population, fitness_values, population_size // 2)
        
        new_population = []
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # Находим лучшую особь в финальной популяции (минимум)
    final_decoded = [(decode_chromosome(chromo_x), decode_chromosome(chromo_y)) 
                    for chromo_x, chromo_y in population]
    final_fitness = [fitness_function(x, y) for x, y in final_decoded]
    
    best_index = final_fitness.index(min(final_fitness))
    best_x, best_y = final_decoded[best_index]
    best_fitness = final_fitness[best_index]
    
    return (best_x, best_y), best_fitness, best_fitness_history

if __name__ == "__main__":
    best_solution, best_fitness, history = genetic_algorithm(
        population_size=100, 
        generations=50, 
        chromosome_length=20, 
        mutation_rate=0.02
    )
    best_x, best_y = best_solution
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Найденный минимум: x={best_x:.6f}, y={best_y:.6f}")
    print(f"Значение функции в минимуме: {best_fitness:.8f}")
    print(f"Проверка: f({best_x:.3f}, {best_y:.3f}) = {-best_x * math.exp(-best_x**2 - best_y**2):.8f}")