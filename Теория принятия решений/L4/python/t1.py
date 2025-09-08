import random
import numpy as np

# Фитнес-функция (целевая функция)
def fitness_function(x):
    return x**2 + 10 * np.sin(x)

# Функция инициализации начальной популяции
def initialize_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        # Генерация случайных двоичных хромосом заданной длины
        chromosome = ''.join(random.choice('01') for _ in range(chromosome_length))
        population.append(chromosome)
    return population

# Декодирование хромосомы обратно в число
def decode_chromosome(chromosome, min_value=0, max_value=10):
    decimal_value = int(chromosome, 2)
    scaled_value = min_value + (decimal_value / ((2**len(chromosome)) - 1)) * (max_value - min_value)
    return scaled_value

# Отбор особей для скрещивания
def selection(population, fitness_values, num_parents):
    parents = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)[:num_parents]
    return [parent[0] for parent in parents]

# Одноточечный кроссовер
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Мутация с определенной вероятностью
def mutation(child, mutation_rate):
    mutated_child = ""
    for bit in child:
        if random.random() < mutation_rate:
            mutated_child += '1' if bit == '0' else '0'
        else:
            mutated_child += bit
    return mutated_child

# Основной цикл генетического алгоритма
def genetic_algorithm(population_size=100, generations=100, chromosome_length=8, mutation_rate=0.01):
    population = initialize_population(population_size, chromosome_length)
    
    for generation in range(generations):
        decoded_population = [decode_chromosome(chromosome) for chromosome in population]
        fitness_values = [fitness_function(x) for x in decoded_population]
        
        best_fitness = max(fitness_values)
        print(f"Generation {generation}: Best Fitness={best_fitness}")
        
        selected_parents = selection(population, fitness_values, population_size // 2)
        
        new_population = []
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    best_chromosome = max(population, key=lambda chromo: fitness_function(decode_chromosome(chromo)))
    best_x = decode_chromosome(best_chromosome)
    best_fitness = fitness_function(best_x)
    
    return best_x, best_fitness

if __name__ == "__main__":
    best_solution, best_fitness = genetic_algorithm()
    print(f"\nBest solution found at x={best_solution}, with fitness value={best_fitness:.2f}")