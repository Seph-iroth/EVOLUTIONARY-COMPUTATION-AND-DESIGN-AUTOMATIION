import random

# Define cities and their coordinates
cities = {"A": (0, 0), "B": (1, 3), "C": (2, 1), "D": (3, 2), "E": (4, 4)}

# Create an initial random tour
def create_random_tour(cities):
    tour = list(cities.keys())
    random.shuffle(tour)
    return tour

# Define the number of generations and population size
num_generations = 100
population_size = 50

# Function to calculate the total distance of a tour
def calculate_total_distance(tour, cities):
    total_distance = 0
    for i in range(len(tour) - 1):
        city1 = cities[tour[i]]
        city2 = cities[tour[i + 1]]
        distance = ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5
        total_distance += distance
    return total_distance

# Main Genetic Algorithm loop
population = [create_random_tour(cities) for _ in range(population_size)]

for generation in range(num_generations):
    # Sort the population by tour length (shortest tours first)
    population.sort(key=lambda tour: calculate_total_distance(tour, cities))

    # Select the top-performing tours for reproduction (based on fitness)
    selected_parents = population[:population_size // 2]

    # Perform crossover (order crossover)
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected_parents, 2)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = parent1[start:end] + [city for city in parent2 if city not in parent1[start:end]]
        new_population.append(child)

    # Perform mutation (swap mutation)
    for i in range(population_size):
        if random.random() < 0.1:  # Adjust mutation rate as needed
            idx1, idx2 = random.sample(range(len(new_population[i])), 2)
            new_population[i][idx1], new_population[i][idx2] = new_population[i][idx2], new_population[i][idx1]

    # Replace the old population with the new population
    population = new_population

# Find the best tour in the final population
best_tour = min(population, key=lambda tour: calculate_total_distance(tour, cities))

# Print the best tour and its distance
print("Best Tour:", best_tour)
print("Total Distance:", calculate_total_distance(best_tour, cities))
