import csv
import random
import operator
import math
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from tqdm import tqdm





# Representation of symbolic expressions as trees
class Node:
    def __init__(self, function=None, terminal=None):
        self.function = function
        self.terminal = terminal
        self.left = None
        self.right = None

    def evaluate(self, x):
        if self.function:
            if self.right:  # Binary operators
                return self.function(self.left.evaluate(x), self.right.evaluate(x))
            return self.function(self.left.evaluate(x))  # Unary operators
        return self.terminal if self.terminal != 'x' else x


FUNCTIONS = [(operator.add, 2), (operator.sub, 2), (operator.mul, 2), (operator.truediv, 2), (math.sin, 1), (math.cos, 1)]
TERMINALS = [str(i) for i in range(-10, 11)] + ['x']


def random_expression(depth=3):
    if depth == 0 or (not depth and random.random() < 0.5):
        return Node(terminal=random.choice(TERMINALS))
    func, arity = random.choice(FUNCTIONS)
    node = Node(function=func)
    node.left = random_expression(depth - 1)
    if arity == 2:
        node.right = random_expression(depth - 1)
    return node


def random_population(size=100, depth=3):
    return [random_expression(depth) for _ in range(size)]


def fitness(expression, data):
    error = 0.0
    for x, y in data:
        try:
            prediction = expression.evaluate(x)
            error += (prediction - y) ** 2
        except:
            error += float('inf')  # For divisions by zero
    return error


def crossover(parent1, parent2):
    if random.random() < 0.7:  # 70% probability of crossover
        point1 = random.choice([parent1.left, parent1.right])
        point2 = random.choice([parent2.left, parent2.right])
        point1, point2 = point2, point1
    return parent1


def mutate(expression, prob=0.1):
    if random.random() < prob:
        return random_expression()
    return expression


def symbolic_regression(data, generations=1000, population_size=100):
    population = random_population(population_size)

    # Only store fitness in heap. Population and heap will have parallel indexing.
    heap = [fitness(expr, data) for expr in population]

    fitness_history = []

    for _ in tqdm(range(generations)):
        new_population = []
        for _ in range(population_size // 2):
            idx1 = heap.index(min(heap))
            parent1 = population[idx1]
            heap[idx1] = float('inf')  # Set the selected fitness to infinity so it's not selected again

            idx2 = heap.index(min(heap))
            parent2 = population[idx2]
            heap[idx2] = float('inf')

            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            child1, child2 = mutate(child1), mutate(child2)

            new_population.extend([child1, child2])

        population = new_population
        heap = [fitness(expr, data) for expr in population]
        fitness_history.append(min(heap))

    best_expression = population[heap.index(min(heap))]
    return best_expression, fitness_history


# Sample data
data = []
with open('data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append([float(row[0]), float(row[1])])
best_expression, fitness_history = symbolic_regression(data, generations=10)
# data = [(i, i**2) for i in range(10)]
# Plotting
# plt.errorbar(range(len(fitness_history)), fitness_history, yerr=0.5, ecolor='red', capsize=5)
y_values = range(len(fitness_history))
plt.plot(fitness_history, y_values, label='Data Line', color='yellow', marker='', linestyle='')

print(fitness_history)
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Fitness over iterations')
plt.show()
