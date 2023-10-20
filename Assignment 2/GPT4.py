import csv
import random
import math
import operator
from heapq import heappush, heappop

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# 1. Representation of symbolic expressions as trees

class Node:
    def __init__(self, function=None, terminal=None):
        self.function = function
        self.terminal = terminal
        self.left = None # initialized to None.
        self.right = None # initialized to None.

    def evaluate(self, x):
        if self.function:
            if self.right:  # Binary operators
                return self.function(self.left.evaluate(x), self.right.evaluate(1))
            return self.function(self.left.evaluate(x))  # Unary operators
        return self.terminal if self.terminal != 'x' else x


    def __str__(self):
        if self.function == operator.add:
            return f"({str(self.left)} + {str(self.right)})"
        elif self.function == operator.sub:
            return f"({str(self.left)} - {str(self.right)})"
        elif self.function == operator.mul:
            return f"({str(self.left)} * {str(self.right)})"
        elif self.function == operator.truediv:
            return f"({str(self.left)} / {str(self.right)})"
        elif self.function == math.sin:
            return f"sin({str(self.left)})"
        elif self.function == math.cos:
            return f"cos({str(self.left)})"
        else:
            return str(self.terminal)


FUNCTIONS = [(operator.add, 2), (operator.sub, 2), (operator.mul, 2), (operator.truediv, 2), (math.sin, 1),
             (math.cos, 1)]
TERMINALS = [str(i* random.random()) for i in range(-10, 11)] + ['x']+ [str(i) for i in range(-11, 11)] #constants (e.g. 1.23) or variables (e.g. “x”
# print(TERMINALS)
csv_file_path = 'data.csv'



data = []
with open('data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append([float(row[0]), float(row[1])])

def exportFile(number_list,list_list):
    with open("fitness.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in number_list:
            csv_writer.writerow([i])
        # csv_writer.writerows(number_list)

def random_expression(depth = 4):
    #if depth == 0 or (not depth and random.random() < 0.5):
    if depth == 0: #random.random() generates numbers between 0-1
        return Node(terminal=random.choice(TERMINALS))
        # return Node(terminal="x")

    func, arity = random.choice(FUNCTIONS)
    node = Node(function=func)
    node.left = random_expression(depth - 1)
    if arity == 2:
        node.right = random_expression(depth - 1)
    return node


# 2. Random generation of initial population

def random_population(size=100, depth=3):
    return [random_expression(depth) for _ in range(size)]


# for i in random_population(size=100, depth=4):
#     print(i.__str__())

# 3. Fitness calculation
def fitness(expression, data):
    error = 0.0

    for x, y in data:
        try:
            prediction = expression.evaluate(x) ################### issue!
            error += (prediction - y) ** 2
            # print(prediction)
        except:
            error += float('inf')  # For divisions by zero
            error += error  # For divisions by zero
            # print(error)
    return error

# def fitness(expression, data):
#     error = 0.0
#     for x, y in data:
#         prediction = expression.evaluate(x)
#         error += (prediction - y) ** 2
#
#     return error


# 4. Selection, crossover, and mutation operations

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

def plot_expression(expression, x_values, iteration):
    y_values = [expression.evaluate(x) for x in x_values]
    plt.plot(x_values, y_values, label=f'Generation {iteration}')

# 5. The main loop for the genetic programming
result = []
def symbolic_regression(data, generations=10, population_size=200):

    #generate 100 expressions
    population = random_population(population_size)

    # calculate it's fitness for each expressions.
    heap = [(fitness(expr, data), idx) for idx, expr in enumerate(population)]
    for i in population:
        print(i.__str__())
    for i in population:
        print(fitness(i, data))

    for generation in tqdm(range(generations)):
        new_population = []

        # print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
        for _ in range(population_size // 2):
            idx1 = heappop(heap)[1]
            idx2 = heappop(heap)[1]
            parent1 = population[idx1]
            parent2 = population[idx2]

            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            child1, child2 = mutate(child1), mutate(child2)

            new_population.extend([child1, child2])

        # Reevaluate the population and reform the heap
        population = new_population
        heap = [(fitness(expr, data), idx) for idx, expr in enumerate(population)]

    return population[heap[0][1]]




# Sample data (x,y) pairs
# data = [(i, i ** 2) for i in range(10)]  # Example: trying to fit x^2
# best_expression = symbolic_regression(data)
# x_values = np.linspace(1,100,1)
# y_values = [best_expression.evaluate(1) for x in x_values]
# print(best_expression)




