import csv
import random
import math
import numpy as np
from tqdm import tqdm

def cos(x):
    return math.cos(x)
def sin(x):
    return math.sin(x)

import concurrent.futures
# Define the Node class that will represent our function or terminal.
class Node:
    def __init__(self, function=None, terminal=None):
        self.function = function
        self.terminal = terminal
        self.left = None
        self.right = None

    def eval(self, x_val):
        if self.function:
            if self.function in ['+', '-', '*', '/']:
                if self.function == '+': return self.left.eval(x_val) + self.right.eval(x_val)
                if self.function == '-': return self.left.eval(x_val) - self.right.eval(x_val)
                if self.function == '*': return self.left.eval(x_val) * self.right.eval(x_val)
                if self.function == '/':
                    denom = self.right.eval(x_val)
                    if denom == 0: return 1
                    return self.left.eval(x_val) / denom
            elif self.function == 'sin':
                return math.sin(self.left.eval(x_val))
            elif self.function == 'cos':
                return math.cos(self.left.eval(x_val))
        else:
            if self.terminal == 'x':
                return x_val
            return self.terminal
    def __str__(self):
        if self.function == '+':
            return f"({str(self.left)} + {str(self.right)})"
        elif self.function ==  '-':
            return f"({str(self.left)} - {str(self.right)})"
        elif self.function == '*':
            return f"({str(self.left)} * {str(self.right)})"
        elif self.function == '/':
            return f"({str(self.left)} / {str(self.right)})"
        elif self.function == 'sin':
            return f"sin({str(self.left)})"
        elif self.function == 'cos':
            return f"cos({str(self.left)})"
        else:
            return str(self.terminal)

    def str_to_node(s):
        # This is a simplistic parser for demonstration purposes.
        # It assumes the format "(value left_child right_child)" for non-leaf nodes.
        s = s.strip()

        # If it's a leaf node (e.g., "x" or "5")
        if s[0] != '(':
            return Node(s)

        # Removing the outer parentheses
        s = s[1:-1].strip()

        # Extracting value (assuming a single character for simplicity)
        value = s[0]

        # Find the position to split the string for left and right child
        # This assumes only single characters for node values and spaces in between.
        pos = s[2:].index(' ') + 2

        left_child = str_to_node(s[2:pos])
        right_child = str_to_node(s[pos + 1:])

        return Node(value, left_child, right_child)
def exportFile(equation_list,error_list):
    with open("equation_list.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(equation_list)

    with open("error_list.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ',escapechar=' ')
        csv_writer.writerow(error_list)
# Generate a random expression tree
def generate_tree(depth):
    if depth == 0:
        if random.random() < 0.5:
            return Node(terminal='x')
        else:
            return Node(terminal=random.uniform(-10, 10))
    else:
        n = Node()
        func = random.choice(['+', '-', '*', '/', 'sin', 'cos'])
        n.function = func
        n.left = generate_tree(depth - 1)
        if func not in ['sin', 'cos']:
            n.right = generate_tree(depth - 1)
        return n

# Compute the mean squared error as fitness function
def mse(tree, x_values, y_values):

    errors = [(tree.eval(x) - y)**2 for x, y in zip(x_values, y_values)]
    # print(errors)
    return np.mean(errors)


# Mutate a given tree by replacing a random node with a new subtree
def mutate_tree(tree, depth=2):
    all_nodes = get_all_nodes(tree)
    node_to_mutate = random.choice(all_nodes)

    # Replacing the node by generating a new subtree
    if random.random() < 0.5:
        node_to_mutate.left = generate_tree(depth)
    else:
        node_to_mutate.right = generate_tree(depth)

    return tree


# Tournament selection
def tournament_select(population, x_values, y_values, tournament_size=5):
    selected = random.sample(population, tournament_size) #varience
    selected.sort(key=lambda t: mse(t, x_values, y_values))
    return selected[0]

# Subtree crossover
def crossover(parent1, parent2):
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)

    swap_node_1 = random.choice(get_all_nodes(child1))
    swap_node_2 = random.choice(get_all_nodes(child2))

    swap_node_1.function, swap_node_2.function = swap_node_2.function, swap_node_1.function
    swap_node_1.terminal, swap_node_2.terminal = swap_node_2.terminal, swap_node_1.terminal
    swap_node_1.left, swap_node_2.left = swap_node_2.left, swap_node_1.left
    swap_node_1.right, swap_node_2.right = swap_node_2.right, swap_node_1.right

    return child1, child2

# def mutate(expression, prob=0.1):
#     if random.random() < prob:
#         return random_expression()
#     return expression
# Get all nodes from a tree
def get_all_nodes(tree):
    nodes = [tree]
    if tree.left:
        nodes.extend(get_all_nodes(tree.left))
    if tree.right:
        nodes.extend(get_all_nodes(tree.right))
    return nodes

from copy import deepcopy

# Genetic programming main loop
def genetic_programming(x_values, y_values, generations=10, pop_size=100, depth=4):
    population = [generate_tree(depth) for _ in range(pop_size)]

    for generation in tqdm(range(generations)):
        new_population = []

        for i in range(pop_size // 2):
            parent1 = tournament_select(population, x_values, y_values)
            parent2 = tournament_select(population, x_values, y_values)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        population = new_population
        # print(new_population[1].eval(1).__str__())
    best_tree = min(population, key=lambda t: mse(t, x_values, y_values))

    return best_tree

def write_data_to_file(data, file_path):

    with open(file_path, 'w') as file:
        # Write the data to the file
        file.write(data)

def best(data):
    output = data[0]
    # print(output)
    for i in data:
        if i < output:
            output = i
    return output

def random_search(x_values, y_values, iterations=1000, depth=4,pop=100):
    best_tree = None
    best_score = float('inf')
    best_score_List = []# for learning curve.
    best_tree_list=[]
    for g in tqdm(range(iterations)):
        candidate_tree = generate_tree(depth)
        candidate_score = mse(candidate_tree, x_values, y_values)
        if candidate_score < best_score:
            best_tree = candidate_tree
            best_score = candidate_score
            best_score_List.append(candidate_score)
            best_tree_list.append(best_tree.__str__())
        best_score_List.append(best_score)
        best_tree_list.append(candidate_tree.__str__())
    write_data_to_file(str(best_score_List),"best_score_List.txt")
    write_data_to_file(str(best_tree_list), "best_tree_list.txt")
    return best_tree_list,best_score_List,best_tree

def hill_climbing(x_values, y_values, iterations=1000, depth=4, mutation_depth=4,pop=100):

    best_score_List = []# for learning curve.
    best_tree_list = []
    nodes_list = []
    best_tree = generate_tree(depth)
    best_score = mse(best_tree, x_values, y_values)

    for _ in tqdm(range(iterations)):
        # Create a new mutated version of the current tree
        new_tree = mutate_tree(deepcopy(best_tree), mutation_depth)
        new_score = mse(new_tree, x_values, y_values)

        # If the new tree is better, update our current tree and score
        if new_score < best_score:
            best_tree, best_score = new_tree, new_score
            nodes_list.append(new_tree.__str__())

    write_data_to_file(str(best_score_List), "best_score_List.txt")
    write_data_to_file(str(best_tree_list), "best_tree_list.txt")
    write_data_to_file(str(nodes_list.__str__()), "nodes_list.txt")
    return best_tree,best_score,nodes_list


# Example usage
if __name__ == "__main__":
    iter = 10
    # Generate random data
    x_values = np.linspace(-10, 10, 1000)
    y_values = 3 * x_values + 2 + np.sin(x_values) + np.random.normal(0, 0.5, size=1000)

    data = []
    with open('D:\Hod evolutionary\Working Area\Assignment 2\data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append([float(row[0]), float(row[1])])


    # best_tree= hill_climbing(x_values, y_values,iterations=iter)
    # best_tree = genetic_programming(x_values, y_values)
    best_tree_list,best_list,nodes_list = hill_climbing(x_values, y_values,iterations=iter)
    print(best_tree_list)
    print(nodes_list)
    print(best_tree_list.eval(1))
#0.8040391840243081
x = 1
b = sin(cos((sin((x * x)) - ((0.19971975244503426 * 4.413773222851702) * cos(x)))))
print(b)