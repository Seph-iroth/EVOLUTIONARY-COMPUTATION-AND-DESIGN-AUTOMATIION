import csv
import random
import math
import ast
import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify, symbols
from tqdm import tqdm
import matplotlib

def cos(x):
    return math.cos(x)
def sin(x):
    return math.sin(x)


def read_txt_inROW(file_path):

    # Open the file for reading ('r' mode)
    points = []
    with open(file_path, 'r') as file:
        # Read each line of the file
        points = []
        for line in file:
            # Split each line into two parts using a comma as the delimiter
            parts = line.strip().split(', ')

            # Ensure there are exactly two parts in the line
            if len(parts) == 2:
                try:
                    # Convert each part to a float and create a tuple
                    point = (float(parts[0]), float(parts[1]))
                    points.append(point)
                except ValueError:
                    print(f"Skipping line: {line.strip()} (invalid format)")
    print(points)
    return points

def read_list_from_file(filename):
    with open(filename, 'r') as file:
        # Read the content of the file
        content = file.read()

        # Use ast.literal_eval to safely evaluate the string and convert it to a list
        return ast.literal_eval(content)

def strExpression_code_eval(exp,input):
    # Define the symbol 'x'
    x = symbols('x')
    # Convert the string into a SymPy expression
    expression = sympify(exp)
    # If you want to evaluate the expression for a specific value of x, say x=2:
    return expression.subs(x, input).evalf()

def get_csv(filename):
    data_list = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each string number to an integer and then form a tuple
            tuple_data = (float(row[0]), float(row[1]))
            data_list.append(tuple_data)

    return data_list




platilum = "D:\Hod evolutionary\Working Area\Assignment 2\Ramdom search\Platinum.csv"

real = get_csv(platilum)
x, y = [list(t) for t in zip(*real)]
plt.plot(x, y,color='green', marker='', linestyle='dashed')

best_tree_list = "best_tree_list.txt"
predict = read_list_from_file(best_tree_list)
print(predict[-1])

plt.title('Customized Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# Display the plot
# plt.show()