import time
import random
import numpy as np
from numpy import sort

import concurrent.futures
import math
import matplotlib.pyplot as plt
# PRIMES = [
#     1,
#     2,
#     3,
#     4,
#     5,
#     6]
#
# core = 8
# PRIMES = [i for i in range(5000)]
# rangeofloop = 10000
# def is_prime(n):
#     for i in range(rangeofloop):
#         end = sum([i for i in range(50)])
#     return end
#
#
# def main():
#     with concurrent.futures.ProcessPoolExecutor(max_workers=core) as executor:
#         executor.map(is_prime, PRIMES)
#
#
# def loop():
#     for i in range(rangeofloop):
#         gaga = sum([i for i in range(50)])
#         # print(gaga)
#     return gaga
#
# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     end= time.time()
#     print(end - start_time)
#
#     start_time = time.time()
#     loop()
#     end = time.time()
#     print(end - start_time)

import ast
def read_list_from_file(file_path):
    """
    Read a list of numbers from a text file and return it as a Python list.

    Args:
        file_path (str): The path to the text file containing the list.

    Returns:
        list: A list of numbers.
    """
    number_list = []

    # Open the file for reading ('r' mode)
    with open(file_path, 'r') as file:
        # Read the content of the file as a string
        file_content = file.read()

        # Use ast.literal_eval to safely evaluate the string as a Python literal
        number_list = ast.literal_eval(file_content)

    return number_list
def best(data):
    output = data[0]
    print(output)
    for i in data:
        if i < output:
            output = i
    return output

# print(len(haha))
# print(best(haha))
# print(haha)

# Specify the file path
# file_path = "Gold.txt"

# Initialize an empty list to store the points


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


errorlist = read_list_from_file("error_list.txt")
print(np.std(errorlist))


# Now, 'points' contains a list of tuples representing the points
Bronze = []
with open("Bronze.txt", 'r') as file:
    # Read each line of the file
    for line in file:
        # Split each line into two parts using a comma as the delimiter
        parts = line.strip().split(', ')

        # Ensure there are exactly two parts in the line
        if len(parts) == 2:
            try:
                # Convert each part to a float and create a tuple
                point = (float(parts[0]), float(parts[1]))
                Bronze.append(point)
            except ValueError:
                print(f"Skipping line: {line.strip()} (invalid format)")

# scatterplot = read_list_from_file("scatter_plot.txt")
scatterplot = read_list_from_file("var_diversity.txt")
x_coords = range(len(scatterplot))
# print(scatterplot)
# plt.loglog(x_coords, scatterplot, marker='', color='b')

plt.plot(range(len(errorlist)), errorlist, marker='', color='r')
# plt.errorbar(errorlist)
std = np.std(errorlist)
plt.errorbar(range(len(errorlist)), errorlist, yerr=[std if i % 1000 == 0 else 0 for i in range(len(errorlist))], color='g', linestyle='-',fmt='|', ms=1,mew=1,capthick=2,capsize=5)

# for i in scatterplot:

#     print(i)
# y_coords,x_coords, = zip(*scatterplot)

# x_coords, y_coords = zip(*points)
# x_coordsb, y_coordsb = zip(*Bronze)
# print(max(y_coords))
# Create a scatter plot
# plt.scatter(x_coords, y_coords, marker='o', color='b')
# plt.scatter(x_coordsb, y_coordsb, marker='o', color='r')
# Add labels and a title
plt.xlabel('Iterations')
plt.ylabel('MSE ')
plt.title('learning curve for Random search')

# Show the plot
plt.grid(True)
plt.show()
