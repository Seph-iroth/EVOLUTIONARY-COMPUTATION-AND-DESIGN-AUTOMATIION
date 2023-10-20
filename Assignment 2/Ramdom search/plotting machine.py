import csv
import random
import math
import ast
import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify, symbols
from tqdm import tqdm
import matplotlib
import winsound
winsound.Beep(500, 100)  # Frequency = 1000 Hz, Duration = 1000 ms = 1 second
import statistics



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

def equation(x):
    y = ((cos((sin(((x - x) - cos(2.3511267615850944))) / cos(cos((x - x))))) + (((sin(cos(x)) - cos(cos(6.8881438899192915))) + cos(cos((x * 1.4845855503051233)))) - (((cos(x) / cos(6.584966212628782)) - (sin(x) - sin(4.706860020769653))) * sin(sin((0.22173165835762276 - x)))))) * sin(((((sin(-6.205405348762733) + sin(x)) * ((x * x) + (x / x))) * ((sin(-8.262864272878685) - sin(x)) + ((x * x) * (x / 7.233569687084863)))) - cos((cos((x + x)) + (cos(-5.381945634049177) - (7.787117480762888 + -7.078444956676353)))))))

    return y


# platilum = "D:\Hod evolutionary\Working Area\Assignment 2\Ramdom search\Platinum.csv"

# real = get_csv(platilum)
# x, y = [list(t) for t in zip(*real)]
# plt.plot(x, y,color='green', marker='', linestyle='dashed',label='True')

# best_tree_list = "best_tree_list.txt"
# predict = read_list_from_file(best_tree_list)
# print(predict[0])


# y_predict =  []
# for i in x:
#     try:
#         y=equation(i)
#         y_predict.append(y)
#     except:
#         y_predict.append(0)




# plt.plot(x, y_predict,color='red', marker='', linestyle='dashed',label='Predict')

# plt.title('Random Search')

hill_climb_1 = read_list_from_file("1_score")
std = np.std(hill_climb_1)
print(std)
x = list(range(len(hill_climb_1)))
plt.semilogy(x, hill_climb_1,color='red',label="hill climb bronze")
plt.errorbar(x, hill_climb_1, yerr=std,xerr = 0, color='red', linestyle='None',capthick=1,capsize=5,errorevery=3200,ecolor='red')

hill_climb_1 = read_list_from_file("2_score")
std = np.std(hill_climb_1)
print(statistics.stdev(hill_climb_1))
x = list(range(len(hill_climb_1)))
plt.semilogy(x, hill_climb_1,color='green',label="hill climb sliver")
plt.errorbar(x, hill_climb_1, yerr=std,xerr = 0, color='green', linestyle='None',capthick=1,capsize=5,errorevery=3300,ecolor='green')

hill_climb_1 = read_list_from_file("3_score")
std = np.std(hill_climb_1)
print(std)
x = list(range(len(hill_climb_1)))
plt.semilogy(x, hill_climb_1,color='black',label="hill climb gold")
plt.errorbar(x, hill_climb_1, yerr=std,xerr = 0, color='black', linestyle='None',capthick=1,capsize=5,errorevery=3400,ecolor='black')

hill_climb_1 = read_list_from_file("4_score")
std = np.std(hill_climb_1)
print(std)
x = list(range(len(hill_climb_1)))
plt.semilogy(x, hill_climb_1,color='Blue',label="hill climb Platinum")
plt.errorbar(x, hill_climb_1, yerr=std,xerr = 0, color='Blue', linestyle='None',capthick=1,capsize=5,errorevery=3500,ecolor='Blue')


plt.title("Hill climb learning curve for 4 set of data")
plt.xlabel('iteration')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()