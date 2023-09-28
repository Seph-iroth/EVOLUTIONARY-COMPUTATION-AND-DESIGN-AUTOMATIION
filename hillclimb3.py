import math
import random
import time

from matplotlib import pyplot as plt


def total_length(data):
    current_total_dist_list = []
    for k in range(len(data) - 1):
        # print(tour_temp)
        x1, y1 = data[k]
        x2, y2 = data[k + 1]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        current_total_dist_list.append(dist)

    return sum(current_total_dist_list)

input = []
data = [(1, 1), (2, 1), (3, 1), (4, 1)]
def shake(data):
    random.shuffle(data)

for j in range(5):

    some = random.sample(data, len(data))
    input.append(some)


# for i in input:
#     print(i)


# input = []
#         #generate input list of successors
#         for j in range(len(data)):
#             some = random.sample(data, len(data))
#             input.append(some)
#         # ThreadPoolExecutor, faster
#         # ProcessPoolExecutor
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             result = list(executor.map(total_length, input))
def swap_ramdom(list_1):
    start = list_1
    index1, index2 = random.sample(range(len(list_1)), 2)
    print(index1, index2)
    list_1[index1], list_1[index2] = list_1[index2], list_1[index1]
    print(start)
    print(list_1)
    print(list_1==start)
    return list_1




l1  = [(1, 2), (3, 4), (5, 6),(7, 8), (9, 10), (11, 12)]




# for i in range(4):
#     l1 = swap_ramdom(l1)


input_tourList = []
temp_best_tour = l1
for j in range(len(temp_best_tour)):
    index1, index2 = random.sample(range(0, len(temp_best_tour)), 2)
    # print(index1,index2)
    tour_temp = temp_best_tour
    temp_best_tour[index1], temp_best_tour[index2] = temp_best_tour[index2], temp_best_tour[index1]
    # swap_ramdom(temp_best_tour)
    input_tourList.append(temp_best_tour)
# print(input_tourList)


# Python3 program to swap elements
# at given positions

# Swap function
def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]



# Driver function
list_1 = [23, 65, 19, 90]

list_1  = [(1, 2), (3, 4), (5, 6),(7, 8), (9, 10), (11, 12)]
list_1  = [[1,2], [3,4], [5,6], [7,8], [9,10],[11,12]]
a, b = list_1[1]
# print(a,b)

# print(list_1[:1] , list_1[1],list_1[2:4], list_1[4],list_1[4:len(list_1)])
index1 = 2
index2 = 4
# print(list_1[:index1] , list_1[index1], list_1[index1 + 1: index2], list_1[index2],list_1[index2:len(list_1)])
gg = []

for i in list_1:
    gg.append(i)
# print(gg)



for i in range(3):
    start = list_1
    pos1,pos2 = random.sample(range(0, len(list_1)), 2)
    # print(pos1,pos2)
    list_1[pos1], list_1[pos2] = list_1[pos2], list_1[pos1]
    # end = list_1
    # print(list_1)
    # print(start==end)

for i in range(3):
    A = list_1
    B = list_1

    pos1,pos2 = random.sample(range(0, len(list_1)), 2)
    # print(pos1,pos2)
    A[pos1] = B[pos2]
    A[pos2] = B[pos1]
    list_1 = A
    # print(list_1)

my_list = [1, 2, 3, 4, 5]

my_list[1]=3


list_1  = [[1,2], [3,4], [5,6], [7,8], [9,10],[11,12]]
def swap_ramdom(list_1):
    # print(list_1)
    index1, index2 = random.sample(range(len(list_1)), 2)
    # print(index1, index2)
    list_1[index1], list_1[index2] = list_1[index2], list_1[index1]

    return list_1
final = []
# for i in range(len(list_1)):
#     swap_ramdom(list_1)
#     print(list_1)
#     final.append(list_1)

for i in range(len(list_1)):
    swap_ramdom(list_1)
    # print(list_1)
    final.append(list_1)
my_list = [1, 2, 3,2,2,324,23,235,25,235,253,23]
my_list.insert(len(my_list), 123123123123)
print(my_list)
# for i in final:
    # print(i)

data = [ (0.03176, 0.03740),
          (0.90296, 0.56026),
          (0.39814, 0.07302),
          (0.94808, 0.97195),
          (0.42961, 0.04406),
          (0.36967, 0.92293),
          (0.08301, 0.07182),
          (0.07212, 0.66948),
          (0.05147, 0.06067),
          (0.85407, 0.41000),
          (0.41072, 0.03573),
          (0.13152, 0.57465)]

a = [561.025176956542, 558.2663135607318, 555.3253552679648, 552.5916947839551, 548.3629214609654, 545.9684641981074, 542.6933297923981, 539.4744102751486, 535.9915379517485, 533.2539884628915, 531.0025232171178, 528.8235193272809, 525.5241798799, 523.6246296734838, 521.3616698735825, 519.2888505973763, 516.9971483005695, 514.257285987384, 511.95032345322335, 509.3643282684146, 506.5285929679254, 504.0877694459912, 501.0345446859498, 498.6811734385126, 496.6334848921986, 494.2504859962513, 491.69388390802686, 488.97194355097645, 486.40647922027176, 484.0517179736167, 482.0160036211221, 478.8940297997248, 476.7267715629929, 474.7958368215925, 472.8505460562945, 470.6433669953387, 468.214908770365, 466.2338790739195, 464.03662603133137, 462.1147254349233, 460.2440532710276, 458.1268355525171, 456.23043767234685, 453.908037800943, 451.5775177419054, 449.473863064514, 447.2657263451977, 445.45903706808394, 443.5448966546169, 441.48472751365654, 439.3667401866316, 437.3175802086699, 435.4366036100293, 433.66880563908603, 431.7616444517149, 429.9167835847898, 427.77723518301514, 426.0669800850273, 423.90615234443925, 422.161580864678]

a = 13
b = 4

c = a % b
print(a, "mod", b, "=",
      c, sep=" ")

from time import sleep
from tqdm import tqdm
# o = [1,2,3,4,5,6,7,8,9,10]
# def shuffle_fun(alist):
#     random.shuffle(alist)
#     return alist
# print(shuffle_fun(o))


x = [0, 1, 2, 3, 4, 5,6]
y = [0, 1, 4, 9, 16, 25,24]

# Create a scatter plot for the path
plt.plot(x, y, label='Path', color='blue', marker='o')

# Customize the plot (add labels, title, legend, etc.)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Path Plot')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()












