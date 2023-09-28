import random
import time
import math
import csv

# import executor as executor
import numpy as np
import networkx as nx
from multiprocessing import pool
import concurrent.futures


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# def swap_ramdom(list_1):
#     print(list_1)
#     index1, index2 = random.sample(range(len(list_1)), 2)
#     print(index1, index2)
#     list_1[index1], list_1[index2] = list_1[index2], list_1[index1]
#     print(list_1)
#
#     return list_1

def swap_ramdom(list_1):
    # print(list_1)
    # print(len(list_1))
    a = random.sample(range(len(list_1)), 2)
    # print(list_1)
    index1 = a[0]
    index2 = a[1]
    # print(index1)
    # print(index1, index2)
    list_1[index1], list_1[index2] = list_1[index2], list_1[index1]

    return list_1

# def swap_ramdom(list_1):
#     A = list_1
#     B = list_1
#     pos1,pos2 = random.sample(range(0, len(list_1)), 2)
#     A[pos1] = B[pos2]
#     A[pos2] = B[pos1]
#     list_1 = A
#     return list_1


def calculate_distance_total(data):
    current_total_dist_list = []
    for i in range(len(data) - 1):
        dist = calculate_distance(data[i], data[i + 1])
        current_total_dist_list.append(dist)

    return sum(current_total_dist_list)


def total_length(data):
    current_total_dist_list = []
    for k in range(len(data) - 1):
        # print(data[k])
        x1, y1 = data[k]
        x2, y2 = data[k + 1]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        current_total_dist_list.append(dist)

    return sum(current_total_dist_list)


def main():
    csv_file_path = 'points.csv'

    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append([float(row[0]), float(row[1])])
            # data.append((float(row[0]), float(row[1])))
    # print(calculate_distance_total([[0,0],[0,1],[1,1],[1,0],[0,0]]))
    # print(calculate_distance_total(data))

    bestTour = data
    shortest = calculate_distance_total(data)

    result_of_each_iteration = []


    # start_time = time.time()
    result_of_each_shortest_length = []
    result_of_each_tour_path = []
    best_value_tour_path = []
    iter = 100
    print("estimated time: " + str(1 * iter/60))
    for times in range(iter):
        temp_best_tour = bestTour
        # generate 1000 successor for the best tour
        input_tourList = []
        for i in temp_best_tour:
            input_tourList.append(temp_best_tour)
        #ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # print(temp_best_tour)
            # print(len(temp_best_tour))
            input_tourList = list(executor.map(swap_ramdom, input_tourList))
        # for i in range(len(input_tourList) - 1):
        #     print(input_tourList[i] == input_tourList[i+1])
        # for i in input_tourList:
        #     print(i)
        # for i in range(2):
        #     # input_tourList.append(temp_best_tour)
        #     # print(temp_best_tour)
        #     swap_ramdom(temp_best_tour)
        #     # print(temp_best_tour)
        #     # print(temp_best_tour)
        #     # if input_tourList[i] == temp_best_tour[i+1]:
        #     #     swap_ramdom(temp_best_tour)
        #     input_tourList.insert(len(input_tourList), temp_best_tour)
            # temp_best_tour = temp_best_tour + swap_ramdom(temp_best_tour)

        # for j in input_tourList:
        #     print(j)

        # print(input_tourList[0]==input_tourList[1])
        # solve for best distance and best tour
        # #ThreadPoolExecutor, faster
        # #ProcessPoolExecutor
        # print(input_tourList)
        # for i in range(len(input_tourList)):
        #     print(input_tourList[i]==input_tourList[i+1])
        list_of_length = []
        list_of_tour = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # list -> (distance, [tour list])
            list_of_length = executor.map(total_length, input_tourList)
            list_of_length = list(list_of_length)
        # print(list_of_length)


        # find the location of the shortest tour by find the index of the shortest length.
        # then map that to input to find it's tour.
        value_of_the_path = min(list_of_length)
        best_value_tour_path = input_tourList[list_of_length.index(value_of_the_path)]

        # update the best if found better
        if value_of_the_path < shortest:
            # shortest = value_of_the_path
            result_of_each_shortest_length.append(value_of_the_path)
            best_value_tour_path = input_tourList[list_of_length.index(value_of_the_path)]
            bestTour = best_value_tour_path
        else:
            result_of_each_shortest_length.append(shortest)
        # print("iteration: " + str(times) )
        # print(best_value)
    # print(shortest)
    print(result_of_each_shortest_length)
    print(shortest)
    # print(best_value_tour_path)

    # csv_file_path = 'output of hill climb 2.csv'
    #
    # with open(csv_file_path, 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     for g in result_of_each_shortest_length:
    #         csv_writer.writerow(g)
    #
    # print(f'Data has been written to {csv_file_path}')


    # print(result_of_each_shortest_length)

    # result = 0
    #
    # #the new way
    # input = []
    # #generate input list of successors
    # for j in range(len(temp_best_tour)):
    #     some = swap_ramdom(temp_best_tour)
    #     input.append(some)
    # # ThreadPoolExecutor, faster
    # # ProcessPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     result = list(executor.map(total_length, input))
    #
    #
    # for i in range(len(data)):
    #     # swap one pair of cities
    #     tour_temp = swap_ramdom(temp_best_tour)
    #     curr = total_length(tour_temp)
    #
    #
    #     # current_total_dist_list = []
    #     # for k in range(len(data) - 1):
    #     #     # print(tour_temp)
    #     #     x1, y1 = tour_temp[k]
    #     #     x2, y2 = tour_temp[k + 1]
    #     #     dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     #     # dist = calculate_distance(tour_temp[k], tour_temp[k + 1])
    #     #     current_total_dist_list.append(dist)
    #     # curr = sum(current_total_dist_list)
    #     if curr < shortestForNow:
    #         shortestForNow = curr
    #         bestTour = tour_temp
    #         result = curr
    #     else:
    #         result = shortestForNow
    #
    #
    # real_result_of_each_iteration.append(result)

    # data = [(1, 1), (2, 1), (3, 1), (4, 1)]
    # input = []
    #
    # for j in range(len(data)):
    #     some = random.sample(data, len(data))
    #     input.append(some)
    # #ThreadPoolExecutor, faster
    # #ProcessPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     result = list(executor.map(total_length, input))

    # end_time = time.time()
    # print(list(result))
    #
    # numbers = list(range(10))
    # result = list(executor.map(square, numbers))

    # print(total_length(data))

    # print(len(real_result_of_each_iteration))
    # print(real_result_of_each_iteration)

    # print(f"Processing time: {end_time - start_time} seconds")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
