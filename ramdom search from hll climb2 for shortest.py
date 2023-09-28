import random
import time
import math
import csv
from tqdm import tqdm
# import executor as executor
# import numpy as np
# import networkx as nx
# from multiprocessing import pool
import concurrent.futures


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def swap_ramdom(list_1):
    a = random.sample(range(len(list_1)), 2)
    index1 = a[0]
    index2 = a[1]
    list_1[index1], list_1[index2] = list_1[index2], list_1[index1]
    return list_1


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


def exportFile(number_list,list_list):
    with open("random_search_shortest_number_list2.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in number_list:
            csv_writer.writerow([i])
        # csv_writer.writerows(number_list)
    with open("random_search_shortest_number_list_list2.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ',escapechar=' ')
        csv_writer.writerows(list_list)
def shuffle_fun(alist):
    random.shuffle(alist)
    return alist
def main():
    csv_file_path = 'points.csv'
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(  [float(row[0]), float(row[1])]  )
    bestTour = data
    shortest = calculate_distance_total(data)

    result_path_of_each_iteration = []

    result_of_each_shortest_length = []
    result_of_each_tour_path = []
    best_value_tour_path = []
    iter = 10000
    one_iter_time = 0.8
    sec = iter*one_iter_time % 60
    minutes = math.floor(iter*one_iter_time / 60)
    hours =  math.floor(minutes/60)
    numberofpoints = len(data)
    print(f"estimated time: {hours} hours {minutes} minutes {sec} seconds")
    print(f'Number of points in one list: {numberofpoints}')
    for times in tqdm(range(iter)):
        temp_best_tour = bestTour
        # generate 1000 successor for the best tour
        input_tourList = []
        # for i in temp_best_tour:
        #     input_tourList.append(temp_best_tour)
        for i in range(2):
            input_tourList.append(temp_best_tour)
        # #ProcessPoolExecutor
        list_of_length = []
        list_of_tour = []
        #ProcessPoolExecutor

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            input_tourList = list(executor.map(shuffle_fun, input_tourList))
            list_of_length = executor.map(total_length, input_tourList)
            list_of_length = list(list_of_length)


        # with concurrent.futures.ProcessPoolExecutor() as executor:
            # list -> (distance, [tour list])


        # print(list_of_length)


        # find the location of the shortest tour by find the index of the shortest length.
        # then map that to input to find it's tour.
        value_of_the_path = min(list_of_length)
        # value_of_the_path = max(list_of_length)
        best_value_tour_path = input_tourList[list_of_length.index(value_of_the_path)]

        # update the best if found better
        # if value_of_the_path < shortest:
        if value_of_the_path < shortest:
            shortest = value_of_the_path
            result_of_each_shortest_length.append(value_of_the_path)
            best_value_tour_path = input_tourList[list_of_length.index(value_of_the_path)]
            bestTour = best_value_tour_path
            # bestTour = input_tourList[list_of_length.index(value_of_the_path)]

        else:
            result_of_each_shortest_length.append(shortest)# [1,2,3,4]
            result_path_of_each_iteration.append(best_value_tour_path) # [[1,2],[3,4]]
        exportFile(result_of_each_shortest_length,result_path_of_each_iteration)
    # print(result_path_of_each_iteration)
    # for i in result_path_of_each_iteration:
    #     print(i)
    # print(result_of_each_shortest_length)
    # print(len(result_path_of_each_iteration))
    # print(result_path_of_each_iteration)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
