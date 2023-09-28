import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

random_search_shortest_number_list2 = []
with open('Hill climb shorest length/number_list.csv', 'r') as file:
    csv_reader = csv.reader(file,quotechar = ' ')
    for row in csv_reader:
        random_search_shortest_number_list2.append(float(row[0]))


y_values = random_search_shortest_number_list2
x_values = range(len(random_search_shortest_number_list2))
std = np.std(y_values)
plt.plot(x_values, y_values, label='Data Line', color='blue', marker='', linestyle='-')
plt.errorbar(x_values, y_values, yerr=[std if i % 900 == 0 else 0 for i in range(len(random_search_shortest_number_list2))], color='blue', linestyle='-',fmt='|', ms=1,mew=1,capthick=2,capsize=5)

# print(data),l


random_search_longest_number_list2 = []
with open('Hill climb longest length/number_list.csv', 'r') as file:
    csv_reader = csv.reader(file,quotechar = ' ')
    for row in csv_reader:
        random_search_longest_number_list2.append(float(row[0]))

y_values = random_search_longest_number_list2
x_values = range(len(random_search_longest_number_list2))
std = np.std(y_values)
plt.plot(x_values, y_values, label='Data Line', color='red', marker='', linestyle='-')
plt.errorbar(x_values, y_values, yerr=[std if i % 900 == 0 else 0 for i in range(len(random_search_longest_number_list2))], color='red', linestyle='-',fmt='|', ms=1,mew=1,capthick=2,capsize=5)



# Add labels and a title
plt.xlabel('iterations')
plt.ylabel('distance')
plt.title('Longest Distance and shortest, Random Search')
# Display the plot
plt.show()