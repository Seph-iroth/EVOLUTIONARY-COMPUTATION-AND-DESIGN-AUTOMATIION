# import heapq
#
# # Create an empty min-heap
# heap = []
#
# # Insert elements into the heap
# heapq.heappush(heap, 5)
# heapq.heappush(heap, 2)
# heapq.heappush(heap, 8)
#
# # # Extract the smallest element
# # smallest = heapq.heappop(heap)
# #
# # print("Smallest element:", smallest)  # Output: Smallest element: 2
# def print_heap(heap, index=0, level=0):
#     if index < len(heap):
#         print("  " * level + str(heap[index]))
#         left_child = 2 * index + 1
#         right_child = 2 * index + 2
#         print_heap(heap, left_child, level + 1)
#         print_heap(heap, right_child, level + 1)
# print([str(i) for i in range(-10, 11)] )
import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

# Sample errors (could represent standard deviations, standard errors, etc.)
y_errors = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

# Create a simple plot with error bars
plt.errorbar(x, y, yerr=y_errors, linestyle='-',fmt='|', label='Data with error bars')
plt.title("Error Bar Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()