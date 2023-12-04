import os

import numpy as np
from multiprocessing import Pool, Process
import time
#generate 2 array, each array has 3 random number from 0 to 1.
A = np.random.rand(2,3)

A = np.random.uniform(1,15)
# print(A)
#
# def f(x, y):
#     # time.sleep(1)
#     return x * y
# start = time.time()
# if __name__ == '__main__':
#     with Pool(16) as p:
#         for i in range(3):
#             results = p.starmap(f, [(1, 2), (3, 4), (5, 6)])
#         print(results)
# end = time.time()
# # print(end - start)

def info(title):
    # print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    time.sleep(1)
    print('hello', name)
# Get the full path of the current script
full_path = __file__
file_name = os.path.splitext(os.path.basename(full_path))[0]

print("Full Path:", type(file_name))
print("File Name:", file_name)

if __name__ == '__main__':
    for i in range(3):
        print(i)
    # info('main line')
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # Process(target=f, args=('bob',)).start()
    # p.join()
