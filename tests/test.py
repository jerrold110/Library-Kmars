import numpy as np

arr = np.empty((0,3), int)
print(arr)
arr = np.append(arr, np.array([[1,2,3]]), axis=0)
arr = np.append(arr, np.array([[4,5,4]]), axis=0)
arr = np.append(arr, np.array([[1,2,3]]), axis=0)
arr = np.append(arr, np.array([[1,2,3]]), axis=0)
arr2 = np.append(arr, np.array([[9,9,9]]), axis=0)
# print(arr.sum(axis=None))
# print(arr.sum(axis=1))
# print(arr.sum(axis=0))
# print(arr.sum(axis=-1))

# arr = np.empty(0, int)

# print(arr)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# print(arr)
    
# random_centroids = np.empty((0,3))
# def foo(x):
#     x = np.append(arr=x, values=np.array([centroid])+i, axis=0)
# print(random_centroids)

print(arr)
print(arr.shape)




