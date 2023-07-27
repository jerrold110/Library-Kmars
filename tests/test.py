import numpy as np

arr = np.empty((0,3), int)
print(arr)
arr = np.append(arr, np.array([[1,2,3]]), axis=0)
arr = np.append(arr, np.array([[4,5,4]]), axis=0)
arr = np.append(arr, np.array([[1,2,3]]), axis=0)
print(arr.sum(axis=None))
print(arr.sum(axis=1))
print(arr.sum(axis=0))
print(arr.sum(axis=-1))

# arr = np.empty(0, int)

# print(arr)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# print(arr)

def _get_nearest_centroids(self, X, centroids):
        nearest_centroids = np.empty(0, int)
        for i in range(X.shape[0]):
            datapoint = X[i]
            ind_nearest_centroid = np.argmin([self._distance_euclidean(datapoint, c) for c in centroids])
            nearest_centroids = np.append(np.array([ind_nearest_centroid]), nearest_centroids, 0)
        
        return nearest_centroids