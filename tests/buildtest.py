import numpy as np
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.title("Mixture of Gaussian Blobs")
plt.suptitle("Ground truth clusters").set_y(0.95)
#plt.show()

# path management
import os
base_directory = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(f"{base_directory}/src")

import kmars
km = kmars.KMedoids(3, dist='euclidean', n_init=10, init='kmeans++', max_iter=300, verb=True)
km.fit(X)
# print(km.init_cluster_centers_)
# print(km.init_labels_[km.init_labels_==1].shape)
# print(km.init_sse_)
print(km.cluster_centers_)
print(km.labels_[km.labels_==1].shape)
print(km.sse_)
# print(km.x_features_)
# print(km.x_samples_)
print(km._n_iter_converge)

# test of sklearn euclidean


