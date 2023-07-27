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

from kmars import KMeansEuclidean
km = KMeansEuclidean(3)
km.fit(X)
print(km.cluster_centers)
print(km.labels)
print(km.sse)

