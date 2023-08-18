import numpy as np
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170

X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# path management
import os
base_directory = os.path.dirname(os.path.dirname(__file__))
print(base_directory)
import sys
sys.path.append(f"{base_directory}")
import kmars

km = kmars.KMedoids(3, init='kmeans++', dist='manhattan', verb=False).fit(X)
print(km.init_cluster_centers_.dtype)
print(km.init_labels_.dtype)
print(km.init_labels_[km.init_labels_==1].shape)
print(km.init_sse_)
print(km.cluster_centers_)
print(km.labels_[km.labels_==1].shape)
print(km.sse_)
print(km.ssr_)
print(km.x_features_)
print(km.x_samples_)
print(km._n_iter_converge)
