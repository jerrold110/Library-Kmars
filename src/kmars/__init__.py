"""
This is an implementation of K-means/medians/medoids with various distance metrics (Euclidean, Manhattan, Cosine...) built over Numpy. 
Read more at https://github.com/jerrold110/Library-Kmars

KMeans parameters:
KMeans(n_clusters, dist='euclidean', mini_ord=3, init='kmeans++', n_init=10, max_iter=300, tol=0.0001, random_state=0, verb=False)
"""
from .kmeans import KMeans
from .kmedians import KMedians
from .kmedoids import KMedoids
