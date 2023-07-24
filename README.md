Create kmeans model as a class with OOP
Use decorators to utilise different error metrics
Incorporate Kmeans++ for centroid initialisation


Algorithm:
Initialise K cluster centroids randomly or with Kmeans++
Repeat:
    For data points 1 -> m:
        assign the data point to the closest centroid
    For centroids 1 -> k:
        move centroids to the average of all points it is assigned to



Features:
Incorporate different distance metrics (euclidean, mahalanobis, cosine similarity...) with decorator functions

fields:
n_clusters
initialisation method
iterations (not max_iterations)
seed
SSE
cluster_centers
labels

future versions:
tolerance and maximum iterations