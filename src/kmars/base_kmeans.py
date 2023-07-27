import numpy as np

class _BaseKMeans:
    """
    Base kmeans class. Assumed to be working with numpy.ndarrays
    
    Parameters
    -----------
    n_clusters: int, default=
        The number of clusters to form and the number of centroids to generate
    init: str, defalt="kmeans++"
    
    max_iter: The numer of iterations to recalculate centroids
    
    seed: The random seed used to initialise the centroids
    
    ----------
    Init methods:
    rand
    kmeans++
    """
    def __init__(self, n_clusters, init="kmeans++", max_iter=100, random_state=0):
        self._n_clusters = n_clusters
        self._init = init
        self._max_iter = max_iter
        self._random_state = random_state
        print("KMeans base class inheritance for __init__ of base class successful")
    
    def _init_random(self, X, n_clusters):
        """
        Returns a 2D array of initial centroid positions initialised randomly between the max and min value of every column
        """
        random_centroids = np.array([])
        column_maxs = X.max(axis=0)
        column_mins = X.min(axis=0)
        column_ranges = column_maxs - column_mins
        for _ in n_clusters:
            centroid =  np.random.uniform(0, 1, X.shape[1])
            centroid = column_mins + (centroid * column_ranges)
            random_centroids = np.append(arr=random_centroids, values=np.array([centroid]), axis=0)
            
        return random_centroids
    
    def _init_kmeansplusplus(self, X, seed):
        """
        Selects initial centroid and returns two arrays. One of the initial centroid, and the candidates. This is only called once for every fit() function
        """
        np.random.seed(seed)
        n_samples, n_features = X.shape
        # Initialise the first centroid with uniform distribution
        centroids = np.empty((0, n_features), np.double)
        candidate_centroids = X.copy()
        first_centroid_ind = np.random.choice(n_samples)
        first_centroid = X[first_centroid_ind]
        # add new centroid to centroids
        centroids = np.append(arr=centroids, values=np.array([first_centroid]), axis=0)
        # remove datapoint from candidate_centroids
        candidate_centroids = np.delete(arr=candidate_centroids, obj=first_centroid_ind, axis=0)
        
        return centroids, candidate_centroids