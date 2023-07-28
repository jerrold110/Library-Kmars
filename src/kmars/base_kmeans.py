import numpy as np

class _BaseK:
    """
    Base kmeans class. Assumed to be working with numpy.ndarrays
    
    Parameters
    -----------
    n_clusters: int
        The number of clusters to form and the number of centroids to generate
    dist: int
        The distance metric to use
    init: str
        
    max_iter: The numer of iterations to recalculate centroids
    
    seed: The random seed used to initialise the centroids
    """
    def __init__(self, n_clusters, dist, init, n_init, max_iter, tol, random_state, verb):
        if dist not in ['euclidean','manhattan','cosine','mahanalobis']:
            raise ValueError("dist argument %s not valid" % (dist))
        if init not in ['kmeans++','rand']:
            raise ValueError("init argument %s not valid" % (init))
        
        self._n_clusters = n_clusters
        self._dist = dist
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._verb = verb
        self._random_state = random_state
    
    def _init_random(self, X):
        """
        Returns a 2D array of initial centroid positions initialised randomly between the max and min value of every column
        
        Args:
            a (_type_): _description_
            b (_type_): _description_
            c (_type_): _description_

        Returns:
            _type_: _description_
        """
        np.random.seed(self._random_state)
        n_features = X.shape[1]
        random_centroids = np.empty((0,n_features))
        column_maxs = X.max(axis=0)
        column_mins = X.min(axis=0)
        column_ranges = column_maxs - column_mins
        for _ in range(self._n_clusters):
            centroid =  np.random.uniform(0, 1, X.shape[1])
            centroid = column_mins + (centroid * column_ranges)
            random_centroids = np.append(arr=random_centroids, values=np.array([centroid]), axis=0)
            
        return random_centroids
    
    def _init_kmeansplusplus(self, X, seed):
        """
        Selects initial centroid and returns two arrays. One of the initial centroid, and the candidates.
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