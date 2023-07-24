import numpy as np

class kmeans:
    """
    Kmeans class.
    Assumed to be working with numpy.ndarrays
    
    Parameters
    -----------
    n_clusters: int, no default
        The number of clusters to form and the number of centroids to generate
    init: str, defalt="kmeans++"

    dist: The distance metric to use
    
    max_iter: The numer of iterations to recalculate centroids
    
    seed: The random seed used to initialise the centroids
    
    ----------
    Methods:
    _dst_euclidean
    _kmeans++
    """

    def __init__(self, n_clusters, init, dist, max_iter, seed):
        self.n_clusters = n_clusters
        self.init = init
        self.dist = dist
        self.max_iter = max_iter
        self.seed = seed
        self.SSE = None
        self.cluster_centers = None
        self.labels = None
    
    def _dst_euclidean(vec1, vec2):
        """
        Euclidean distance between two points
        Distance methods take in two single dimension vectors (ndarrays)
        """
        dist = np.linalg.norm(v1 - v2, ord=2, axis=0)
        return dist
    
    def _init_kmeans_random(m1, n_clusters):
        """
        Returns an array of initial centroid positions initialised randomly
        """
        centroids = np.array([])
        
        column_maxs = m1.max(axis=0)
        column_mins = m1.min(axis=0)
        column_ranges = column_maxs - column_mins
        
        for _ in n_clusters:
            for _ in range(m1.shape[1]):
                centroid = np.array([])
                
                
            
        
    
    def _init_kmeans_plus(x):
        """
        Returns an array of initial centroid positions initialised with kmeans++
        """
    
    
