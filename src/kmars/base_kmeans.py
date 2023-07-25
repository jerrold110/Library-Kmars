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
    
    def _distance(self, v1, v2):
        """
        Does nothing. Meant to be overwritten in subclass
        """
        None
    
    def _init_random(self, m1, n_clusters):
        """
        Returns a 2D array of initial centroid positions initialised randomly between the max and min value of every column
        """
        random_centroids = np.array([])
        column_maxs = m1.max(axis=0)
        column_mins = m1.min(axis=0)
        column_ranges = column_maxs - column_mins
        
        for _ in n_clusters:
            centroid =  np.random.uniform(0, 1, m1.shape[1])
            centroid = column_mins + (centroid * column_ranges)
            random_centroids = np.append(arr=random_centroids, values=np.array([centroid]), axis=0)

        return random_centroids
    
    def _init_kmeansplusplus(self, m1:np.ndarray, n_clusters:int):
        """
        Returns an array of initial centroid positions initialised with kmeans++
        Kmeans++ seeks to speard out the k initial clusters
        """
        n_samples, n_features = m1.shape
        # Initialise the first centroid with uniform distribution
        centroids = np.empty((0, n_features), np.double)
        candidate_centroids = m1.copy()
        first_centroid_ind = np.random.choice(n_samples)
        first_centroid = m1[first_centroid_ind]
        centroids = np.append(arr=centroids, values=np.array([first_centroid]), axis=0)
        candidate_centroids = np.delete(arr=candidate_centroids, obj=first_centroid_ind, axis=0)
        
        # Loop over remaining centroids
        # Loop over all candidate centroids
        # Loop over all centroids. Calculate the min squared distance of all candidate centroids to existing centroids
        # Create the probability distribution and select the next centroid
        for i in range(self._n_centroids_ - 1):
            distances = []
            n_candidate_centroids = candidate_centroids.shape[0]
            n_centroids = centroids.shape[0]
            for j in range(n_candidate_centroids):
                point_to_centroid = []
                for c in range(n_centroids):
                    point_to_centroid.append(self._distance(candidate_centroids[j], centroids[c]))
                distances.append(min(point_to_centroid))
            distances = np.array(distances)
            probabilities = distances / np.sum(distances)
            next_centroid_ind = np.random.choice(a=n_candidate_centroids, p=probabilities)
            # add new centroid to centroids
            centroids = np.append(centroids, candidate_centroids[next_centroid_ind], 0)
            # remove datapoint from candidate_centroids
            candidate_centroids = np.delete(candidate_centroids, next_centroid_ind, 0)
            
        return centroids