import numpy as np
from .modules.distance_metrics import _distance_euclidean, _distance_manhattan, _distance_minikowski, _distance_cosine, _distance_hamming

"""
To do:
Move all the centroid initialisation methods into the parent class because all the variables needed are there

"""

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
    def __init__(self, n_clusters, dist, mini_ord, init, n_init, max_iter, tol, random_state, verb):
        if dist not in ['euclidean','manhattan','minikowski','cosine','hamming']:
            raise ValueError("dist argument %s not valid" % (dist))
        if init not in ['kmeans++','rand']:
            raise ValueError("init argument %s not valid" % (init))
        
        self._n_clusters = n_clusters
        self._dist = dist
        self._mini_ord = mini_ord
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._verb = verb
        self._random_state = random_state
        
    def _distance(self, v1, v2):
        """
        Returns vector distance between two points based on distance metric during initialisation.
        """
        if self._dist == 'euclidean':
            distance = _distance_euclidean(v1, v2)
        elif self._dist == 'manhattan':
            distance = _distance_manhattan(v1, v2)
        elif self._dist == 'minikowski':
            distance = _distance_minikowski(v1, v2, self._mini_ord)
        elif self._dist == 'cosine':
            distance = _distance_cosine(v1, v2)
        elif self._dist == 'hamming':
            distance = _distance_hamming(v1, v2)
        else:
            raise ValueError("We should not be able to get here. Error in _distance function with dist input %s" % (self._dist))

        return distance
    
    def _sse_error(self, X, cluster_centers, x_labels):
        """
        Calculates the sum of squared distances of samples to their closest cluster center based on distance metric during initialisation.

        Args:
            X (2-dimension ndarray): X data
            cluster_centers (2-dimension ndarray): array of vectors, shape[0]=_n_cluster_centers, shape[1] = n_features
            x_labels (1-dimension ndarray): array of ints representing nearest cluster, shape[0] = n_samples

        Returns:
            float: The SSE error
        """
        sse_error = .0
        for i in range(X.shape[0]):
            nearest_cluster_ind = x_labels[i]
            point_to_centroid_distance = self._distance(X[i], cluster_centers[nearest_cluster_ind])
            se_error = point_to_centroid_distance**2
            sse_error += se_error
            
        return sse_error
    
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
    
    def _init_kmeansplusplus(self, X, start_seed):
        """
        Returns an array of initial centroid positions initialised with kmeans++. Based on best SSE with multiple seeds
        Kmeans++ seeks to spread out the k initial clusters.

        Args:
            X (2-dimension ndarray): X data
            start_seed (int): The first RNG seed to be used

        Returns:
            (2-dimension ndarray): array of vectors, shape[0]=_n_cluster_centers, shape[1] = n_features
        """
        # This array stores tuples (see, centroids)
        results = []
        for i in range(self._n_init):
            seed = start_seed + i
            # Initialise the first centroid with random uniform distribution. 
            # Centroids is the array that will be returned
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
            # Loop over remaining centroids
            # Loop over all candidate centroids
            # Loop over all centroids. Calculate the min squared distance of all candidate centroids to existing centroids
            # Create the probability distribution and select the next centroid
            np.random.seed(seed)
            for _ in range(self._n_clusters - 1):
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
                # add new centroid to centroids. T
                centroids = np.append(centroids, np.expand_dims(candidate_centroids[next_centroid_ind], 0), 0)
                # remove datapoint from candidate_centroids
                candidate_centroids = np.delete(candidate_centroids, next_centroid_ind, 0)
            # Calculate SSE of the selected candidate centroids
            nearest_centroids = self._get_nearest_centroids(X, centroids)
            sse = self._sse_error(X, centroids, nearest_centroids)
            results.append((sse, centroids))
        # Sort results by see
        results.sort(key=lambda x:x[0])
        print("Kmeans++ initial centroids:")
        print(results[0][1])
        print(results[0][0])

        return results[0][1]