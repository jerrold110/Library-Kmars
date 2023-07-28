import numpy as np
from .base_k import _BaseK
from .modules.distance_metrics import _distance_euclidean, _distance_manhattan, _distance_minikowski, _distance_cosine, _distance_hamming
from .modules.verbose import blockPrint, enablePrint

class KMedians(_BaseK):
    """
    K-Medians clustering object.

    Args:
        n_clusters (int): The number of clusters to form
        dist (str, optional): The distance metric used {'euclidean','manhattan','minikowski','cosine','hamming'}. Defaults to 'manhattan'.
        mini_ord (int, optional): The order for minikowski distance. Ignored if dist is not 'minikowski'. Defaults to 3.
        init (str, optional): The initialisation method used to select initial centroids, 'kmeans++' or 'rand' . Defaults to 'kmeans++'.
        n_init (int, optional): The number of different seeds and times to run the kmeans++ initialisation of which best result is selected. Defaults to 10.
        max_iter (int, optional): The maximum number of iterations the K-Means algorithm can run without convergence. Defaults to 300.
        tol (float, optional): The relative tolerance of the Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. Defaults to 0.0001.
        random_state (int, optional): The RNG seed used for centroid initialisation. Defaults to 0.
        verb (bool, optional): Enable or disable verbose output. Defaults to False.
        
    Methods:
        fit()
    
    Getter methods call-able after calling fit():
        init_cluster_centers_
        init_labels_
        init_sse_
        cluster_centers_
        labels_
        sse_
        n_iter_converge_
        x_samples_
        x_features_
    """
    def __init__(self, n_clusters, dist='manhattan', mini_ord=3, init='kmeans++', n_init=10, max_iter=300, tol=0.0001, random_state=0, verb=False):
        
        super().__init__(n_clusters, dist, mini_ord, init, n_init, max_iter, tol, random_state, verb)
        self._n_samples = None
        self._n_features = None
        self._init_cluster_centers = None
        self._init_labels = None
        self._init_sse = None
        self._cluster_centers = None
        self._labels = None
        self._sse = None
        self._n_iter_converge = None
        print("KMedians object initialised with %s distance metric"%(self._dist))
    
    def _init_random(self, X):
        """
        Returns a 2D array of initial centroid positions initialised randomly between the max and min value of every column
        """
        random_centroids = super()._init_random(X)
        
        return random_centroids
    
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
        for i in range(self._n_samples):
            nearest_cluster_ind = x_labels[i]
            point_to_centroid_distance = self._distance(X[i], cluster_centers[nearest_cluster_ind])
            se_error = point_to_centroid_distance**2
            sse_error += se_error
            
        return sse_error
    
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
            centroids, candidate_centroids = super()._init_kmeansplusplus(X, seed)
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
        print("")

        return results[0][1]
    
    def _centroid_new_pos(self, points):
        """
        Takes in a 2D array of all the points belonging to a centroid, returns a vector that is the mean of all points.
        Mean for KMeans, Median for KMedians
        """
        return np.median(points, axis=0)
    
    def _centroids_update(self, n_centroids, X, x_nearest_centroids, current_centroids):
        """
        Returns a vector of the updated centroids.
        x_nearest_centroids denotes the index of the nearest centroid in centroids for each data point.
        This function changes the parameter current_centroid (ndarray) because it is being passed by reference
        
        Args:
            n_centroids (int): the number of centroids
            X (2-dimension ndarray): data
            x_nearest_centroids (1-dimension ndarray): index positions of nearest centroids for each data point
            current_centroids (2-dimension ndarray): the array containing the current positions of the centroids
        """
        current_centroids_copy = current_centroids.copy()
        for i in range(n_centroids):
            centroid_connected_points = X[x_nearest_centroids==i]
            new_centroid = self._centroid_new_pos(centroid_connected_points)
            current_centroids_copy[i] = new_centroid
            
        return current_centroids_copy
    
    def _get_nearest_centroids(self, X, centroids):
        """
        Returns a vector of the index of the closest centroid to each data point in x. Ndarray of ints

        Args:
            X (2-dimension ndarray): data
            centroids (2-dimension ndarra): the array containing the positions of the centroids
        """
        nearest_centroids = np.empty(0, int)
        # Loop over data points
        # Loop over centroids, get closest distance, get index position
        # Add index position to nearest_centroids
        for i in range(X.shape[0]):
            datapoint = X[i]
            distances = np.array([self._distance(datapoint, c) for c in centroids])
            ind_nearest_centroid = np.argmin(distances)
            nearest_centroids = np.append(nearest_centroids, np.array([ind_nearest_centroid]), 0)

        return nearest_centroids
    
    def fit(self, X):
        """
        Compute k-medians clustering

        Args:
            X (2-dimension ndarray): The X data as a matrix.
        """
        if self._verb == False:
            blockPrint()
        elif self._verb == True:
            pass
        else:
            raise ValueError("Error in verbose with input %s" % (self._verb))
        
        self._n_samples, self._n_features = X.shape  
        if self._init == "rand":
            initial_centroids = self._init_random(X)
        elif self._init == "kmeans++":
            initial_centroids = self._init_kmeansplusplus(X, self._n_clusters)
        else:
            raise ValueError("The init variable %s is invalid " % (self._init))
        print("Initial centroid via %s successful" % (self._init))
        # These variables are for testing the sse of the initial centroids
        initial_labels = self._get_nearest_centroids(X, initial_centroids)
        initial_sse = self._sse_error(X, initial_centroids, initial_labels)
        self._init_cluster_centers = initial_centroids
        self._init_labels = initial_labels
        self._init_sse = initial_sse
        # Loop over the max_iterations
        current_centroids = np.copy(initial_centroids)
        # 1: Calculate positions of new centroids based on median of points that belong to it
        # 2: Update current_centroids to new_centroids
        print('Starting KMedians iterations...')
        for i in range(self._max_iter):
            nearest_centroids = self._get_nearest_centroids(X, current_centroids)
            new_centroids = self._centroids_update(self._n_clusters, X, nearest_centroids, current_centroids)
            # 3: Calculate frobenius norm against tolerance to declare convergence
            diff = new_centroids - current_centroids
            fn_update_diff = np.sqrt(np.sum(np.square(diff)))
            print(fn_update_diff)
            if (fn_update_diff < self._tol):
                print(f"Convergence reached at iteration {i}")
                break
            else:
                current_centroids = new_centroids
        self._n_iter_converge = i + 1
        print("KMedians iterations complete...")
        self._cluster_centers = current_centroids
        self._labels = self._get_nearest_centroids(X, self._cluster_centers)
        self._sse = self._sse_error(X, self._cluster_centers, self._labels)
        
        if self._verb == False:
            enablePrint()
        elif self._verb == True:
            pass
    
    @property
    def init_cluster_centers_(self):
        return self._init_cluster_centers
    
    @property
    def init_labels_(self):
        return self._init_labels
    
    @property
    def init_sse_(self):
        return self._init_sse
        
    @property
    def cluster_centers_(self):
        return self._cluster_centers
    
    @property
    def labels_(self):
        return self._labels
    
    @property
    def sse_(self):
        return self._sse
    
    @property
    def n_iter_converge(self):
        return self._n_iter_converge
    
    @property
    def x_samples_(self):
        return self._n_samples
    
    @property
    def x_features_(self):
        return self._n_features
    

    