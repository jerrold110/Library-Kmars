import numpy as np
from .base_k import _BaseK

class KMedoids(_BaseK):
    """
    K-Medoids clustering object.
    Maximum iterations is set at 50 by default because of the longer complexity of Kmedoids. 
    Convergence is reached if none of the medoids move in a single iteration of n medoids.

    Args:
        n_clusters (int): The number of clusters to form
        dist (str, optional): The distance metric used {'euclidean','manhattan','minikowski','cosine','hamming'}. Defaults to 'euclidean'.
        mini_ord (int, optional): The order for minikowski distance. Ignored if dist is not 'minikowski'. Defaults to 3.
        m_update(str, optional): The type {'se', 'sse'} of metric to decide on which is the best datapoint to swap the medoid and whether to update the medoid
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
        x_samples
        x_features
    """
    def __init__(self, n_clusters, dist='manhattan', mini_ord=3, m_update='se', init='kmeans++', n_init=10, max_iter=50, tol=0.0001, random_state=0, verb=False):
        
        super().__init__(n_clusters, dist, mini_ord, init, n_init, max_iter, tol, random_state, verb)
        self._fit_score = m_update
        self._n_samples = None
        self._n_features = None
        self._init_cluster_centers = None
        self._init_labels = None
        self._init_sse = None
        self._cluster_centers = None
        self._labels = None
        self._sse = None
        self._ssr = None
        self._n_iter_converge = None
        if self._verb:
            print("KMedoids object initialised with %s distance metric and %s fit score "%(self._dist, self._fit_score))
    
    def _se_error(self, X, cluster_centers, x_labels):
        """
        Calculates the sum of distances of samples to their closest cluster center based on distance metric during initialisation.

        Args:
            X (2-dimension ndarray): X data
            cluster_centers (2-dimension ndarray): array of vectors, shape[0]=_n_cluster_centers, shape[1] = n_features
            x_labels (1-dimension ndarray): array of ints representing nearest cluster, shape[0] = n_samples

        Returns:
            float: The SE error
        """
        se_error = .0
        for i in range(X.shape[0]):
            nearest_cluster_ind = x_labels[i]
            point_to_centroid_distance = self._distance(X[i], cluster_centers[nearest_cluster_ind])
            se_error += point_to_centroid_distance
        return se_error
    
    def _cost(self, X, cluster_centers, x_labels):
        if self._fit_score == 'se':
            return self._se_error(X, cluster_centers, x_labels)
        elif self._fit_score == 'sse':
            return self._sse_error(X, cluster_centers, x_labels)
    
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
            distances = []
            for c in centroids:
                distances.append(super()._distance(datapoint, c))
            distances = np.array(distances)
            ind_nearest_centroid = np.argmin(distances)
            nearest_centroids = np.append(nearest_centroids, np.array([ind_nearest_centroid]), 0)

        return nearest_centroids
    
    def cluster_centroid_update(self, reference_centroid, cluster_data):
        """
        Selects the best medoid in the cluster to update the medoid to

        Args:
            reference_centroid (1-dimension ndarray): The starting centroid centroid
            cluster_data (2-dimension ndarray): data

        Returns:
            return_centroid: The best centroid to update the data to
        """
        current_cluster_fit_score = self._cost(cluster_data, np.array([reference_centroid]), np.zeros(cluster_data.shape[0], dtype=int))
        best_cluster_fit_score = current_cluster_fit_score
        return_centroid = reference_centroid.copy()
        for new_centroid in cluster_data:
            new_cluster_fit_score = self._cost(cluster_data, np.array([new_centroid]), np.zeros(cluster_data.shape[0], dtype=int))
            if new_cluster_fit_score < best_cluster_fit_score:
                return_centroid = new_centroid
                best_cluster_fit_score = new_cluster_fit_score
                
        return return_centroid
    
    def fit(self, X):
        """
        Compute k-medoids clustering

        Args:
            X (2-dimension ndarray): The X data as a matrix.
        """
        X = self._validate_data(X)
        self._n_samples, self._n_features = X.shape  
        if self._init == "rand":
            initial_centroids = super()._init_random(X)
        elif self._init == "kmeans++":
            initial_centroids = super()._init_kmeansplusplus(X, self._n_clusters)
        else:
            raise ValueError("The init variable %s is invalid " % (self._init))
        if self._verb:
            print("Initial centroid via %s successful" % (self._init), "\n")
        # These variables are for testing the sse of the initial centroids
        initial_labels = self._get_nearest_centroids(X, initial_centroids)
        initial_sse = super()._sse_error(X, initial_centroids, initial_labels)
        self._init_cluster_centers = initial_centroids
        self._init_labels = initial_labels
        self._init_sse = initial_sse
        # Loop over the max_iterations
        current_centroids = np.copy(initial_centroids)
        # assign current_fit_score
        current_fit_score = self._cost(X, initial_centroids, initial_labels)
        # 1: For each iteration, for each cluster
        # 2: Check against all data points, change the medoid in that cluster with that data point
        if self._verb:
            print('Starting KMedoids iterations...')
        for i in range(self._max_iter):
            if self._verb:
                print('Current fit score: %s' % (current_fit_score))
            iteration_centroids = current_centroids
            for n in range(self._n_clusters):
                nearest_centroids = self._get_nearest_centroids(X, current_centroids)
                new_centroids = iteration_centroids.copy()
                new_centroid_n = self.cluster_centroid_update(current_centroids[n], X[nearest_centroids==n])
                new_centroids[n] = new_centroid_n
                new_labels = self._get_nearest_centroids(X, new_centroids)
                new_fit_score = self._cost(X, new_centroids, new_labels)
                #  3: If an iteration improves overall fit score, update the medoid
                if new_fit_score < current_fit_score:
                    current_centroids = new_centroids
                    current_fit_score = new_fit_score
            self._n_iter_converge = i + 1
            # 4: Converge if no change in centroids for an iteration
            if np.array_equal(new_centroids, iteration_centroids):
                if self._verb:
                    print(f"Convergence reached at iteration {i}")
                break
        if self._verb:
            print("KMedoids iterations complete...")
        self._cluster_centers = current_centroids
        self._labels = self._get_nearest_centroids(X, self._cluster_centers)
        self._sse = super()._sse_error(X, self._cluster_centers, self._labels)
        self._ssr = super()._sse_residual(X, self._cluster_centers, self._labels)

        return self
        
    def fit_transform(self, X):
        """
        Transform X into a cluster-distance matrix. Columns represent cluster centers, rows represent datapoints. 
        Each dimension represents the distance of the datapoint to the cluster center generated after fitting the data with fit() based on the distance metric during intialisation

        Args:
            X (2-dimension ndarray): The X data as a matrix.

        Returns:
            (2-dimension ndarray): cluster-distance matrix
        """

        self.fit(X)
        distance_to_cluster = [[self._distance(i, j) for i in self._cluster_centers] for j in X]
        distance_to_cluster = np.array(distance_to_cluster)
        
        return distance_to_cluster
    
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
    def ssr_(self):
        return self._ssr
    
    @property
    def n_iter_converge(self):
        return self._n_iter_converge
    
    @property
    def x_samples_(self):
        return self._n_samples
    
    @property
    def x_features_(self):
        return self._n_features
    

    