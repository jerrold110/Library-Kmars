import numpy as np
from .base_k import _BaseK
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
            distances = []
            for c in centroids:
                distances.append(super()._distance(datapoint, c))
            distances = np.array(distances)
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
            initial_centroids = super()._init_random(X)
        elif self._init == "kmeans++":
            initial_centroids = super()._init_kmeansplusplus(X, self._n_clusters)
        else:
            raise ValueError("The init variable %s is invalid " % (self._init))
        print("Initial centroid via %s successful" % (self._init), "\n")
        # These variables are for testing the sse of the initial centroids
        initial_labels = self._get_nearest_centroids(X, initial_centroids)
        initial_sse = super()._sse_error(X, initial_centroids, initial_labels)
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
        self._sse = super()._sse_error(X, self._cluster_centers, self._labels)
        
        if self._verb == False:
            enablePrint()
        elif self._verb == True:
            pass
        
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
        print(X.shape)
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
    def n_iter_converge(self):
        return self._n_iter_converge
    
    @property
    def x_samples_(self):
        return self._n_samples
    
    @property
    def x_features_(self):
        return self._n_features
    

    