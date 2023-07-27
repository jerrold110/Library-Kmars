from .base_kmeans import _BaseKMeans
import numpy as np

class KMeansEuclidean(_BaseKMeans):
    def __init__(self, n_clusters, init='kmeans++', n_init=10, cluster_update='mean', max_iter=300, tol=0.0001, random_state=0):
        super().__init__(n_clusters, init, n_init, cluster_update, max_iter, tol, random_state)
        self._n_samples = None
        self._n_features = None
        self._cluster_centers = None
        self._labels = None
        self._sse = None
    
    def _init_random(self, X, n_clusters):
        """
        Returns a 2D array of initial centroid positions initialised randomly between the max and min value of every column
        """
        random_centroids = super()._init_random()
        
        return random_centroids
    
    def _distance_euclidean(self, v1, v2):
        """
        Euclidean distance between two points
        Distance methods take in two single dimension vectors (ndarrays)
        """
        dist = np.linalg.norm(v1 - v2, ord=2, axis=0)
        
        return dist
    
    def _sse_error(self, X, cluster_centers, x_labels):
        """
        Calculates the sum of squared distances of samples to their closest cluster center

        Args:
            X (_type_): data
            cluster_centers (_type_): array of vectors, of length _n_cluster_centers
            x_labels (_type_): array of ints representing nearest cluster

        Returns:
            float: The SSE error
        """
        sse_error = .0
        for i in range(self._n_samples):
            nearest_cluster_ind = x_labels[i]
            point_to_centroid_distance = self._distance_euclidean(X[i], cluster_centers[nearest_cluster_ind])
            se_error = point_to_centroid_distance**2
            sse_error += se_error
        return sse_error
    
    def _init_kmeansplusplus(self, X, start_seed):
        """
        Returns an array of initial centroid positions initialised with kmeans++ and the sse
        Kmeans++ seeks to spread out the k initial clusters.
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
                        point_to_centroid.append(self._distance_euclidean(candidate_centroids[j], centroids[c]))
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
        print(results[0][1])
        print(results[0][0])

        return results[0][1]
    
    def _centroid_new_pos(self, points):
        """
        Takes in a 2D numpy array of all the points belonging to a centroid, returns an array that is mean 
        """
        if self._cluster_update == "mean":
            return np.mean(points, axis=0)
        elif self._cluster_update == "median":
            return np.median(points, axis=0)
        else:
            raise AssertionError("This value of cluster_update %s ain't possible" % (self._cluster_update))
    
    def _centroids_update(self, n_centroids, X, x_nearest_centroids, current_centroids):
        """
        Returns a vector of the updated centroids.
        x_nearest_centroids denotes the index of the nearest centroid in centroids for each data point.
        
        Args:
            n_centroids (int): the number of centroids
            x (matrix): data
            x_nearest_centroids (vector): index positions of nearest centroids for each data point
            current_centroids (matrix): the array containing the current positions of the centroids
        """
        for i in range(n_centroids):
            centroid_connected_points = X[x_nearest_centroids==i]
            new_centroid = self._centroid_new_pos(centroid_connected_points)
            current_centroids[i] = new_centroid
            
        return current_centroids
    
    def _get_nearest_centroids(self, X, centroids):
        """
        Returns a vector of the index of the closest centroid to each data point in x. Ndarray of ints

        Args:
            x (matrix): data
            centroids (_type_): the array containing the positions of the centroids
        """
        nearest_centroids = np.empty(0, int)
        # Loop over data points
        # Loop over centroids, get closest distance, get index position
        # Add index position to nearest_centroids
        for i in range(X.shape[0]):
            datapoint = X[i]
            distances = np.array([self._distance_euclidean(datapoint, c) for c in centroids])
            ind_nearest_centroid = np.argmin(distances)
            nearest_centroids = np.append(nearest_centroids, np.array([ind_nearest_centroid]), 0)

        return nearest_centroids
    
    def fit(self, X):
        """
        Compute k-means clustering

        Args:
            x (_type_): A 2D ndarray
        """
        self._n_samples, self._n_features = X.shape  
        if self._init == "rand":
            initial_centroids = self._init_random(X, self.n_clusters)
        elif self._init == "kmeans++":
            initial_centroids = self._init_kmeansplusplus(X, self._n_clusters)
        else:
            raise TypeError("The init variable %s is invalid " % (self._init))
        print("Initial centroid via %s successful" % (self._init))
        # Loop over the max_iterations
        # tolerance for breakage will be added later on
        current_centroids = np.copy(initial_centroids)
        # Update current_centroids by recalculating the average position of each centroid
        print('Starting iterations...')
        for i in range(self._max_iter):
            nearest_centroids = self._get_nearest_centroids(X, current_centroids)
            new_centroids = self._centroids_update(self._n_clusters, X, nearest_centroids, current_centroids)
            # Calculate frobenius norm against tolerance to declare convergence
            diff = new_centroids - current_centroids
            fn_update_diff = np.sqrt(np.sum(np.square(diff)))
            if (fn_update_diff < self._tol):
                print(f"Convergence reached at iteration {i}")
                break
            else:
                current_centroids = new_centroids
        if self._cluster_update == "mean":
            print("KMeans iterations complete")
        elif self._cluster_update == "median":
            print("KMedians iterations complete")
        self.cluster_centers = current_centroids
        self.labels = self._get_nearest_centroids(X, self.cluster_centers)
        self.sse = self._sse_error(X, self.cluster_centers, self.labels)
        
    @property
    def cluster_centers_(self):
        return self._cluster_centers
    
    @property
    def labels_(self):
        return self._labels
    
    @property
    def sse_(self):
        return self._sse