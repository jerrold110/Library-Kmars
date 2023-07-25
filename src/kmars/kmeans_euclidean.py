from base_kmeans import _BaseKMeans
import numpy as np


class KMeansEuclidean(_BaseKMeans):
    def __init__(self, n_clusters, init, max_iter, seed):
        super().__init__(n_clusters, init, max_iter, seed)
        self._n_samples = None
        self._n_features = None
        self._cluster_centers = None
        self._labels = None
        self._sse = None
        print('KMeansEuclidean object creation successful')
    def _distance(self, v1, v2):
        """
        Euclidean distance between two points
        Distance methods take in two single dimension vectors (ndarrays)
        """
        dist = np.linalbg.norm(v1 - v2, ord=2, axis=0)
        return dist
    
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
    
    def _centroid_new_pos(self, points):
        """
        Takes in a 2D numpy array of all the points belonging to a centroid, returns an array that is mean 
        """
        return points.mean(axis=0)
    
    def _centroids_update(self, n_centroids, x, x_nearest_centroids, current_centroids):
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
            centroid_connected_points = x[x_nearest_centroids==i]
            new_centroid = self._centroid_new_pos(centroid_connected_points)
            current_centroids[i] = new_centroid
            
        return current_centroids
    
    def _get_nearest_centroids(self, x, centroids):
        """
        Returns a vector of the index of the closest centroid to each data point in x.

        Args:
            x (matrix): data
            centroids (_type_): the array containing the positions of the centroids
        """
        nearest_centroids = np.empty(0, int)
        # Loop over data points
        # Loop over centroids, get closest distance, get index position
        # Add index position to nearest_centroids
        for i in range(x.shape[0]):
            datapoint = x[i]
            ind_nearest_centroid = np.argmin(self._distance(datapoint, c) for c in centroids)
            nearest_centroids = np.append(ind_nearest_centroid, nearest_centroids, 0)
        
        return nearest_centroids
    
    def _sse_error(self, x, cluster_centers, x_labels):
        """
        Calculates the sum of squares error between data points and cluster centers after fitting the data.

        Args:
            x (_type_): _description_
            cluster_centers (_type_): _description_
            x_labels (_type_): _description_

        Returns:
            float: The SSE error
        """
        sse_error = .0
        for i in self._n_samples:
            nearest_cluster_ind = x_labels[i]
            se_error = np.square(x[i]-cluster_centers[nearest_cluster_ind])
            sse_error += se_error
            
        return sse_error
    
    def fit(self, X):
        """
        Compute k-means clustering

        Args:
            x (_type_): A 2D ndarray
        """
        self._n_samples, self._n_features = X.shape
        np.random.seed(self._random_state)
        
        if self._init == "rand":
            initial_centroids = self._init_random(X, self.n_clusters)
        elif self._init == "kmeans++":
            initial_centroids = self._init_kmeansplusplus(X, self._n_clusters)
        else:
            raise TypeError("The init variable %s is invalue "(self._init))
        print("Initial centroid via %s successful" % (self._init))
        # Loop over the max_iterations
        # tolerance for breakage will be added later on
        current_centroids = initial_centroids.copy()
        # update current_centroids by recalculating the average position of each centroid
        print('Starting iterations')
        counter = 0
        for _ in range(self._max_iter):
            print("Iteration %s" % (counter)); counter += 1
            nearest_centroids = self._get_nearest_centroids(X, current_centroids)
            current_centroids = self._centroids_update(self._n_clusters, X, nearest_centroids, current_centroids)
        
        print("KMeans iterations complete")
        self.cluster_centers = current_centroids
        self.labels = self._get_nearest_centroids(X, self.cluster_centers)
        self.sse = self._sse_error(X, self.cluster_centers_, self.labels_)
        
    @property
    def cluster_centers_(self):
        return self._cluster_centers
    
    @property
    def labels_(self):
        return self._labels
    
    @property
    def sse_(self):
        return self._sse