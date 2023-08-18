## Introduction
This is an implementation of K-means/medians/medoids with various distance metrics (Euclidean, Manhattan, Cosine...) built over Numpy.

I created this library for my own use, because the KMeans class of the Scikit-learn library lacks options for customising the variant of the K-algorithm and options for using distance metrics besides Euclidean distance. 

This library was developed with the intention to replicate the functionality and parameter convention used in Scikit-learn as close as possible so that end users, such as myself, will have little difficulty writing code for K-algorithm clustering machine learning tasks with KMars. The additional distance metrics not present in sklearn's clustering module such as manhattan, minikowski, and cosine enable better results when working with high dimensional data.

Jupyter notebook comparison with sklearn:

https://github.com/jerrold110/Library-Kmars/blob/main/notebooks/Comparison%20of%20Sklearn%20and%20Kmars.ipynb

## Example:
```pip install kmars```

https://pypi.org/project/kmars/


```python
import numpy as np
from kmars import KMeans, KMedians, KMedoids

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

km = KMeans(4, dist='euclidean', init='kmeans++')
KMeans.fit(X)
labels = km.labels_
cetroids = km.cluster_centers_
print(help(KMeans))
```

## Features:
- Algorithms: KMeans, KMedians, KMedoids
- Distance metrics: 'euclidean','manhattan','minikowski','cosine','hamming'
- K-means++ centroid initialisation with seed search in selected distance metric
- Frobenius (L2) norm convergence, and tolerance parameter
- Getter methods for many metrics after fitting for initial and final centroids
- Selection of Sum-Square-Error or Sum-Error metric for KMedoids cluster centroid update and overall fit score
- Data type changes to float64 during distance calculation to avoid numerical overflow

## Distance metrics
The distance metric selected at initialisation is the same metric used for: 
- Centroid initialisation with kmeans++
- Sum squares error metrics in distance metric at initialisation
- Kmedoids centroid selection and all-cluster-centroid-update-approval
- Manhattan distance (L1 norm) SSE residuals as a common metric to compare different distance metrics of the same model

## Available metrics after fitting model to data
- init_cluster_centers_: position of initial centroids
- init_labels_: closest initial centroid of each data point
- init_sse_: sum of squared errors of initial clusters in the chosen distance metric
- cluster_centers_: positions of final centroids
- labels_: closest final centroid of each data point
- sse_: sum of squared errors of final clusters in the chosen distance metric
- ssr_: sum of squared errors of final clusters in Manhattan distance (L1 norm)
- n_iter_converge_: number of iterations for convergence
- x_samples_: number of samples of data during fit
- x_features_: number of features in each datum during fit

## Future features
- Data validation to take in pandas dataframes
- More algorithms, algorithm upgrades (FastPAM for Kmedoids instead of Naive)
- More distance metrics (eg: improved sqrt cosine)
- Heuristic centroid initialisation: picks the n_clusters points with the smallest sum distance to every other point

## Issues
- Currently only accepts 2 dimensional numpy arrays as input
