## Introduction
This is an implementation of K-means/medians/medoids with various distance metrics (Euclidean, Manhattan, Cosine...) built over Numpy.

I created this library for my own use, because the KMeans class of the Scikit-learn library lacks options for customising the variant of the K-algorithm and options for using distance metrics besides Euclidean distance. 

This library was developed with the intention to replicate the functionality and parameter convention used in Scikit-learn as close as possible so that end users, such as myself, will have little difficulty writing code for K-algorithm clustering machine learning tasks with KMars. The additional distance metrics not present in sklearn's clustering such as manhattan, minikowski, and cosine enable better results when working with high dimensional data.

Jupyter notebook comparison with sklearn
https://github.com/jerrold110/Library-Kmars/blob/main/notebooks/Comparison%20of%20Sklearn%20and%20Kmars.ipynb

## Example:
```python
import numpy as np
from kmars import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

km = KMeans(4, dist='euclidean', init='kmeans++')
KMeans.fit(X)
labels = km.labels_
cetroids = km.cluster_centers_

```

## Features:
- K-means++ centroid initialisation with seed search 
- Frobenius norm convergence
- Distance metrics: 'euclidean','manhattan','minikowski','cosine','hamming'
- Algorithms: KMeans, KMedians, KMedoids
- Getter methods for positions/error_scores/close centroid for each datapoint label for initial centroids and final centroids
- Selection of Sum-Square-Error or Sum-Error metric for KMedoids cluster centroid update and overall fit score
- Data type changes to float64 during distance calculation to avoid numerical overflow

## Distance metric
The distance metric selected at initialisation is the same metric used for: 
- Centroid initialisation with kmeans++
- Overall SSE in original distance metric, and euclidean distance (for comparing different distance metrics on the same algorithm)
- Kmedoids centroid selection and all-cluster-centroid-update-approval

## Future features
- Data validation to take in pandas dataframes
- More algorithms, algorithm upgrades (FastPAM for Kmedoids instead of Naive)
- More distance metrics

## Issues
Currently only accepts numpy arrays as input.