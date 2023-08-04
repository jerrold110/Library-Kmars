## Introduction
This is an implementation of K-means/medians/medoids with various distance metrics (Euclidean, Manhattan, Cosine...) built over Numpy.

I created this library for my own use, because the KMeans class of the Scikit-learn library lacks options for customising the variant of the K-algorithm and options for using distance metrics besides Euclidean distance. 

This library was developed with the intention to replicate the functionality and parameter convention used in Scikit-learn as close as possible so that end users, such as myself, will have little difficulty writing code for K-algorithm clustering machine learning tasks with KMars. New parameters are for controlling added functionality with organisation in mind. Currently only accepts numpy arrays as input.

## Example:
```python
from kmars import KMeans, KMedians

km = KMeans(3, dist='cosine')
KMeans.fit(X)
labels = km.labels_
cetroid_positions = km.cluster_centers_

km2 = KMedians(4, dist='manhattan', init='rand')
km2.fit(X)
init_labels = km2.init_labels_
cetroid_positions = km2.cluster_centers_
```

## Features:
- K-means++ centroid initialisation with seed search 
- Frobenius norm convergence
- Distance metrics: 'euclidean','manhattan','minikowski','cosine','hamming'
- Algorithms: KMeans, KMedians, KMedoids
- Getter methods for positions/error_scores/close centroid for each datapoint label for initial centroids and final centroids
- Selection of Sum-Square-Error or Sum-Error metric for KMedoids cluster centroid update and overall fit score

## Distance metric
The distance metric selected at initialisation is the same metric used for: 
- Centroid initialisation
- Overall SSE (between all cluster centers and data points) calculation
- Kmedoids centroid selection and all-cluster-centroid-update-approval

## Future features
- Data validation to take in pandas dataframes
- More algorithms
- More distance metrics