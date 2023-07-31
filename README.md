## Introduction
This is an implementation of K-means/medians/medoids with various distance metrics (Euclidean, Manhattan, Cosine...).

I created this library for my own use, out of frustratation that the KMeans class of the Scikit-learn library lacks options for customising the variant of the K-algorithm and options for using distance metrics besides Euclidean distance. 

This library was developed with the intention to replicate the functionality and parameter convention used in Scikit-learn as close as possible so that end users, such as myself, will have little difficulty writing code for K-algorithm clustering machine learning tasks with KMars. New parameters are for controlling added functionality with organisation in mind.

## Example:
from kmars import KMeans

kmeans = KMeans(3, dist='cosine')
KMeans.fit(x)
results = KMeans.labels_

## Features:
K-means++ centroid initialisation with seed search 
Frobenius norm convergence
Distance metrics: 'euclidean','manhattan','minikowski','cosine','hamming'
Algorithms: KMeans, KMedians, KMedoids
Getter methods for positions/error_scores/closest_centroid_foreach_data for initial centroids and final centroids

## Future features:
Silhouette score for Kmedoids cost computation at each iteration
Silhouette score getter method
