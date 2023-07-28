import numpy as np

def _distance_euclidean(v1, v2):
    """
    Euclidean distance between two points
    Distance methods take in two single dimension vectors (ndarrays)
    """
    dist = np.linalg.norm(v1 - v2, ord=2, axis=0)
    
    return dist

def _manhattan_euclidean(v1, v2):
    """
    Euclidean distance between two points
    Distance methods take in two single dimension vectors (ndarrays)
    """
    dist = np.sum(np.abs(v1 - v2))
    
    return dist

def _cosine_distance(v1, v2):
    
    dist = 