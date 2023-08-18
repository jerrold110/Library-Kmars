import numpy as np

def _distance_euclidean(v1, v2):
    """
    Euclidean distance between two vectors. Calculating L2 norm of vector difference
    Distance methods take in two single dimension vectors (ndarrays)
    """
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    dist = np.linalg.norm(v1 - v2, ord=2, axis=0)
    
    return dist.astype(np.float32)

def _distance_manhattan(v1, v2):
    """
    Manhattan distance between two vectors
    Take in two vectors (ndarrays)
    """
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    dist = np.sum(np.abs(v1 - v2))
    
    return dist.astype(np.float32)

def _distance_minikowski(v1, v2, o):
    """
    Minikowski distance between two vectors
    Take in two vectors (ndarrays) and order o
    """
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    dist = np.linalg.norm(v1 - v2, ord=o, axis=0)
    
    return dist.astype(np.float32)

def _distance_cosine(v1, v2):
    """
    Cosine distance between two vectors. 
    Cosine distance = 1 - Cosine similarity. [0, 2]
    Same direction: 0
    Perpendicular: 1
    Opposite: 2
    """
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    cosine_similarity = np.dot(v1, v2)/(np.linalg.norm(v1, ord=2, axis=0)*np.linalg.norm(v2, ord=2, axis=0))
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance.astype(np.float32)

def _distance_hamming(v1, v2):
    """
    Hamming distance between two vectors with binary values.
    Take in two binary vectors (ndarrays)
    """
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    dist = np.sum(np.abs(v1-v2))/np.shape(v1)[0]
    
    return dist.astype(np.float32)


# a = np.array([-7.5114837, -6.0005107])
# b = np.array([-8.027044,  -6.4166536])
# print(_distance_cosine(a, b))