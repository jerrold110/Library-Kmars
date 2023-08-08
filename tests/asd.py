import csv
import unittest
import numpy as np
import os
import sys
base_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(f"{base_directory}/src")
from kmars import KMeans

# Prepare the data 
X = []
with open('tests/blobs.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        X.append(row)
X = np.array(X, np.float16)

data = []
with open('tests/kmeans_euclidean_centroids.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        data.append(row)


# Fit model to data
km = KMeans(3, dist='euclidean', init='kmeans++', random_state=0)
cc = km.fit(X).cluster_centers_

print(data)
print(cc)

class TestKMeansMethods(unittest.TestCase):
    def test_strings_a(self):
        
        self.assertEqual(data, list(cc))
  
if __name__ == '__main__':
    unittest.main()