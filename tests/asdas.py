import numpy as np

x = np.array([[1,2], [3, 4], [5, 6]]).astype(np.float32)
y = np.array([[1,2], [3, 4], [5, 6]]).astype(np.float64)
z = (x+y).mean(axis=0)

def shit(x):
    x = x.astype(np.int16)
    return x

xx = shit(x)

print(x.dtype)