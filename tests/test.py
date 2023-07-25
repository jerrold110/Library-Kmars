import numpy as np

# arr = np.empty((0,3), int)
# print(arr)
# arr = np.append(arr, np.array([[1,2,3]]), axis=0)
# arr = np.append(arr, np.array([[4,5,4]]), axis=0)
# arr = np.append(arr, np.array([[1,2,3]]), axis=0)

# print(arr.sum(axis=None))
# print(arr.sum(axis=1))
# print(arr.sum(axis=0))
# print(arr.sum(axis=-1))

# arr = np.empty(0, int)

# print(arr)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# arr = np.append(arr, 1)
# print(arr)

# x = 'asdasd'
# print("Cannot clone object %s, as the constructor " % (x))

class ReadOnlyObject:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

# Create an instance of the ReadOnlyObject class
obj = ReadOnlyObject(42)

# Access the read-only variable
print(obj.value)  # Output: 42

# Attempt to modify the read-only variable (this will raise an AttributeError)
#obj.value = 100 

