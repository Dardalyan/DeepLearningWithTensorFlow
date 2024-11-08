import tensorflow as tf
import numpy as np

tensor = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(tensor)
"""
tf.Tensor(
[[1 2 3]
 [4 5 6]
 [7 8 9]], shape=(3, 3), dtype=int64)
"""


print(np.array(tensor))
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

print(type(np.array(tensor))) # <class 'numpy.ndarray'>

print(tensor.numpy())
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

print(tensor.numpy()[1]) #  [4 5 6]




