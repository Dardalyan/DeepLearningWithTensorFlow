import tensorflow as tf

tf.random.set_seed(0)
tensor = tf.constant(tf.random.uniform(shape=[50]),shape=[1,1,1,1,50])
print(f"Shape of tensor: {tensor.shape}")
print(tensor)
"""
Shape of tensor: (1, 1, 1, 1, 50)
tf.Tensor(
[[[[[0.29197514 0.20656645 0.53539073 0.5612575  0.4166745  0.80782795
     0.4932251  0.99812925 0.69673514 0.1253736  0.7098167  0.6624156
     0.57225657 0.36475348 0.42051828 0.630057   0.913813   0.6616472
     0.83347356 0.08395803 0.2797594  0.0155232  0.72637355 0.7655387
     0.6798667  0.53272796 0.7565141  0.04742193 0.05037141 0.75174344
     0.1727128  0.3119352  0.29137385 0.10051239 0.16567075 0.7696651
     0.58567977 0.98200965 0.9148327  0.14166534 0.09756553 0.6062784
     0.17792177 0.518052   0.9821211  0.17577946 0.04563165 0.59754145
     0.5629543  0.80507433]]]]], shape=(1, 1, 1, 1, 50), dtype=float32)
"""

squeezed_tensor = tf.squeeze(tensor) # Removes dimensions of size 1
print(f"Shape of squeezed_tensor: {squeezed_tensor.shape}")
print(squeezed_tensor)
"""
Shape of squeezed_tensor: (50,)
tf.Tensor(
[0.29197514 0.20656645 0.53539073 0.5612575  0.4166745  0.80782795
 0.4932251  0.99812925 0.69673514 0.1253736  0.7098167  0.6624156
 0.57225657 0.36475348 0.42051828 0.630057   0.913813   0.6616472
 0.83347356 0.08395803 0.2797594  0.0155232  0.72637355 0.7655387
 0.6798667  0.53272796 0.7565141  0.04742193 0.05037141 0.75174344
 0.1727128  0.3119352  0.29137385 0.10051239 0.16567075 0.7696651
 0.58567977 0.98200965 0.9148327  0.14166534 0.09756553 0.6062784
 0.17792177 0.518052   0.9821211  0.17577946 0.04563165 0.59754145
 0.5629543  0.80507433], shape=(50,), dtype=float32)
"""