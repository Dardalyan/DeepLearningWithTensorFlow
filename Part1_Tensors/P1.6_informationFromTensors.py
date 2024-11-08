import tensorflow as tf

rank_4_tensor = tf.zeros(shape=(2,3,4,5))
print(rank_4_tensor)
"""
OUTPUT:

tf.Tensor(
[[[[0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]]

  [[0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]]

  [[0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]]]


 [[[0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]]

  [[0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]]

  [[0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0.]]]], shape=(2, 3, 4, 5), dtype=float32)

"""

print(f'Shape : {rank_4_tensor.shape} \n Dimension : {rank_4_tensor.ndim} \n Size: {tf.size(rank_4_tensor)}')
# OUTPUT : Shape : (2, 3, 4, 5) Dimension : 4  Size: 120

print(f'Elements along the 0 axis: {rank_4_tensor.shape[0]}') # 2
print(f'Elements along the last axis: {rank_4_tensor.shape[-1]}') # 5 -> if rank_4_tensor.shape[2] = 4 
