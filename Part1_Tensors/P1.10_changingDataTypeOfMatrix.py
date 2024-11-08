import tensorflow as tf


fmatrix = tf.constant([[2.4,4.5,7.9],[1.,3.5,6.1]])
print(fmatrix)
"""
tf.Tensor(
[[2.4 4.5 7.9]
 [1.  3.5 6.1]], shape=(2, 3), dtype=float32)
"""

fmatrix = tf.cast(fmatrix, dtype=tf.float16)
print(fmatrix)
"""
tf.Tensor(
[[2.4 4.5 7.9]
 [1.  3.5 6.1]], shape=(2, 3), dtype=float16) 
"""

# 16bit precisions can run more faster in modern GPUs

