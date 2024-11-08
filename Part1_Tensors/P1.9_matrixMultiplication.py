import tensorflow as tf

m1 = tf.constant([[7,5,3],[1,3,9]])
m2 = tf.constant([[2,6],[4,8],[6,4]])

# RANDOM TENSORS BUT JUST AS AN EXAMPLE COMMENT LINES

print(m1)
"""
tf.Tensor(
[[7 5 3]
 [1 3 9]], shape=(2, 3), dtype=int32)
"""
print(m2)
"""
tf.Tensor(
[[2 6]
 [4 8]
 [6 4]], shape=(3, 2), dtype=int32)
"""

print(tf.matmul(m1,m2))
"""
tf.Tensor(
[[52 94]
 [68 66]], shape=(2, 2), dtype=int32)
"""

print(f"\n OR FOR MULTIPLICATION -> m1 @ m2 -> \n {m1 @ m2}") # THE SAME OUTPUT


result = m1 @ m2
print(f"\n RESHAPING THE RESULT -> {tf.reshape(result,shape=(4,))}") # OUTPUT :  RESHAPING THE RESULT -> [52 94 68 66]

# TRANSPORING MATRIX
m3 = tf.constant([[2,4],[7,6],[5,4]],shape=[3,2])
print(f"\n The initial version of the matrix m3 -> {m3}")
"""
 The initial version of the matrix m3 -> [  [2 4]
                                            [7 6]
                                            [5 4] ]
"""
print(f" \n Transposed the matrix m3 -> {tf.transpose(m3)} ")
"""
Transposed the matrix m3 -> [   [2 7 5]
                                [4 6 4]  ]
"""