import tensorflow  as tf


#Create tensor with tf.constant()

scalar = tf.constant(7)
print(scalar) # OUTPUT : tf.Tensor(7, shape=(), dtype=int32)
#Checking number of dimensions
print(scalar.ndim)  # OUTPUT: 0

vector = tf.constant([10,10])
print(vector)   # OUTPUT: tf.Tensor([10 10], shape=(2,), dtype=int32)
#Checking number of dimensions
print(vector.ndim)  # OUTPUT: 1

matrix = tf.constant([[5,5]])
print(matrix)   # OUTPUT: tf.Tensor([[5 5]], shape=(1,2), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 2

matrix = tf.constant([[[8,8],[9,9]]])
print(matrix)   # OUTPUT: tf.Tensor([[[8 8] [9,9]]], shape=(1,2,2), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 3

matrix = tf.constant([[3,5],[9,4]])
print(matrix)   # OUTPUT: tf.Tensor([[3 5] [9 4]], shape=(2, 2), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 2

matrix = tf.constant([[3,5],[9,4],[2,7]])
print(matrix)   # OUTPUT: tf.Tensor([   [3 5]
#                                       [9 4]
#                                       [2 7]], shape=(3, 2), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 2

matrix = tf.constant([[3,5,9],[4,2,7]])
print(matrix)   # OUTPUT: tf.Tensor([[3 5 9]
#                                    [4 2 7]], shape=(2, 3), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 2


matrix = tf.constant([[[3],[5],[9],[4],[2],[7]]])
print(matrix)   # OUTPUT: tf.Tensor([[ [3]
#                                      [5]
#                                      [9]
#                                      [4]
#                                      [2]
#                                      [7]]], shape=(1,6,1), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 3


matrix = tf.constant([
    [
        [24,65],[21,72],[24,93],[12,53]
    ],
    [
        [23,65],[12,68],[38,62],[48,19]
    ],
    [
        [64,29],[49,25],[41,33],[12,54]
    ]

                    ])
print(matrix)   # OUTPUT: tf.Tensor([[[24 65]
#                                     [21 72]
#                                     [24 93]
#                                     [12 53]]
#
#                                    [[23 65]
#                                     [12 68]
#                                     [38 62]
#                                     [48 19]]
#
#                                    [[64 29]
#                                     [49 25]
#                                     [41 33]
#                                     [12 54]]], shape=(3,4,2), dtype=int32)
#Checking number of dimensions
print(matrix.ndim)  # OUTPUT: 3




