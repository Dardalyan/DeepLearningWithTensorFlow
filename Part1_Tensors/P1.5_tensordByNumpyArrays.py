import numpy as np
import tensorflow as tf

#tf.zeros() and tf.ones()

print(tf.ones(shape=(3,2)))
# OUTPUT:  tf.Tensor(
#                   [[1. 1.]
#                    [1. 1.]
#                    [1. 1.]], shape=(3, 2), dtype=float32)


print(tf.zeros(shape=(2,3)))
# OUTPUT: tf.Tensor(
#                   [[0. 0. 0.]


#                    [0. 0. 0.]], shape=(2, 3), dtype=float32)

print("\n*********************************************************************")

numpy_array = np.arange(1,25,dtype=np.int32)
print(numpy_array) #OUTPUT: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]

y = tf.constant(numpy_array)
print(y) # OUTPUT: tf.Tensor([ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24], shape=(24,), dtype=int32)


X = tf.constant(numpy_array,shape=(3,2,4))
print(X)
#OUTPUT : tf.Tensor(
#                   [[[ 1  2  3  4]
#                     [ 5  6  7  8]]
#
#                    [[ 9 10 11 12]
#                     [13 14 15 16]]
#
#                    [[17 18 19 20]
#                     [21 22 23 24]]], shape=(3, 2, 4), dtype=int32)


