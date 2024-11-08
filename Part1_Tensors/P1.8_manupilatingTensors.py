import tensorflow as tf

tensor = tf.constant([[2,3,4],[5,6,7]])
print(tensor)
# OUTPUT: tf.Tensor(
#                   [[2 3 4]
#                    [5 6 7]], shape=(2, 3), dtype=int32)

# ADDITION
print(tensor+10)
print(tf.add(tensor,10))
# OUTPUT: tf.Tensor(
#                   [[12 13 14]
#                    [15 16 17]], shape=(2, 3), dtype=int32)

# SUBTRACTION
print(tensor-2)
print(tf.subtract(tensor,2))
# OUTPUT: tf.Tensor(
#                   [[0 1 2]
#                    [3 4 5]], shape=(2, 3), dtype=int32)

# MULTIPLICATION
print(tensor*3)
print(tf.multiply(tensor,3))
# OUTPUT: tf.Tensor(
#                   [[6 9 12]
#                    [15 18 21]], shape=(2, 3), dtype=int32)

# DIVISION
print(tensor/2)
print(tf.divide(tensor,2))
# OUTPUT: tf.Tensor(
#                   [[1 1.5 2.]
#                    [2.5 3. 3.5]], shape=(2, 3), dtype=int32)




