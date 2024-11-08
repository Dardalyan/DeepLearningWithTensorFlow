import tensorflow as tf


tensor = tf.constant([[-1, 2, -3], [4, -5, -6]])
print(tensor)
"""
tf.Tensor(
[[-1  2 -3]
 [ 4 -5 -6]], shape=(2, 3), dtype=int32)
"""
# Get Absoulute
tensor = tf.abs(tensor)
print(tensor)
"""
tf.Tensor(
[[1  2 3]
 [ 4 5 6]], shape=(2, 3), dtype=int32)
"""


# Get The Min   Of A Tensor
print(tf.reduce_min(tensor)) # tf.Tensor(1, shape=(), dtype=int32)

# Get the Max   Of A Tensor
print(tf.reduce_max(tensor)) # tf.Tensor(6, shape=(), dtype=int32)

# Get The Mean  Of A Tensor
print(tf.reduce_mean(tensor)) #tf.Tensor(3, shape=(), dtype=int32)

# Get The Sum  Of A Tensor
print(tf.reduce_sum(tensor)) #tf.Tensor(21, shape=(), dtype=int32)

# Get The Variance  Of A Tensor
print(tf.math.reduce_variance(tf.cast(tensor,dtype=tf.float16))) # tf.Tensor(2.916, shape=(), dtype=float16)


tf.random.set_seed(42)
tensor = tf.random.uniform(shape=[50])
print(tensor)
"""
tf.Tensor(
[0.6645621  0.44100678 0.3528825  0.46448255 0.03366041 0.68467236
 0.74011743 0.8724445  0.22632635 0.22319686 0.3103881  0.7223358
 0.13318717 0.5480639  0.5746088  0.8996835  0.00946367 0.5212307
 0.6345445  0.1993283  0.72942245 0.54583454 0.10756552 0.6767061
 0.6602763  0.33695042 0.60141766 0.21062577 0.8527372  0.44062173
 0.9485276  0.23752594 0.81179297 0.5263394  0.494308   0.21612847
 0.8457197  0.8718841  0.3083862  0.6868038  0.23764038 0.7817228
 0.9671384  0.06870162 0.79873943 0.66028714 0.5871513  0.16461694
 0.7381023  0.32054043], shape=(50,), dtype=float32)
"""

# Get The Positional Max   Of A Tensor
print(tf.argmax(tensor)) # tf.Tensor(42, shape=(), dtype=int64) -> represents the position
print(tensor[tf.argmax(tensor)]) # tf.Tensor(0.9671384, shape=(), dtype=float32) -> the value at position 42

# Get The Positional Max   Of A Tensor
print(tf.argmin(tensor)) # tf.Tensor(16, shape=(), dtype=int64) ->  represents the position
print(tensor[tf.argmin(tensor)]) # tf.Tensor(0.009463668, shape=(), dtype=float32) -> the value at position 16






