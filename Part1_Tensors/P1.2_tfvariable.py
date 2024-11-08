import tensorflow as tf

changable_tensor = tf.Variable([7,5],dtype=tf.int32)
print(changable_tensor) # OUTPUT: <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([7, 5], dtype=int32)>

changable_tensor[0].assign(5)
print(changable_tensor) #OUTPUT: <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([5, 5], dtype=int32)>