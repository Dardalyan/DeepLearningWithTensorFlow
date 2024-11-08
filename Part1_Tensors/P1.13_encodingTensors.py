import tensorflow as tf

#one_hot encoding tensors
ls = [0,1,2,3,4,5,6]

print(tf.one_hot(ls, depth=3)) # specify depth
"""
tf.Tensor(
[[1. 0. 0.]   -> refers to 0  in the list 'ls'
 [0. 1. 0.]   -> refers to 1  in the list 'ls'
 [0. 0. 1.]   -> refers to 2  in the list 'ls'
 [0. 0. 0.]   -> refers to 3  in the list 'ls'
 [0. 0. 0.]   -> refers to 4  in the list 'ls'
 [0. 0. 0.]   -> refers to 5  in the list 'ls'
 [0. 0. 0.]], -> refers to 6  in the list 'ls'
 shape=(7, 3), dtype=float32)
"""

#specify your own character or value
print(tf.one_hot(ls, depth=5,on_value='x',off_value='y'))
"""
tf.Tensor(
[[b'x' b'y' b'y' b'y' b'y']
 [b'y' b'x' b'y' b'y' b'y']
 [b'y' b'y' b'x' b'y' b'y']
 [b'y' b'y' b'y' b'x' b'y']
 [b'y' b'y' b'y' b'y' b'x']
 [b'y' b'y' b'y' b'y' b'y']
 [b'y' b'y' b'y' b'y' b'y']], shape=(7, 5), dtype=string)
"""


