import tensorflow as tf
#Generate 2 random tensors (but same)

generator = tf.random.Generator.from_seed(42) # set seed for reproducibility | returns new generator
random_1 = generator.normal(shape=(3,2))
print(random_1)
# OUTPUT: something like -> [[-0.7565803  -0.06854702]
#                               [ 0.07595026 -1.2573844 ]
#                               [-0.23193763 -1.8107855 ]], shape=(3, 2), dtype=float32)

generator = tf.random.Generator.from_seed(42)
random_2 = generator.normal(shape=(3,2))
print(random_2)
#OUTPUT :  tf.Tensor(
#           [[-0.7565803  -0.06854702]
#            [ 0.07595026 -1.2573844 ]
#            [-0.23193763 -1.8107855 ]], shape=(3, 2), dtype=float32)


print(f"\n {random_1 == random_2}")
#OUTPUT :  [[ True  True]
#           [ True  True]
#           [ True  True]]
