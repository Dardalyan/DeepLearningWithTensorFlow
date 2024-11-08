import tensorflow as tf

not_shuffled = tf.constant([
    [1,4],[8,2],[7,5]
])
print(not_shuffled)

shuffled = tf.random.shuffle(not_shuffled)
print(shuffled) #in every execution different orders !

shuffled = tf.random.shuffle(not_shuffled, seed=42)
print(shuffled) #in every execution different orders !

tf.random.set_seed(42)
shuffled = tf.random.shuffle(not_shuffled, seed=42)
print(shuffled) #the same order once it is shuffled in every execution
