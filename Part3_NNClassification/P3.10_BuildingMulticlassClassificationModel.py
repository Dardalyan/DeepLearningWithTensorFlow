import tensorflow as tf
import numpy as np
from keras.src.datasets import fashion_mnist



# The data has already been sorted into training and test sets for us
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()
# train_data : holds data with shape (6000,28,28)
# train_labels : holds data with shape (6000,)
# test_data : holds data with shape (10000, 28, 28)
# test_labels : holds data with shape (10000,)


class_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

"""
# understandng how it is working again
plt.figure(figsize=(2,2))
index = random.randint(0,9)
plt.imshow(train_data[index])
print("\nTRAIN DATA : ",train_data[index])
print("\nTRAIN LABEL : ",train_labels[index])
print(class_names[train_labels[index]])
plt.show()
"""




# Create the Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), # shape 28*28 = (None,784) ndim :)
    tf.keras.layers.Dense(units=4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=10,activation=tf.keras.activations.softmax), # for multi-class
])


# Compile the  Model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy()],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))


non_norm_history = model.fit(train_data,tf.one_hot(train_labels,depth=10), epochs=1, validation_data=(test_data,tf.one_hot(test_labels,depth=10)))


#improving performance

#summary of the mmodel
model.summary()

#min and max data of training and test sets
print((train_data.min(),train_data.max()))
print((test_data.min(),test_data.max()))

# NNs prefer data to be scaled (or normalized) , this means they like to have numbers in the tensors between 0 an 1

# We can get our training and testing data between 0 and 1
train_data_norm = train_data/train_data.max()
test_data_norm = test_data/test_data.max()

print((train_data_norm.min(),train_data_norm.max()))
print((test_data_norm.min(),test_data_norm.max()))


# set a random seed
tf.random.set_seed(42)

# create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=10,activation=tf.keras.activations.softmax),
])

#compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ['accuracy'])

# fit the normalized model
norm_history = model.fit(train_data_norm,train_labels,epochs=10,validation_data=(test_data_norm,test_labels))
print(norm_history)


