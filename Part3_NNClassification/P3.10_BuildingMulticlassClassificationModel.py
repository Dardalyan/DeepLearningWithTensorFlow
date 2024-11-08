import random

import keras
import numpy as np
from keras._tf_keras.keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

# The data has already been sorted into training and test sets for us
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()
# train_data : holds data with shape (6000,28,28)
# train_labels : holds data with shape (6000,)
# test_data : holds data with shape (10000, 28, 28)
# test_labels : holds data with shape (10000,)


class_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

# understandng how it is working again
"""
plt.figure(figsize=(2,2))
index = random.randint(0,9)
plt.imshow(train_data[index])
print(class_names[train_labels[index]])
plt.show()
"""


# Create the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # shape 28*28 = (None,784) ndim :)
    keras.layers.Dense(units=100,activation=keras.activations.relu),
    keras.layers.Dense(units=100,activation=keras.activations.relu),
    keras.layers.Dense(units=100,activation=keras.activations.relu),
    keras.layers.Dense(units=1,activation=keras.activations.softmax), # for multi-class
])


# Compile the  Model
model.compile(loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.Accuracy()],
              optimizer=keras.optimizers.Adam(learning_rate=0.001))


#non_nor_history = model.fit(train_data,train_labels,epochs=50,validation_data=(test_data,test_labels))

# Improving performance




