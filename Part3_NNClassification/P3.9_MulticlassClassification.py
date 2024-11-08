import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random

# The data has already been sorted into training and test sets for us
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()
# train_data : holds data with shape (6000,28,28)
# train_labels : holds data with shape (6000,)
# test_data : holds data with shape (10000, 28, 28)
# test_labels : holds data with shape (10000,)

"""
    | Label | Description |
    |:-----:|-------------|
    |   0   | T-shirt/top |
    |   1   | Trouser     |
    |   2   | Pullover    |
    |   3   | Dress       |
    |   4   | Coat        |
    |   5   | Sandal      |
    |   6   | Shirt       |
    |   7   | Sneaker     |
    |   8   | Bag         |
    |   9   | Ankle boot  |
"""

class_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
print(class_names.size)

# Check the shape of a single example
plt.imshow(train_data[0])
plt.show()


print(f"Label:{train_labels[0]} and belongs to the class:{class_names[train_labels[0]]}")

plt.figure(figsize=(10,10))
for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    rand_index = random.choice(range(len(train_labels)))
    plt.imshow(train_data[rand_index],cmap='gray')
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)



plt.show()