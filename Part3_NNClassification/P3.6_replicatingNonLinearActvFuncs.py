import tensorflow as tf
import keras
from matplotlib import pyplot as plt
# !!!!!!!!!!!!!  Replicating non-linear activation functions !!!!!!!!!!!!!
print("\n !!!!!!!!!!!!!  Replicating non-linear activation functions !!!!!!!!!!!!! \n")

def sigmoid(x):
    return 1/(1+tf.exp(-x))

def relu(x):
    return tf.maximum(0,x)

def linear(x):
    return x

# Creating a toy tensor
A = tf.cast(tf.range(-10,10),dtype=tf.float32)

print(f"Sigmoid A: {sigmoid(A)}")
print(f"ReLu A: {relu(A)}")
print(f"Linear A: {linear(A)}")

plt.plot(sigmoid(A))
plt.show()

plt.plot(relu(A))
plt.show()

plt.plot(linear(A))
plt.show()
