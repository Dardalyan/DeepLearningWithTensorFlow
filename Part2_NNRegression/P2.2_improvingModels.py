import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam


X = np.array([-7.,-4.,-1.,2.,5.,8.,11.,14.]) #features
y = np.array([3.,6.,9.,12.,15.,18.,21.,24.]) #labels

X = tf.cast(tf.constant(X),tf.float32) # tf.Tensor([-7 -4 -1  2  5  8 11 14], shape=(8,), dtype=int64)
y = tf.cast(tf.constant(y),tf.float32) # tf.Tensor([ 3  6  9 12 15 18 21 24], shape=(8,), dtype=int64)


#Set Random seed
tf.random.set_seed(42)

# 1-Create a model un sing the Sequential API -> https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,activation='relu'), # hidden layers with activation function 'relu'
    tf.keras.layers.Dense(100,activation='relu'), # hidden layers with activation function 'relu'
    tf.keras.layers.Dense(100,activation='relu'), # hidden layers with activation function 'relu'
    tf.keras.layers.Dense(100,activation='relu'), # hidden layers with activation function 'relu'
    tf.keras.layers.Dense(100,activation='relu'), # hidden layers with activation function 'relu'
    tf.keras.layers.Dense(100,activation='relu'), # hidden layers with activation function 'relu'
    tf.keras.layers.Dense(1,activation=None), # Output layer
])

# 2-Compile model
model.compile(loss=tf.keras.losses.mae,
              optimizer=Adam(learning_rate=0.001),
              metrics=['mae'])

# 3-Fit model
model.fit(X,y,epochs=100)

y_prediction = model.predict(tf.constant([17.0]))
print(y_prediction)


