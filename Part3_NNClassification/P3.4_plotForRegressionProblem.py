import keras
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

# CREATE REGRESSION DATA
x_reg = tf.range(0,1000,5)
y_reg = tf.range(100,1100,5)

x_reg_train = x_reg[:150]
x_reg_test = x_reg[150:]
y_reg_train = y_reg[:150]
y_reg_test = y_reg[150:]


# Create the Model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(units=100,activation=keras.activations.linear),
    keras.layers.Dense(units=100,activation=keras.activations.linear),
    keras.layers.Dense(units=100,activation=keras.activations.linear),
    keras.layers.Dense(units=1),
])


# Compile the  Model
model.compile(loss=keras.losses.MeanSquaredError().name,
              metrics=[keras.metrics.MeanSquaredError().name],
              optimizer=keras.optimizers.Adam())


model.fit(x_reg_train,y_reg_train,epochs=100,verbose=0)
model.evaluate(x_reg_test,y_reg_test)

#predictions
y_reg_pred = model.predict(x_reg_test)
y_reg_pred = tf.squeeze(tf.constant(y_reg_pred))


#plot
plt.scatter(x_reg_train,y_reg_train,c="b",label= "Training Data")
plt.scatter(x_reg_test,y_reg_test,c="g",label="Test Data")
plt.scatter(x_reg_test,y_reg_pred,c="r",label="Predictions")
plt.legend()
plt.show()











