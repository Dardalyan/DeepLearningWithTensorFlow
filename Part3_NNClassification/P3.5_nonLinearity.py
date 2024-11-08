import keras
import keras.src.layers
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

from Part3_NNClassification import plot_desicion_boundary

x,y = make_circles(1000,noise=0.03,random_state=42)

x= tf.constant(x)
y= tf.constant(y)

# Create the Model
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(units=100,activation=keras.activations.relu),
    keras.layers.Dense(units=100,activation=keras.activations.relu),
    keras.layers.Dense(units=100,activation=keras.activations.relu),
    keras.layers.Dense(units=1,activation=keras.activations.sigmoid),
])



# Compile the  Model
model.compile(loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.Accuracy()],
              optimizer=keras.optimizers.Adam())


model.fit(x,y,epochs=100,verbose=1)

model.evaluate(x,y)

#plot
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.show()

plot_desicion_boundary(model=model,x=x.numpy(),y=y.numpy())














