import keras
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf

#Make 1000 examples
n_samples = 1000

#Create circles
x,y = make_circles(n_samples,noise=0.03,random_state=42)

# Note: x shape-> (1000,2) so like [-0.7542462506997276, 0.23148073787097836] is one of the data from count of 1000
# so we can get the values like -> x[:,0][0] = -0.7542462506997276 , x[:,1][0] = 0.23148073787097836
# Created 1000 x (features) and 1000 y (labels)


circles = pd.DataFrame({"X0":x[:,0], "X1":x[:,1], "label":y })
print(circles.head())
"""
         X0        X1  label
0  0.754246  0.231481      1
1 -0.756159  0.153259      1
2 -0.815392  0.173282      1
3 -0.393731  0.692883      1
4  0.442208 -0.896723      0
"""

plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
#plt.show()

#Input and output Shapes
print(f"\nFeature Shape:{x.shape} \n Output Shape:{y.shape}")

print("\n****************************** Creating Model ******************************\n")


tf.random.set_seed(42)

x = tf.constant(x)
y = tf.constant(y)

# Create the Model
model = keras.Sequential([
    keras.layers.Input(shape=[2,]),
    keras.layers.Dense(units=100),
    keras.layers.Dense(units=10),
    keras.layers.Dense(units=1),
])


# Compile the  Model
model.compile(loss=keras.losses.BinaryCrossentropy(), #Binary Classification we have x[:,0] and x[:,1] as a feature
              metrics=[keras.metrics.Accuracy().name], # Classification accuracy
              optimizer=keras.optimizers.Adam())

# Fit the Model
model.fit(x,y,epochs=100,verbose=0) # ~ 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 283us/step - accuracy: 0.5177 - loss: 0.6918

# Evaluate the Model
model.evaluate(x,y)

print("\n****************************** Improving Model ******************************\n")

# Create the Model
model = keras.Sequential([
    keras.layers.Input(shape=[2,]),
    keras.layers.Dense(units=100),
    keras.layers.Dense(units=80),
    keras.layers.Dense(units=60),
    keras.layers.Dense(units=40),
    keras.layers.Dense(units=20),
    keras.layers.Dense(units=1),
])


# Compile the  Model
model.compile(loss=keras.losses.BinaryCrossentropy(), #Binary Classification we have x[:,0] and x[:,1] as a feature
              metrics=[keras.metrics.Accuracy().name], # Classification accuracy
              optimizer=keras.optimizers.Adam())

# Fit the Model
model.fit(x,y,epochs=300,verbose=0) # ~ 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 283us/step - accuracy: 0.5177 - loss: 0.6918

# Evaluate the Model
model.evaluate(x,y)









