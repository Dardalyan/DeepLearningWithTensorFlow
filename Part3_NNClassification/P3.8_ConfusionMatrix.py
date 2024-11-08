import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
import itertools

# TRUE POSITIVE = model predicts 1 when truth is 1
# TRUE NEGATIVE = model predicts 0 when truth is 0
# FALSE POSITIVE = model predicts 1 when truth is 0
# FALSE NEGATIVE = model predicts 0 when truth is 1

# Evaluating and Improving Classification Model
x,y = make_circles(1000,noise=0.03,random_state=42)

x = tf.constant(x)
y = tf.constant(y)

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


x_train, y_train = x[:800], y[:800]
x_test, y_test = x[800:], y[800:]

lr_callback = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch/20))
history = model.fit(x_train,y_train,epochs=100,verbose=1,callbacks=[lr_callback])


# Checking accuracy of our model
loss, accuracy = model.evaluate(x_test,y_test)
print(f"\n Loss: {loss} | Accuracy: {(accuracy*100):.2f}%")

# Making prediction
y_pred = model.predict(x_test)

print(y_test[:10])
print(y_pred[:10],"\n")

#converting prediction probabilities to binary format and view the first 10
# NOTE : If we don't do this we will get this error in confusion matrix creation -> ValueError: Classification metrics can't handle a mix of binary and continuous targets.

y_pred_binary = tf.round(y_pred)
print("The first 10 element of Y prediction binary:" , y_pred_binary[:10])
print("The first 10 element of Y test :", y_test[:10])

# Creating confusion matrix
conf_matrix = confusion_matrix(y_test,y_pred_binary)
print(f"Confusion Matrix: {conf_matrix}")

# Normalize
cm_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# getting number of class
n_classes = conf_matrix.shape[0]
print(n_classes)

# lets prettify it
fig, ax = plt.subplots(figsize=(10,10))

# Create a matrix plot
cax = ax.matshow(conf_matrix,cmap=plt.cm.Blues)
fig.colorbar(cax)

# craete classses
classes = False

if classes:
    labels = classes
else:
    labels = np.arange(conf_matrix.shape[0])

# Label the axes
ax.set(
    title='Confusion Matrix',
    xlabel="predicted Label",
    ylabel= "True Label",
    xticks = np.arange(n_classes),
    yticks = np.arange(n_classes),
    xticklabels=labels,
    yticklabels=labels,
)

# Set threshold for different colors
threshold = (conf_matrix.max() + conf_matrix.min()) / 2.0
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, f"{conf_matrix[i, j]} ({conf_matrix[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > threshold else "black",size=15)
plt.show()























