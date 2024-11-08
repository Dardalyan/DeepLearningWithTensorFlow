import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_circles

from Part3_NNClassification import plot_desicion_boundary

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
              optimizer=keras.optimizers.Adam(learning_rate=0.01))


x_train, y_train = x[:800], y[:800]
x_test, y_test = x[800:], y[800:]

print(f"\n x_train shape:{x_train.shape} | y_train shape:{y_train.shape} | x_test shape: {x_test.shape} | y_test shape: {y_test.shape} \n")

history = model.fit(x_train,y_train,epochs=25,verbose=0)
model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
print(y_pred[0], y_test.numpy()[0])

plt.subplot(1,2,1)
plt.title("Train Data")
plot_desicion_boundary(model,x_train.numpy(),y_train.numpy())
plt.subplot(1,2,2)
plt.title("Test Data")
plot_desicion_boundary(model,x_test.numpy(),y_test.numpy())
#plt.show()

model.summary()

#Plot loss (or training) curves

history_frame = pd.DataFrame(history.history)
print(history_frame)

history_frame.plot()
#plt.show()

# Checking Learning Rate Differences

# Compile the  Model again
model.compile(loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.Accuracy()],
              optimizer=keras.optimizers.Adam())


lr_callback = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch/20))
history = model.fit(x_train,y_train,epochs=100,verbose=1,callbacks=[lr_callback])

pd.DataFrame(history.history).plot(figsize=(9,7),xlabel= 'epochs')
plt.show()

#ploting lr vs loss
lrs = 1e-4 * (10*(tf.range(100)/20))
plt.figure(figsize=(9,7))
plt.semilogx(lrs,history.history['loss'])
plt.xlabel('learning rate')
plt.ylabel('loss')
#plt.show()

# Checking accuracy of our model
loss, accuracy = model.evaluate(x_test,y_test)
print(f"Loss: {loss} | Accuracy: {(accuracy*100):.2f}%")




















