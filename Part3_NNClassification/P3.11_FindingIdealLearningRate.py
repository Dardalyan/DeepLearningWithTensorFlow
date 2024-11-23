import tensorflow as tf
from keras.src.datasets import fashion_mnist
from matplotlib import pyplot as plt


(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()



# We can get our training and testing data between 0 and 1
train_data_norm = train_data/train_data.max()
test_data_norm = test_data/test_data.max()

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

#findng best learning rate
lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3*(10**(epoch)/20))
lr_history = model.fit(train_data_norm,train_labels,epochs=40,validation_data=(test_data_norm,test_labels),callbacks=[lr_schedular])


# Ploting to find best learning rate

lrts = 1e-3 * (10**(tf.range(40)/20))
plt.semilogx(lrts,lr_history.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Finding the best suitable learning rate...')
plt.show()





