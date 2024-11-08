import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#House problem features = bedroom , garage ,bathroom
#Predictable variable : price = 939700

X = np.array([-7.,-4.,-1.,2.,5.,8.,11.,14.]) #features
y = np.array([3.,6.,9.,12.,15.,18.,21.,24.]) #labels

print(y == X + 10) # OUTPUT: [ True  True  True  True  True  True  True  True] -> the relation is y = X +10

plt.scatter(X,y)
#plt.show()

print("\n************************************************\n")
# Creating demo tensor
house_info = tf.constant(['bedroom','bathroom','garage'])
house_price = tf.constant([937700])
print(house_info) # OUTPUT : tf.Tensor([b'bedroom' b'bathroom' b'garage'], shape=(3,), dtype=string)
print(house_price) # OUTPUT : tf.Tensor([937700], shape=(1,), dtype=int32)

print("\n************************************************\n")

#Convert numpy arrays to tensors
X = tf.cast(tf.constant(X),tf.float32) # tf.Tensor([-7 -4 -1  2  5  8 11 14], shape=(8,), dtype=int64)
y = tf.cast(tf.constant(y),tf.float32) # tf.Tensor([ 3  6  9 12 15 18 21 24], shape=(8,), dtype=int64)
print("X: ",X)
print("y: ",y)


#Set Random seed
tf.random.set_seed(42)

# 1-Create a model un sing the Sequential API -> https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1) # We give -7 then we get 3 as above "X and y example"
])

# 2-Compile model
model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

# 3-Fit model
model.fit(tf.expand_dims(X,axis=1),y,epochs=5)

# 4- Check out X and y
print(X,y) # tf.Tensor([-7. -4. -1.  2.  5.  8. 11. 14.], shape=(8,), dtype=float32)
           # tf.Tensor([ 3.  6.  9. 12. 15. 18. 21. 24.], shape=(8,), dtype=float32)

y_prediction = model.predict(tf.constant([17.0]))
print(y_prediction)

