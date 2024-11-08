import tensorflow as tf


print(f'\n*************** CREATING A MODEL *************************\n')

x = tf.range(1,100,2,dtype=tf.float32)
y = x * 2


x_train = x[:40]
x_test = x[40:]


y_train = x[:40]
y_test = x[40:]


model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,name="first_layer"),
    tf.keras.layers.Dense(100,name="second_layer"),
    tf.keras.layers.Dense(100,name="third_layer"),
    tf.keras.layers.Dense(1,name="output_layer"),
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.fit(x_train,y_train,epochs=100,verbose=1)

y_pred   = tf.constant( tf.squeeze(model.predict(x_test)))

print(f"Evaluating the model_3: {model.evaluate(x_test,y_test)}")

mae          = tf.keras.metrics.MeanAbsoluteError()
mae.update_state(y_test, y_pred)

print(f'\n*************** MAE *************************\n')


print(f"MAE for the model: {mae.y_pred()}")



print(f'\n*************** SAVING AND LOADING MODEL AGAIN FROM A FILE ... *************************\n')

# !!!! SAVING MODEL !!!!
model.save("saved_models/my_first_model.h5")

loaded_model = tf.keras.models.load_model('./saved_models/my_first_model.h5')
print(loaded_model.summary())

print(f'\n*************** CHECKING LOADED MODEL WHETHER IT IS THE SAME WITH THE MODEL RIGHT BEFORE SAVED *************************\n')

loaded_model_pred = tf.constant(tf.squeeze(loaded_model.predict(x_test)))
print(loaded_model_pred == y_pred)
#OUTPUT : tf.Tensor([ True  True  True  True  True  True  True  True  True  True], shape=(10,), dtype=bool)



