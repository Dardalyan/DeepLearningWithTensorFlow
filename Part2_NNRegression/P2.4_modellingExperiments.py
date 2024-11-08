import tensorflow as tf
import pandas as pd

x = tf.range(1,100,2,dtype=tf.float32)
y = x * 2


x_train = x[:40]
x_test = x[40:]


y_train = x[:40]
y_test = x[40:]

print(f'\n*************** ORIGINAL MODEL *************************\n')


original_model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,name="first_layer"),
    tf.keras.layers.Dense(1)
])

original_model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

original_model.fit(x_train,y_train,epochs=50,verbose=0) # 1 hidden unit, 100 neurone per unit, epoch=50

y_pred_original = tf.constant( tf.squeeze(original_model.predict(x_test)))

print(f"Evaluating the original model: {original_model.evaluate(x_test,y_test)}")

mae_original_model  = tf.keras.metrics.MeanAbsoluteError()
mae_original_model.update_state(y_test, y_pred_original)

print(f"MAE for the original model: {mae_original_model.y_pred()}")

print(f'\n*************** MODEL_1 *************************\n')


model_1 = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,name="first_layer"),
    tf.keras.layers.Dense(100,name="second_layer"),
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model_1.fit(x_train,y_train,epochs=50,verbose=0)        # 2 hidden unit, 100 neurone per unit  epoch=50

y_pred_model1   = tf.constant( tf.squeeze(model_1.predict(x_test)))

print(f"Evaluating the model_1: {model_1.evaluate(x_test,y_test)}")

mae_model1          = tf.keras.metrics.MeanAbsoluteError()
mae_model1.update_state(y_test, y_pred_model1)

print(f"MAE for the model_1: {mae_model1.y_pred()}")

print(f'\n*************** MODEL_2 *************************\n')

model_2 = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(200,name="first_layer"),
    tf.keras.layers.Dense(200,name="second_layer"),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model_2.fit(x_train,y_train,epochs=50,verbose=0)        # 2 hidden unit, 200 neurone per unit  epoch=50

y_pred_model2   = tf.constant( tf.squeeze(model_2.predict(x_test)))

print(f"Evaluating the model_2: {model_2.evaluate(x_test,y_test)}")

mae_model2          = tf.keras.metrics.MeanAbsoluteError()
mae_model2.update_state(y_test, y_pred_model2)

print(f"MAE for the model_2: {mae_model2.y_pred()}")

print(f'\n*************** MODEL_3 *************************\n')

model_3 = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,name="first_layer"),
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model_3.fit(x_train,y_train,epochs=200,verbose=0)       # 1 hidden unit, 100 neurone per unit  epoch = 200

y_pred_model3   = tf.constant( tf.squeeze(model_3.predict(x_test)))

print(f"Evaluating the model_3: {model_3.evaluate(x_test,y_test)}")

mae_model3          = tf.keras.metrics.MeanAbsoluteError()
mae_model3.update_state(y_test, y_pred_model3)

print(f"MAE for the model_3: {mae_model3.y_pred()}")

#Comparing
model_results = [
    ["original_model", mae_original_model.y_pred()],
    ["mae_model1", mae_model1.y_pred(), ],
    ["mae_model2", mae_model2.y_pred()],
    ["mae_model3", mae_model3.y_pred()]
]

results = pd.DataFrame(model_results,columns=['model','mae'])
print(results)

