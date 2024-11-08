import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

print(f'\n*************** MAKE AGAIN PREDICTION ***********************\n')


x = tf.range(-100,100,4)
y = x +10

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(1,activation=None),
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=Adam(learning_rate=0.001),
              metrics=['mae'])

model.fit(x,y,epochs=50)

y_predict = model.predict(np.array([100]))
print(y_predict)

print(f'\n************** 3 SETS - TRAINING, VALIDATION AND TEST SETS **************************\n')

# 3 sets !!!
# Training Set : 70-80% of the total data available
# Validation Set : 10-15% of the data available
# Test Set : 10-15% of the total data  available

x_train = x[:40] # first 40 elements 80% of the total data
y_train = y[:40] # first 40 elements 80% of the total data
print("Training Set X:")
print(x_train)
print("Training Set Y:")
print(y_train)

x_test = x[40:] # last 10 elements 20% of the total data
y_test = y[40:] # last 10 elements 20% of the total data
print("Test Set X:")
print(x_test)
print("Test Set Y:")
print(y_test)
"""
Training Set X:
tf.Tensor(
[-100  -96  -92  -88  -84  -80  -76  -72  -68  -64  -60  -56  -52  -48
  -44  -40  -36  -32  -28  -24  -20  -16  -12   -8   -4    0    4    8
   12   16   20   24   28   32   36   40   44   48   52   56], shape=(40,), dtype=int32)
   
Training Set Y:
tf.Tensor(
[-90 -86 -82 -78 -74 -70 -66 -62 -58 -54 -50 -46 -42 -38 -34 -30 -26 -22
 -18 -14 -10  -6  -2   2   6  10  14  18  22  26  30  34  38  42  46  50
  54  58  62  66], shape=(40,), dtype=int32)
  
Test Set X:
tf.Tensor([60 64 68 72 76 80 84 88 92 96], shape=(10,), dtype=int32)

Test Set Y:
tf.Tensor([ 70  74  78  82  86  90  94  98 102 106], shape=(10,), dtype=int32)
"""


# Visualizing training data
plt.scatter(x_train,y_train,c='b',label='Train')
# Visualizing test data
plt.scatter(x_test,y_test,c='g',label='Test')

plt.legend()
#plt.show()

print(f'\n*************** MODEL SUMMARY SEQUENTIAL_1 *************************\n')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1,input_shape=[1]),
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.summary()
"""

Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_5 (Dense)                 │ (None, 1)              │             2 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2 (8.00 B)
 Trainable params: 2 (8.00 B)
 Non-trainable params: 0 (0.00 B)
 
 
 WARNING: !!! UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs) !!!
"""

# SO WE DO ->
print(f'\n*************** MODEL SUMMARY SEQUENTIAL_2 *************************\n')

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(10,name="first layer"), #!!! NOTE: Dense means that fully connected. (All the neurones are connected to all the neurones in the next layer.)!!!
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.summary()

"""
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ first layer (Dense)             │ (None, 10)             │            20 │ Param =    10 neurones in the layer
└─────────────────────────────────┴────────────────────────┴───────────────┘             *
 Total params: 20 (80.00 B)                                                              1 input node in x_train like -> [1,2,3,4], not like [[1,2],[2,3]]
 Trainable params: 20 (80.00 B)                                                          = Weights (so we've got 10 )
 Non-trainable params: 0 (0.00 B)                                                            +
                                                                                            10 biases (neurones in the layer)         
                                                                                            = 20 count of param
"""
print(f'\n****************************************\n')

# SO WE DO ->
print(f'\n*************** MODEL SUMMARY SEQUENTIAL_3 *************************\n')

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(10,name="first layer"),
    tf.keras.layers.Dense(10,name="second layer"),
    tf.keras.layers.Dense(10,name="third layer")
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.summary()

"""
Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩-> the first one connected to input layer which shape is 2 so we have 
│ first layer (Dense)             │ (None, 10)             │            30 │->  10 * 2 + 10 = 30
├─────────────────────────────────┼────────────────────────┼───────────────┤-> But the next layer  is connected to the 2nd one so we have and each neurone connect each other because we are using dense
│ second layer (Dense)            │ (None, 10)             │           110 │->  10 * 1 (Weights) + 100 (Biases) = 110
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ third layer (Dense)             │ (None, 10)             │           110 │-> 10 * 1 (Weights) + 100 (Biases) = 110
└─────────────────────────────────┴────────────────────────┴───────────────┘TOTAL PARAM : 110+110+30 = 250params
 Total params: 250 (1000.00 B)
 Trainable params: 250 (1000.00 B)
 Non-trainable params: 0 (0.00 B)
"""

print(f'\n*************** MODEL SUMMARY SEQUENTIAL_4 *************************\n')


model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100,name="hidden_layer"),
    tf.keras.layers.Dense(1,name="output_layer"),
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.summary()

"""

Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ hidden_layer (Dense)            │ (None, 10)             │            20 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_layer (Dense)            │ (None, 1)              │            11 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 31 (124.00 B)
 Trainable params: 31 (124.00 B)
 Non-trainable params: 0 (0.00 B)
 
 """
print(f"\n Fitting the model \n")
model.fit(x_train,y_train,epochs=50,verbose=1)


print(f'\n*****************  PLOT MODEL ->(SAVED IN A FILE NAMED "model.png")   ***********************\n')
tf.keras.utils.plot_model(model,show_shapes=True) # uploaded into model.png file


print(f'\n*************** VISUALIZING PREDICTIONS *************************\n')

y_pred = model.predict(x_test)
print(f"y_test: \n {y_test}")
print(f"y_pred: \n {y_pred}") # -> data type is np.ndarray not a tensor

def plot_predictions(
        train_data=x_train,
        train_label=y_train,
        test_data=x_test,
        test_label=y_test,
        predictions=y_pred
):
    plt.figure(figsize=(10,10))
    plt.scatter(train_data,train_label,c="b",label="Training Data")
    plt.scatter(test_data,test_label,c="g",label="Testing Data")
    plt.scatter(test_data,predictions,c="r",label="Predictions")
    plt.legend()
    plt.show()

#plot_predictions()

print(f'\n*************** EVALUATION MATRIX *************************\n')

print(model.evaluate(x_test,y_test))
"""
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - loss: 16.1746 - mae: 16.1746
[16.17461585998535, 16.17461585998535]
"""

print("MAE(Mean Absolute Error): ")
y_pred = tf.squeeze(tf.constant(y_pred))
y_test = tf.cast(y_test,dtype=tf.float32)

mae = keras.metrics.MeanAbsoluteError()
mae.update_state(y_test,y_pred)
print(mae.y_pred())
"""
MAE(Mean Absolute Error): 
tf.Tensor(10.887206, shape=(), dtype=float32)
"""

print("MAE(Mean Squared Error): ")
mse = keras.metrics.MeanSquaredError()
mse.update_state(y_test,y_pred)
print(mse.y_pred())




