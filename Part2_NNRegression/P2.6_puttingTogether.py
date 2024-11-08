import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


insurance = pd.read_csv('insurance.csv')

insurance_dummy = pd.get_dummies(insurance,dtype=int)
#print(insurance_dummy.head()) # head() returns the first n rows default value is 5 .

#Create X and Y values (features and labels)
x = insurance_dummy.drop("charges",axis=1)
y = insurance_dummy['charges']

#Create training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train,y_train)


#Build a neural network (sort of like model ex. from prev file)

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = Adam(learning_rate=0.1),
    metrics = ['mae']
)

model.fit(x_train, y_train,epochs=200)




