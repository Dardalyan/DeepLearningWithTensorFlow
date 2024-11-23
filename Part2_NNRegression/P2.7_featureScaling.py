import keras
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import tensorflow as tf



insurance = pd.read_csv('insurance.csv')

# Preprocessing data (normalization and standardization)
# In terms of scaling values, neural networks tend to prefer normalization
# If you re not sure on which to choose, then try both


ct= make_column_transformer(
    (MinMaxScaler(),['age','bmi','children']), # turn all values in these columns between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)

#Create x & y
x = insurance.drop('charges',axis=1)
y = insurance['charges']

#Build our train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
ct.fit(x_train)

# Transform training and test date with data with normalization (MinScaler) and OneHotEncoder
X_Train_Normal = ct.transform(x_train)
X_Test_Normal = ct.transform(x_test)



#What does our data look like before ?

"""
age                19
sex            female
bmi              27.9
children            0
smoker            yes
region      southwest
Name: 0, dtype: object
"""

#What does our data look like now ?
print(X_Train_Normal)
print(x_train.shape,X_Train_Normal.shape) # x_train :(1070, 6) , X_Train_Normal: (1070, 11)
"""
[[0.60869565 0.10734463 0.4        ... 1.         0.         0.        ]
 [0.63043478 0.22491256 0.         ... 0.         0.         0.        ]
 [0.73913043 0.23944041 0.         ... 0.         1.         0.        ]
 ...
 [0.86956522 0.24791499 0.         ... 0.         0.         0.        ]
 [0.41304348 0.85122411 0.4        ... 0.         0.         1.        ]
 [0.80434783 0.37503363 0.         ... 0.         0.         1.        ]]
"""

# Now our data has been normalized and one hot encoded. Now let's build a model

model = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = Adam(learning_rate=0.1),
    metrics = [keras.metrics.MeanAbsoluteError()]
)

model.fit(X_Train_Normal, y_train,epochs=200)

model.evaluate(X_Test_Normal, y_test)











