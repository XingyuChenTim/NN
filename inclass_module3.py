import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

####################################################################################
# Example 1
# Oranges vs. Grapefruit: diameter, weight, and color data
####################################################################################

# This is a 2-label dataset (0 and 1)
url ="https://drive.google.com/file/d/1Pf7wp0PhskFbbNZaAkNgDRbBOcwu47PV/view?usp=sharing"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
data = pd.read_csv(url)
print("Check first 10 rows of dataset\n", data.head(10))

# split target and features
x = data.iloc[:, 1:]
y = pd.DataFrame([1 if each == "orange" else 0 for each in data['name']], columns=["target"])

# Feature Scaling, normalize data
sc = StandardScaler()
x = sc.fit_transform(x)
print('The normalized features: ', x)

# Then lets create x_train, y_train, x_test, y_test arrays
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# MODEL 1 - for the two-label dataset
model1 = keras.Sequential([
    keras.layers.Dense(3, input_shape=(5,), activation="relu"),  # first hidden layer
    keras.layers.Dense(2, activation="relu"),  # This is the second hidden layer with two units
    keras.layers.Dense(1, activation="sigmoid")])  # we are using 0 or 1 here, so we have output size of 1.

model1.compile(
    # optimizer='SGD',
    optimizer="Adam",
    loss=keras.losses.MeanSquaredError(),
    # loss="categorical_crossentropy",
    metrics=["accuracy"])

model1.fit(x_train, y_train, epochs=50)

Validation_Loss, Validation_Accuracy = model1.evaluate(x_test, y_test)

print('The validation loss and accuracy: ', Validation_Loss, Validation_Accuracy)

# 93.1% sgd + meansquareerror
# 49.3% sgd + categorical_crossentropy
# 50.1% adam + categorical_crossentropy
# 94.4% adam + meansquareerror

# 3 units in first layer 2 units in the second layer output for sigmoid 94.4
# 3 units in first layer then output for sigmoid 93.3%

# 3 units in first layer 2 units in the second layer output for sigmoid 94.4
# 3 units in first layer 3 units in the second layer output for sigmoid 93.4
# 2 units in first layer 3 units in the second layer output for sigmoid 93.1
# 2 units in first layer 2 units in the second layer output for sigmoid 93.3

####################################################################################
# Example 2
# academic success w/ one-hot encoding, labels will be 0, 1, 2 (dropout, graduate, enrolled)
####################################################################################

# This is a 3-label dataset (0, 1, 2)
url = "https://drive.google.com/file/d/1qFziUtrehFwy4vHa67Q3taJs_zQ6EMuQ/view?usp=sharing"
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
data = pd.read_csv(url)

print("Check first 10 rows of dataset\n", data.head(10))

# split target and features
x = data.iloc[:, :-1]
le = LabelEncoder().fit(data.iloc[:, -1])
y = le.transform(data.iloc[:, -1])

# Feature Scaling
sc = StandardScaler()
x = sc.fit_transform(x)

# one hot labels
temp = y  # the training set labels
one_hot_labels = np.zeros((len(y), 3))
for i in range(len(y)):
    one_hot_labels[i, temp[i] - 1] = 1
y = one_hot_labels
print("The one-hot for training labels\n", y)

# Then lets create x_train, y_train, x_test, y_test arrays
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# MODEL 2 - for the three-label dataset
model2 = keras.Sequential([
    keras.layers.Dense(18, input_shape=(36,), activation="relu"),  # Hidden layer 1
    keras.layers.Dense(12, activation='sigmoid'),  # Hidden layer 2
    keras.layers.Dense(8, activation='sigmoid'),  # Hidden layer 3
    keras.layers.Dense(6, activation="relu"),  # Hidden layer 4
    keras.layers.Dense(3, activation='softmax')  # output layer
])
model2.compile(
    # optimizer='SGD',
    optimizer="Adam",
    # loss=keras.losses.MeanSquaredError(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model2.fit(x_train, y_train, epochs=100)

Validation_Loss, Validation_Accuracy = model2.evaluate(x_test, y_test)

print('The validation loss and accuracy: ', Validation_Loss, Validation_Accuracy)

# 5 layer, relu, relu, sigmoid, sigmoid, softmax; 10, 5, 3, 3, 3  73.2%
# 5 layer, relu, relu, sigmoid, sigmoid, softmax; 18, 12, 8, 6, 3  74.5%
# 4 layer, relu, relu, sigmoid, softmax; 18, 12, 6, 3  73.5%
# 4 layer, relu, relu, sigmoid, softmax; 10, 5, 3, 3  73.3%
# 5 layer, sigmoid, sigmoid, relu, relu, softmax; 18, 12, 8, 6, 3  75.2%

# 59.4% sgd + meansquareerror
# 74.9% sgd + categorical_crossentropy
# 75.7% adam + categorical_crossentropy
# 75.2% adam + meansquareerror
