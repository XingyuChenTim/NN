# Use the MNIST dataset, and TF/Keras to build a NN with two hidden layers. Use Softmax/Cat Crossentropy for the
# output activation and Loss function. For the middle layers, use ReLU for one and Sigmoid for the other. The goal
# for this part is to simply assure that the functions and methods work for you. You can locate the solution online
# on the Tensorflow site.


import tensorflow as tf

mnist = tf.keras.datasets.mnist
Data_ = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = Data_.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
#
# Epoch 1/5
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.5756 - accuracy: 0.8007
# Epoch 2/5
# 1875/1875 [==============================] - 3s 2ms/step - loss: 0.3982 - accuracy: 0.8566
# Epoch 3/5
# 1875/1875 [==============================] - 3s 2ms/step - loss: 0.3600 - accuracy: 0.8689
# Epoch 4/5
# 1875/1875 [==============================] - 3s 2ms/step - loss: 0.3339 - accuracy: 0.8788
# Epoch 5/5
# 1875/1875 [==============================] - 3s 2ms/step - loss: 0.3131 - accuracy: 0.8844
# 313/313 [==============================] - 1s 1ms/step - loss: 0.3450 - accuracy: 0.8756
