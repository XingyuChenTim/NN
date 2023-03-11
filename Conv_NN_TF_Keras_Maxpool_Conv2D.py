# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:45:05 2022
For this part of the Assignment, your goal is to dig in a bit deeper. Therefore, you will take several steps.

Step 1: Use THIS CODELinks to an external site. to get a basic multilayer 2D convolutional NN working for a color
dataset. Everything you need is in the code.

Step 2: Once you get the code working, create a PowerPoint Tutorial that includes all the code and explanations
for what each line of code is doing. Include why sizes and shapes are as noted in the code. Include visual examples
as often as possible. Pretend that it is your job to EXPLAIN and ILLUSTRATE each step to a person who is new to the
topic.

For example - each slide in your PowerPoint Set might have 1 - 5 (with 5 as the max) lines of code that work together
or individually to perform a task. You will include and explain the code. When possible, offer insightful
illustrations. Include OUTPUT on slides as you go. Really assure that a viewer can SEE and understand what is going on.

There are 1000s of ways to do this :) Be creative and clear.
@author: prof a
"""
# Image Processing Python
# https://note.nkmk.me/en/python-numpy-image-processing/

import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

print(type(train_images))
print(train_images.shape)  # 50000 rows, 32 by 32, depth 3
plt.imshow(train_images[2])
plt.show()

print(train_images[0, :, :, 0])
print(train_images[0, :, :, 0].shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = tf.keras.models.Sequential()
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
