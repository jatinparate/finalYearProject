from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class_names = ['jatin', 'not jatin']

train_images = []
train_labels = []
for user in os.listdir('images'):
    for item in os.listdir('images/' + user):
        train_images.append(cv2.resize(
            plt.imread('images/' + user + '/' + item),
            (180, 180)
        ))
        train_labels.append(class_names.index(user))

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images = train_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(180,180)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
