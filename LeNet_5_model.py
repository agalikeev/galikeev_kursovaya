import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape)

model = keras.Sequential([
    Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', input_shape=(14, 14, 6)),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dense(units=120, activation='sigmoid'),
    Dense(units=84, activation='sigmoid'),
    Dense(units=10, activation='softmax'),
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat, batch_size=64, epochs=10, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

model.save('test_lenet.h5')

plt.figure()

plt.subplot(211)
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.grid(True)

plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.show()