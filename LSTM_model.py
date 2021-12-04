import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Activation
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential( [
    LSTM(units=128,input_shape=(28, 28)),
    Dense(units=10)
    ])


print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)
model.evaluate(x_test, y_test)

model.save('test_lmst.h5')

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
