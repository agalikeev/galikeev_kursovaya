import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import  load_model
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_test_cat = keras.utils.to_categorical(y_test, 10)

x_test1 = np.expand_dims(x_test, axis=3)

print("LeNet-5:")
model_lenet5 = load_model('test_lenet.h5')
model_lenet5.evaluate(x_test1, y_test_cat)



print("LMST:")
model_lstm = load_model('test_lmst.h5')
model_lstm.evaluate(x_test,y_test)
