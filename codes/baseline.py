import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras
from keras.datasets import mnist
import classical_part
from keras.utils import np_utils

x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_mnist(3600, 1200, 1200)
cmodel = classical_part.classical_model()
hmodel = classical_part.hybrid_model()

cmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H1 = cmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=4, epochs=30, verbose=0)
hmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
          batch_size=4, epochs=30, verbose=0)

H3 = hmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=4, epochs=30, verbose=0)

print("H1 model: ", cmodel.evaluate(x_test, y_test))
print("H2 model: ", hmodel.evaluate(np.array(xq_test), y_test))
print("H3 model: ", hmodel.evaluate(np.array(x_test), y_test))
np.savetxt('h1history.txt', H1.history['loss'])
np.savetxt('h2history.txt', H2.history['loss'])
np.savetxt('h3history.txt', H3.history['loss'])
