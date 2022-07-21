import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras
from keras.datasets import mnist
import classical_part

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_val, y_val = x_train[600:800,:], y_train[600:800]
x_train, y_train = x_train[:600,:], y_train[:600]
x_test, y_test = x_test[:200,:], y_test[:200]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

xq_train = []
for x_train_item in x_train:
    xq_train.append(classical_part.quantum_model(x_train_item))
xq_train = np.array(xq_train)

xq_val = []
for x_val_item in x_val:
    xq_val.append(classical_part.quantum_model(x_val_item))
xq_val = np.array(xq_val)

xq_test = []
for x_test_item in x_test:
    xq_test.append(classical_part.quantum_model(x_test_item))

y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)

cmodel = classical_part.classical_model()
hmodel = classical_part.hybrid_model()

cmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H1 = cmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=16, epochs=30, verbose=1)
hmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
          batch_size=1, epochs=10, verbose=1)
print(x_test.shape)
print(xq_test.shape)
print(y_test.shape)
print("Classical_model: ", cmodel.evaluate(x_test, y_test))
print("Hybrid_model: ", hmodel.evaluate(np.array(xq_test), y_test))
np.savetxt('h1history.txt', H1.history['loss'])
np.savetxt('h2history.txt', H2.history['loss'])