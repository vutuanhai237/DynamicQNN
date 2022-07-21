
import qiskit
import numpy as np, matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras
from keras.datasets import mnist
import classical_part
# %load_ext autoreload
# %autoreload 2

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_val, y_val = x_train[500:600,:], y_train[500:600]
x_train, y_train = x_train[:500,:], y_train[:500]

plt.imshow(x_train[0])

# 3. Reshape lại dữ liệu cho đúng kích thước mà keras yêu cầu
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

plt.imshow(x_train[0])

# # 3. Reshape lại dữ liệu cho đúng kích thước mà keras yêu cầu
# xq_train = []
# for x_train_item in x_train:
#     xq_train.append(classical_part.quantum_model(x_train_item))
# xq_test = []
# for x_test_item in x_train:
#     xq_test.append(classical_part.quantum_model(x_test_item))
xq_train = []
for x_train_item in x_train:
    xq_train.append(classical_part.quantum_model(x_train_item))
xq_train = np.array(xq_train)

xq_val = []
for x_val_item in x_val:
    xq_val.append(classical_part.quantum_model(x_val_item))
xq_val = np.array(xq_val)

print(xq_val.shape)

print(x_val.shape)

# 4. One hot encoding label (Y)
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('Dữ liệu y ban đầu ', y_train[0])
print('Dữ liệu y sau one-hot encoding ',y_train[0])

plt.imshow(x_val[0])

cmodel = classical_part.classical_model()
hmodel = classical_part.hybrid_model()

cmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H1 = cmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=16, epochs=20, verbose=1)

hmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
          batch_size=16, epochs=20, verbose=1)

# results = model.evaluate(X_test, Y_test, batch_size=128)
# print("test loss, test acc:", results)

# # 8. Vẽ đồ thị loss, accuracy của traning set và validation set
# fig = plt.figure()
# numOfEpoch = 10
# plt.plot(np.arange(0, numOfEpoch), H1.history['loss'], label='training loss')
# plt.plot(np.arange(0, numOfEpoch), H1.history['val_loss'], label='validation loss')
# plt.plot(np.arange(0, numOfEpoch), H1.history['accuracy'], label='accuracy')
# plt.plot(np.arange(0, numOfEpoch), H1.history['val_accuracy'], label='validation accuracy')
# plt.title('Accuracy and Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss|Accuracy')
# plt.legend()

# # 8. Vẽ đồ thị loss, accuracy của traning set và validation set
# fig = plt.figure()
# numOfEpoch = 10
# plt.plot(np.arange(0, numOfEpoch), H2.history['loss'], label='training loss')
# plt.plot(np.arange(0, numOfEpoch), H2.history['val_loss'], label='validation loss')
# plt.plot(np.arange(0, numOfEpoch), H2.history['accuracy'], label='accuracy')
# plt.plot(np.arange(0, numOfEpoch), H2.history['val_accuracy'], label='validation accuracy')
# plt.title('Accuracy and Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss|Accuracy')
# plt.legend()
np.savetxt('h1history.txt', H1.history['loss'])
np.savetxt('h2history.txt', H2.history['loss'])