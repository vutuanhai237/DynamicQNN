import numpy as np
import keras
import keras.layers as krl
import qiskit
import constant
import types
import random
from keras.utils import np_utils
from keras.datasets import mnist, fashion_mnist
def normalize_count(counts, n_qubits):
    for i in range(0, 2**n_qubits):
        x = (str(bin(i)[2:]))
        x = (n_qubits - len(x))*"0" + x
        if x not in counts:
            counts[x] = 0
            
    normalized_counts = np.array(list(dict(sorted(counts.items())).values()))
    return normalized_counts / np.linalg.norm(normalized_counts)

def measure(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """Measuring the quantu circuit which fully measurement gates
    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (np.ndarray): List of measured qubit
    Returns:
        - float: Frequency of 00.. cbit
    """
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
  
    counts = qiskit.execute(
        qc, backend = constant.backend,
        shots = constant.num_shots).result().get_counts()
    return counts

def quantum_model(image):
    n_image = image.shape[0]
    kernal_size = 2
    out = np.zeros((n_image // 2, n_image // 2, kernal_size * 2))
    for i in range(0, n_image, kernal_size):
        for j in range(0, n_image, kernal_size):  
            exp_values = quanvolutional([
                image[i, j],
                image[i, j + 1],
                image[i + 1, j],
                image[i + 1, j + 1]
            ])
            for c in range(4):
                out[i // 2, j // 2, c] = exp_values[c]
    return out

def classical_model():
    model = keras.models.Sequential()
    model.add(krl.Conv2D(1, (5, 5), activation='relu', input_shape=(28,28,1)))
    model.add(krl.MaxPooling2D(pool_size=(2,2)))
    model.add(krl.Conv2D(1, (5, 5), activation='relu'))
    model.add(krl.MaxPooling2D(pool_size=(2,2)))
    model.add(krl.Flatten())
    model.add(krl.Dense(1024, activation='relu'))
    model.add(krl.Dropout(0.4))
    model.add(krl.Dense(10, activation='softmax'))
    return model

def hybrid_model():
    model = keras.models.Sequential()
    # model.add(krl.MaxPooling2D(pool_size=(2,2)))
    # model.add(krl.Conv2D(1, (5, 5), activation='relu'))
    # model.add(krl.MaxPooling2D(pool_size=(2,2)))
    model.add(krl.Flatten())
    model.add(krl.Dense(1024, activation='relu'))
    model.add(krl.Dropout(0.4))
    model.add(krl.Dense(10, activation='softmax'))
    return model

def quanvolutional(vector):
    if np.sum(vector) == 0:
        return [0, 0, 0, 0]
    vector = np.squeeze(vector)
    vector = vector / np.linalg.norm(vector)
    n = int(np.log2(vector.shape[0]))
    qc = qiskit.QuantumCircuit(n, n)
    qc.initialize(vector, range(0, n))
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    for i in range(1, n):
        qc.cry(thetas[i], 0, i)
    counts = measure(qc, list(range(0, n)))
    normalized_count = normalize_count(counts, n)
    return normalized_count

def converter(data: np.ndarray, quanv: types.FunctionType):
    quantum_datas = []
    for quantum_data in data:
        quantum_datas.append(quanv(quantum_data))
    quantum_datas = np.array(quantum_datas)
    return quantum_datas

def load_mnist(n_train: int, n_val: int, n_test: int, quanv: types.FunctionType = quantum_model):
    """_summary_

    Args:
        n_train (int): number of train items
        n_val (int): number of validation items
        n_test (int): number of test items
        quanv (types.FunctionType, optional): _description_. Defaults to quantum_model.

    Returns:
        tuple: Splitted dataset
    """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Get k random item in whole MNIST has 60000 train / 10000 test
    random_itrain = random.sample(range(0, 60000), n_train + n_val)
    random_itest = random.sample(range(0, 10000), n_test)
    x_train1 = np.asarray([x_train[i] for i in random_itrain])
    y_train1 = np.asarray([y_train[i] for i in random_itrain])
    x_test = np.asarray([x_test[i] for i in random_itest])
    y_test = np.asarray([y_test[i] for i in random_itest])
    # Split train / val / test
    x_train, y_train = x_train1[:n_train,:], y_train1[:n_train]
    x_val, y_val = x_train1[n_train:n_train + n_val,:], y_train1[n_train:n_train + n_val]
    x_test, y_test = x_test[:n_test,:], y_test[:n_test]
    # Reshape for fitting with Keras input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # One-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # Create post-processing data (the data that has gone through the quanvolutional layer)
    xq_train = converter(x_train, quanv)
    xq_val = converter(x_val, quanv)
    xq_test = converter(x_test, quanv)

    return x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test

def load_mnist_fashion(n_train: int, n_val: int, n_test: int, quanv: types.FunctionType = quantum_model):
    """_summary_

    Args:
        n_train (int): number of train items
        n_val (int): number of validation items
        n_test (int): number of test items
        quanv (types.FunctionType, optional): _description_. Defaults to quantum_model.

    Returns:
        tuple: Splitted dataset
    """
    
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Get k random item in whole MNIST has 60000 train / 10000 test
    random_itrain = random.sample(range(0, 60000), n_train + n_val)
    random_itest = random.sample(range(0, 10000), n_test)
    x_train1 = np.asarray([x_train[i] for i in random_itrain])
    y_train1 = np.asarray([y_train[i] for i in random_itrain])
    x_test = np.asarray([x_test[i] for i in random_itest])
    y_test = np.asarray([y_test[i] for i in random_itest])
    # Split train / val / test
    x_train, y_train = x_train1[:n_train,:], y_train1[:n_train]
    x_val, y_val = x_train1[n_train:n_train + n_val,:], y_train1[n_train:n_train + n_val]
    x_test, y_test = x_test[:n_test,:], y_test[:n_test]
    # Reshape for fitting with Keras input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # One-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # Create post-processing data (the data that has gone through the quanvolutional layer)
    xq_train = converter(x_train, quanv)
    xq_val = converter(x_val, quanv)
    xq_test = converter(x_test, quanv)

    return x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test