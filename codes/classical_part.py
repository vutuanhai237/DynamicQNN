import numpy as np
import keras
import keras.layers as krl
import qiskit
import constant
import types
import random, math
from keras.utils import np_utils
from keras.datasets import mnist, fashion_mnist
import numpy as np
import entangled_circuit
from typing import List, Tuple, Union


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
        qc, backend=constant.backend,
        shots=constant.num_shots).result().get_counts()
    return counts


def add_padding(matrix: np.ndarray,
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 
    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix
    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix
    return padded_matrix


def connector(vector, filter):
    n = int(np.log2(vector.shape[0]))
    qc = qiskit.QuantumCircuit(n, n)
    qc.initialize(vector, range(0, n))
    qc = filter(qc)
    counts = measure(qc, list(range(0, n)))
    normalized_count = normalize_count(counts, n)
    return normalized_count


def quanv(image, filter: types.FunctionType):
    n_image = image.shape[0]
    kernel_size = 4
    if n_image % 4 != 0:
        image = add_padding(image, ((n_image % 4) // 2, (n_image % 4) // 2))
        n_image = image.shape[0]
    num_deep = constant.get_num_quanv_filter(kernel_size)
    out = np.zeros((n_image // kernel_size, n_image // kernel_size,
                   num_deep))
    for i in range(0, n_image, kernel_size):
        for j in range(0, n_image, kernel_size):
            sub_image = image[i:i + kernel_size, j:j + kernel_size]
            # Turn normal image to quantum state
            if np.all(sub_image == 0):
                sub_image[0] = 1
            sub_image = np.squeeze(sub_image)
            sub_image = sub_image / np.linalg.norm(sub_image)
            # Convert quantum state to quantum probabilities
            # If required deep size > default, add more quanv circuit
            num_filter = math.ceil(num_deep / kernel_size**2)
            exp_valuess = np.asarray([])
            for i in range(num_filter):
                exp_values = connector(sub_image.flatten(), filter)
                exp_valuess = np.concatenate((exp_valuess, exp_values), axis=None)
            
            for c in range(out.shape[2]):
                out[i // kernel_size, j // kernel_size, c] = exp_valuess[c]
    return out


def classical_model():
    model = keras.models.Sequential()
    model.add(krl.Conv2D(constant.num_conv_filter, (4, 4), activation='relu', input_shape=(28, 28, 1)))
    model.add(krl.MaxPooling2D(pool_size=(2, 2)))
    model.add(krl.Conv2D(constant.num_conv_filter, (4, 4), activation='relu'))
    model.add(krl.MaxPooling2D(pool_size=(2, 2)))
    model.add(krl.Flatten())
    model.add(krl.Dense(1024, activation='relu'))
    model.add(krl.Dropout(0.4))
    model.add(krl.Dense(10, activation='softmax'))
    return model


def hybrid_model():
    model = keras.models.Sequential()
    model.add(krl.Flatten())
    model.add(krl.Dense(1024, activation='relu'))
    model.add(krl.Dropout(0.4))
    model.add(krl.Dense(10, activation='softmax'))
    return model


def converter(data: np.ndarray, filter: types.FunctionType):
    quantum_datas = []
    for quantum_data in data:
        quantum_datas.append(quanv(quantum_data, filter))
    quantum_datas = np.array(quantum_datas)
    return quantum_datas


def load_mnist(n_train: int, n_val: int, n_test: int, filter: types.FunctionType, is_take_xq=False):
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
    x_train, y_train = x_train1[:n_train, :], y_train1[:n_train]
    x_val, y_val = x_train1[n_train:n_train +
                            n_val, :], y_train1[n_train:n_train + n_val]
    x_test, y_test = x_test[:n_test, :], y_test[:n_test]
    # Reshape for fitting with Keras input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # One-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    if is_take_xq:
        # Create post-processing data (the data that has gone through the quanvolutional layer)
        xq_train = converter(x_train, filter)
        xq_val = converter(x_val, filter)
        xq_test = converter(x_test, filter)
        return x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test
    else:

        return x_train, y_train, x_val, y_val, x_test, y_test


def load_mnist_fashion(n_train: int, n_val: int, n_test: int, filter: types.FunctionType, is_take_xq=False):
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
    x_train, y_train = x_train1[:n_train, :], y_train1[:n_train]
    x_val, y_val = x_train1[n_train:n_train +
                            n_val, :], y_train1[n_train:n_train + n_val]
    x_test, y_test = x_test[:n_test, :], y_test[:n_test]
    # Reshape for fitting with Keras input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # One-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # Create post-processing data (the data that has gone through the quanvolutional layer)
    if is_take_xq:
        xq_train = converter(x_train, filter)
        xq_val = converter(x_val, filter)
        xq_test = converter(x_test, filter)

        return x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test
