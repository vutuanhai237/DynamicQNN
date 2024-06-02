
import numpy as np
from keras import utils
from keras.datasets import fashion_mnist
from typing import List, Tuple, Union
from qiskit.primitives import Sampler
import types, random, math, os, qiskit
import keras
import keras.layers as krl
import json

quanv_num_filter = 16 # should be 2^n
quanv_size_filter = 4 # should be 2, 4, 

def processing(invocation_input):
    num_train = invocation_input['num_train']
    num_test = invocation_input['num_test']
    num_val = invocation_input['num_val']
    x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = load_mnist_fashion(
        num_train, num_val, num_test, quanvolutional, True)

    hmodel = hybrid_model()
    hmodel.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    global H2
    H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
        batch_size=1, epochs=100, verbose=0)

    _, test_accuracy = hmodel.evaluate(xq_test, y_test)

    with open("result.json", "w") as outfile: 
        json.dump(H2.history, outfile)
    return qiskit.QuantumCircuit(4)


def post_processing(job_result):
    return H2.history


def hybrid_model():
    model = keras.models.Sequential()
    model.add(krl.Flatten())
    model.add(krl.Dense(1024, activation='relu'))
    model.add(krl.Dropout(0.4))
    model.add(krl.Dense(10, activation='softmax'))
    return model

def get_quanv_num_filter(kernel_size):
    if quanv_num_filter == -1:
        return kernel_size**2
    else:
        return quanv_num_filter
    
def normalize_count(counts, n_qubits):
    for i in range(0, 2**n_qubits):
        # x = (str(bin(i)[2:]))
        # x = (n_qubits - len(x))*"0" + x
        if i not in counts:
            counts[i] = 0
    normalized_counts = np.array(list(dict(sorted(counts.items())).values()))
    return normalized_counts / np.linalg.norm(normalized_counts)


def measure(qc: qiskit.QuantumCircuit):
    """Measuring the quantu circuit which fully measurement gates
    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (np.ndarray): List of measured qubit
    Returns:
        - float: Frequency of 00.. cbit
    """
    qc.measure_all()
    sampler = Sampler()
    result = sampler.run(qc, shots = 10000).result().quasi_dists[0]
    return result


def add_padding(matrix: np.ndarray,
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 
    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix
    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    matrix = np.squeeze(matrix)
    n, m = matrix.shape
    r, c = padding
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix
    return padded_matrix


def connector(vector, filter: types.FunctionType):
    """If sub-image has the size
    2 x 2 => require 2 qubits
    3 x 3 => require 4 qubits
    4 x 4 => require 4 qubits
    5 x 5 => require 5 qubits
    6 x 6 => require 6 qubits
    7 x 7 => require 6 qubits
    8 x 8 => require 6 qubits
    Args:
        vector (np.ndarray): quantum state
        filter (types.FunctionType): quantum circuit

    Returns:
        np.ndarray: probability vector
    """
    
    n = math.ceil(np.log2(vector.shape[0]))
    if vector.shape[0] < 2**n:
        vector = np.concatenate([vector, np.array([0]*(2**n-vector.shape[0]))])
    qc = qiskit.QuantumCircuit(n)
    qc.initialize(vector, range(0, n))
    qc = filter(qc)
    counts = measure(qc)
    normalized_count = normalize_count(counts, n)
    return normalized_count



def quanv(image, filter: types.FunctionType):
    n_image = image.shape[0]
    kernel_size = quanv_size_filter
    if n_image % kernel_size != 0:
        padding_size = kernel_size - n_image % kernel_size
        image = add_padding(image, (int(np.ceil(padding_size / 2)), int(np.ceil(padding_size / 2))))
        n_image = image.shape[0]
    num_deep = get_quanv_num_filter(kernel_size)
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
            #print(sub_image.flatten())
            for c in range(out.shape[2]):
                try:
                    out[i // kernel_size, j // kernel_size, c] = exp_valuess[c]
                except:
                    pass
    return out



def converter(data: np.ndarray, filter: types.FunctionType):
    quantum_datas = []
    for quantum_data in data:
        quantum_datas.append(quanv(quantum_data, filter))
    quantum_datas = np.array(quantum_datas)
    return quantum_datas

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
    y_train = utils.to_categorical(y_train, 10)
    y_val = utils.to_categorical(y_val, 10)
    y_test = utils.to_categorical(y_test, 10)
    # Create post-processing data (the data that has gone through the quanvolutional layer)
    if is_take_xq:
        xq_train = converter(x_train, filter)
        xq_val = converter(x_val, filter)
        xq_test = converter(x_test, filter)

        return x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test

def quanvolutional(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    for i in range(1, n):
        qc.cry(thetas[i], 0, i)
    return qc

# keras
qiskit
tensorflow