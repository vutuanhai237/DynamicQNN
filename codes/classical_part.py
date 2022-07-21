import numpy as np
import keras
import keras.layers as krl
import qiskit
import constant

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
    model.add(krl.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28,28,1)))
    model.add(krl.Conv2D(32, (3, 3), activation='sigmoid'))
    model.add(krl.MaxPooling2D(pool_size=(2,2)))
    model.add(krl.Dropout(0.1))
    model.add(krl.Flatten())
    model.add(krl.Dense(128, activation='sigmoid'))
    model.add(krl.Dense(10, activation='softmax'))
    return model

def hybrid_model():
    model = keras.models.Sequential()
    model.add(krl.MaxPooling2D(pool_size=(2,2)))
    model.add(krl.Dropout(0.1))
    model.add(krl.Flatten())
    model.add(krl.Dense(128, activation='sigmoid'))
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
