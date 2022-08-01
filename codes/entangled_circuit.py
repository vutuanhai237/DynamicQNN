# https://arxiv.org/pdf/1905.10876.pdf
import qiskit, classical_part
import numpy as np
def quanvolutional1(vector):
    if np.sum(vector) == 0:
        return np.ones(len(vector))
    vector = np.squeeze(vector)
    vector = vector / np.linalg.norm(vector)
    n = int(np.log2(vector.shape[0]))
    qc = qiskit.QuantumCircuit(n, n)
    qc.initialize(vector, range(0, n))
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(2*n,))
    k = 0
    for i in range(0, n):
        qc.rx(thetas[k], i)
        k += 1
    for i in range(0, n):
        qc.rz(thetas[k], 0, i)
        k += 1
    counts = classical_part.measure(qc, list(range(0, n)))
    normalized_count = classical_part.normalize_count(counts, n)
    return normalized_count

def quanvolutional2(vector):
    if np.sum(vector) == 0:
        return np.ones(len(vector))
    vector = np.squeeze(vector)
    vector = vector / np.linalg.norm(vector)
    n = int(np.log2(vector.shape[0]))
    qc = qiskit.QuantumCircuit(n, n)
    qc.initialize(vector, range(0, n))
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(2*n,))
    k = 0
    for i in range(0, n):
        qc.rx(thetas[k], i)
        k += 1
    for i in range(0, n):
        qc.rz(thetas[k], 0, i)
        k += 1
    
    for i in range(0, n):
        qc.cnot(n - i, n - 1 - i)
    counts = classical_part.measure(qc, list(range(0, n)))
    normalized_count = classical_part.normalize_count(counts, n)
    return normalized_count
