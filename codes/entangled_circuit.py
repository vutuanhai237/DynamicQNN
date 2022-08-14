# https://arxiv.org/pdf/1905.10876.pdf
import qiskit, classical_part
import numpy as np
import math
def xz_layer(qc: qiskit.QuantumCircuit, thetas) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        qc (qiskit.QuantumCircuit): _description_
        thetas (_type_): _description_

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    k = 0
    for i in range(0, n):
        qc.rx(thetas[k], i)
        k += 1
    for i in range(0, n):
        qc.rz(thetas[k], i)
        k += 1
    return qc
def entangled_r_layer(qc: qiskit.QuantumCircuit, thetas, type, num_upsidedown = 0) -> qiskit.QuantumCircuit:
    n = qc.num_qubits
    k = 0
    for i in range(0, n - 1):
        if type == "rx":
            if i < num_upsidedown:
                qc.crx(thetas[k], n - 2 - i, n - i - 1)
            else:
                qc.crx(thetas[k], n - i - 1, n - 2 - i)
           
        if type == "ry":
            if i < num_upsidedown:
                qc.crx(thetas[k], n - 2 - i, n - i - 1)
            else:
                qc.crx(thetas[k], n - i - 1, n - 2 - i)
        if type == "rz":
            if i < num_upsidedown:
                qc.crx(thetas[k], n - 2 - i, n - i - 1)
            else:
                qc.crx(thetas[k], n - i - 1, n - 2 - i)
        k += 1
    return qc

def decrease_r_layer(qc: qiskit.QuantumCircuit, thetas, type, control_index = 0) -> qiskit.QuantumCircuit:
    n = qc.num_qubits
    k = 0
    controlled_indexs = (list(range(n - 1, -1, -1)))
    controlled_indexs.remove(control_index) 
    for i in range(0, n - 1):
        if type == 'rx':
            qc.crx(thetas[k], control_index, controlled_indexs[i])
        if type == 'ry':
            qc.cry(thetas[k], control_index, controlled_indexs[i])
        if type == 'rz':
            qc.crz(thetas[k], control_index, controlled_indexs[i])
        k += 1
    return qc

def entangled_cnot_layer(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    n = qc.num_qubits
    k = 0
    for i in range(0, n - 1):
        qc.cnot(n - i - 1, n - 2 - i)
        k += 1
    return qc

def quanvolutional(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    for i in range(1, n):
        qc.cry(thetas[i], 0, i)
    return qc

def quanvolutional1(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(2*n,))
    qc = xz_layer(qc, thetas)
    return qc

def quanvolutional2(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(2*n,))
    qc = xz_layer(qc, thetas)
    qc = entangled_cnot_layer(qc)
    return qc

def quanvolutional3(qc):
    n = qc.num_qubits 
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n - 1,))
    qc = xz_layer(qc, thetas[:2*n])
    qc = entangled_r_layer(qc, thetas[2*n:], 'rz')
    return qc

def quanvolutional4(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n - 1,))
    qc = xz_layer(qc, thetas[:2*n])
    qc = entangled_r_layer(qc, thetas[2*n:], 'rx')
    return qc

def quanvolutional5(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n*n + 3*n,))
    qc = xz_layer(qc, thetas[:2*n])   
    qc = decrease_r_layer(qc, thetas[2*n:2*n + n - 1], type = 'rz', control_index = n - 1)
    qc = decrease_r_layer(qc, thetas[2*n + n - 1:2*n + 2*n - 2], type = 'rz', control_index = n - 2)
    qc = decrease_r_layer(qc, thetas[2*n + 2*n - 2:2*n + 3*n - 3], type = 'rz', control_index = n - 3)
    qc = decrease_r_layer(qc, thetas[2*n + 3*n - 3:2*n + 4*n - 4], type = 'rz', control_index = n - 4)
    qc = xz_layer(qc, thetas[2*n + 4*n - 4:2*n + 4*n - 4 + 2*n]) 
    return qc

def quanvolutional6(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n*n + 3*n,))
    qc = xz_layer(qc, thetas[:2*n])   
    qc = decrease_r_layer(qc, thetas[2*n:2*n + n - 1], type = 'rx', control_index = n - 1)
    qc = decrease_r_layer(qc, thetas[2*n + n - 1:2*n + 2*n - 2], type = 'rx', control_index = n - 2)
    qc = decrease_r_layer(qc, thetas[2*n + 2*n - 2:2*n + 3*n - 3], type = 'rx', control_index = n - 3)
    qc = decrease_r_layer(qc, thetas[2*n + 3*n - 3:2*n + 4*n - 4], type = 'rx', control_index = n - 4)
    qc = xz_layer(qc, thetas[2*n + 4*n - 4:2*n + 4*n - 4 + 2*n]) 
    return qc


def quanvolutional7(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(5*n - 1,))
    qc = xz_layer(qc, thetas[:2*n])
    k = 2*n
    for i in range(0, n - 1, 2):
        qc.crz(thetas[k], i + 1, i)
        k += 1
    qc = xz_layer(qc, thetas[2*n + n // 2: 4*n + n // 2])
    k = 4*n + n // 2
    for i in range(0, n - 2, 2):
        qc.crz(thetas[k], i + 2, i + 1)
        k += 1
    return qc

def quanvolutional8(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(5*n - 1,))
    qc = xz_layer(qc, thetas[:2*n])
    k = 2*n
    for i in range(0, n - 1, 2):
        qc.crx(thetas[k], i + 1, i)
        k += 1
    qc = xz_layer(qc, thetas[2*n + n // 2: 4*n + n // 2])
    k = 4*n + n // 2
    for i in range(0, n - 2, 2):
        qc.crx(thetas[k], i + 2, i + 1)
        k += 1
    return qc

def quanvolutional9(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    for i in range(0, n):
        qc.h(i)
    
    for i in range(0, n - 1, 1):
        qc.cz(n - i - 1, n - 2 - i)
    
    k = 0
    for i in range(0, n):
        qc.rx(thetas[k], i)
        k += 1
    return qc

def quanvolutional10(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(2*n,))
    k = 0
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    for i in range(0, n - 1, 1):
        qc.cz(n - i - 1, n - 2 - i)
    
    qc.cz(0, n - 1)
    
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    return qc

def quanvolutional11(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(4*n - 4,))
    k = 0
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    for i in range(0, n):
        qc.rz(thetas[k], i)
        k += 1
    for i in range(0, n - 1, 2):
        qc.cnot(i + 1, i)
    for i in range(1, n - 1):
        qc.ry(thetas[k], i)
        k += 1
    for i in range(1, n - 1):
        qc.rz(thetas[k], i)
        k += 1
    
    for i in range(1, n - 2, 2):
        qc.cnot(i + 1, i)
    return qc

def quanvolutional12(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(4*n - 4,))
    k = 0
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    for i in range(0, n):
        qc.rz(thetas[k], i)
        k += 1
    for i in range(0, n - 1, 2):
        qc.cz(i + 1, i)
    for i in range(1, n - 1):
        qc.ry(thetas[k], i)
        k += 1
    for i in range(1, n - 1):
        qc.rz(thetas[k], i)
        k += 1
    
    for i in range(1, n - 2, 2):
        qc.cz(i + 1, i)
    return qc

def quanvolutional13(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n + n // math.gcd(n, 3),))
    k = 0
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    qc.crz(thetas[k], n - 1, 0)
    k += 1
    for i in range(0, n - 1):
        qc.crz(thetas[k], n - 2 - i, n - 1 - i)
        k += 1
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    
    qc.crz(thetas[k], n - 1, n - 2)
    k += 1
    qc.crz(thetas[k], 0, n -1)
    k += 1
    qc.crz(thetas[k], 1, 0)
    k += 1
    qc.crz(thetas[k], 2, 1)
    k += 1
    return qc

def quanvolutional14(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n + n // math.gcd(n, 3),))
    k = 0
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    qc.crx(thetas[k], n - 1, 0)
    k += 1
    for i in range(0, n - 1):
        qc.crx(thetas[k], n - 2 - i, n - 1 - i)
        k += 1
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    
    qc.crx(thetas[k], n - 1, n - 2)
    k += 1
    qc.crx(thetas[k], 0, n - 1)
    k += 1
    qc.crx(thetas[k], 1, 0)
    k += 1
    qc.crx(thetas[k], 2, 1)
    k += 1
    return qc

def quanvolutional15(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(2*n,))
    k = 0
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    qc.cx(n - 1, 0)
    for i in range(0, n - 1):
        qc.cx(n - 2 - i, n - 1 - i)
    for i in range(0, n):
        qc.ry(thetas[k], i)
        k += 1
    
    qc.cx(n - 1, n - 2)
    qc.cx(0, n - 1)
    qc.cx(1, 0)
    qc.cx(2, 1)
    return qc

def quanvolutional16(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n - 1,))
    qc = xz_layer(qc, thetas[:2*n])
    k = 2*n
    for i in range(0, n - 1, 2):
        qc.crz(thetas[k], i + 1, i)
        k += 1
    k = 2*n + n // 2
    for i in range(1, n - 1, 2):
        qc.crz(thetas[k], i + 1, i)
        k += 1
    return qc

def quanvolutional17(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n - 1,))
    qc = xz_layer(qc, thetas[:2*n])
    k = 2*n
    for i in range(0, n - 1, 2):
        qc.crx(thetas[k], i + 1, i)
        k += 1
    k = 2*n + n // 2
    for i in range(1, n - 1, 2):
        qc.crx(thetas[k], i + 1, i)
        k += 1
    return qc

def quanvolutional18(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n,))
    qc = xz_layer(qc, thetas[:2*n])
    k = 2*n
    qc.crz(thetas[k], n - 1, 0)
    k += 1
    for i in range(0, n - 1):
        qc.crz(thetas[k], n - i - 2, n - 1 - i)
        k += 1
    return qc

def quanvolutional19(qc):
    n = qc.num_qubits
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(3*n,))
    qc = xz_layer(qc, thetas[:2*n])
    k = 2*n
    qc.crx(thetas[k], n - 1, 0)
    k += 1
    for i in range(0, n - 1):
        qc.crx(thetas[k], n - i - 2, n - 1 - i)
        k += 1
    return qc