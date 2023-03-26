# https://arxiv.org/pdf/1905.10876.pdf
import qiskit, classical_part
import numpy as np
import math

def create_Wchain(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create W_chain ansatz
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[-1], qc.num_qubits - 1, 0)
    return qc

def create_Walternating(qc: qiskit.QuantumCircuit, thetas: np.ndarray, index_layer):
    """Create W_alternating ansatz
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - index_layer (int)
    Returns:
        - qiskit.QuantumCircuit
    """
    t = 0
    if index_layer % 2 == 0:
        # Even
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
        qc.cry(thetas[-1], 0, qc.num_qubits - 1)
    else:
        # Odd
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
    return 

def create_Walltoall(qc: qiskit.QuantumCircuit, thetas: np.ndarray, limit=0):
    """Create Walltoall
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - limit (int): limit layer
    Returns:
        - qiskit.QuantumCircuit
    """
    if limit == 0:
        limit = len(thetas)
    t = 0
    for i in range(0, qc.num_qubits):
        for j in range(i + 1, qc.num_qubits):
            qc.cry(thetas[t], i, j)
            t += 1
            if t == limit:
                return qc
    return qc

def create_Wchain_layered_ansatz(qc: qiskit.QuantumCircuit):
    """Create Alternating layered ansatz
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers
    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    thetas = np.random.uniform(low=0, high=2*np.pi, size=(num_layers * (n * 3)))
    qc = create_Wchain(qc, thetas[:n])
    qc.barrier()
    qc = xz_layer(qc, thetas[n:])
    return qc


def calculate_n_walternating(index_layers, num_qubits):
    if index_layers % 2 == 0:
        n_walternating = int(num_qubits / 2)
    else:
        n_walternating = math.ceil(num_qubits / 2)

    return n_walternating

def create_Walternating_layered_ansatz(qc: qiskit.QuantumCircuit):
    """Create Walternating layered ansatz
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (Int): numpy of layers
    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    n_alternating = calculate_n_walternating(0, n)
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n_alternating + (n * 2)))
    qc = create_Walternating(qc, thetas[:n_alternating], 0)
    qc.barrier()
    qc = xz_layer(qc, thetas[n_alternating:])
    return qc

def calculate_n_walltoall(n):
    n_walltoall = 0
    for i in range(1, n):
        n_walltoall += i
    return n_walltoall


def create_Walltoall_layered_ansatz(qc: qiskit.QuantumCircuit,
                                  ):
    """Create W all to all ansatz
    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - num_layers (int): numpy of layers
    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    n_walltoall = calculate_n_walltoall(n)
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(n_walltoall + (n * 2)))
    qc = create_Walltoall(qc, thetas[0:n_walltoall])
    qc.barrier()
    qc = xz_layer(qc, thetas[n_walltoall:])
    return qc

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

def trainable_quanvolutional(qc, thetas):
    n = qc.num_qubits
    for i in range(1, n):
        qc.cry(thetas[i], 0, i)
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