import qiskit
import numpy as np




# Training hyperparameter
conv_num_filter = 50
quanv_num_filter = 4
quanv_size_filter = 2
conv_size_filter = 2
num_shots = 10000
learning_rate = 0.01
noise_prob = 0.01
backend = qiskit.Aer.get_backend('qasm_simulator')

parameterized_gate = [
    'rx',
    'ry',
    'rz',
    'crx',
    'cry',
    'crz'
]

nonparameterized_gate = [
    'x',
    'y',
    'z',
    'cx',
    'cy',
    'cz',
    'cswap',
    'h'
]

def get_quanv_num_filter(kernel_size):
    if quanv_num_filter == -1:
        return kernel_size**2
    else:
        return quanv_num_filter