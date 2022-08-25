import qiskit
import numpy as np




# Training hyperparameter
num_conv_filter = 1
num_quanv_filter = 1

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

def get_num_quanv_filter(kernel_size):
    if num_quanv_filter == -1:
        return kernel_size**2
    else:
        return num_quanv_filter