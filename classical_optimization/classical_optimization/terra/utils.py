import dill
import hashlib
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
import os


def write_graph(graph, attributes=None, where=None, noisy=False):
    if attributes is None:
        attributes = {}
    h = hashlib.md5()
    arr = adjacency_matrix(graph).toarray()
    h.update(arr)
    hash_ = h.hexdigest()
    num_qubits = len(graph.nodes)
    name = 'hari'
    if noisy:
        name += 'noisy'
    if where is not None:
        filename = where
    else:
        filename = f'{name}/{num_qubits}_graphs/{hash_}.pkl'
    try:
        os.mkdir(f'{name}')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{name}/{num_qubits}_graphs')
    except FileExistsError:
        pass
    try:
        with open(filename, 'rb') as filehandle:
            data = dill.load(filehandle)
            print("Fetching existing file...")
    except FileNotFoundError:
        data = {'graph': graph}
    for k, v in attributes.items():
        if data.get(k) is None:
            data[k] = v
        else:
            data[k] = v
            print(f"File {filename} already has attribute {k}, STILL OVERWRITING!!!")
    with open(filename, 'wb') as filehandle:
        dill.dump(data, filehandle)
    return filename


def read_graph(filename):
    with open(filename, 'rb') as filehandle:
            data = dill.load(filehandle)
    return data


# def cost(density_matrix, num_qubits, weights):
#     rtn = 0
#     for edge, weight in weights.items():
#         rtn += .5 * weight * (1 - np.trace(Z(*edge, num_qubits).dot(density_matrix)))
#     return rtn

def cost(statevector, num_qubits, weights):
    rtn = 0
    for edge, weight in weights.items():
        rtn += weight * (np.conj(statevector.T).dot(Z(*edge, num_qubits).dot(statevector)))
    return rtn


def density_cost(density_matrix, num_qubits, weights):
    rtn = 0
    for edge, weight in weights.items():
        rtn += weight * (np.trace(Z(*edge, num_qubits).dot(density_matrix)))
    return rtn


def Z(i, j, num_qubits):
    rtn = np.eye(1)
    z = np.array([[1, 0], [0, -1]])
    # Holy eff order matters be careful.
    for k in reversed(range(num_qubits)):
        if k == i or k == j:
            rtn = np.kron(rtn, z)
        else:
            rtn = np.kron(rtn, np.eye(2))
    return rtn


def weights(graph):
    rtn = {}
    for e in graph.edges:
        try:
            weight = graph.get_edge_data(e[0], e[1])['weight']
        except KeyError:
            weight = 1
        rtn[e] = weight
    return rtn
