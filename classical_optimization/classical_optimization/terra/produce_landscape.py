"""
python produce_landscape.py filename discretization
"""
from classical_optimization.qaoa_circuits import produce_gammas_betas, maxcut_qaoa_circuit
from classical_optimization.terra.utils import write_graph, read_graph, weights, cost
import numpy as np
from qiskit import Aer, execute
# I may have modified my qiskit - this adds on an attribute when I import.
from qiskit.providers.aer.extensions import snapshot_density_matrix
import sys
import time

max_gamma = 2*np.pi
max_beta = np.pi
discretization = int(sys.argv[2])
gammas, betas = produce_gammas_betas(discretization, max_gamma, max_beta)

filename = sys.argv[1]
landscape_string = f"landscape_d{discretization}_b{max_beta}_g{max_gamma}"
if landscape_string not in read_graph(filename).keys():
    graph = read_graph(filename)['graph']
    num_qubits = len(graph.nodes)
    start = time.time()
    simulator = Aer.get_backend('statevector_simulator')
    experiments = []
    for gamma in gammas:
        for beta in betas:
            circuit = maxcut_qaoa_circuit(gammas=[gamma], betas=[beta], p=1, num_qubits=num_qubits, weights=weights(graph), measure=False)
            experiments.append(circuit)
    job = execute(experiments, backend=simulator)
    expectations = [np.real(cost(job.result().get_statevector(experiment), num_qubits=num_qubits, weights=weights(graph))) for experiment in experiments]

    landscape = np.zeros((2*discretization, discretization))
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            landscape[i][j] = expectations[i*len(betas) + j]
    stop = time.time()
    write_graph(graph, {landscape_string: landscape, landscape_string + '_time': stop-start})
