"""
python produce_landscape.py filename discretization
"""
from classical_optimization.qaoa_circuits import produce_gammas_betas, maxcut_qaoa_circuit
from classical_optimization.terra.utils import write_graph, read_graph, weights, cost
from coldquanta.qiskit_tools.modeling.neutral_atom_noise_model import create_noise_model
import numpy as np
from qiskit import Aer, execute
# I may have modified my qiskit - this adds on an attribute when I import.
from qiskit.providers.aer.extensions import snapshot_density_matrix
from recirq.qaoa.simulation import exact_qaoa_values_on_grid
import sys
import time

min_gamma = -np.pi
max_gamma = np.pi
min_beta = -np.pi/4
max_beta = np.pi/4
discretization = int(sys.argv[2])
discretization = 40
gammas, betas = produce_gammas_betas(discretization, max_gamma, max_beta, min_gamma, min_beta)
noisy = True

filename = sys.argv[1]
landscape_string = f"landscape_d{discretization}_b{max_beta}_g{max_gamma}_b{min_beta}_g{min_gamma}"
if landscape_string not in read_graph(filename).keys():
    graph = read_graph(filename)['graph']
    num_qubits = len(graph.nodes)
    start = time.time()
    if noisy:
        simulator = Aer.get_backend('qasm_simulator')
        noise_model = create_noise_model(cz_fidelity=.8)
        experiments = []
        for gamma in gammas:
            for beta in betas:
                circuit = maxcut_qaoa_circuit(gammas=[gamma], betas=[beta], p=1, num_qubits=num_qubits, weights=weights(graph), measure=False)
                experiments.append(circuit)
        job = execute(experiments, backend=simulator, noise_model=noise_model)
        expectations = [np.real(cost(job.result().get_statevector(experiment), num_qubits=num_qubits, weights=weights(graph))) for experiment in experiments]

        landscape = np.zeros((2*discretization, discretization))
        for i, gamma in enumerate(gammas):
            for j, beta in enumerate(betas):
                landscape[i][j] = expectations[i*len(betas) + j]
    else:
        landscape = exact_qaoa_values_on_grid(graph, num_processors=int(sys.argv[3]), xlim=(min_gamma, max_gamma), ylim = (min_beta, max_beta),
                                              x_grid_num=2 * discretization, y_grid_num=discretization)
    stop = time.time()
    write_graph(graph, {landscape_string: landscape, landscape_string + '_time': stop-start}, noisy=noisy)
else:
    print("Already computed ths one!")