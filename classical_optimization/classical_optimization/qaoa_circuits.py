from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import execute
import numpy as np

SCIPY_METHODS =\
    ['Nelder-Mead',
     'Powell',
     'CG',
     'BFGS',
     # 'Newton-CG',
     'L-BFGS-B',
     'TNC',
     'COBYLA',
     'SLSQP',
     'trust-constr',
     #'dogleg',
     #'trust-ncg',
     #'trust-krylov',
     #'trust-exact'
]


def maxcut_qaoa_circuit(*, gammas: list, betas: list, weights: dict, rows: int, cols: int, p: int = 1):
    """
    :param gammas: A list of gamma values as defined in the original paper by Farhi et al.
    :param betas: A list of beta values as defined in the original paper by Farhi et al.
    :param weights: A dictionary of weights for each edge. The graph is undirected, so this assumes each edge is given
     only once. EXP(-iZZ\theta) is the only entangling gate, and is symmetric, so order does not matter.
    :param rows: The number of rows in the the trap.
    :param cols: The number of columns in the trap.
    :param p: The number of effective trotter steps in this QAOA circuit.
    :return: The qiskit QuantumCircuit for this QAOA instance.
    """
    QAOA = QuantumCircuit(rows * cols)
    # apply the layer of Hadamard gates to all qubits, and then fence all qubits.
    QAOA.h(range(rows * cols))
    QAOA.barrier()

    # apply the Ising type gates with angle gamma along the edges in E
    for i in range(p):
        for edge, weight in weights.items():
            k, l = edge
            # The following is a gate decomposition for a exp(-i*\gamma*ZZ). See the mathematica notebook cphase.nb.
            QAOA.h(l)
            QAOA.cz(k, l)
            QAOA.h(l)
            QAOA.rz(2 * gammas[i] * weight, l)
            QAOA.h(l)
            QAOA.cz(k, l)
            QAOA.h(l)
            QAOA.barrier()
        # then apply the single qubit X - rotations with angle beta to all qubits
        QAOA.rx(betas[i], list(range(rows * cols)))
    QAOA.measure_all()
    return QAOA


def estimate_cost(counts: dict, weights: dict, func=None):
    """
    Given counts and weights, return func of the cut values.

    :param counts: Dictionary mapping bitstrings to number of occurrences in output.
    :param weights: Dictionary mapping tuples of edges to the associated weight of the edge in the MAXCUT problem.
    :param func: The function to apply the cut values. By default this returns the average, however you could instead
     consider a function like max.
    :return: The cost of the cut.
    """
    cut_values = []
    counts_list = []
    for assignment, count in counts.items():
        cut_value = 0
        for edge, weight in weights.items():
            if assignment[edge[0]] != assignment[edge[1]]:
                # We've cut this edge.
                cut_value += weight
        cut_values.append(cut_value)
        counts_list.append(count)
    if func is None:
        return sum(map(lambda tup: tup[0]*tup[1], list(zip(counts_list, cut_values))))/sum(counts_list)
    else:
        return func(cut_values, counts)


def plot_landscape(landscape):
    plt.imshow(landscape)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,
        labelleft=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel(r"$\gamma$")
    plt.xlabel(r"$\beta$")
    plt.colorbar()


def execute_qaoa_circuit_and_estimate_cost(gamma, beta, num_shots, simulator, coupling_map, weights, rows, cols,
                                           *, noise_model=None):
    circuit = maxcut_qaoa_circuit(gammas=[gamma], betas=[beta], p=1, rows=rows, cols=cols, weights=weights)
    job = execute(circuit,
                  simulator,
                  noise_model=noise_model,
                  coupling_map=coupling_map,
                  optimization_level=0,
                  shots=num_shots)
    all_counts = job.result().get_counts()
    return estimate_cost(all_counts, weights)


def gamma_beta_to_index(gamma, beta, discretization, max_gamma, max_beta):
    return (gamma * discretization/max_gamma)%(discretization-1), (beta * discretization/max_beta)%(discretization-1)


def plot_history_over_landscape(history, landscape, discretization, max_gamma, max_beta):
    plot_landscape(landscape)
    for i, point in enumerate(history[1:-1]):
        plt.scatter(*gamma_beta_to_index(*point, discretization, max_gamma, max_beta), c='r')
    plt.scatter(*gamma_beta_to_index(*history[0], discretization, max_gamma, max_beta), marker='.', s=200, c='w')
    plt.scatter(*gamma_beta_to_index(*history[-1], discretization, max_gamma, max_beta), marker='*', s=200, c='w')


def try_optimizer(optimizer, simulator, coupling_map, shots_per_point, weights, max_gamma, max_beta, rows, cols,
                  history=None):
    if history is None:
        history = []

    def store_log(func):
        def logged_func(x):
            history.append(x)
            return func(x)
        return logged_func

    @store_log
    def gamma_beta_objective(gamma_beta):
        return -execute_qaoa_circuit_and_estimate_cost(gamma=gamma_beta[0], beta=gamma_beta[1], num_shots=shots_per_point,
                                                       simulator=simulator,
                                                       coupling_map=coupling_map,
                                                       weights=weights,
                                                       rows=rows,
                                                       cols=cols)

    initial_gamma_beta = [np.random.rand() * max_param for max_param in (max_gamma, max_beta)]
    result = minimize(gamma_beta_objective, x0=initial_gamma_beta, method=optimizer)
    print(fr'$\gamma$,$\beta$={result.x}')
    print(f'Max cut is {-result.fun}')
    return result