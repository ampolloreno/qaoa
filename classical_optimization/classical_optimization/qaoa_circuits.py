from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import execute
import numpy as np
#from recirq.optimize.mgd import model_gradient_descent
# I may have modified my qiskit - this adds on an attribute when I import.
from qiskit.providers.aer.extensions import snapshot_density_matrix


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


def density_cost(density_matrix, num_qubits, weights):
    rtn = 0
    for edge, weight in weights.items():
        rtn += weight * (np.trace(Z(*edge, num_qubits).dot(density_matrix)))
    return rtn


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


def maxcut_qaoa_circuit(*, gammas: list, betas: list, weights=None, rows=None, cols=None, p: int = 1, measure=True,
                        num_qubits=None, density_matrix=False):
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
    if rows is not None and cols is not None:
        num_qubits = rows * cols
    QAOA = QuantumCircuit(num_qubits)
    # apply the layer of Hadamard gates to all qubits, and then fence all qubits.
    QAOA.h(range(num_qubits))
    QAOA.barrier()

    # apply the Ising type gates with angle gamma along the edges in E
    for i in range(p):
        for edge, weight in weights.items():
            k, l = edge
            # The following is a gate decomposition for a exp(-i*\gamma*ZZ). See the mathematica notebook cphase.nb.
            QAOA.h(l)
            QAOA.cz(k, l)
            QAOA.h(l)
            # Both factors of 2 seems to come from SU(2) being a double cover.
            QAOA.rz(2*gammas[i] * weight, l)
            QAOA.h(l)
            QAOA.cz(k, l)
            QAOA.h(l)
            QAOA.barrier()
        # then apply the single qubit X - rotations with angle beta to all qubits
        QAOA.rx(2*betas[i], list(range(num_qubits)))
    if measure:
        QAOA.measure_all()
    else:
        if density_matrix:
            QAOA.snapshot_density_matrix('output')
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


def plot_landscape(landscape, max_gamma, max_beta, colorbar=True, history=None, discretization=None):
    """Given a QAOA landscape, a 2d numpy array of values coming from sweeping gamma and beta, plot it."""
    ax = plt.imshow(landscape)

    ax.figure.canvas.draw()
    subplot = ax.figure.get_axes()[0]

    ticks = subplot.get_xticklabels()
    scale = 1/max([float(tick.get_text().replace('−', '-')) for tick in ticks if tick.get_text() != ''])
    scale *= max_beta
    for tick in ticks:
        if tick.get_text() != '':
            tick.set_text(round(scale * float(tick.get_text().replace('−', '-')), 2))
        subplot.set_xticklabels(ticks)

    ticks = subplot.get_yticklabels()
    scale = 1 / max([float(tick.get_text().replace('−', '-')) for tick in ticks if tick.get_text() != ''])
    scale *= max_gamma
    for tick in ticks:
        if tick.get_text() != '':
            tick.set_text(round(scale * float(tick.get_text().replace('−', '-')), 2))
        subplot.set_yticklabels(ticks)

    plt.ylabel(r"$\gamma$")
    plt.xlabel(r"$\beta$")
    if colorbar:
        plt.colorbar()
    # if this is put before, it messes up the scale of the colorbar... I'm not sure why.
    if history is not None:
        for i, point in enumerate(history[1:-1]):
            plt.scatter(*beta_gamma_to_index(*point, discretization, max_gamma, max_beta), c='r')
        plt.scatter(*beta_gamma_to_index(*history[0], discretization, max_gamma, max_beta), marker='.', s=200, c='w')
        plt.scatter(*beta_gamma_to_index(*history[-1], discretization, max_gamma, max_beta), marker='*', s=200, c='w')


def execute_qaoa_circuit_and_estimate_cost(gamma, beta, num_shots, simulator, coupling_map, weights, rows, cols,
                                           *, noise_model=None, seed=None):
    """Build and run the a qaoa circuit with the given parameters on the given simulator."""
    circuit = maxcut_qaoa_circuit(gammas=[gamma], betas=[beta], p=1, rows=rows, cols=cols, weights=weights, measure=False, density_matrix=True)
    job = execute(circuit,
                  simulator,
                  noise_model=noise_model,
                  coupling_map=coupling_map,
                  optimization_level=0,
                  shots=num_shots,
                  seed_simulator=seed)
    #all_counts = job.result().get_counts()
    outputs = [result.data.snapshots.density_matrix['output'][0]['value'] for result in job.result().results]
    # The diagonal is real, so we take the first element.
    num_qubits = rows*cols
    expectations = [density_cost(np.array(output)[:, :, 0], num_qubits=num_qubits, weights=weights) for output in
                   outputs]
    return expectations[0]


    #return estimate_cost(all_counts, weights)


def beta_gamma_to_index(beta, gamma, discretization, max_gamma, max_beta, relative=False):
    """Given a discretization for a maximum beta and gamma, return the sample index that a given gamma, and beta are
     associated with."""
    if not relative:
        return (beta % max_beta * (discretization-1)/max_beta), (gamma % max_gamma * (discretization-1)/max_gamma)
    else:
        return (beta % max_beta * (discretization - 1) / max_beta), (gamma % max_gamma * (2*discretization-1)/max_gamma)


def plot_history_over_landscape(history, landscape, discretization, max_gamma, max_beta, result):
    """Given a history and a landscape, plot the trajectory of the optimizer."""
    plot_landscape(landscape, max_gamma, max_beta)
    for i, point in enumerate(history[1:-1]):
        plt.scatter(*beta_gamma_to_index(*point, discretization, max_gamma, max_beta), c='r')
    plt.scatter(*beta_gamma_to_index(*history[0], discretization, max_gamma, max_beta), marker='.', s=200, c='w')
    plt.scatter(*beta_gamma_to_index(*history[-1], discretization, max_gamma, max_beta), marker='*', s=200, c='w')
    plt.scatter(*beta_gamma_to_index(*result.x, discretization, max_gamma, max_beta), marker='*', s=200, c='r')


def try_optimizer(optimizer, simulator, coupling_map, shots_per_point, weights, max_gamma, max_beta, rows, cols,
                  history=None, noise_model=None, initial_gamma_beta=None):
    """Try optimizer to optimize QAOA from a random initial point."""
    if history is None:
        history = []

    def store_log(func):
        def logged_func(x):
            history.append(x)
            return func(x)
        return logged_func

    @store_log
    def gamma_beta_objective(gamma_beta):
        return -execute_qaoa_circuit_and_estimate_cost(gamma=gamma_beta[1], beta=gamma_beta[0],
                                                       num_shots=shots_per_point,
                                                       simulator=simulator,
                                                       coupling_map=coupling_map,
                                                       weights=weights,
                                                       rows=rows,
                                                       cols=cols,
                                                       noise_model=noise_model)
    if initial_gamma_beta is None:
        initial_gamma_beta = [np.random.rand() * max_param for max_param in (max_gamma, max_beta)]
    if optimizer == 'mgd':
        result = model_gradient_descent(gamma_beta_objective,
                                        x0=np.array(initial_gamma_beta),
                                        tol=.1,
                                        n_sample_points=20)
    else:
        result = minimize(gamma_beta_objective, x0=np.array(initial_gamma_beta), method=optimizer)
    print(fr'$\gamma$,$\beta$={result.x}')
    print(f'Max cut is {-result.fun}')
    return result


def produce_gammas_betas(discretization, max_gamma, max_beta, min_gamma=0, min_beta=0):
    """Discretization is the number of points per range of pi (over the beta range).
     We double this density for gamma."""
    return np.linspace(min_gamma, max_gamma, 2*discretization), np.linspace(min_beta, max_beta, discretization)
