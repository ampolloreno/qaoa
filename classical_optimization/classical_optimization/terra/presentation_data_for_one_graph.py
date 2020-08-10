import os
from networkx.algorithms.isomorphism import is_isomorphic
from classical_optimization.terra.utils import read_graph, write_graph
from scipy.optimize import dual_annealing
from classical_optimization.qaoa_circuits import execute_qaoa_circuit_and_estimate_cost
import numpy as np
from qiskit import Aer
from coldquanta.qiskit_tools.modeling.neutral_atom_noise_model import create_noise_model
from es import OpenES
import sys

path = sys.argv[1]
graph_data = read_graph(path)
graph = graph_data['graph']

dirs = ['../../../3_regular', '../../../complete']
dir_ = dirs[1]
graph_folders = [folder for folder in os.listdir(dir_)]
shots_per_point = 10


def max_landscape(data):
    for k, _ in data.items():
        if 'landscape' in k:
            max_beta = float(k.split('_')[2][1:])
            max_gamma = float(k.split('_')[3][1:])
            min_beta = float(k.split('_')[4][1:])
            min_gamma = float(k.split('_')[5][1:])
            return data.get(k), max_beta, min_beta, max_gamma, min_gamma


graph_data = read_graph(os.path.join(dir_, '4_graphs', '13bd629afa6b97e0d4cabfca5ed72ffb.pkl'))
graph = graph_data['graph']


def isomorphism_classes(graphs):
    classes = []
    for g in graphs:
        graph = g[0]
        appended = False
        for class_ in classes:
            if is_isomorphic(graph, class_[0][0]):
                class_.append(g)
                appended = True
                break
        if not appended:
            classes.append([g])
    return classes


def prune_graphs(classes):
    for class_ in classes:
        while len(class_) > 1:
            duplicate = class_.pop()
            os.remove(duplicate[1])


def clean_up_graphs():
    for folder in graph_folders:
        path = os.path.join(dir_, folder)
        files = [f for f in os.listdir(path)]
        graphs = []
        for f in files:
            if 'pkl' in f:
                f = os.path.join(dir_, folder, f)
                data = read_graph(f)
                graphs.append((data['graph'], f))
                #DELETES FILES, UNCOMMENT CAREFULLY
                if len(data) == 1:
                    os.remove(f)
        prune_graphs(isomorphism_classes(graphs))


np.random.seed(666)
reprate = 50
one_hour = 60 * 60
max_gamma = 2 * np.pi
max_beta = np.pi
simulator = Aer.get_backend('qasm_simulator')
noise_model = create_noise_model(cz_fidelity=1)


def weights(graph):
    rtn = {}
    for e in graph.edges:
        weight = graph.get_edge_data(e[0], e[1])['weight']
        rtn[e] = weight
    return rtn


def objective(graph):
    #Hack for backwards compatibility.
    num_rows = len(graph.nodes)
    num_cols = 1

    history = []

    def store_log(func):
        def logged_func(x):
            history.append(x)
            return func(x)
        return logged_func

    @store_log
    def gamma_beta_objective(gamma_beta):
        # The cut value is the expectation value, minima of the negation correspond to maxima.
        return execute_qaoa_circuit_and_estimate_cost(gamma=gamma_beta[1], beta=gamma_beta[0],
                                                       num_shots=shots_per_point,
                                                       simulator=simulator,
                                                       coupling_map=None,
                                                       weights=weights(graph),
                                                       rows=num_rows,
                                                       cols=num_cols,
                                                       noise_model=noise_model,
                                                       # Just a fully random seed, in the full range.
                                                       seed=np.random.randint(0, 2**32 - 1))
    return gamma_beta_objective, history

bounds = [(-np.pi, np.pi), (-np.pi/4, np.pi/4)]
func, history = objective(graph)
initial_gamma_beta = [np.random.rand() * max_param for max_param in (max_gamma, max_beta)]
result = dual_annealing(
    lambda x: -1 * func(x),
    bounds=bounds,
    x0=np.array(initial_gamma_beta),
    # One annealing attempt.
    maxiter=1,
    initial_temp=10,
    maxfun=one_hour * reprate,
    restart_temp_ratio=1E-10,
    no_local_search=True)
result.fun = -result.fun
# gamma, beta; reported average cut after sampling many times
annealing_result = (result.x, result.fun)

NPARAMS = 2
NPOPULATION = 100
oes = OpenES(NPARAMS,                  # number of model parameters
            sigma_init=0.025,            # initial standard deviation
            sigma_decay=0.999,         # don't anneal standard deviation
            learning_rate=0.005,         # learning rate for standard deviation
            learning_rate_decay = 0.0, # annealing the learning rate
            popsize=NPOPULATION,       # population size
            antithetic=False,          # whether to use antithetic sampling
            weight_decay=0.00,         # weight decay coefficient
            rank_fitness=False,        # use rank rather than fitness numbers
            forget_best=False)

MAX_ITERATION = 500
fit_func, history = objective(graph)


def test_solver(solver):
    history = []
    for j in range(MAX_ITERATION):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = fit_func(solutions[i])
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j + 1) % 100 == 0:
            print("fitness at iteration", (j + 1), result[1])
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history, result


history, result = test_solver(oes)
# gamma, beta; reported average cut after sampling many times
es_result = (result[0], result[1])


def cutsize(set1, set2, g):
    cut = 0
    for s1 in set1:
        for s2 in set2:
            if g.get_edge_data(s1, s2) is not None:
                cut += g.get_edge_data(s1, s2)['weight']
    return cut


def maxcut(g, a=None, b=None, used=None):
    if a is None:
        a = []
    if b is None:
        b = []
    if used is None:
        used = []
    for node in g.nodes:
        if node not in used:
            left = maxcut(g, list(a) + [node], list(b), list(used) + [node])[0]
            right = maxcut(g, list(a), list(b) + [node], list(used) + [node])[0]
            if left > right:
                a = list(a) + [node]
                b = list(b)
            else:
                a = list(a)
                b = list(b) + [node]
    # There are no unused nodes, we've reached a leaf.
    return cutsize(a, b, g), a, b


maxcut_result = maxcut(graph)
landscape, max_beta, min_beta, max_gamma, min_gamma = max_landscape(graph_data)
# gamma, beta; reported average cut after sampling many times
maxarg_result = (np.argmax(landscape), np.max(landscape))

write_graph(graph, attributes={'annealing_result': annealing_result,
                               'es_result': es_result,
                               'maxcut_result': maxcut_result,
                               'maxarg_result': maxarg_result})

