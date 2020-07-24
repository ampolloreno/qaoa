"""
python dispatch_jobs.py num_graphs_gen d num_qubits discretization
"""
from classical_optimization.terra.utils import write_graph
import networkx as nx
import numpy as np
from subprocess import call
import sys


seed = 137
np.random.seed(seed)
# We might want more graphs, additionally we want to make sure that the graphs are saved in a way that it is easy
# to recover the data, and make sure that new graphs are non-isomorphic.
num_graphs_gen = int(sys.argv[1])
graphs = []
# We might want larger d.
d = int(sys.argv[2])
# We will scan over this parameter, it's not clear if it should scale multiplicatively, or additively.
num_qubits = int(sys.argv[3])
discretization = int(sys.argv[4])

for _ in range(num_graphs_gen):
    graphs.append(nx.generators.random_graphs.random_regular_graph(d, num_qubits))

for graph in graphs:
    filename = write_graph(graph)
    cmd = f"python produce_landscape.py {filename} {discretization}"
    call(cmd, shell=True)
