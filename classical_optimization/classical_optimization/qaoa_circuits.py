from qiskit import QuantumCircuit


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
        QAOA.rx(betas, list(range(rows * cols)))
        QAOA.measure_all()
    return QAOA
