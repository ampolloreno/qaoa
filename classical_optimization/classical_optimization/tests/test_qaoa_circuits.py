from classical_optimization.qaoa_circuits import maxcut_qaoa_circuit, estimate_cost


def test_maxcut_qaoa():
    rows = 1
    cols = 2
    weights = {(0, 1): 1}
    p1 = maxcut_qaoa_circuit(gammas=[1], betas=[1],  p=1, weights=weights, rows=rows, cols=cols)
    p2 = maxcut_qaoa_circuit(gammas=[1, 1], betas=[1, 1],  p=2, weights=weights, rows=rows, cols=cols)
    # Increasing p should make the circuit deeper.
    assert len(p2) > len(p1)

    def count_gate(circ, name):
        count = 0
        for g, _, __ in circ:
            if g.name == name:
                count += 1
        return count
    # There should be a number of RX gates equal to the number of qubits * p
    assert count_gate(p1, 'rx') == rows * cols
    assert count_gate(p2, 'rx') == 2 * rows * cols
    assert count_gate(p2, 'measure') == rows * cols


def test_estimate_cost():
    weights = {(0, 1): 1}
    counts = {'00': 1, '01': 1, '10': 1, '11': 1}
    # The uniform distribution over two bit strings with uniform weight is half of the weight.
    assert estimate_cost(counts, weights) == .5
    assert estimate_cost(counts, weights, lambda a, b: max(a)) == 1
    assert estimate_cost(counts, weights, lambda a, b: min(a)) == 0
