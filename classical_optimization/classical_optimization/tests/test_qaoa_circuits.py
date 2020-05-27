from classical_optimization.qaoa_circuits import maxcut_qaoa_circuit


def test_maxcut_qaoa():
    rows = 1
    cols = 2
    weights = {(0, 1): 1}
    p1 = maxcut_qaoa_circuit(gammas=[1], betas=[1],  p=1, weights=weights, rows=rows, cols=cols)
    p2 = maxcut_qaoa_circuit(gammas=[1, 1], betas=[1, 1],  p=2, weights=weights, rows=rows, cols=cols)
    # Increasing p should make the circuit deeper.
    assert len(p2) > len(p1)
