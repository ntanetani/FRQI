import numpy as np
def frqiEncoder(circ, img, target, controls, anc):
    size = len(img)
    assert len(controls) >= np.log2(size), "You need more control qubits."

    