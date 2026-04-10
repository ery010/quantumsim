import numpy as np


class StateVector:

    def __init__(self, 
                 n_qubits: int, 
                 batch_size: int = 1,
                 xp = None,
                 init: str = "zero"
                 ):
        if xp is None:
            import numpy as as _np
            xp = _np