import numpy as np
np.set_printoptions(precision = 4)
class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray):
        return (x > 0).astype(int)
