import cupy as cp

class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        return cp.maximum(0, x)

    def derivative(self, x: cp.ndarray):
        return (x > 0).astype(int)
