import cupy as cp

class Flatten:
    def __init__(self):
        pass

    def __call__(self, x: cp.ndarray):
        self.result = x.reshape(x.shape[0], -1)
        return self.result 

    def derivative(self, x: cp.ndarray):
        pass
