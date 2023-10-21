import cupy as cp

class Sigmoid:

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return 1 / (1 + cp.exp(-x))

    def derivative(self, x):
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)
