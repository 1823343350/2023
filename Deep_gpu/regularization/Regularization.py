from typing import Any
import cupy as cp

class Regularization():

    def __init__(self, lamda = 0.0) -> None:
        self.lamda = lamda

    def __call__(self, layer = None) -> Any:
        col_sum = cp.array([cp.sum(layer.weight, axis=1)]).T
        layer.re  = self.lamda*(col_sum-layer.weight)
