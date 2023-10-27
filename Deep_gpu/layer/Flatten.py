import cupy as cp
from .model import CustomModel

class Flatten(CustomModel):

    def __init__(self,name: str  = 'flatten') -> None:
        super().__init__(name = name)
        self.type = 'flatten'
        self.weight = 0
        self.bias = 0
        self.lwgrad = 0
        self.lbgrad = 0

    def __call__(self, x):
        self.result = x
        return x.reshape(x.shape[0], -1)

    def derivative(self, x: cp.ndarray):
        # 打平操作的还原
        # root_matrix = cp.zeros_like(self.result)
        root_matrix = x.reshape(self.result.shape[0],self.result.shape[1],self.result.shape[2])
        return root_matrix