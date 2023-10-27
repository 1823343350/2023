from cmath import sqrt
import cupy as cp
from .model import CustomModel

class Pool(CustomModel):

    def __init__(self,name: str  = 'pool',kernel_size: int = 1, stride: int = 1,  pool_function = None, activation = None) -> None:
        super().__init__(name=name, activation_function = activation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_function = pool_function
        self.weight = 0
        self.bias = 0
        self.is_pool = False
        self.type = 'pool'

    def __call__(self, x):
        # 保存输入矩阵,反向传播使用
        self.uw_grad = x
        self.u = x
        self.bias = 0
        # 开始滑动计算结果
        if x.shape[1]>=self.kernel_size and x.shape[2]>=self.kernel_size: 
            # 判断是否池化过
            self.is_pool = True
            row = int((x.shape[1]-self.kernel_size)/self.stride) + 1
            col = int((x.shape[2]-self.kernel_size)/self.stride) + 1

            if self.pool_function == 'Max':
                self.u = cp.zeros((x.shape[0], row, col)) # 用来存放结果
                for i in range(row):
                    for k in range(col):
                        self.u[:, i, k] = cp.max(x[:, i * self.stride:i * self.stride + self.kernel_size, k * self.stride:k * self.stride + self.kernel_size], axis=(1,2))

        if self.activation_function:
            self.y = self.activation_function(self.u + self.bias)
        else:
            self.y = self.u + self.bias
        return self.y
