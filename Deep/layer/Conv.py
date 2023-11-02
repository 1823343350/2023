import numpy as np
from numpy import sqrt
from .model import CustomModel
np.set_printoptions(precision = 4)
class Conv(CustomModel):
    """
    创建一个普通的全连接层, 能够得到相应的输出
    
    input_features_nums: int, 每个神经元的参数量（输入通道数）
    Number_of_neurons: int, 神经元的数量（输出通道数）
    bias: float = 0, 偏置项

    """

    def __init__(self, name: str = 'conv',in_channels: int = 0, out_channels: int = 0, bias: float = 0, kernel_size: int = None, activation_function: any = None, stride: int = None, padding: int = None, regularization = None) -> None:
        super().__init__(name=name, activation_function=activation_function, regularization=regularization)

        a = sqrt(6 / (in_channels + out_channels))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = 0.01 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.ones((out_channels, 1, 1))
        self.size.append([in_channels, out_channels])
        self.stride = stride
        self.padding = padding
        self.row = 0
        self.col = 0
        self.temp = []
        self.uw_next_grad = None
        self.type = 'conv'

    def __call__(self, x):
        # 根据pad进行填充
        if self.padding:
            matrix = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        else:
            matrix = x

        # 开始滑动计算结果
        self.row = int((matrix.shape[2]+2*self.padding-self.kernel_size)/self.stride) + 1
        self.col = int((matrix.shape[3]+2*self.padding-self.kernel_size)/self.stride) + 1
        self.u = np.zeros((x.shape[0], self.weight.shape[0], self.row, self.col)) # 用来存放结果

        # 遍历所有图片
        for r in range(x.shape[0]):
            # 遍历一个卷积和的所有卷积位置
            # now_list = []
            for i in range(self.row):
                for k in range(self.col):
                    # now_list.append(matrix[r, : , i * self.stride:i * self.stride + self.kernel_size, k * self.stride:k * self.stride + self.kernel_size])
                    self.u[r, : , i, k] = np.sum(self.weight * matrix[r, : , i * self.stride:i * self.stride + self.kernel_size, k * self.stride:k * self.stride + self.kernel_size], axis=(-3,-2,-1))
            # self.temp.append(now_list)

        # 记录输入矩阵x, 求参数时用到
        self.uw_grad = matrix
        # 记录实际输入矩阵matrix, 求前一层梯度时用到
        self.uw_next_grad = x
        self.uy_grad = self.weight

        if self.activation_function:
            self.y = self.activation_function(self.u + self.bias)
        else:
            self.y = self.u + self.bias
        return self.y
