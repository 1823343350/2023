import cupy as cp
from cupy import sqrt
from .model import CustomModel

class Conv(CustomModel):
    """
    创建一个普通的全连接层, 能够得到相应的输出
    
    input_features_nums: int, 每个神经元的参数量（输入通道数）
    Number_of_neurons: int, 神经元的数量（输出通道数）
    bias: float = 0, 偏置项

    """

    def __init__(self, in_channels: int, out_channels: int, bias: float = 0, kernel_size: int = None, activation_function: any = None, stride: int = None, padding: int = None, regularization = None) -> None:
        super().__init__('Conv', activation_function, regularization)

        a = sqrt(6 / (in_channels + out_channels))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = cp.random.uniform(low=-a, high=a, size=(out_channels, kernel_size, kernel_size))
        self.bias = cp.ones((out_channels,1,1))
        self.size.append([in_channels, out_channels])
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        # 根据pad进行填充
        if self.padding:
            matrix = cp.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        else:
            matrix = x
        # 开始滑动计算结果
        row = int((matrix.shape[1]+2*self.padding-self.kernel_size)/self.stride) + 1
        col = int((matrix.shape[2]+2*self.padding-self.kernel_size)/self.stride) + 1
        self.u = cp.zeros((matrix.shape[0], row, col)) # 用来存放结果

        for z in range(self.weight.shape[0]):
            for i in range(row):
                for k in range(col):
                    self.u[z, i, k] = cp.sum(self.weight[z] * matrix[:, i * self.stride:i * self.stride + self.kernel_size, k * self.stride:k * self.stride + self.kernel_size])

        self.y = self.u + self.bias
        self.uw_grad = matrix
        self.uy_grad = self.weight
        if self.activation_function:
            self.y = self.activation_function(self.u + self.bias)
        else:
            self.y = self.u + self.bias
        return self.y
