import numpy as np
from .model import CustomModel

class Conv(CustomModel):
    """
    创建一个普通的全连接层, 能够得到相应的输出
    
    input_features_nums: int, 每个神经元的参数量（输入通道数）
    Number_of_neurons: int, 神经元的数量（输出通道数）
    bias: float = 0, 偏置项

    """

    def __init__(self, in_channels: int, out_channels: int, bias: float = 0, kernel_size: int = None, activation_function: any = None, stride: int = None, padding: int = None) -> None:
        super().__init__('Conv', activation_function)

        self.weight = np.random.uniform(low=-1, high=1, size=(10, 2, 3))

    def __call__(self, x):
        if self.activation_function:
            self.loss = self.activation_function(np.dot(self.weight, x) + self.bias)
        else:
            self.loss = np.dot(self.weight, x) + self.bias
        return self.loss
