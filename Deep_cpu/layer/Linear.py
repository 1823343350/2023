import numpy as np

from Deep_gpu.activation import ReLU
from .model import CustomModel

class Linear(CustomModel):
    """
    创建一个普通的全连接层, 能够得到相应的输出
    
    input_features_nums: int, 每个神经元的参数量（输入通道数）
    Number_of_neurons: int, 神经元的数量（输出通道数）
    bias: float = 0, 偏置项

    """

    def __init__(self, input_features_nums: int, Number_of_neurons: int, activation_function: any = None, bias: float = 0) -> None:
        super().__init__('Linear', activation_function)
        # 生成随机的初始权重, 均匀分布在[-a, a]之间, 一共有Number_of_neurons个神经元，每个神经元有in_features这么多参数
        a = np.sqrt(6 / (input_features_nums + Number_of_neurons))
        self.weight = np.random.uniform(low=-a, high=a, size=(Number_of_neurons, input_features_nums))
        self.bias = np.full((1, Number_of_neurons), 1.0)
        self.size.append([input_features_nums, Number_of_neurons])

    def __call__(self, x):
        self.u = np.dot(self.weight, x)
        self.uw_grad = x
        self.uy_grad = self.weight
        if self.activation_function:
            self.y = self.activation_function(self.u + self.bias.T)
        else:
            self.y = self.u + self.bias.T
        return self.y
    
    def __eq__(self, other):
        if other == "Linear":
            return True
        else:
            return False
