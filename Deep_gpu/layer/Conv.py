import cupy as cp
from .model import CustomModel

class Conv(CustomModel):
    """
    创建一个普通的全连接层, 能够得到相应的输出
    
    input_features_nums: int, 每个神经元的参数量（输入通道数）
    Number_of_neurons: int, 神经元的数量（输出通道数）
    bias: float = 0, 偏置项

    """

    def __init__(self, input_features_nums: int, Number_of_neurons: int, bias: float = 0, loss_fn: any = None, activation_function: any = None) -> None:
        super().__init__('Linear', input_features_nums, Number_of_neurons, bias, loss_fn, activation_function)

    def __call__(self, x):
        if self.activation_function:
            self.loss = self.activation_function(cp.dot(self.weight, x) + self.bias)
        else:
            self.loss = cp.dot(self.weight, x) + self.bias
        return self.loss
