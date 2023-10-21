from math import sqrt
import numpy as np
from Deep.activation import ReLU

class CustomModel:
    """
    定义模型每一层的底层

    变量:
        name: 该层的类型\n
        weight: 该层的参数项\n
        bias: 该层的偏置项\n
        size: 该层的大小\n
        activation_function: 该层的激活函数\n
        loss_fn: 该层的损失函数\n

    函数:
        __init__: 初始化一层网络\n

    """
    
    def __init__(self, name: str, input_features_nums: int = 0, Number_of_neurons: int = 0, activation_function: any = None, bias: bool = True):
        """
        参数:
            name: 这一层的类型, 目前只支持全连接层\n
            input_features_nums: 每一个神经元的参数个数\n
            Number_of_neurons: 神经元数量\n
            bias: 偏置项\n
            activation_function: 该层激活函数\n
            weight: 自动生成随机的初始权重, 均匀分布在[-1, 1]之间\n
        """
        self.size = []
        self.name = name
        self.input_features_nums = input_features_nums
        self.Number_of_neurons = Number_of_neurons
        self.activation_function = activation_function
        
        # 每一层的两个权重梯度
        self.lwgrad: np.ndarray = None
        self.lbgrad: np.ndarray = None
        # 仅用于保存最后一层的损失
        self.loss: float = None
        # 反向传播时使用的参数--其他的梯度
        self.u: np.ndarray = None
        self.y: np.ndarray = None

        # 损失函数的导数
        self.ly_grad: np.ndarray = None
        # 激活函数的导数
        self.lc_grad: np.ndarray = None
        # wx对w的导数
        self.uw_grad: np.ndarray = None
        # wx+b对b的导数
        self.uy_grad: np.ndarray = None
        a = sqrt(6 / (input_features_nums + Number_of_neurons))
        # 生成随机的初始权重, 均匀分布在[-a, a]之间, 一共有Number_of_neurons个神经元，每个神经元有in_features这么多参数
        self.weight = np.random.uniform(low=-a, high=a, size=(Number_of_neurons, input_features_nums))
        self.bias = np.full((1, Number_of_neurons), 1.0)
        self.size.append([input_features_nums, Number_of_neurons])
