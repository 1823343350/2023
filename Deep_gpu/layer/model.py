from math import sqrt
import cupy as cp
from Deep_gpu.activation import ReLU

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
    
    def __init__(self, name: str, input_features_nums: int = 0, Number_of_neurons: int = 0, activation_function: any = None, bias: bool = True, regularization = None):
        """
        参数:
            name: 这一层的类型, 目前只支持全连接层\n
            input_features_nums: 每一个神经元的参数个数\n
            Number_of_neurons: 神经元数量\n
            bias: 偏置项\n
            activation_function: 该层激活函数\n
            weight: 自动生成随机的初始权重, 均匀分布在[-1, 1]之间\n
            regularization: 正则化选项\n
        """
        self.size = []
        self.name = name
        self.input_features_nums = input_features_nums
        self.Number_of_neurons = Number_of_neurons
        self.activation_function = activation_function
        self.regularization = regularization
        
        # 每一层的两个权重梯度
        self.lwgrad: cp.ndarray = None
        self.lbgrad: cp.ndarray = None
        # 仅用于保存最后一层的损失
        self.loss: float = None
        # 反向传播时使用的参数--其他的梯度
        self.u: cp.ndarray = None
        self.y: cp.ndarray = None

        # 损失函数的导数
        self.ly_grad: cp.ndarray = None
        # 激活函数的导数
        self.lc_grad: cp.ndarray = None
        # wx对w的导数
        self.uw_grad: cp.ndarray = None
        # wx+b对b的导数
        self.uy_grad: cp.ndarray = None

        # 正则化减少项目
        self.re: cp.ndarray = None
        # self.lamda: float = regularization.lamda

        a = sqrt(6 / (input_features_nums + Number_of_neurons))
        # 生成随机的初始权重, 均匀分布在[-a, a]之间, 一共有Number_of_neurons个神经元，每个神经元有in_features这么多参数
        self.weight = cp.random.uniform(low=-a, high=a, size=(Number_of_neurons, input_features_nums))
        self.bias = cp.full((1, Number_of_neurons), 1.0)
        self.size.append([input_features_nums, Number_of_neurons])
