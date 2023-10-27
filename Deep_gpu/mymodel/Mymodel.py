from typing import Any
from Deep_gpu import Linear, ReLU, MeanSquaredError, GradientDescentOptimizer, CustomModel, Conv
from Deep_gpu import Sigmoid, Softmax
from Deep_gpu.layer import Pool
from Deep_gpu.layer.Flatten import Flatten

class MyModel():

    def __init__(self, layers_dict: dict = None, optimizer: any = None, regularization: any = None) -> Any:
        self.x_data = None
        self.y_data = None
        self.optimizer = optimizer
        self.cur_loss = 0
        self.loss = []
        self.layers = []
        self.is_first_linear = True
        
        activation_functions = {
            "ReLU": ReLU(),
            "Sigmoid": Sigmoid(),
            "Softmax": Softmax(),
        }

        for layer_name, layer_info in layers_dict.items():
            layer_type = layer_info.get('type', None)
            
            activation_name = layer_info.get('activation', None)
            # 输出层获得损失函数
            if layer_name == 'output':
                self.loss_fn = layer_info.get('loss_fn', None)

            activation = activation_functions.get(activation_name)

            if layer_type == 'linear':
                layer = Linear(
                    name=layer_info.get('name', 'linear'),
                    input_features_nums = layer_info.get('input_features_nums', None),
                    Number_of_neurons = layer_info.get('Number_of_neurons', None),
                    activation_function = activation,
                    regularization = layer_info.get('regularization', None),
                )
            elif layer_type == 'conv':
                layer = Conv(
                    name=layer_info.get('name', 'conv'),
                    in_channels = layer_info.get('in_channels', None),
                    out_channels = layer_info.get('out_channels', None),
                    kernel_size = layer_info.get('kernel_size', None),
                    activation_function = activation,
                    stride = layer_info.get('stride', None),
                    padding = layer_info.get('padding', None),
                    bias = layer_info.get('bias', 0),
                )
            elif layer_type == 'pool':
                layer = Pool(
                    name=layer_info.get('name', 'pool'),
                    kernel_size = layer_info.get('kernel_size', 1),
                    stride = layer_info.get('stride', 1),
                    pool_function= layer_info.get('pool_function', None),
                    activation = activation, # 传递激活函数
                )
            elif layer_type == 'flatten':
                layer = Flatten(
                    name=layer_info.get('name', 'flatten'),
                )
            else:
                raise ValueError(f"未知类型的层: {layer_type}")
            self.layers.append(layer)

    def fit(self, x, y):
        self.x_data = x
        self.y_data = y.T
        for i in range(self.optimizer.max_iterations):
            self.forward(self.x_data, self.y_data)
            self.back_propagation()
            if self.loss[-1] < self.optimizer.epsilon:
                break
        self.is_first_linear = True

    def forward(self, x, y):
        # 根据输入进行前向传播
        for layer in self.layers:
            if layer.type == 'linear' and self.is_first_linear:
                x = x.T
                self.is_first_linear = False
            x = layer(x)
            # 计算该层正则化参数
            if layer.regularization:
                layer.regularization(layer)
        # 获得最后一层的损失的梯度, 以用于反向传播
        if self.loss_fn:
            self.layers[-1].loss, self.optimizer.root_grad = self.loss_fn(y, x)
            self.loss.append(self.layers[-1].loss)
        else:
            raise(f"请添加损失函数到模型里面, 检查layers_dict参数的output参数的值!")
        # 返回预测值
        return x

    def back_propagation(self):
        self.optimizer(self.layers)

    def predict(self, x):
        for layer in self.layers:
            if layer.type == 'linear' and self.is_first_linear:
                x = x.T
                self.is_first_linear = False
            x = layer(x)
        self.is_first_linear = True
        return x
