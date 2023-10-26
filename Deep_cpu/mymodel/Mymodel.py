from typing import Any
import numpy as np
from Deep_cpu import Linear, ReLU, MeanSquaredError, GradientDescentOptimizer, CustomModel, Conv
from Deep_cpu import Sigmoid, Softmax

class MyModel():

    def __init__(self, layers_dict: dict = None, optimizer: any = None) -> Any:
        self.x_data = None
        self.y_data = None
        self.optimizer = optimizer
        self.cur_loss = 0
        self.loss = []
        self.layers = []

        activation_functions = {
            "ReLU": ReLU(),
            "Sigmoid": Sigmoid(),
            "Softmax": Softmax(),
        }

        for layer_name, layer_info in layers_dict.items():
            layer_type = layer_info.get('type', None)
            in_features = layer_info.get('in_features', None)
            out_features = layer_info.get('out_features', None)
            activation_name = layer_info.get('activation', None)
            bias = layer_info.get('bias', 0)

            # 输出层获得损失函数
            if layer_name == 'output':
                self.loss_fn = layer_info.get('loss_fn', None)

            activation = activation_functions.get(activation_name)

            if layer_type == 'linear':
                layer = Linear(in_features, out_features, activation, bias)
            elif layer_type == 'conv':
                layer = Conv(
                    in_channels=layer_info['in_channels'],
                    out_channels=layer_info['out_channels'],
                    kernel_size=layer_info['kernel_size'],
                    activation=activation,  # 传递激活函数
                    stride=layer_info['stride'], 
                    padding=['padding'],
                    bias=bias
                )
            else:
                raise ValueError(f"未知类型的层: {layer_type}")
            self.layers.append(layer)

    def fit(self, x, y):
        self.x_data = x.T
        self.y_data = y.T
        for i in range(self.optimizer.max_iterations):
            self.forward(self.x_data, self.y_data)
            self.back_propagation()
            if self.loss[-1] < self.optimizer.epsilon:
                break
    
    def forward(self, x, y):
        # 根据输入进行前向传播
        for layer in self.layers:
            x = layer(x)
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
        x = x.T
        for layer in self.layers:
            x = layer(x)
        return x
