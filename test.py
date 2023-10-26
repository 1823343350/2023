from typing import Any
import cupy as cp
import torch
from DL import plot_graph
from Deep_gpu import MyModel
from Deep_gpu.activation import ReLU
from Deep_gpu.activation.loss import MeanSquaredError, CrossEntropyLoss
from Deep_gpu.optimizer.gdo import GradientDescentOptimizer

import torchvision
import torchvision.transforms as transforms

loss_fn = MeanSquaredError()
op = GradientDescentOptimizer(lr=0.1, max_iterations=1)

layer_dict = {
    'first': {
        'type': 'conv',
        'in_channels': 3,
        'out_channels': 3,
        'kernel_size': 2, # 卷积核大小11*11
        'stride': 1, # 步长
        'padding': 0,# 填充
        'activation': "ReLU"
    },
    'second': {
        'type': 'pool',
        'kernel_size': 2,
        'stride': 1, # 步长
        'pool_function': 'Max',
        'activation': "Flatten"
    },
    'output': {
        'type': 'linear',
        'input_features_nums': 4,
        'Number_of_neurons': 1,
        # 'activation': "Softmax",
        'loss_fn': loss_fn
    }
}

model = MyModel(layers_dict=layer_dict, optimizer = op)

a = cp.array([
    [[1,2,3,4],
     [4,5,6,5],
     [7,8,9,6],
     [1,3,4,5]],

    [[124,12,3,4],
     [4,5,6,5],
     [7,8,9,46],
     [1,33,4,5]],

    [[1,122,3,4],
     [4,5,6,5],
     [7,8,9,146],
     [1,323,4,5]],
])
y = cp.array([
    [1,2,3]
])
myconv = model.layers[0]
mypool = model.layers[1]
myline = model.layers[2]
ss1 = myconv(a)
ss2 = mypool(ss1)
print(ss1)
print(ss2)
ss3 = myline(ss2)

print(ss3.shape)

print(loss_fn(y, ss3))
