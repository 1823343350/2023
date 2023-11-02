import sys
sys.path.append('G:\学习文件\python学习\CODE\Deep') 
from typing import Any
import numpy as np
import torch

from Deep import MyModel
from Deep import ReLU
from Deep import MeanSquaredError, CrossEntropyLoss
from Deep import GradientDescentOptimizer
from Deep import Regularization

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

"""
多层感知机进行MINIST手写数据集的识别
"""

np.set_printoptions(precision = 4)

photo_nums = 64

transform = transforms.Compose([transforms.ToTensor()])

# 获取MNIST数据集
# windows
train_dataset = datasets.MNIST(root='G:\学习文件\python学习\CODE\data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='G:\学习文件\python学习\CODE\data', train=False, transform=transform, download=True)
# linux
# train_dataset = datasets.MNIST(root='/media/xc/学习/学习文件/python学习/CODE/data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='/media/xc/学习/学习文件/python学习/CODE/data', train=False, transform=transform, download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=photo_nums, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=photo_nums, shuffle=False)

loss_fn = CrossEntropyLoss()
reg = Regularization(0.001)

layer_dict = {
    'first': {
        'type': 'conv',
        'name': 'conv1',
        'in_channels': 1,
        'out_channels': 64,
        'kernel_size': 11, # 卷积核大小
        'stride': 1, # 步长
        'padding': 0,# 填充
        'activation': "ReLU"
    },
    'second': {
        'type': 'pool',
        'name': 'pool1',
        'kernel_size': 2,
        'stride': 1, # 步长
        'pool_function': 'Max',
    },
    'first2': {
        'type': 'conv',
        'name': 'conv2',
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': 3, # 卷积核大小
        'stride': 1, # 步长
        'padding': 0,# 填充
        'activation': "ReLU"
    },
    'second2': {
        'type': 'pool',
        'name': 'pool2',
        'kernel_size': 2,
        'stride': 1, # 步长
        'pool_function': 'Max',
    },
    'first3': {
        'type': 'conv',
        'name': 'conv2',
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': 2, # 卷积核大小
        'stride': 1, # 步长
        'padding': 0,# 填充
        'activation': "ReLU",
    },
    'second3': {
        'type': 'flatten',
        'name': 'flatten',
    },
    'linear1': {
        'type': 'linear',
        'name': 'linear1',
        'input_features_nums': 10816,
        'Number_of_neurons': 512,
        'activation': "Softmax",
        'loss_fn': loss_fn
    },
    'output': {
        'type': 'linear',
        'name': 'linear1',
        'input_features_nums': 512,
        'Number_of_neurons': 10,
        'activation': "Softmax",
        'loss_fn': loss_fn
    }
}
op = GradientDescentOptimizer(lr=0.001, max_iterations=1)
model = MyModel(layers_dict=layer_dict, optimizer = op, regularization=reg)
np.set_printoptions(precision = 4)
epochs = 10
for epoch in range(epochs):
    k = 0
    for batch in train_loader:
        inputs, labels = batch

        x = inputs.numpy()
        y = labels.numpy()

        y_one_hot = np.zeros((len(y), 10))

        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1

        model.fit(x, y_one_hot)

        k += 1
        if k >= 10:
            break
        running_loss = model.loss[-1]
        print(f"{64*10} photos, Epoch {epoch + 1}, Loss: {running_loss}")