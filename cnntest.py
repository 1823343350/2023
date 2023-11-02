import sys
sys.path.append('../..') 
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
        'out_channels': 10,
        'kernel_size': 5, # 卷积核大小
        'stride': 1, # 步长
        'padding': 0,# 填充
        'activation': "ReLU"
    },
    'second': {
        'type': 'pool',
        'name': 'pool1',
        'kernel_size': 2,
        'stride': 2, # 步长
        'pool_function': 'Max',
    },
    'first2': {
        'type': 'conv',
        'name': 'conv2',
        'in_channels': 10,
        'out_channels': 20,
        'kernel_size': 5, # 卷积核大小
        'stride': 1, # 步长
        'padding': 0,# 填充
        'activation': "ReLU"
    },
    'second2': {
        'type': 'pool',
        'name': 'pool2',
        'kernel_size': 2,
        'stride': 2, # 步长
        'pool_function': 'Max',
    },
    'second3': {
        'type': 'flatten',
        'name': 'flatten',
    },
    'linear': {
        'type': 'linear',
        'name': 'linear1',
        'input_features_nums': 320,
        'Number_of_neurons': 50,
        'activation': "ReLU"
    },
    'output': {
        'type': 'linear',
        'name': 'linear2',
        'input_features_nums': 50,
        'Number_of_neurons': 10,
        'activation': "Softmax",
        'loss_fn': loss_fn
    }
}
op = GradientDescentOptimizer(lr=0.000001, max_iterations=1)
model = MyModel(layers_dict=layer_dict, optimizer = op, regularization=reg)
np.set_printoptions(precision = 4)
epochs = 1
for epoch in range(epochs):
    k = 0
    times = 0
    for batch in train_loader:
        inputs, labels = batch

        x = inputs.numpy()
        y = labels.numpy()

        y_one_hot = np.zeros((len(y), 10))

        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1

        model.fit(x, y_one_hot)

        k += 1
        if k >= 500:
            break
        running_loss = model.loss[-1]
        if len(model.loss) >=10 and model.loss[-1] - model.loss[-9] < 1e-4:
            times += 1
            if times%3==0:
                model.optimizer.learning_rate = model.optimizer.learning_rate/2
                times = 0
        else:
            times = 0
        print(f"{64*k} photos, Epoch {epoch + 1}, Loss: {running_loss}")