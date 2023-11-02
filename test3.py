import sys
sys.path.append('../..') 
from typing import Any
import numpy as cp
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
photo_nums = 64

transform = transforms.Compose([transforms.ToTensor()])

# 获取MNIST数据集
# windows
# train_dataset = datasets.MNIST(root='G:\学习文件\python学习\CODE\data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='G:\学习文件\python学习\CODE\data', train=False, transform=transform, download=True)
# linux
train_dataset = datasets.MNIST(root='/media/xc/学习/学习文件/python学习/CODE/data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='/media/xc/学习/学习文件/python学习/CODE/data', train=False, transform=transform, download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=photo_nums, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=photo_nums, shuffle=False)

loss_fn = CrossEntropyLoss()
reg = Regularization(0.001)

layer_dict = {
    'first': {
        'type': 'linear',
        'input_features_nums': 784,
        'Number_of_neurons': 128,
        'activation': "ReLU"
    },
    'output': {
        'type': 'linear',
        'input_features_nums': 128,
        'Number_of_neurons': 10,
        'activation': "Softmax",
        'loss_fn': loss_fn
    }
}

op = GradientDescentOptimizer(lr=0.001, max_iterations=1)
model = MyModel(layers_dict=layer_dict, optimizer = op, regularization=reg)

epochs = 10
for epoch in range(epochs):
    k = 0
    for batch in train_loader:
        inputs, labels = batch

        x = inputs.numpy()
        y = labels.numpy()
        x = x.reshape(photo_nums, -1)
        x = cp.asarray(x)
        y_one_hot = cp.zeros((len(y), 10))
        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1

        if x.shape[1] == 784:
            model.fit(x, y_one_hot)

        k += 1  
        if k >= 100:
            break
    running_loss = model.loss[-1]
    print(f"{64*500} photos, Epoch {epoch + 1}, Loss: {running_loss}")