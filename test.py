from typing import Any
import numpy as np
import torch
from DL import plot_graph
from Deep_cpu import MyModel
from Deep_cpu.activation import ReLU
from Deep_cpu.activation.loss import MeanSquaredError, CrossEntropyLoss
from Deep_cpu.optimizer.gdo import GradientDescentOptimizer

import torchvision
import torchvision.transforms as transforms

photo_nums = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=photo_nums, shuffle=True)

loss_fn = CrossEntropyLoss()
op = GradientDescentOptimizer(lr=0.1, max_iterations=1)

layer_dict = {
    'first': {
        'type': 'linear',
        'in_features': 784,
        'out_features': 128,
        'activation': "ReLU"
    },
    'output': {
        'type': 'linear',
        'in_features': 128,
        'out_features': 10,
        'activation': "Softmax",
        'loss_fn': loss_fn
    }
}

model = MyModel(layers_dict=layer_dict, optimizer = op)

# 数据预处理--One-Hot Encoding
k = 1
aaas2 = []
issss = True
for i in range(10):
    j = 0
    for batch in train_loader:
        inputs, labels = batch

        x = inputs.numpy()
        y = labels.numpy()
        x = x.reshape(photo_nums, -1)
        y_one_hot = np.zeros((len(y), 10))

        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1

        while issss:
            aaas1 = x
            aaas2.append(y_one_hot)
            issss = False

        if x.shape[1] == 784:
            model.fit(x, y_one_hot)
            # print(f"Epoch {i + 1}, Loss: {model.loss[-1] / photo_nums}")
        if j > 10:
            break
        j += 1
    k += 1
    if k > 10:
        break

result = model.predict(aaas1)
max_values = np.max(result, axis=0)
max_indices = np.argmax(result, axis=0)
for row, (max_value, max_index) in enumerate(zip(max_values, max_indices)):
    print(f"Row {row}: Max Value = {max_value}, Index = {max_index}")
print(result[::, 0])
print()