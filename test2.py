
from numpy import convolve
import torch
import cupy as cp

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

inp = 3
outp = 2
s = 3

matrix = cp.array([
    [[1,1,1],
     [1,1,1],
     [1,1,1],],
    [[1,0,0],
     [0,0,0],
     [0,0,0],],
])
staa = 1
pad = 0
# a = cp.pad(a, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)


row = int((a.shape[1]+2*pad-s)/staa) + 1
col = int((a.shape[2]+2*pad-s)/staa) + 1

weight = cp.random.uniform(-1, 1, size=(2, 2, 2))

print(weight)

# result = cp.zeros((a.shape[0], row, col))

# k = matrix * a[:, 0:3, 0:3]
# print(k)
# print(cp.sum(k))

# for i in range(row):
#     for k in range(col):
#         result[:, i, k] = cp.max(a[:, i * staa:i * staa + s, k * staa:k * staa + s], axis=(1,2))

# print(result)

# print(result.reshape(result.shape[0], -1))


