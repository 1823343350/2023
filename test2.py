import numpy as np
import cupy as cp
# 创建一个3x3的矩阵
matrix = cp.array([
    [1, 2]
])

# 创建一个包含9个3x3矩阵的列表
b = cp.array([
    cp.array([[1, 1],
              [1, 1]]),
    cp.array([[2, 3],
              [4, 5]]),
])

a = cp.array([
    cp.array([[1, 1],
              [5, 2]]),
    cp.array([
              [2, 1],
              [1, 1]]),
])

print(a)
print(a.T)

