import numpy as np
import cupy as cp
from Deep_gpu.regularization import Regularization

# 创建一个示例矩阵
a = cp.array([
    [1,2],
    [1,4],
])

col_sum = cp.sum(a, axis=1)
col_sum = cp.array([col_sum]).T
print(col_sum.shape)
re  = 1*(col_sum-a)
print(re)
