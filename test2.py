import torch
import numpy as np
import time
# 卷积示例
a = np.random.uniform(low=-1, high=1, size=(16, 3, 4, 4))
b = np.random.uniform(low=-1, high=1, size=(10,5,3,3))

# x1 = np.zeros((a.shape[0], b.shape[0], 2, 2))
# bais = np.ones((10,1,1))

# start_time1 = time.time()

# for i in range(a.shape[0]):
#     for z in range(2):
#         for r in range(2):
#             x1[i, : , z, r] = np.sum(b * a[i,:,0+z:3+z,0+r:3+r], axis=(-3, -2, -1))

# end_time1 = time.time()
# time_diff1 = end_time1 - start_time1

# start_time2 = time.time()
# 转换为PyTorch张量, pytorch验证卷积操作的正确性
# a = torch.from_numpy(a).double()
# b = torch.from_numpy(b).double()

# 使用PyTorch的卷积操作
# x = torch.nn.functional.conv2d(a, b, stride=1, padding=0)
# end_time2 = time.time()
# time_diff2 = end_time2 - start_time2
# print("代码块1的运行时间:", time_diff1)
# print("代码块2的运行时间:", time_diff2)
# 如果需要转换为NumPy数组：
# x_np = x.detach().numpy()

# print(x_np)
# print(x_np.shape)
# print(x1[0])
# print(x_np[0])
# print((x1 - x_np)[0]) # 结果全部接近0, 所以结果正确

# 卷积层到全连接层, 打平操作的还原
# result = x
# tt = x.reshape(x.shape[0], -1)
# root_matrix = tt.reshape(16, 10, 2, 2)
# print(root_matrix)
# root_matrix = tt.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

# 梯度反向传播示例
# print(x.shape)
# print((x[0, : , 0, 0].reshape(x.shape[1],1,1)*x[0,:,0:2,0:2]).shape)
# for i in range(a.shape[0]):
#     for z in range(2):
#         for r in range(2):
#             k = np.sum(b * a[i,:,0+z:3+z,0+r:3+r], axis=(-3, -2, -1))
#             x[i, : , z, r] = k


# 对b求导示例
# zz = np.sum(x, axis=0)
# zz = np.sum(zz, axis=(1,2)).reshape(x.shape[1],1,1)

# print(zz)
# print(zz.shape)

# print(x.shape)


# qq = np.ones((10,1,1,1))
# print((np.sum(b, axis=0).shape))

# 测试池化
# print(a.shape)
# u = np.zeros((16,3,3,3))
# print(u[0, :, 0, 0].shape)
# for r in range(a.shape[0]):
#     for i in range(3):
#         for k in range(3):
#             u[r, :, i, k] = np.max(a[r, :, i:i*1+2, k:k*1+2], axis=(1,2))
# print(u.shape)

# 逆池化过程传播梯度
# print(a.shape)
# k = a[0, : , 0, 0].reshape(1, a.shape[1] , 1, 1)

# d=[c.argmax() for c in ab]
# d = np.array(d)
# print(d)
# x = d//2
# y = d%2
# _xy = np.array([x, y]).T
# z: int = 0
# # 给对应的最大值所在位置的值赋予梯度, 其余位置梯度为0
# root_grad = np.zeros_like(ab)
# for c,d in _xy:
#     print(c,d)


# a = np.random.uniform(low=-1, high=1, size=(64, 3, 3, 3))
# b = np.random.uniform(low=-1, high=1, size=(64, 1, 1, 1))
# c = np.sum(a*b, axis=0)
# print(c.shape)

# a = np.ones((2,3,2,2))

# a[0][0][1][0] = -999
# a[0][2][0][1] = 56

# c = (a > 0).astype(int)

# print(c)

a = np.ones((2,1,2,2))

b = a*2
c = a*2

print(b*c)

