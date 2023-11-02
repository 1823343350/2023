import numpy as np
from .model import CustomModel
np.set_printoptions(precision = 4)
np.seterr(divide='ignore',invalid='ignore') # 忽略除以0的警告
class Pool(CustomModel):

    def __init__(self,name: str  = 'pool',kernel_size: int = 1, stride: int = 1,  pool_function = None, activation = None) -> None:
        super().__init__(name=name, activation_function = activation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_function = pool_function
        self.weight = 0
        self.bias = 0
        self.is_pool = False
        self.type = 'pool'
        # 用来保存反向传播的矩阵, 方便反向传播不在逆池化
        self.same_t = None

    def __call__(self, x):
        # 保存输入矩阵,反向传播使用
        self.uw_grad = x
        self.u = x
        self.bias = 0

        # 开始滑动计算结果
        if x.shape[2]>=self.kernel_size and x.shape[3]>=self.kernel_size:
            # 加上池化过的标志
            self.same_t = np.zeros_like(x)
            self.is_pool = True
            row = int((x.shape[2]-self.kernel_size)/self.stride) + 1
            col = int((x.shape[3]-self.kernel_size)/self.stride) + 1

            if self.pool_function == 'Max':
                self.u = np.zeros((x.shape[0], x.shape[1], row, col)) # 用来存放结果
                for r in range(x.shape[0]):
                    for i in range(row):
                        for k in range(col):
                            # 得到当前窗口
                            aera = x[r, :, i * self.stride:i * self.stride + self.kernel_size, k * self.stride:k * self.stride + self.kernel_size]
                            # 取得窗口最大值
                            max_value = np.max(aera, axis=(1,2))
                            self.u[r, :, i, k]  = max_value
                            # 记录采样点的位置, 变成矩阵的形式
                            aera += 1e-99 # 防止除以0
                            aera = aera/(max_value.reshape(max_value.shape[0],1,1))
                            self.same_t[r, :, i * self.stride:i * self.stride + self.kernel_size, k * self.stride:k * self.stride + self.kernel_size] += np.where(aera < 1, 0, aera)
            self.same_t = (self.same_t > 0).astype(int)
        else:
            self.same_t = 1                    

        if self.activation_function:
            self.y = self.activation_function(self.u + self.bias)
        else:
            self.y = self.u + self.bias
        return self.y
