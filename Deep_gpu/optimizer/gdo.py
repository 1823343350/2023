import cupy as cp


class GradientDescentOptimizer:
    """
    梯度下降优化器

    例如: 
        x = GradientDescentOptimizer()\n
        x.minimize(): 梯度下降优化参数 (非神经网络时使用)---暂时未实现\n
        x.step(): 反向传播 (神经网络时使用)

    参数:
        - learning_rate (float): 学习率，控制每次参数更新的步长。
        - max_iterations (int): 最大迭代次数，控制优化过程的结束。
        - epsilon (float): 停止条件, 当梯度的范数小于epsilon时停止优化。

    返回:
    float: 待定
    """

    def __init__(self, lr: float = 0.01, max_iterations: int = 1000, epsilon: float = 1e-5):
        self.root_grad = None
        self.learning_rate = lr  # 学习率
        self.max_iterations = max_iterations  # 最大迭代次数
        self.epsilon = epsilon  # 停止条件，当梯度的范数小于epsilon时停止

    def __call__(self, layers):
        # 每一层的梯度清零
        self.zero_grad(layers)
        # 开始计算梯度
        self.get_grad(layers)
        # 更新梯度
        self.step(layers)

    def zero_grad(self, layers):
        """
        清零所有参数的梯度。
        """
        for layer in layers:
            # if layer.ly_grad is not None:
            layer.ly_grad = None
            layer.lwgrad = None
            layer.lbgrad = None
            layer.lc_grad = None

    def get_grad(self, layers):
        ly_grad = self.root_grad
        for layer in layers[::-1]:
            # 更新每一层的梯度
            if layer.type == 'linear':
                ly_grad = self._grad_linear(layer, ly_grad)
            elif layer.type == 'conv':
                ly_grad = self._grad_conv(layer, ly_grad)
            elif layer.type == 'pool':
                ly_grad = self._grad_pool(layer, ly_grad)
            elif layer.type == 'flatten':
                ly_grad = self._grad_flatten(layer, ly_grad)

    def _grad_flatten(self, layer, lygrad):
        layer.lwgrad = 0
        layer.lbgrad = 0
        return layer.derivative(lygrad)

    def _grad_conv(self, layer, ly_grad):

        layer.lwgrad = cp.zeros_like(layer.weight)
        # 得到参数w的梯度

        par_x = cp.array(layer.temp)
        for i in range(ly_grad.shape[0]):
            result = 0
            for k in range(par_x.shape[0]):
                x = k // ly_grad.shape[1]
                y = k  % ly_grad.shape[2]
                result += par_x[k] * ly_grad[i][x][y]
            layer.lwgrad[i] = result

        # 得到参数b的梯度
        layer.lbgrad = cp.sum(ly_grad, axis=(1,2))
        layer.lbgrad = layer.lbgrad.reshape(ly_grad.shape[0],1,1)

        # 得到下一层的梯度
        next_lygrad = cp.zeros_like(layer.uw_grad)
        next_lygrad = next_lygrad.astype(float)
        row = int((layer.uw_grad.shape[1]-layer.kernel_size)/layer.stride) + 1
        col = int((layer.uw_grad.shape[2]-layer.kernel_size)/layer.stride) + 1

        for i in range(row):
            for k in range(col):
                for z in range(layer.weight.shape[0]):
                    next_lygrad[0, i*layer.stride:i*layer.stride+layer.kernel_size, k*layer.stride:k*layer.stride+layer.kernel_size] += layer.weight[z] * ly_grad[z,i,k]

        for i in range(1, next_lygrad.shape[0]):
            next_lygrad[i] = next_lygrad[0]

        layer.temp = []
        return next_lygrad

    def _grad_pool(self, layer, ly_grad):

        layer.lwgrad = 0
        layer.lbgrad = 0

        # 激活函数还原梯度, 比如矩阵打平的还原
        if layer.activation_function:
            ly_grad = layer.activation_function.derivative(ly_grad)

        # TODO 应该放到pool层里面
        if layer.is_pool:
           
            # 逆向池化, 还原梯度到对应位置
            root_grad = cp.zeros_like(layer.uw_grad)
            row = int((layer.uw_grad.shape[1]-layer.kernel_size)/layer.stride) + 1
            col = int((layer.uw_grad.shape[2]-layer.kernel_size)/layer.stride) + 1
        
            tempx = 0
            tempy = 0
            for i in range(row):
                for k in range(col):
                    aera = layer.uw_grad[:, i * layer.stride:i * layer.stride + layer.kernel_size, k * layer.stride :k * layer.stride  + layer.kernel_size]
                    # 找到矩阵中最大的值
                    max_value_pos = cp.argmax(aera, axis=(1,2))
                    x = i * layer.stride + max_value_pos//layer.kernel_size
                    y = k * layer.stride + max_value_pos%layer.kernel_size
                    _xy = cp.array([x, y]).T
                    z: int = 0
                    # 给对应的最大值所在位置的值赋予梯度, 其余位置梯度为0
                    for c,d in _xy:
                        root_grad[z][int(c)][int(d)] += ly_grad[z][tempx][tempy]
                        z += 1
                    tempy += 1
                    if tempy >= ly_grad.shape[1]:
                        tempx += 1
                        tempy = 0
                        
        else:
            root_grad = ly_grad

        return root_grad

    def _grad_linear(self, layer, ly_grad):
        # 损失函数的导数
        layer.ly_grad = ly_grad

        # 如果存在激活函数, 求激活函数的偏导数
        if layer.activation_function:
            layer.lc_grad = layer.activation_function.derivative(
                layer.u + layer.bias.T)
        else:
            layer.lc_grad = 1

        # 加入激活函数的导数
        if layer.activation_function == 'Softmax':
            part = []
            for dim in range(layer.lc_grad.shape[0]):
                part.append(
                    cp.sum(layer.ly_grad[::, dim] * layer.lc_grad[dim], axis=1))
            layer.ly_grad = cp.stack(part, axis=1)
        else:
            layer.ly_grad = layer.ly_grad * layer.lc_grad

        # 损失函数对参数的导数
        layer.lwgrad = (cp.dot(layer.uw_grad, layer.ly_grad.T)/(layer.uw_grad.shape[1])).T
        layer.lbgrad = cp.sum(layer.ly_grad, axis=1)/(layer.uw_grad.shape[1])
        # 损失函数对下一层的导数
        next_lygrad = cp.dot(layer.ly_grad.T, layer.weight).T

        return next_lygrad

    def step(self, layers):
        """
        更新所有参数的梯度。
        """
        for layer in layers:
            layer.bias -= self.learning_rate * layer.lbgrad
            if layer.regularization:
                layer.weight -= self.learning_rate * layer.lwgrad + layer.re
            else:
                layer.weight -= self.learning_rate * layer.lwgrad
