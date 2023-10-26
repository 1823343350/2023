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
            ly_grad = self._grad(layer, ly_grad)

    def _grad(self, layer, ly_grad):
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
