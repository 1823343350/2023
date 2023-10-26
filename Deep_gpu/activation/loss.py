from typing import Any
import cupy as cp

class CrossEntropyLoss:
    """
    计算 交叉熵损失

    例如: 
        x = CrossEntropyLoss()\n
        loss = x(predicted, target)

    返回: float: 交叉熵损失值

    参数:\n
    predicted (numpy.ndarray): 模型的预测概率分布 (例如softmax输出)\n
    target (numpy.ndarray): 实际的标签分布


    """
    def __init__(self):
        pass

    def __call__(self, actual_values: cp.ndarray, predicted_values: cp.ndarray, Regularization: str = None):
        # 防止概率值为0
        # predicted_values = predicted_values.T
        epsilon = 1e-2
        predicted_values = cp.maximum(epsilon, predicted_values)
        # 计算交叉熵损失
        cross_entropy = -(actual_values * cp.log(predicted_values))
        # 计算这次前向传播的损失
        loss = cp.sum(cross_entropy) / actual_values.shape[0]
        # 计算这次前向传播最后一层的梯度
        self.root_grad = -(actual_values/predicted_values)/actual_values.shape[0]
        return loss, self.root_grad

    def L1(self):
        pass


class MeanSquaredError:
    """
    计算 均方误差

    例如: 
        x = MeanSquaredError()\n
        loss = x(predicted_values, actual_values)
    
    返回:
    float: 均方误差

    参数:\n
    actual_values (numpy.ndarray): 实际值\n
    predicted_values (numpy.ndarray): 预测值

    """

    def __init__(self, Regularization: dict = None):
        pass

    def __call__(self, actual_values: cp.ndarray, predicted_values: cp.ndarray):
        # predicted_values = predicted_values.T
        if actual_values.shape == predicted_values.shape:
            differences = actual_values - predicted_values
            squared_differences = (cp.power(differences, 2)) / 2
            mse = squared_differences.mean()
            # 计算这次前向传播最后一层的梯度
            self.root_grad = predicted_values - actual_values
            # 计算这次前向传播的损失
            return mse, self.root_grad
        else:
            raise TypeError(f"实际值矩阵大小{actual_values.shape }与预测值矩阵大小{predicted_values.shape}不匹配, 无法计算均方误差!")
