from typing import Any
import numpy as np

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

    def __call__(self, actual_values: np.ndarray, predicted_values: np.ndarray):
        # 防止概率值为0
        epsilon = 1e-2
        predicted_values = np.maximum(epsilon, predicted_values)
        # 计算交叉熵损失
        cross_entropy = -(actual_values * np.log(predicted_values))
        # 计算这次前向传播的损失
        loss = np.sum(cross_entropy) / actual_values.shape[0]
        # 计算这次前向传播最后一层的梯度
        self.root_grad = -(actual_values/predicted_values)/actual_values.shape[0]
        return loss, self.root_grad


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

    def __init__(self):
        pass

    def __call__(self, actual_values: np.ndarray, predicted_values: np.ndarray):
        if actual_values.shape == predicted_values.shape:
            # 计算这次前向传播最后一层的梯度
            self.root_grad = predicted_values - actual_values
            # 计算这次前向传播的损失
            differences = actual_values - predicted_values
            squared_differences = (np.power(differences, 2)) / 2
            mse = squared_differences.mean()
            return mse, self.root_grad
        else:
            raise TypeError(f"实际值矩阵大小{actual_values.shape }与预测值矩阵大小{predicted_values.shape}不匹配, 无法计算均方误差!")
