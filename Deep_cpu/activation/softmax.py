import numpy as np

class Softmax:

    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = x.T
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        result = e_x / np.sum(e_x,axis=-1, keepdims=True)
        return result.T

    def __eq__(self, other) -> bool:
        if other == 'Softmax':
            return True
        else:
            return False

    def derivative(self, x):
        # 返回一个n2*n1*n1的矩阵, n2表示有多少个样本, n1*n1表示单个样本的偏导
        # 其中每一列表示一个输出对所有输入按顺序的偏导
        # 因为每一个输出都对输入有一个偏导, 所以用一列来表示这个输出对每一个输入的偏导
        output = self.__call__(x)

        result = []

        for col in range(output.shape[1]):
            y_sample = output[::, col]

            # 获取向量的长度
            vector_length = len(y_sample)
            identity_matrix = np.eye(vector_length)
            # 使用np.tile将向量变成一个矩阵
            matrix = np.tile(y_sample, (vector_length, 1))
            result.append(identity_matrix - matrix.T)

        big_matrix = np.stack(result)

        return big_matrix
