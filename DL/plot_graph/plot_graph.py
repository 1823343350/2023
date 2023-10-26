import matplotlib.pyplot as plt
import numpy as np

def plot_graph(matrix_data: 'np.ndarray', save_path: str ="", x_label: str ="", y_label: str ="", line_labels: list =None, graph_title: str ="", graph_show: bool = False):
    """
    绘制矩阵数据的折线图。

    参数:
    - matrix_data (numpy.ndarray): 包含要绘制的数据的二维矩阵，每一行表示一组数据。
    - save_path (str, 可选): 图形保存的文件路径, 最后一个名称为保存的文件名
    - x_label (str, 可选): X轴的标签, 默认为空字符串。
    - y_label (str, 可选): Y轴的标签, 默认为空字符串。
    - line_labels (list, 可选): 数据线的标签列表，与矩阵数据的行数相匹配。如果未提供，则不显示图例。
    - graph_title (str, 可选): 图形的标题，默认为空字符串。
    - graph_show (bool, 可选): 是否显示绘图, 默认为False

    返回值:
    无，函数将绘制图形并保存到指定路径。

    示例用法:
    data_matrix = np.array([[1, 2, 3, 4, 5],
                           [2, 3, 4, 5, 6],
                           [3, 4, 5, 6, 7]])
    line_labels = ["数据线1", "数据线2", "数据线3"]

    # 可以选择性地传入参数
    plot_matrix_data(data_matrix, x_label="X轴标签", y_label="Y轴标签", line_labels=line_labels)
    """

    # 创建图形并且判断输入数据类型是否正确
    if isinstance(matrix_data, np.ndarray):
        for i in range(len(matrix_data)):
            plt.plot(matrix_data[i], label=line_labels[i] if line_labels else None)
    else:
        raise TypeError("输入数据的类型不匹配, 应该输入一个numpy.ndarray的数据")
    
    # 判断图例的数量是否和数据类别的数量匹配
    if line_labels and len(line_labels) != len(matrix_data):
        raise ValueError("line_labels参数的数量应与矩阵数据的行数匹配")

    # 添加标签和标题（如果提供）
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if graph_title:
        plt.title(graph_title)

    # 添加图例
    if line_labels:
        plt.legend()

    # 保存图形到指定位置
    if save_path:
        plt.savefig(save_path)

    # 显示图形（可选）
    if graph_show:
        plt.show()
