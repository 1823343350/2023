import torch
from torch import nn
import torch.nn.functional as F

class MyAlexNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(MyAlexNet, self).__init__(*args, **kwargs)
        # 定义第一个卷积层，输入通道为3,输出通道为96,卷积核的步长为4,大小为11*11，开启长度为2的边缘填充
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 使用ReLU作为激活函数
        self.ReLU = nn.ReLU()
        # 2d的最大池化层，大小为3*3,步长为2
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第二个卷积层，大小为5*5,步长为1，扩充边缘为2
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 第二个池化层，大小为3*3,步长为2
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第三个卷积层，大小为3*3,步长1,扩充边缘1
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 第四个卷积层，大小为3*3,步长1,扩充边缘1
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 第五个卷积层，大小为3*3,步长1,扩充边缘1
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 第三个池化层，大小为3*3,步长2，最后得到6*6*256输出
        self.s3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        # 全连接层
        self.f6 = nn.Linear(6*6*256, 4096)
        self.f7 = nn.Linear(4096, 4096)
        # SoftMax层
        self.f8 = nn.Linear(4096, 1000)
        self.f9 = nn.Linear(1000, 5)

    def forward(self, x):
        # 卷积和池化
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s3(x)
        # 全连接
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)

        return x
    
if __name__ == '__main__':
    # rand：返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，此处为四维张量
    x = torch.rand([1, 3, 224, 224])
    # 模型实例化
    model = MyAlexNet()
    y = model(x)
    print(y)
