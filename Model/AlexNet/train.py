import torch
from torch import nn
from model import MyAlexNet
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = '/root/workspace/xucong/data/processed_dataset/flower_photos/train' # 数据集
ROOT_TEST = '/root/workspace/xucong/data/processed_dataset/flower_photos/val'

# 数据增强和归一化
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(2) # 在编号为1的GPU上运行
model = MyAlexNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        # 正向传播并且计算损失和准确率
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.true_divide(torch.sum(pred == y), output.shape[0])
        # cur_acc = torch.sum(y == pred)//output.shape[0]
        
        # print(f"pred is {pred} and y is {y}")
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        # 计算每一轮的误差和准确率
        loss += cur_loss.item()
        current += cur_acc.item()
        n += 1

    train_loss = loss / n
    train_acc = current / n

    print('train_loss=' + str(train_loss))
    # 计算训练的准确率
    print('train_acc=' + str(train_acc))
    return train_loss, train_acc


# 定义验证函数
def val(dataloader, model, loss_fn):
    loss, current, n = 0.0, 0.0, 0
    # eval()：如果模型中有Batch Normalization和Dropout，则不启用，以防改变权值
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.true_divide(torch.sum(pred == y), output.shape[0])
            # cur_acc = torch.sum(y == pred)//output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n+1
 
    val_loss = loss / n
    val_acc = current / n
    # 计算验证的错误率
    print('val_loss=' + str(val_loss))
    # 计算验证的准确率
    print('val_acc=' + str(val_acc))
    return val_loss, val_acc


# 定义画图函数
# 错误率
def matplot_loss(train_loss, val_loss):
    plt.clf()
    # 参数label = ''传入字符串类型的值，也就是图例的名称
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）
    plt.legend(loc='best')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的loss值对比图")
    plt.savefig('/root/workspace/xucong/Model/AlexNet/photo/loss_plot.png')


# 准确率
def matplot_acc(train_acc, val_acc):
    plt.clf()
    plt.plot(train_acc, label = 'train_acc')
    plt.plot(val_acc, label = 'val_acc')
    plt.legend(loc = 'best')
    plt.xlabel('acc')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的acc值对比图")
    plt.savefig('/root/workspace/xucong/Model/AlexNet/photo/acc_plot.png')


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

# 训练次数
epoch = 30
# 用于判断最佳模型
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t+1}\n----------")
    # 训练模型
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    # 验证模型
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
 
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)
 
    # 保存最好的模型权重
    if val_acc > min_acc:
        folder = '/root/workspace/xucong/Model/AlexNet/Weight file'
        # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
        if not os.path.exists(folder):
            # os.mkdir() 方法用于以数字权限模式创建目录
            os.mkdir('/root/workspace/xucong/Model/AlexNet/Weight file')
        min_acc = val_acc
        print(f"save best model,第{t+1}轮")
        # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
        torch.save(model.state_dict(), '/root/workspace/xucong/Model/AlexNet/Weight file/best_model.pth')

    # 保存最后一轮权重
    if t == epoch-1:
        torch.save(model.state_dict(), '/root/workspace/xucong/Model/AlexNet/Weight file/best_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)

print('done')
