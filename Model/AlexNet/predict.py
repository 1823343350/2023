import torch
from model import MyAlexNet
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
 
ROOT_TRAIN = '/root/workspace/xucong/data/processed_dataset/flower_photos/train'
ROOT_TEST = '/root/workspace/xucong/data/processed_dataset/flower_photos/val'
 
# 将图像的像素值归一化到[-1,1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
 
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
 
# 加载训练数据集
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(2)
# 模型实例化，将模型转到device
model = MyAlexNet().to(device)
 
# 加载train.py里训练好的模型
model.load_state_dict(torch.load(r'/root/workspace/xucong/Model/AlexNet/Weight file/best_model.pth'))
 
# 结果类型
classes = [
    "daisy",
    "dandelion",
    "roese",
    "sunflowers",
    "tulips"
]
 
# 把Tensor转化为图片，方便可视化
show = ToPILImage()
 
# 进入验证阶段
model.eval()
cur_acc = 0
times = 10 # 32张图片一次
show = False
for batch, (x, y) in enumerate(val_dataloader):
    if batch >= times:
        print(f"{predict_photo.shape[0]*batch}张图片的准确率为:{torch.true_divide(cur_acc, predict_photo.shape[0]*batch)}")
        break
    else:
        image, y = x.to(device), y.to(device)
        with torch.no_grad():
            predict_photo = model(image)
            _, pred = torch.max(predict_photo, axis=1)
            if show:
                for val_class, true in zip(pred, y):
                    print(f'predicted:"{classes[val_class]}", actual:"{classes[true.int()]}"')
            cur_acc += torch.sum(pred == y)

