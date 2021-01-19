import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

logPath = '.\\log.json'
bestPath = '.\\nowTheBest.json'
# 训练
# isTrain = True
# isValidation = False
# 测试
isTrain = False
isValidation = True
imageSize = 28

batchSize = 15
epoch = 20
r = 0.001
decay = 0
p = 0.5  # dropout层丢弃神经元的概率
print('batchSize: ' + str(batchSize) + ', epoch: ' + str(epoch) + ', r: ' + str(r) + ', decay: ' + str(decay)
      + ', p: ' + str(p))


# 写训练图片路径到txt文件
def createTrainTxt():
    with open('..\\data\\train.txt', 'r+') as f:
        for i in range(12):
            for j in range(510):
                f.write("..\\data\\train\\%d\\%d.bmp" % (i + 1, j + 1) + "\n")


# 写测试图片路径到txt文件
def createTestTxt():
    with open('..\\data\\test.txt', 'r+') as f:
        for i in range(12):
            for j in range(511, 621):
                f.write("..\\data\\test\\%d\\%d.bmp" % (i + 1, j) + "\n")


def createdValidationTxt():
    with open('..\\validation\\validation.txt', 'r+') as f:
        for i in range(1, 1801):  # 面试时为1801
            f.write("..\\validation\\%d.bmp" % i + "\n")


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        super(MyDataset, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，在本例中是数字
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                images.append(line)
                # labels.append(int(line.split('\\')[-2]) - 1)  # 面试时注释
                labels.append(0)
        self.images = images
        self.labels = labels
        self.transforms = transform  # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # 用PIL.Image读取图像
        label = self.labels[index]
        image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 卷积层1，输入通道个数为1，输出通道个数为6，卷积核大小为3 * 3，步长默认为1
        self.bn1 = nn.BatchNorm2d(6)  # bn1层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，两个卷积层共用一层，max pooling窗口大小为2 * 2 ，max pooling窗口移动的步长为2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层2，输入通道个数为6，输出通道个数为16，卷积核大小为5 * 5，步长默认为1
        self.bn2 = nn.BatchNorm2d(16)  # bn2层
        self.fc1 = nn.Linear(256, 120)  # 全连接层1，输入向量为256维（16 * 16），输出为120维
        self.fc2 = nn.Linear(120, 60)  # 全连接层2，输入向量为120维，输出为60维
        self.fc3 = nn.Linear(60, 12)  # 全连接层3，输入向量为60维，输出为12维
        self.dropout = nn.Dropout(p=p)  # dropout层

    # 每次运行时都会执行forward
    def forward(self, x):
        """
        卷积，激活，池化，卷积，激活，池化，全连接，dropout层，激活，全连接，dropout层，激活，全连接
        """
        # x = self.pool(torch.sigmoid(self.conv1(x)))
        # x = self.pool(torch.sigmoid(self.conv2(x)))
        # x = x.view(-1, 256)  # 将前面多维度的tensor展平成一维
        # x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        # x = self.fc3(x)

        # x = self.pool(torch.tanh(self.conv1(x)))
        # x = self.pool(torch.tanh(self.conv2(x)))
        # x = x.view(-1, 256)  # 将前面多维度的tensor展平成一维
        # x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        # x = self.fc3(x)

        # x = self.pool(Func.leaky_relu(self.conv1(x)))
        # x = self.pool(Func.leaky_relu(self.conv2(x)))
        # x = x.view(-1, 256)  # 将前面多维度的tensor展平成一维
        # x = Func.leaky_relu(self.fc1(x))
        # x = Func.leaky_relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = Func.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = Func.relu(x)
        x = self.pool(x)
        x = x.view(-1, 256)  # 将前面多维度的tensor展平成一维
        x = self.fc1(x)
        x = self.dropout(x)
        x = Func.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = Func.relu(x)
        x = self.fc3(x)
        return x


# 训练模型
def train():
    transform = transforms.Compose(  # Compose()类会将transforms列表里面的transform操作进行遍历
        [
            transforms.Resize((imageSize, imageSize)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    )

    trainSet = MyDataset('..\\data\\train.txt', transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)  # 将数据按batchSize封装成Tensor
    device = torch.device('cpu')
    model = net()
    model.to(device)
    model.train()

    lossFunction = nn.CrossEntropyLoss()  # 损失函数为交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=r, weight_decay=decay)
    tmpEpoch = 0
    while tmpEpoch < epoch:
        runningLoss = 0.0
        # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, data in enumerate(trainLoader):
            inputs, labels = data[0].to(device), data[1].to(device)  # 放入数据和标签
            # 注意inputs是一个四维向量[N,C,L,W], N:batch_size; C:channel number; L:img length W:img weight
            optimizer.zero_grad()  # 将模型参数梯度初始化为0
            outs = model(inputs)  # 前向传播计算预测值
            loss = lossFunction(outs, labels)  # 计算当前损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            runningLoss += loss.item()  # 累加每个batch里每个样本的loss
            # 每个batch打印loss
            print('epoch: %d, batch: %d, loss为: %f' % (tmpEpoch + 1, i + 1, runningLoss))
            runningLoss = 0.0
        tmpEpoch += 1
    torch.save(
        {
            'model_state_dict': model.state_dict(),
        },
        logPath
    )
    print('Finish training')


# 验证模型，计算预测精度
def validation():
    transform = transforms.Compose([transforms.Resize((imageSize, imageSize)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])  # 设置图片
    # testSet = MyDataset('..\\data\\test.txt', transform=transform)  # 面试时注释
    testSet = MyDataset('..\\validation\\validation.txt', transform=transform)
    testLoader = DataLoader(testSet, batch_size=batchSize)
    device = torch.device('cpu')
    model = net()
    model.to(device)

    # checkpoint = torch.load(logPath)
    checkpoint = torch.load(bestPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 数据切换到测试模式

    # total = 0.0
    # correct = 0.0
    # for i, data in enumerate(testLoader):
    #     inputs = data[0].cpu()
    #     labels = data[1].cpu()
    #     outputs = model(inputs)
    #     _, predict = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predict == labels).sum().item()
    #     print('batch: %d, 目前的总准确率: %f' % (i + 1, correct / total))
    # print('总准确率: %.2f%%' % (correct / total * 100))

    for i, data in enumerate(testLoader):
        inputs = data[0].cpu()
        outputs = model(inputs)
        _, predict = torch.max(outputs.data, 1)
        with open('pred.txt', 'a') as f:
            for j in range(len(predict)):
                print((predict[j] + 1).item())
                f.write(str((predict[j] + 1).item()) + "\n")


if __name__ == '__main__':
    if isTrain:
        createTrainTxt()
        train()
    if isValidation:
        # createTestTxt()  # 面试时注释
        createdValidationTxt()
        validation()
