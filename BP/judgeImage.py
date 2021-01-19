import cv2
import numpy as np
import random
import torch


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def make_matrix(m, n, f=0.0):  # 创建一个m * n的矩阵
    mat = []
    for i in range(m):
        mat.append([f] * n)
    return mat


def getImageVector(dir, file):
    img = cv2.imread("../train/" + str(dir) + "/" + str(file) + ".bmp", 0)  # 28 * 28
    list = []  # 1 * 784的数组
    #  黑的为1，白的为0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 0:
                list.append(1)
            else:
                list.append(0)
    return list


class BPNetWork:
    def __init__(self, inputNodeNum, hiddenLayerNum, hiddenLayerNodeNum, outputNodeNum, r, weightDecay):  # 初始化变量
        self.inputNodeNum = inputNodeNum  # 输入层节点数
        self.hiddenLayerNum = hiddenLayerNum  # 隐藏层的数目
        self.hiddenLayerNodeNum = hiddenLayerNodeNum  # 隐藏层节点数
        self.outputNodeNum = outputNodeNum  # 输出层节点数
        self.r = r  # learning rate
        self.weightDecay = weightDecay
        self.inputCell = [1.0] * self.inputNodeNum
        # 初始化权重
        self.allWeight = []
        for t in range(0, self.hiddenLayerNum + 1):
            if t == 0:  # 输入层与第一个隐层之间的权重
                tmp = make_matrix(self.inputNodeNum, self.hiddenLayerNodeNum)
                for i in range(0, self.inputNodeNum):
                    for j in range(0, self.hiddenLayerNodeNum):
                        tmp[i][j] = random.uniform(-0.1, 0.1)
                self.allWeight.append(tmp)
            elif 0 < t < self.hiddenLayerNum:  # 各个隐层之间的权重
                tmp = make_matrix(self.hiddenLayerNodeNum, self.hiddenLayerNodeNum)
                for i in range(self.hiddenLayerNodeNum):
                    for j in range(self.hiddenLayerNodeNum):
                        tmp[i][j] = random.uniform(-0.1, 0.1)
                self.allWeight.append(tmp)
            else:  # 最后一个隐层和输出层之间的权重
                tmp = make_matrix(self.hiddenLayerNodeNum, self.outputNodeNum)
                for i in range(self.hiddenLayerNodeNum):
                    for j in range(self.outputNodeNum):
                        tmp[i][j] = random.uniform(-0.1, 0.1)
                self.allWeight.append(tmp)
        # 初始化隐藏层
        self.hiddenCells = []
        for t in range(0, hiddenLayerNum):
            self.hiddenCells.append([1.0] * self.hiddenLayerNodeNum)
        # 初始化输出层
        self.outputCell = [1.0] * self.outputNodeNum
        #  初始化bias
        self.allBias = []
        for t in range(0, self.hiddenLayerNum + 1):
            if t == self.hiddenLayerNum:  # 输出层的bias
                tmp = make_matrix(self.outputNodeNum, 1)
                for o in range(self.outputNodeNum):
                    tmp[o] = random.uniform(-0.1, 0)  # 这样子的操作，把这个地方的list类型变成了数字类型
                self.allBias.append(tmp)
            else:
                tmp = make_matrix(self.hiddenLayerNodeNum, 1)
                for h in range(self.hiddenLayerNodeNum):
                    tmp[h] = random.uniform(-0.1, 0)
                self.allBias.append(tmp)

    def forwardPropagate(self, inputs):
        # 输入层的值
        for i in range(0, self.inputNodeNum):
            self.inputCell[i] = inputs[i]
        for t in range(0, len(self.hiddenCells)):
            if t == 0:  # 输入层与第一层隐藏层之间的传输
                for j in range(self.hiddenLayerNodeNum):
                    total = 0.0
                    for i in range(self.inputNodeNum):
                        total += self.inputCell[i] * self.allWeight[t][i][j]
                    self.hiddenCells[t][j] = sigmoid(total + self.allBias[t][j])
            elif 0 < t <= len(self.hiddenCells) - 1:  # 其他隐藏层的值
                for j in range(self.hiddenLayerNodeNum):
                    total = 0.0
                    for i in range(self.hiddenLayerNodeNum):
                        total += self.hiddenCells[t - 1][i] * self.allWeight[t][i][j]
                    self.hiddenCells[t][j] = sigmoid(total + self.allBias[t][j])
        for j in range(self.outputNodeNum):
            total = 0.0
            for i in range(self.hiddenLayerNodeNum):
                total += self.hiddenCells[self.hiddenLayerNum - 1][i] * self.allWeight[self.hiddenLayerNum][i][j]
            total += self.allBias[self.hiddenLayerNum][j]
            self.outputCell[j] = total + self.allBias[self.hiddenLayerNum][j]
        print("过softmax前： " + str(self.outputCell))
        self.outputCell = softmax(self.outputCell)
        print("过softmax后： " + str(self.outputCell))
        return self.outputCell

    def backPropagate(self, case, label):
        self.forwardPropagate(case)
        # 输出层误差
        outputDelta = [0.0] * self.outputNodeNum
        for i in range(self.outputNodeNum):
            outputDelta[i] = label[i] - self.outputCell[i]
            '''上面这一行相当于以下代码：
            if label[i] == 1:
                outputDelta[i] = 1 - self.outputCell[i]
            else:
                outputDelta[i] = -self.outputCell[i]
            '''
        # 隐藏层误差
        allHiddenDeltas = [[0.0] * self.hiddenLayerNodeNum] * self.hiddenLayerNum
        for k in range(self.hiddenLayerNum - 1, -1, -1):  # 倒序到0
            if k == self.hiddenLayerNum - 1:  # 倒数第一个隐藏层的误差
                for h in range(self.hiddenLayerNodeNum):
                    error = 0.0
                    for o in range(self.outputNodeNum):
                        error += outputDelta[o] * self.allWeight[self.hiddenLayerNum][h][o]
                    allHiddenDeltas[k][h] = sigmoid_derivative(self.hiddenCells[k][h]) * error
            else:
                for h_low in range(self.hiddenLayerNodeNum):
                    error = 0.0
                    for h_high in range(self.hiddenLayerNodeNum):
                        error += allHiddenDeltas[k + 1][h_high] * self.allWeight[k + 1][h_low][h_high]
                    allHiddenDeltas[k][h_low] = sigmoid_derivative(self.hiddenCells[k][h_low]) * error
        # 更新权重
        for k in range(len(self.allWeight) - 1, -1, -1):
            if k == len(self.allWeight) - 1:  # 调整倒数第一个隐藏层和输出层之间的权重
                for h in range(self.hiddenLayerNodeNum):
                    for o in range(self.outputNodeNum):
                        # change是为了防止三个浮点数相乘
                        change = outputDelta[o] * self.hiddenCells[k - 1][h]
                        self.allWeight[k][h][o] = (1 - self.weightDecay) * self.allWeight[k][h][o] + change * self.r
            elif k == 0:  # 调整第一个隐藏层和输入层之间的权重
                for i in range(self.inputNodeNum):
                    for h in range(self.hiddenLayerNodeNum):
                        change = allHiddenDeltas[0][h] * self.inputCell[i]
                        self.allWeight[k][i][h] = (1 - self.weightDecay) * self.allWeight[k][i][h] + change * self.r
            else:  # 调整其他隐藏层之间的权重
                for h_low in range(self.hiddenLayerNodeNum):
                    for h_high in range(self.hiddenLayerNodeNum):
                        change = allHiddenDeltas[k][h_high] * self.hiddenCells[k - 1][h_low]
                        self.allWeight[k][h_low][h_high] = (1 - self.weightDecay) * self.allWeight[k][h_low][
                            h_high] + change * self.r
        # 更新bias
        for k in range(len(self.allBias) - 1, -1, -1):
            if k == len(self.allBias) - 1:  # 调整输出层的bias
                for o in range(self.outputNodeNum):
                    self.allBias[k][o] += self.r * outputDelta[o]
            else:
                for h in range(self.hiddenLayerNodeNum):
                    self.allBias[k][h] += self.r * allHiddenDeltas[k][h]

    def train(self, cases, labels, epoch=1000):
        for i in range(epoch):
            for j in range(len(cases)):
                if i == 0:  # 第一次调整调大一点，所以学习率大一点
                    self.r = 0.01
                else:
                    self.r = 0.01
                print("epoch: 第" + str(i + 1) + "遍")
                case = cases[j]
                label = labels[j]
                self.backPropagate(case, label)


if __name__ == '__main__':
    bpNetwork = BPNetWork(784, 1, 200, 12, 0.01, 0)
    trainImage = []
    for t in range(1, 13, 1):
        for k in range(1, 511, 1):  # 前510张图片
            imgVector = getImageVector(t, k)
            kind = [0.0] * 12
            kind[t - 1] = 1.0
            print((imgVector, kind))
            trainImage.append((imgVector, kind))
    random.shuffle(trainImage)  # 打乱

    cases = []
    labels = []
    for i in range(len(trainImage)):
        cases.append(trainImage[i][0])
        labels.append(trainImage[i][1])
    bpNetwork.train(cases, labels, 5)  # 6120张图片，过5遍网络

    torch.save(
        {
            'bias': bpNetwork.allBias,
            'weight': bpNetwork.allWeight
        },
        "my.dat"
    )

    test = []
    result = []
    countTrueNumber = 0
    for t in range(1, 13, 1):
        for k in range(511, 621, 1):  # 后110张图片
            imgVector = getImageVector(t, k)
            resultList = list(bpNetwork.forwardPropagate(imgVector))
            if (resultList.index(max(resultList))) + 1 == t:
                countTrueNumber += 1
    print("在测试集上的正确率: " + str(countTrueNumber / 1320))
