import cv2
import numpy as np
import torch


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_matrix(m, n, f=0.0):  # 创建一个m * n的矩阵
    mat = []
    for i in range(m):
        mat.append([f] * n)
    return mat


def getImageVector(file):
    img = cv2.imread("../validation/" + str(file) + ".bmp", 0)  # 28 * 28
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

        my = torch.load("my.dat")
        bias = my['bias']
        weight = my['weight']

        # 初始化权重
        self.allWeight = weight
        #  初始化bias
        self.allBias = bias
        # 初始化隐藏层
        self.hiddenCells = []
        for t in range(0, hiddenLayerNum):
            self.hiddenCells.append([1.0] * self.hiddenLayerNodeNum)
        # 初始化输出层
        self.outputCell = [1.0] * self.outputNodeNum

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
        list = self.outputCell
        kind = list.index(max(list)) + 1
        print(kind)
        with open("pred.txt", 'a') as f:
            f.write(str(kind) + "\n")


if __name__ == '__main__':
    bpNetwork = BPNetWork(784, 1, 200, 12, 0.01, 0)
    test = []
    result = []
    countTrueNumber = 0
    for t in range(1, 1801, 1):
        imgVector = getImageVector(t)
        bpNetwork.forwardPropagate(imgVector)
