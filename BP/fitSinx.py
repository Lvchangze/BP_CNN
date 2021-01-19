import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def make_matrix(m, n, f=0.0):  # 创建一个m * n的矩阵
    mat = []
    for i in range(m):
        mat.append([f] * n)
    return mat


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
                        tmp[i][j] = random.uniform(0, 0.001)
                self.allWeight.append(tmp)
            elif 0 < t < self.hiddenLayerNum:  # 各个隐层之间的权重
                tmp = make_matrix(self.hiddenLayerNodeNum, self.hiddenLayerNodeNum)
                for i in range(self.hiddenLayerNodeNum):
                    for j in range(self.hiddenLayerNodeNum):
                        tmp[i][j] = random.uniform(0, 0.001)
                self.allWeight.append(tmp)
            else:  # 最后一个隐层和输出层之间的权重
                tmp = make_matrix(self.hiddenLayerNodeNum, self.outputNodeNum)
                for i in range(self.hiddenLayerNodeNum):
                    for j in range(self.outputNodeNum):
                        tmp[i][j] = random.uniform(0, 0.001)
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
                    tmp[o] = random.uniform(-0.001, 0)  # 这样子的操作，把这个地方的list类型变成了数字类型
                self.allBias.append(tmp)
            else:
                tmp = make_matrix(self.hiddenLayerNodeNum, 1)
                for h in range(self.hiddenLayerNodeNum):
                    tmp[h] = random.uniform(-0.001, 0)
                self.allBias.append(tmp)

    def forwardPropagate(self, inputs):
        # 输入层的值
        for i in range(0, self.inputNodeNum):
            self.inputCell[i] = inputs[i]  # 要归一化吗？
        # print(self.inputCell)
        # print("weight: " + str(self.allWeight))
        # 各个隐藏层的值
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
        # print(self.hiddenCells)
        #  输出层的值
        for j in range(self.outputNodeNum):
            total = 0.0
            for i in range(self.hiddenLayerNodeNum):
                total += self.hiddenCells[self.hiddenLayerNum - 1][i] * self.allWeight[self.hiddenLayerNum][i][j]
            self.outputCell[j] = total + self.allBias[self.hiddenLayerNum][j]
        #  计算误差
        yhat = np.array(self.outputCell)
        label = np.array(np.sin(self.inputCell))
        error = ((yhat - label) ** 2).sum()
        print("The loss is " + str(error))
        return self.outputCell

    def backPropagate(self, case, label):
        self.forwardPropagate(case)
        # 输出层误差
        outputDelta = [0.0] * self.outputNodeNum
        for i in range(self.outputNodeNum):
            error = label[i] - self.outputCell[i]
            outputDelta[i] = error  # 最后一层没有经过函数
            # * sigmoid_derivative(self.outputCell[i])
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

    def train(self, cases, labels, times=1000):
        for i in range(times):
            for j in range(len(cases)):
                case = cases[j]
                label = labels[j]
                self.backPropagate(case, label)


if __name__ == '__main__':
    bpNetwork = BPNetWork(1, 1, 50, 1, 0.03, 0)
    cases = []  # 训练集
    labels = []  # 正确的值
    for i in range(1000):
        tmp = random.uniform(-1, 1)
        cases.append([tmp * np.pi])
        labels.append([np.sin(tmp * np.pi)])
    bpNetwork.train(cases, labels, 2000)
    test = []
    testResult = []
    for i in range(-100, 101, 1):
        test.append([i * np.pi / 100])
    for case in test:
        print("case: " + str(case))
        out = bpNetwork.forwardPropagate(case)
        print("out: " + str(out))
        testResult.append(out[0])
    plt.figure()
    x = np.arange(-1.0, 1.0, 0.03)
    sinx, = plt.plot(x * np.pi, np.sin(x * np.pi), color='red')
    testData = []
    for i in range(len(test)):
        testData.append(test[i][0])
    mysinx, = plt.plot(testData, testResult, color='green')
    plt.legend(handles=[sinx, mysinx, ], labels=['real sinx', 'my sinx'], loc='best')
    plt.show()
