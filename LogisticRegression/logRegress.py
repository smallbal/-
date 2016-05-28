# -*-coding:utf-8-*-


from numpy import *


def loadData():
    data_list = []
    label_list = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_spare = line.strip().split()
        data_list.append([1.0, float(line_spare[0]), float(line_spare[1])])
        # data_list.append([float(line_spare[0]), float(line_spare[1])])
        label_list.append(int(line_spare[-1]))
    return data_list, label_list


def sigmoid(z):
    return 1.0/(1+exp(-z))


def gradAscent(data_mat_in, class_label):
    data_mat = mat(data_mat_in)  # 将输入的数据特征list变为一个m x n 的mat类
    label_mat = mat(class_label).transpose()  # 将标签list变为一个 n x 1的mat类，.transpose()为转置函数
    m, n = shape(data_mat)
    alpha = 0.001
    max_cycle = 500  # 我的电脑里迭代400次，后面两个回归系数值就不变了
    weights = ones((n, 1))  # n x 1 的array
    for k in range(max_cycle):
        h = sigmoid(data_mat * weights)
        # print("in %dth cycle, h= " % k,h)  #print()一个mat或者array类，结果是一个list，原因不明
        error = (label_mat - h)
        weights = weights + alpha * data_mat.transpose() * error  # 将梯度上升法和偏导数结合后得出的结论
    return weights


def stocGradAscent(data_mat_in, class_label):
    """
    随机梯度上升法估计回归系数。属于在线学习算法；相对概念是“批处理”，即依次处理所有数据
    :param data_mat_in:
    :param class_label:
    :return:
    """
    m, n = shape(data_mat_in)
    weights = ones(n)
    alpha = 0.001
    for k in range(m):
        h = sigmoid(sum(data_mat_in[k] * weights))
        error = class_label[k] - h
        weights = weights + alpha * data_mat_in[k] * error
    return weights


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()  # 什么意思。。。查
    data_mat, label_mat = loadData()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]  # 未知错误
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 10.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



'''
测试程序
'''
data, label = loadData()
weights = gradAscent(data, label)
plotBestFit(weights)

#weights = stocGradAscent(array(data), label)
#plotBestFit(weights)