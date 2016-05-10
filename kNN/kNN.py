# -*-coding:utf-8-*-

'''
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: smallbal
'''
from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    """
输入:
    inX: 1xn 的行矩阵,n为特征维度
    dataSet: 测试样本,mxn 的矩阵,n为特征纬度,m为测试样本的个数
    labels: 测试样本的标签信息,是一个list
    k: k邻域参数,为正整数
输出: kNN分类器的分类结果
    """
    dataSetSize = dataSet.shape[0]
    # .shape属性返回array类型的维度信息,是一个tuple
    # 3行2列的array的shape属性为(3,2),则这里的*.shape[0]指的是array的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # tile()函数为numpy.lib.shape_base中的函数,此处效果为将inX以矩阵块的形式拓展为dataSetSize x 1的矩阵
    sqDiffMat = diffMat**2  # 因为inX 和 dataSet均为array类型,所以 diffMat也为array类型, diffMat**2为element-wise的平方
    sqDistances = sqDiffMat.sum(axis=1)  #.sum()为array的求和函数.axis默认为None,此时为将array的各项相加求和;
    # axis=1表示将每一行相加求和,返回由各行的和组成的一个list; axis=1表示将各列相加求和并返回.axis不能超过维数(注意是从0开始数)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    # .argsort()默认返回以"最后一维" 从小到达排序的矩阵相应元素的索引号,最小值的索引号为0.
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # python3.x中,上一句应该变为下一句,原因是python3.x中取消了dict.iteritems()方法,因为.items()方法变为与.iteritems()返回值相同.
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

print("Begin")
datingClassTest()