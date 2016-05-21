# -*-coding:utf-8-*-

from numpy import *


def loadData():
    """
    创建一组词库和相应的测试标签
    :return: posting_list:词库，类型list
             class_vec: 测试标签，类型list, 1表示有侮辱性词语，0表示没有。
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], \
                     ['maybe', 'not', 'take', 'him', 'to,', 'dog', 'park', 'stupid'],  \
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], \
                     ['stop', 'posint', 'stupid', 'worthless', 'garbage', 'to', 'stop', 'him'], \
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], \
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def createVocabList(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def setOfWords2Vec(vocab_list, input_set):
    """
    将输入的词集转换为词集模型列表，词集列表中的每个元素表示句子中是否出现该词，1表示出现，0表示没出现。
    :param vocab_list:  词库字典，list类型，包含所有词语
    :param input_set:   需要转换成字典向量的词语list
    :return: 返回转换成的词集模型列表
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("词语 \"%s\" 不在字典列表中" % word)
    return return_vec


def bagOfWord2Vec(vocab_list, input_set):
    """
    将输入的词集转换为词袋列表，词袋列表中的每个元素表示出现句子中该词出现的次数，>=0。
    :param vocab_list:  词库字典，list类型，包含所有词语
    :param input_set:   需要转换成字典向量的词语list
    :return:  返回转换成的词袋模型列表
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print("词语 \"%s\"不在字典列表中" % word)
    return return_vec


def trainNB0(train_matrix, train_category):
    """
    朴素贝叶斯分类器训练函数
    :param train_matrix: 2维list, 是词集模型列表或者词袋模型列表
    :param train_category: 测试标签
    :return: p0_vect:
             P1_vect:
             p_abusive:
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category)/float(num_train_docs)  # 计算属于侮辱性文档的概率
    p0_num = zeros(num_words); p1_num = zeros(num_words)
    # 创建array是只有一个参数(num_words)，则会创建出1 x num_words的array。zeros()创建的是一个全 0 array矩阵对象.
    p0_denom = p1_denom = 0.0
    # p0_num = ones(num_words); p1_num = ones(num_words)
    # p0_denom = p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]  # array类可与同维list相加，规则为矩阵加法规则，结果类型为array类
            p1_denom += sum(train_matrix[i])  # 计算Abusive句子中所有词条的总数
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = p1_num/p1_denom
    p0_vect = p0_num/p0_denom
    return p0_vect, p1_vect, p_abusive


