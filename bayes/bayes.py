# -*-coding:utf-8-*-

from numpy import *


def loadData():
    """
    创建一组词库和相应的测试标签
    :return: posting_list:词条列表，多位数组形式，但每行元素没有限制，不需要相同，类型list
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
    """
    将词条列表中的元素集中成一个词条库，供之后统一向量使用。
    :param data_set: 词条列表，每条词条句子的词条个数没有要求, 类型list。
    :return: 词条库，类型list
    """
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
    :param train_matrix: 2维array, 是词集模型列表或者词袋模型列表。其实用list也可以。
    :param train_category: 测试标签，是一个1xn的array。其实用list也可以。
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
            p1_num += train_matrix[i]  # 拓展：array类可与同维list相加，规则为矩阵加法规则，结果类型为array类 。
            p1_denom += sum(train_matrix[i])  # 计算Abusive句子中所有词条的总数
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = p1_num/p1_denom
    p0_vect = p0_num/p0_denom
    return p0_vect, p1_vect, p_abusive


def classifyNB(vec2classify, p0_vec, p1_vec, p_class1):
    """
    朴素贝叶斯二分类函数
    :param vec2classify:    测试词集模型向量(list)或者词袋模型向量(list)
    :param p0_vec:  词集模型或词袋模型的概率list，每个元素是p0(wi|ci), 即第i个词条的出现的概率
    :param p1_vec:  词集模型或词袋模型的概率list, 每个元素是p1(wi|ci)，即第i个词条的出现的概率
    :param p_class1:只需要p_class1, 因为p_class0 + p_class1 = 1, 所以p_class0可以由p_class1得出
    :return: 若含侮辱性词条，返回1；否则返回0.
    """
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    list_of_posts, list_classes = loadData()
    my_vocab_list = createVocabList(list_of_posts)
    train_mat = []  # 虽然叫mat，但是是一个list。存储所有转换成词条库向量的list。
    for post_in_doc in list_of_posts:
        # 上面先用list_of_posts生成词条库，这里再将list_of_posts中的每个元素转换成词条库的向量。
        train_mat.append(setOfWords2Vec(my_vocab_list, post_in_doc))
    p0V, p1V, pAb = trainNB0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(setOfWords2Vec(my_vocab_list, test_entry))
    print(test_entry, " classified as: ", classifyNB(this_doc, p0V, p1V, pAb))
    test_entry = ['stupid', 'garbage']
    this_doc = array(setOfWords2Vec(my_vocab_list, test_entry))
    print(test_entry, "classified as: ", classifyNB(this_doc, p0V, p1V, pAb))


def textParse(big_string):
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spamTest():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = textParse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = textParse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = createVocabList(doc_list)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(setOfWords2Vec(vocab_list, doc_list[doc_index]))
        train_classes.append((class_list[doc_index]))
    p0V, p1V, pSpam = trainNB0(array(train_mat, array(train_classes)))
    error_count = 0
    for doc_index in test_set:
        word_vector = setOfWords2Vec(vocab_list, doc_list[doc_index])
        if classifyNB(array(word_vector, p0V, p1V, pSpam)) != class_list[doc_index]:
            error_count += 1
    print('the error rate is:', float(error_count)/len(test_set))