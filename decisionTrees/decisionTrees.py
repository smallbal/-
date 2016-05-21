# !/usr/bin python3
# -*-coding:utf-8-*-

from math import log
import operator  # majorityCnt()函数使用


def createDataSet():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def clacShannonEnt(data_set):
    """
    计算香农熵
    输入: data_set: 是一个二维list,形如[[12,3,4,'A'],[2,3,4,'B']]
    输入: shannon_set: 香农熵,一个float型
    """
    num_entry = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entry
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent


def splitDataSet(data_set, axis, value):
    """
    将数据在指定特征上按指定规则划分
    :param data_set: 要划分成2部分的数据集,类型为二维list
    :param axis: 以第axis列为标准划分,从0开始
    :param value: 在axis列中的划分标准,即将axis列中内容是value的相应行挑出
    :return: 按规则从data_set中挑选出的元素所组成的新的list,其中元素为data_set中挑出元素全掉axis轴中元素后新组成的元素
    example: data_set = [[1,0,1,'a'],[1,1,1,'b'],[1,0,0,'a']];axis=1;
             splitDataDSet(data_set, axis, value) --> [[1,1,'b']] if value==1, [[1,1,'a'],[1,0,'a']] if value==0
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def chooseBestFeatureToSplit(data_set):
    """
    计算出数据集使划分数据后的熵最小的特征列
    :param data_set: list类型,每一个元素也必须是list并且元素list的长度必须相等.
同时,元素list的最后元素必须是类别标签;每个元素list中的特征信息的数据类型没有要求.
    :return:返还使信息增益(或者熵)最大的一列特征标号,第一列标号为0,以此类推.
    """
    num_features = len(data_set[0]) - 1
    base_entropy = clacShannonEnt(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        featurelist = [example[i] for example in data_set]
        unique_vals = set(featurelist)  # set()函数将对象元素转换为set类型,set类型是一个无序的不重复元素的集合
        # 从列表中创建集合是python语言得到列表中唯一元素的最快方法
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = splitDataSet(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * clacShannonEnt(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
            # 因为熵越小,则信息越有序.而信息增益是数据划分之前和之后信息熵的差值,
            # 所以当信息增益越大,则说明划分数据后熵变的越小,即数据更加有序,此时说
            # 明按照此特征划分数据更好.
    return best_feature


def majorityCnt(class_list):
    """
    计算出分类名称列表中类型标签个数最多的那个标签名
    :param class_list: list类型,内容为分类名称标签
    :return: class_list中个数最多的那个标签
    """
    class_count = {}
    for vote in class_list:
        print(vote)
        if vote not in class_count.keys():
            # dict.keys()返回由dict的key组成的list
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # sored()为python的built-in函数
    # dict.items()返回以dict的key-value为tuple型元素的list.例如dict={'a':1,'b':2},dict.items()-->[('a',1),('b',2)]
    # key=operator.itemgetter(1)表示以每个tuple的第二个为基准进行排序
    # reverse(翻转)=True代表排序后是最大的在第一个
    print(sorted_class_count)
    return sorted_class_count[0][0]


def createTree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    '''
    print('class_list:\n', class_list)
    print('class_list.count(class_list[0]) = ',class_list.count(class_list[0]))
    print('len(class_list) = ', len(class_list))
    '''
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majorityCnt(class_list)
    best_feat = chooseBestFeatureToSplit(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = createTree(splitDataSet(data_set, best_feat, value), sub_labels)
    return my_tree



print("begin")
myDat, labels = createDataSet()
print('myDat: \n', myDat)
print('labels: \n', labels)
myTree = createTree(myDat, labels)

print('myTree = \n', myTree)
print("Done")
