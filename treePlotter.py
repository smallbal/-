# !/usr/bin python3
# -*-coding:utf-8-*-

from math import log


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
    :param data_set: 要划分成2部分的数据集,类型为二维list
    :param axis: 以第axis列为标准划分,从0开始
    :param value: 在axis列中的划分标准,即将axis列中内容是value的相应行挑出
    :return: 按规则从data_set中挑选出的元素所组成的新的list,其中元素为data_set中挑出元素全掉axis轴中元素后新组成的元素
    example: data_set = [[1,0,1,'a'],[1,1,1,'b'],[1,0,0,'a']];axis=1;
             splitDataDSet(data_set, axis, value) --> [[1,1,'b']] if value==1, [[1,1,'a'],[1,0,'a']] if value==0
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def chooseBestFeatureToSplit(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = clacShannonEnt(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        featurelist = [example[i] for example in data_set]
        unique_vals = set(featurelist)
        new_entropy = 0.0
        for value in unique_vals:
            sub_Data_set = splitDataSet(data_set, i, value)
            prob = len(sub_Data_set)/float(len(data_set))
            new_entropy += prob * clacShannonEnt(sub_Data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature