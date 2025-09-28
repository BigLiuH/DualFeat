import csv
import torch as t
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        # return t.FloatTensor(md_data)
        return md_data


def get_edge_indexmi(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] > 0.2:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)

def get_edge_indexcirc(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] > 0.1:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def prepare_data():
    dataset = dict()
    dataset['md_p'] = read_csv('../../mydataset/association/ori_circRNAs_miRNAs_association_withoutindex.csv')
    dataset['md_true'] = read_csv('../../mydataset/association/ori_circRNAs_miRNAs_association_withoutindex.csv')

    zero_index = []# 存储md_p中小于1的索引
    one_index = []# 存储md_p中大于等于1的索引
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = t.LongTensor(zero_index)
    one_tensor = t.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    def createSimilarityInfo(Dataset=dataset, path='', name='',path2=''):


        data_matrix = read_txt(path2)
        data_matrix=t.FloatTensor(data_matrix)



        temp=pd.read_csv(path, header=None)
        mm_edge_index = temp.values.T.astype(int)
        # print(mm_edge_index.shape)

        Dataset[name] = {'data_matrix': data_matrix, 'edges': mm_edge_index}
        #print(mm_edge_index)


    #只需维护四个相似矩阵路径即可
    createSimilarityInfo(dataset, '../../dataop/myCMI/CMI_index.csv', 'seq','../../mydataset/node2vec/sim_embedding.txt')
    createSimilarityInfo(dataset, '../../dataop/myCMI/CMI_index.csv', 'gip','../../mydataset/node2vec/GIP_embedding.txt')
    createSimilarityInfo(dataset, '../../dataop/myCMI/CMI_index.csv', 'fun','../../mydataset/node2vec/function_embedding.txt')
    # os._exit(0)
    #edges第一维表示起点，第二维表示终点

    return dataset


prepare_data()