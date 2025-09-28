import time

# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import os
from sklearn.metrics import average_precision_score

os.environ["PYTORCH_JIT"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 可用于调试
import numpy as np
import time
import pandas as pd
import os
from catboost import CatBoostClassifier
import torch
from matplotlib import pyplot as plt
from numpy import interp
import matplotlib.patches as patches

# import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io

from keras.layers import merge

from keras.utils import np_utils, generic_utils

from xgboost import XGBClassifier
# from keras.layers import containers, normalization

from GATpredata import prepare_data

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>这里用的是cbam模块
# from model import GATModel
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>下面用的是eca模块
from GAT import GATModel

from torch import nn, optim
from param2 import parameter_parser

'''
label1是positive样本label
label2是未知样本的label
'''
seed = 1
text = "原始"
filename = "评估信息(单独GAT).txt"


class CMI():
    def __init__(self):
        super().__init__()

        # circ-mi关联矩阵路径
        self.path_interaction = "../../mydataset/association/ori_circRNAs_miRNAs_association_withoutindex.csv"
        # circ、mi embedding的路径
        # self.path_circ_embedding = "../../mydataset/embedding/单独GAT/ciRNAEmbed"  # 用的时候记得加上"mv"
        # self.path_mi_embedding = "../../mydataset/embedding/单独GAT/miRNAEmbed"  # 用的时候记得加上"mv"

        # self.path_circ_embedding = "../../mydataset/node2vec(3)/ciRNAEmbed"  # 用的时候记得加上"mv"
        # self.path_mi_embedding = "../../mydataset/node2vec(3)/miRNAEmbed"  # 用的时候记得加上"mv"
        self.path_circ_embedding = "../../mydataset/embedding/单独GAT/ciRNAEmbed"  # 用的时候记得加上"mv"
        self.path_mi_embedding = "../../mydataset/embedding/单独GAT/miRNAEmbed"  # 用的时候记得加上"mv"
        # 存储核加载分类模块数据的路径
        self.path_cat_model = "../../code/单独GAT/GATCatModel/catmodel_"
        # cat_boost分类模块超参数
        self.cat_epoch = 425  # 代表
        self.cat_depth = 4

        self.threshold = 0.5  # 判断正负样本的阈值常熟
        # circ-mi关联矩阵
        self.interaction = np.loadtxt(self.path_interaction, dtype=float, delimiter=",")

        self.is_loading_embeddings = False  # 判断是否加载了embeddings
        self.is_loading_traindatas = False  # 判断是否加载了训练数据
        # 存储circRNA和miRNA的embedding
        self.circ_embedding = []
        self.mi_embedding = []
        # circRNA和miRNA的数量
        self.circ_number = 504
        self.mi_number = 420
        # circRNA和miRNA的列表
        self.circ_list = list(np.loadtxt('../../dataop/myCMI/circRNA.txt', dtype=str))
        self.mi_list = list(np.loadtxt('../../dataop/myCMI/miRNA.txt', dtype=str))
        # 二者的序列
        self.mi_seq_list = list(np.loadtxt('../../dataop/myCMI/miRNA_seq.txt', dtype=str))
        self.circ_seq_list = list(np.loadtxt('../../dataop/myCMI/circRNA_seq.txt', dtype=str))

    def load_embbedings(self):
        for i in range(1, 2):
            # 为了从1编号
            self.circ_embedding.append(
                np.loadtxt(self.path_circ_embedding + str(i) + '.csv', dtype=float, delimiter=","))
            self.mi_embedding.append(
                np.loadtxt(self.path_mi_embedding + str(i) + '.csv', dtype=float, delimiter=","))

        for i in range(1, 4):

            print('loading embbedings, mv=', i)
            self.circ_embedding.append(
                np.loadtxt(self.path_circ_embedding + str(i) + '.csv', dtype=float, delimiter=","))
            self.mi_embedding.append(np.loadtxt(self.path_mi_embedding + str(i) + '.csv', dtype=float, delimiter=","))

            print('loading over, mv=', i)
            print()

    def prepare_data3(self):
        if not self.is_loading_embeddings:
            self.load_embbedings()
        self.is_loading_embeddings = True
        # 先把正负样本全部搞出来，然后取全部的正样本，随机取等量的负样本。在此基础上进行训练集和测试集的划分
        # 首先进行测试集和训练集的划分
        n = 504 * 420
        arr = np.zeros((n, 2), dtype=int)
        cnt = 0
        for i in range(0, 504):
            for j in range(0, 420):
                arr[cnt][0] = i  # 获取对应circRNA的索引
                arr[cnt][1] = j  # 获取对应miRNA的索引
                cnt += 1
        # 一共有n对，那么我用后面20%作为测试集，前面80%作为训练集

        np.random.shuffle(arr)
        interaction = self.interaction

        # 收集所有正样本
        positive_samples = []# 存储正样本的索引
        for x in range(0, n):
            i = arr[x][0]
            j = arr[x][1]
            if interaction[i][j] == 1:
                positive_samples.append((i, j))

        psnumber = len(positive_samples)
        print(f"正样本数量: {psnumber}")
        # 负样本采样
        # 加载相似矩阵
        miRNAs_max_path = "../../mydataset/matrix/miRNAs_max_matrix.csv"
        miRNAs_avg_path = "../../mydataset/matrix/miRNAs_avg_matrix.csv"
        circRNAs_avg_path = "../../mydataset/matrix/circRNAs_avg_matrix.csv"
        circRNAs_max_path = "../../mydataset/matrix/circRNAs_max_matrix.csv"

        # # 读取相似矩阵文件
        miRNAs_max_matrix = np.loadtxt(miRNAs_max_path, delimiter=',')
        miRNAs_avg_matrix = np.loadtxt(miRNAs_avg_path, delimiter=',')
        circRNAs_avg_matrix = np.loadtxt(circRNAs_avg_path, delimiter=',')
        circRNAs_max_matrix = np.loadtxt(circRNAs_max_path, delimiter=',')
        """
        # 为每个正样本生成一个负样本
        negative_samples = []

        # 对每个正样本对(circRNA-miRNA)，生成一个负样本
        for circ_idx, mi_idx in positive_samples:
            # 随机决定替换circRNA还是miRNA
            if np.random.rand() > 0.5:
                # 使用circRNA相似矩阵找到与当前circRNA相似度最低的circRNA
                similarities = circRNAs_avg_matrix[circ_idx]
                # 将自身的相似度设置为最高，避免选中自己
                similarities[circ_idx] = float('inf')

                # 按相似度升序排序
                sorted_indices = np.argsort(similarities)

                # 选择相似度最低的circRNA作为负样本，确保形成的对是负样本
                for idx in sorted_indices:
                    if interaction[idx][mi_idx] == 0:  # 确保是负样本
                        negative_samples.append((idx, mi_idx))
                        break
            else:
                # 使用miRNA相似矩阵找到与当前miRNA相似度最低的miRNA
                similarities = miRNAs_avg_matrix[mi_idx]
                # 将自身的相似度设置为最高，避免选中自己
                similarities[mi_idx] = float('inf')

                # 按相似度升序排序
                sorted_indices = np.argsort(similarities)

                # 选择相似度最低的miRNA作为负样本，确保形成的对是负样本
                for idx in sorted_indices:
                    if interaction[circ_idx][idx] == 0:  # 确保是负样本
                        negative_samples.append((circ_idx, idx))
                        break
        """

        """
        # 为每个正样本生成一个负样本
        negative_samples = []
        negative_set = set()

        for circ_idx, mi_idx in positive_samples:
            # -------- circRNA 部分 --------
            circ_similarities = circRNAs_avg_matrix[circ_idx].copy()
            circ_similarities[circ_idx] = float('inf')

            circ_sorted_indices = np.lexsort((np.arange(len(circ_similarities)), circ_similarities))

            min_circ_idx = -1
            min_circ_similarity = float('inf')
            for idx in circ_sorted_indices:
                if interaction[idx][mi_idx] == 0 and (idx, mi_idx) not in negative_set:
                    min_circ_idx = idx
                    min_circ_similarity = circ_similarities[idx]
                    break  # 找到第一个合法且不重复的负样本，退出
            # -----------------------------------

            # -------- 替换 miRNA 部分 --------
            mi_similarities = miRNAs_avg_matrix[mi_idx].copy()
            mi_similarities[mi_idx] = float('inf')

            mi_sorted_indices = np.lexsort((np.arange(len(mi_similarities)), mi_similarities))

            min_mi_idx = -1
            min_mi_similarity = float('inf')
            for idx in mi_sorted_indices:
                if interaction[circ_idx][idx] == 0 and (circ_idx, idx) not in negative_set:
                    min_mi_idx = idx
                    min_mi_similarity = mi_similarities[idx]
                    break
            # -----------------------------------

            # -------- 比较两种方案并添加 --------
            if min_circ_similarity <= min_mi_similarity and min_circ_idx != -1:
                negative_samples.append((min_circ_idx, mi_idx))
                negative_set.add((min_circ_idx, mi_idx))
            elif min_mi_idx != -1:
                negative_samples.append((circ_idx, min_mi_idx))
                negative_set.add((circ_idx, min_mi_idx))
            else:
                print("有一个没有找到负样本的正样本对:", (circ_idx, mi_idx))
                # 极少数情况：如果两种方式都没找到合适的负样本，随机生成一个不重复的
                max_try = 1000  # 防止死循环
                try_count = 0
                while try_count < max_try:
                    try_count += 1
                    if np.random.rand() > 0.5:
                        new_circ_idx = np.random.randint(0, len(circRNAs_avg_matrix))
                        if (new_circ_idx != circ_idx and
                                interaction[new_circ_idx][mi_idx] == 0 and
                                (new_circ_idx, mi_idx) not in negative_set):
                            negative_samples.append((new_circ_idx, mi_idx))
                            negative_set.add((new_circ_idx, mi_idx))
                            break
                    else:
                        new_mi_idx = np.random.randint(0, len(miRNAs_avg_matrix))
                        if (new_mi_idx != mi_idx and
                                interaction[circ_idx][new_mi_idx] == 0 and
                                (circ_idx, new_mi_idx) not in negative_set):
                            negative_samples.append((circ_idx, new_mi_idx))
                            negative_set.add((circ_idx, new_mi_idx))
                            break
        """

        """
        alpha = 0.5  # 可调整，也可交叉验证优化

        negative_samples = []

        for circ_idx, mi_idx in positive_samples:
            if np.random.rand() > 0.5:
                # --- 替换 circRNA ---
                avg_similarities = circRNAs_avg_matrix[circ_idx]
                max_similarities = circRNAs_max_matrix[circ_idx]

                # 融合相似度
                similarities = alpha * avg_similarities + (1 - alpha) * max_similarities
                similarities[circ_idx] = float('inf')  # 排除自身

                sorted_indices = np.argsort(similarities)

                for idx in sorted_indices:
                    if interaction[idx][mi_idx] == 0:
                        negative_samples.append((idx, mi_idx))
                        break
            else:
                # --- 替换 miRNA ---
                avg_similarities = miRNAs_avg_matrix[mi_idx]
                max_similarities = miRNAs_max_matrix[mi_idx]

                similarities = alpha * avg_similarities + (1 - alpha) * max_similarities
                similarities[mi_idx] = float('inf')  # 排除自身

                sorted_indices = np.argsort(similarities)

                for idx in sorted_indices:
                    if interaction[circ_idx][idx] == 0:
                        negative_samples.append((circ_idx, idx))
                        break
        """

        """
        # 合并正负样本
        balanced_samples = positive_samples + negative_samples
        np.random.shuffle(balanced_samples)

        # 创建新的样本数组
        tot = len(balanced_samples)
        arr = np.zeros((tot, 2), dtype=int)
        for i, (circ_idx, mi_idx) in enumerate(balanced_samples):
            arr[i][0] = circ_idx
            arr[i][1] = mi_idx

        # 划分训练集和测试集
        train_number = int(tot * 0.8)
        test_number = tot - train_number
        print("#############DEBUGE#############")
        print(f"正样本数: {psnumber}, 负样本数: {len(negative_samples)}, 总样本数: {tot}")
        print(f"训练集大小: {train_number}, 测试集大小: {test_number}")
        print("################################")
        """


        # 统计正负样本数量
        ngnumber = 0
        psnumber = 0
        for x in range(0, n):
            i = arr[x][0]
            j = arr[x][1]  # 获取对应作用对的索引
            if interaction[i][j] == 0:  # 判断是否为负样本
                ngnumber += 1
            elif interaction[i][j] == 1:
                psnumber += 1

        old_arr = arr
        arr = np.zeros((psnumber * 2, 2), dtype=int)  # 创建新数组存储平衡后的样本
        print('len(arr):', len(arr))
        ngnumber = 0
        tot = 0 # 总数
        # 随机选取负样本
        for x in range(0, n):
            i = old_arr[x][0]
            j = old_arr[x][1]
            if interaction[i][j] == 1:
                arr[tot] = old_arr[x]
                tot += 1
            elif interaction[i][j] == 0 and ngnumber < psnumber:
                arr[tot] = old_arr[x]
                tot += 1
                ngnumber += 1

        train_number = int(tot * 0.8)
        test_number = tot - train_number
        print("#############DEBUGE#############")
        print(psnumber, ngnumber, tot, train_number, test_number)
        print("################################")



        np.random.shuffle(arr)  # 打乱顺序
        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        # # 创建测试集文件
        # # 收集测试集中出现的所有唯一RNA
        # test_circrnas = []
        # test_circrna_seqs = []
        # test_mirnas = []
        # test_mirna_seqs = []
        # positive_pairs = []
        # negative_pairs = []
        #
        # # 测试集RNA索引映射
        # circ_index_map = {}  # circ名称到索引的映射
        # mirna_index_map = {}  # mirna名称到索引的映射
        #
        # for k in range(train_number,tot):
        #     i = arr[k][0]
        #     j = arr[k][1]
        #
        #     circ_name = self.circ_list[i]
        #     circ_seq = self.circ_seq_list[i]
        #     mirna_name = self.mi_list[j]
        #     mirna_seq = self.mi_seq_list[j]
        #
        #     # 添加到唯一列表（如果不存在）
        #     if circ_name not in circ_index_map:
        #         circ_index = len(test_circrnas)
        #         test_circrnas.append(circ_name)
        #         test_circrna_seqs.append(circ_seq)
        #         circ_index_map[circ_name] = circ_index
        #
        #     if mirna_name not in mirna_index_map:
        #         mirna_index = len(test_mirnas)
        #         test_mirnas.append(mirna_name)
        #         test_mirna_seqs.append(mirna_seq)
        #         mirna_index_map[mirna_name] = mirna_index
        #
        #     # 获取索引
        #     circ_idx = circ_index_map[circ_name]
        #     mirna_idx = mirna_index_map[mirna_name]
        #
        #     # 添加相互作用对（使用索引）
        #     if interaction[i, j] == 1:  # 正样本
        #         positive_pairs.append((circ_idx, mirna_idx))
        #     else:  # 负样本
        #         negative_pairs.append((circ_idx, mirna_idx))
        #
        # # 保存文件
        # output_dir = "../../mydataset/testset/"
        # os.makedirs(output_dir, exist_ok=True)
        #
        # # 创建CSV数据框并保存
        # # 保存circRNA名称及索引
        # circ_data = [(idx, name) for idx, name in enumerate(test_circrnas)]
        # circ_df = pd.DataFrame(circ_data, columns=["index", "name"])
        # circ_df.to_csv(os.path.join(output_dir, "test_circrnas.csv"), index=False)
        #
        # # 保存miRNA名称及索引
        # mirna_data = [(idx, name) for idx, name in enumerate(test_mirnas)]
        # mirna_df = pd.DataFrame(mirna_data, columns=["index", "name"])
        # mirna_df.to_csv(os.path.join(output_dir, "test_mirnas.csv"), index=False)
        #
        # # 保存circRNA序列及索引
        # circ_seq_data = [(idx, seq) for idx, seq in enumerate(test_circrna_seqs)]
        # circ_seq_df = pd.DataFrame(circ_seq_data, columns=["index", "sequence"])
        # circ_seq_df.to_csv(os.path.join(output_dir, "test_circrna_seqs.csv"), index=False)
        #
        # # 保存miRNA序列及索引
        # mirna_seq_data = [(idx, seq) for idx, seq in enumerate(test_mirna_seqs)]
        # mirna_seq_df = pd.DataFrame(mirna_seq_data, columns=["index", "sequence"])
        # mirna_seq_df.to_csv(os.path.join(output_dir, "test_mirna_seqs.csv"), index=False)
        #
        # # 保存相互作用文件（使用索引）
        # positive_df = pd.DataFrame(positive_pairs, columns=["circ_index", "mirna_index"])
        # positive_df.to_csv(os.path.join(output_dir, "test_positive_interactions.csv"), index=False)
        #
        # negative_df = pd.DataFrame(negative_pairs, columns=["circ_index", "mirna_index"])
        # negative_df.to_csv(os.path.join(output_dir, "test_negative_interactions.csv"), index=False)
        #
        # print(f"已生成测试集circRNA文件，包含 {len(test_circrnas)} 个唯一circRNA")
        # print(f"已生成测试集miRNA文件，包含 {len(test_mirnas)} 个唯一miRNA")
        # print(f"测试集正样本相互作用文件: {len(positive_pairs)}条记录")
        # print(f"测试集负样本相互作用文件: {len(negative_pairs)}条记录")
        # os._exit(0)
        """
        # 初始化保存数据的容器
        test_circrnas = []  # 保存 (原始编号, 名称)
        test_circrna_seqs = []  # 保存 (原始编号, 序列)
        test_mirnas = []  # 保存 (原始编号, 名称)
        test_mirna_seqs = []  # 保存 (原始编号, 序列)
        positive_pairs = []  # 保存 (原始circ编号, 原始miRNA编号)
        negative_pairs = []

        # 标记是否已添加过某个 RNA
        circ_index_map = {}
        mirna_index_map = {}

        for k in range(0, train_number):
            i = arr[k][0]  # 原始 circRNA 编号
            j = arr[k][1]  # 原始 miRNA 编号

            circ_name = self.circ_list[i]
            circ_seq = self.circ_seq_list[i]
            mirna_name = self.mi_list[j]
            mirna_seq = self.mi_seq_list[j]

            # 如果当前 circRNA 没有被记录，则添加
            if i not in circ_index_map:
                test_circrnas.append((i, circ_name))
                test_circrna_seqs.append((i, circ_seq))
                circ_index_map[i] = True

            # 如果当前 miRNA 没有被记录，则添加
            if j not in mirna_index_map:
                test_mirnas.append((j, mirna_name))
                test_mirna_seqs.append((j, mirna_seq))
                mirna_index_map[j] = True

            # 添加交互对（保留原始编号）
            if interaction[i, j] == 1:
                positive_pairs.append((i, j))
            else:
                negative_pairs.append((i, j))

        # 输出目录
        output_dir = "../../mydataset/trainset2/"
        os.makedirs(output_dir, exist_ok=True)

        # 保存 circRNA 名称
        circ_df = pd.DataFrame(test_circrnas, columns=["index", "name"])
        circ_df.to_csv(os.path.join(output_dir, "train_circrnas.csv"), index=False)

        # 保存 circRNA 序列
        circ_seq_df = pd.DataFrame(test_circrna_seqs, columns=["index", "sequence"])
        circ_seq_df.to_csv(os.path.join(output_dir, "train_circrna_seqs.csv"), index=False)

        # 保存 miRNA 名称
        mirna_df = pd.DataFrame(test_mirnas, columns=["index", "name"])
        mirna_df.to_csv(os.path.join(output_dir, "train_mirnas.csv"), index=False)

        # 保存 miRNA 序列
        mirna_seq_df = pd.DataFrame(test_mirna_seqs, columns=["index", "sequence"])
        mirna_seq_df.to_csv(os.path.join(output_dir, "train_mirna_seqs.csv"), index=False)

        # 保存正负样本交互对（保留原始编号）
        positive_df = pd.DataFrame(positive_pairs, columns=["circ_index", "mirna_index"])
        positive_df.to_csv(os.path.join(output_dir, "train_positive_interactions.csv"), index=False)

        negative_df = pd.DataFrame(negative_pairs, columns=["circ_index", "mirna_index"])
        negative_df.to_csv(os.path.join(output_dir, "train_negative_interactions.csv"), index=False)

        # 打印信息
        print(f"✅ 已生成测试集 circRNA 文件，共 {len(test_circrnas)} 个唯一 circRNA")
        print(f"✅ 已生成测试集 miRNA 文件，共 {len(test_mirnas)} 个唯一 miRNA")
        print(f"✅ 正样本数: {len(positive_pairs)}，负样本数: {len(negative_pairs)}")
        os._exit(0)
        """

        # 下面读取数据,
        for mv in range(1, 4):
            circRNA_fea = self.circ_embedding[mv]
            disease_fea = self.mi_embedding[mv]

            # 下面遍历训练集
            link_number = 0
            train = []  # 存储训练集的特征向量
            testfnl = []
            label1 = []  # 存储训练集样本的标签
            label2 = []
            label22 = []
            ttfnl = []
            for k in range(train_number):
                i = arr[k][0]
                j = arr[k][1]
                if interaction[i, j] == 1:  # for associated
                    label1.append(interaction[i, j])  # label1= labels for association(1)
                    link_number = link_number + 1  # no. of associated samples
                    circRNA_fea_tmp = list(circRNA_fea[i])
                    disease_fea_tmp = list(disease_fea[j])
                    tmp_fea = (circRNA_fea_tmp, disease_fea_tmp)  # concatnated feature vector for an association
                    train.append(tmp_fea)  # train contains feature vectors of all associated samples
                elif interaction[i, j] == 0:  # for no association
                    label1.append(interaction[i, j])  # label2= labels for no association(0)
                    circRNA_fea_tmp1 = list(circRNA_fea[i])
                    disease_fea_tmp1 = list(disease_fea[j])
                    tmp_fea = (
                    circRNA_fea_tmp1, disease_fea_tmp1)  # concatenated feature vector for not having association
                    train.append(tmp_fea)  # testfnl contains feature vectors of all non associated samples

            print("len(train)", len(train))

            train = np.array(train)
            X_train.append(train)  # 存储训练集特征向量
            Y_train.append(label1)  # 存储训练集样本标签
            print('prepare train data over, mv=', mv)
            print(len(train))
            print(len(label1))

            # 下面遍历测试集
            link_number = 0
            train = []
            testfnl = []
            label1 = []
            label2 = []
            label22 = []
            ttfnl = []
            for k in range(train_number, tot):
                i = arr[k][0]
                j = arr[k][1]
                if interaction[i, j] == 1:  # for associated
                    label1.append(interaction[i, j])  # label1= labels for association(1)
                    link_number = link_number + 1  # no. of associated samples
                    # link_position.append([i, j])
                    circRNA_fea_tmp = list(circRNA_fea[i])
                    disease_fea_tmp = list(disease_fea[j])
                    tmp_fea = (circRNA_fea_tmp, disease_fea_tmp)  # concatnated feature vector for an association
                    train.append(tmp_fea)  # train contains feature vectors of all associated samples

                elif interaction[i, j] == 0:  # for no association
                    label1.append(interaction[i, j])  # label2= labels for no association(0)
                    # nonlink_number = nonlink_number + 1
                    # nonLinksPosition.append([i, j])
                    circRNA_fea_tmp1 = list(circRNA_fea[i])
                    disease_fea_tmp1 = list(disease_fea[j])
                    test_fea = (
                    circRNA_fea_tmp1, disease_fea_tmp1)  # concatenated feature vector for not having association
                    train.append(test_fea)  # testfnl contains feature vectors of all non associated samples

            print('prepare test data over, mv=', mv)
            train = np.array(train)
            X_test.append(train)
            Y_test.append(label1)
            print(len(train))
            print(len(label1))

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def calculate_performace(self, test_num, pred_y, labels):  # pred_y = proba, labels = real_labels
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index in range(test_num):
            if labels[index] == 1:
                if labels[index] == pred_y[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if labels[index] == pred_y[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1

        acc = float(tp + tn) / test_num

        if tp == 0 and fp == 0:
            precision = 0
            MCC = 0
            f1_score = 0
            sensitivity = float(tp) / (tp + fn)
            specificity = float(tn) / (tn + fp)
        else:
            precision = float(tp) / (tp + fp)
            sensitivity = float(tp) / (tp + fn)
            specificity = float(tn) / (tn + fp)
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
        print("该测试集数目：",test_num)  # test_num=108
        print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)
        return acc, precision, sensitivity, specificity, MCC, f1_score, tp, tn, fp, fn

    def transfer_array_format(self, data):  # preResult=X  , X= all the miRNA features, disease features
        formated_matrix1 = []
        formated_matrix2 = []
        # 返回分割后的两个数组，分别存储circRNA和miRNA的特征
        for val in data:# data type:<class 'numpy.ndarray'>
            formated_matrix1.append(val[0])  # contains circRNA features
            formated_matrix2.append(val[1])  # contains miRNA features

        return np.array(formated_matrix1), np.array(formated_matrix2)

    def preprocess_labels(self, labels, encoder=None, categorical=True):
        if not encoder:
            encoder = LabelEncoder()
            encoder.fit(labels)
        y = encoder.transform(labels).astype(np.int32)
        if categorical:
            y = np_utils.to_categorical(y)
        return y, encoder

    class Config(object):
        def __init__(self):
            self.data_path = '../../datasets'
            self.validation = 1
            self.save_path = '../preResult'

            self.epoch = 100# 用不到
            self.alpha = 0.2

    class Sizes(object):
        def __init__(self, dataset):
            self.m = dataset['mm']['preResult'].size(0)
            self.d = dataset['dd']['preResult'].size(0)
            self.fg = 256
            self.fd = 256
            self.k = 32

    def train(self, model, train_data, optimizer, opt):
        model.train()
        # regression_crit = Myloss()
        # one_index = train_data[2][0FS].cuda().t().tolist()
        # zero_index = train_data[2][1].cuda().t().tolist()

        def train_epoch():
            model.zero_grad()
            # 下面这一行出问题了(已解决）
            score, ciRNAEmbed, disEmbed = model(train_data)
            loss = torch.nn.MSELoss(reduction='mean')
            # loss = loss(score, train_data['md_p'].cuda())
            loss = loss(score, train_data['md_p'])
            loss.backward()
            optimizer.step()
            return loss

        def getEmbedding():
            model.zero_grad()
            score, ciRNAEmbed, disEmbed = model(train_data)
            return ciRNAEmbed, disEmbed

        for epoch in range(1, opt.epoch + 1):
            train_reg_loss = train_epoch()
        ciRNAEmbed, disEmbed = getEmbedding()
        print('after model.train()')
        return ciRNAEmbed, disEmbed

    opt = Config()



    def work_on_test_set(self):
        info = f"固定所有随机数({seed})，使用{text}，cat为425,epoch为800（work_on_test_set)nodecirc=0,mi=0"
        if self.is_loading_traindatas == False:
            self.prepare_data3()
        self.is_loading_traindatas = True

        X = self.X_train
        labels = self.Y_train
        X_data1 = [] # 存储训练集miRNA和circRNA的特征
        X_test_data1 = [] # 存储测试集miRNA和circRNA的特征
        y = [] # 存储训练集样本的标签
        y_test = [] # 存储测试集样本的标签
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for j in range(3):
            # 这里的3是指有3个miRNA和circRNA的embedding
            X_data1_, X_data2_ = self.transfer_array_format(
                X[j])  # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)

            X_test_data1_, X_test_data2_ = self.transfer_array_format(self.X_test[j])

            X_data1_ = np.concatenate((X_data1_, X_data2_), axis=1)  # axis=1 , rowwoise concatenation
            X_test_data1_ = np.concatenate((X_test_data1_, X_test_data2_), axis=1)  # axis=1 , rowwoise concatenation

            y_ = np.array(labels[j])
            y_test_ = np.array(self.Y_test[j])
            t = 0
            X_data1.append(X_data1_)
            X_test_data1.append(X_test_data1_)
            y.append(y_)
            y_test.append(y_test_)

        train1 = []
        test1 = []
        train_label = []
        test_label = []
        realLabel = []
        trainLabelNew = []
        probaList = []
        probaCoefList = []

        for i in range(3):
            trainTmp = np.array([x for i, x in enumerate(X_data1[i]) if True])
            testTmp = np.array([x for i, x in enumerate(X_test_data1[i]) if True])
            train_labelTmp = np.array([x for i, x in enumerate(y[i]) if True])
            test_labelTmp = np.array([x for i, x in enumerate(y_test[i]) if True])

            train1.append(trainTmp)
            test1.append(testTmp)
            train_label.append(train_labelTmp)
            test_label.append(test_labelTmp)

        for i in range(3):
            real_labelTmp = []
            for val in test_label[i]:
                if val == 0:  # tuples in array, val[0]- first element of tuple
                    real_labelTmp.append(0)
                else:
                    real_labelTmp.append(1)

            train_label_newTmp = []
            for val in train_label[i]:
                if val == 0:
                    train_label_newTmp.append(0)
                else:
                    train_label_newTmp.append(1)
            class_index = 0
            prefilter_train = train1[i]
            prefilter_test = test1[i]

            # clf = XGBClassifier(n_estimators=self.cat_epoch, max_depth=self.cat_depth)
            clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed, verbose=0,
                                     task_type='CPU')
            clf.fit(prefilter_train, train_label_newTmp)  # ** *Training
            ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # **testing

            clf.save_model(self.path_cat_model + str(i + 1) + '.model')

            proba = self.transfer_label_from_prob(ae_y_pred_prob)
            probaList.append(proba)
            probaCoefList.append(ae_y_pred_prob)
            realLabel.append(real_labelTmp)
            trainLabelNew.append(train_label_newTmp)

        # 计算测试集上的结果
        avgProbCoef = probaCoefList[0]
        for i in range(1, 3):
            tempProb = probaCoefList[i]
            for j in range(len(avgProbCoef)):
                avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
        for i in range(len(avgProbCoef)):
            avgProbCoef[i] = avgProbCoef[i] / 3

        avgProb = self.transfer_label_from_prob(avgProbCoef)
        acc, precision, sensitivity, specificity, MCC, f1_score, tp, tn, fp, fn = self.calculate_performace(
            len(realLabel[1]),
            avgProb,
            realLabel[1])
        # avgProb = transfer_label_from_prob(probaCoefList[9])
        # acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(realLabel[1]), avgProb,
        #                                                                                realLabel[1])

        fpr, tpr, auc_thresholds = roc_curve(realLabel[1], avgProbCoef)
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('raw_DNN', {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score})

        precision1, recall, pr_threshods = precision_recall_curve(realLabel[1], avgProbCoef)

        # plt.plot(recall, precision1, label= 'ROC fold %d (AUC = %0.4f)' % (t, auc_score))

        aupr_score = auc(recall, precision1)
        print("testing_set:\n", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
        print('tp=', tp, 'fp=', fp, 'tn=', tn, 'fn=', fn)
        # 测试集结果保存
        metrics_dict = {
            "准确率": acc, "精确率": precision,
            "敏感性": sensitivity, "特异性": specificity,
            "MCC": MCC, "AUC": auc_score,
            "AUPR": aupr_score, "F1分数": f1_score,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }
        # print(metrics_dict)
        self.save_metrics_to_file(metrics_dict, f"{filename}", info)

    def embedding(self):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args = parameter_parser()
        # 以下为GAT的Embedding过程
        print('embedding.......')

        dataset = prepare_data()
        train_data = dataset

        for k in range(1, 4):
            print('k=', k)
            for i in range(self.opt.validation):
                print('-' * 50)
                model = GATModel(args, k)  # parameter_parser()
                # model.cuda()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # 下面进行模型训练
                ciRNAEmbed, disEmbed = self.train(model, train_data, optimizer, args)
                print()
            ciRNAEmbed = ciRNAEmbed.detach().cpu().numpy()
            diseaseEmbed = disEmbed.detach().cpu().numpy()

            circPath = self.path_circ_embedding + str(k) + '.csv'
            disPath = self.path_mi_embedding + str(k) + '.csv'  # 其实是miRNA的embedding路径
            np.savetxt(circPath, ciRNAEmbed, delimiter=',')
            np.savetxt(disPath, diseaseEmbed, delimiter=',')
        print('embedding over........')



    def cross_validate(self):
        def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color='blue'):
            def MyFrame(x0, y0, width, height):
                import matplotlib.pyplot as plt
                import numpy as np

                x1 = np.linspace(x0, x0, num=20)
                y1 = np.linspace(y0, y0, num=20)
                xk = np.linspace(x0, x0 + width, num=20)
                yk = np.linspace(y0, y0 + height, num=20)

                xkn = []
                ykn = []
                counter = 0
                while counter < 20:
                    xkn.append(x1[counter] + width)
                    ykn.append(y1[counter] + height)
                    counter = counter + 1

                plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)
                plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)
                plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)
                plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)

                return

            width2 = times * width
            height2 = times * height
            MyFrame(x0, y0, width, height)
            MyFrame(x1, y1, width2, height2)

            xp = np.linspace(x0 + width, x1, num=20)
            yp = np.linspace(y0, y1 + height2, num=20)
            plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

            # 筛选区域内的所有点
            XDottedLine = []
            YDottedLine = []

            for counter in range(len(mean_fpr)):
                # 包含区域边界上的所有点
                if (mean_fpr[counter] >= x0 and mean_fpr[counter] <= (x0 + width) and
                        mean_tpr[counter] >= y0 and mean_tpr[counter] <= (y0 + height)):
                    XDottedLine.append(mean_fpr[counter])
                    YDottedLine.append(mean_tpr[counter])

            # 坐标变换到放大区域
            XDottedLine_transformed = []
            YDottedLine_transformed = []

            for i in range(len(XDottedLine)):
                # 线性变换：将原始区域映射到放大区域
                x_transformed = (XDottedLine[i] - x0) / width * width2 + x1
                y_transformed = (YDottedLine[i] - y0) / height * height2 + y1
                XDottedLine_transformed.append(x_transformed)
                YDottedLine_transformed.append(y_transformed)

            # 绘制放大后的曲线
            if len(XDottedLine_transformed) > 0:
                plt.plot(XDottedLine_transformed, YDottedLine_transformed, color=color, lw=thickness, alpha=1)

            return

        # 颜色列表
        colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'orange', 'brown', 'pink', 'gray', 'olive']

        # 在方法开始时重新设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        info = f"固定所有随机数({seed})，使用{text}，cat为425,epoch为800（cross_validate）nodecirc=0,mi=0"
        if self.is_loading_traindatas == False:
            self.prepare_data3()
        self.is_loading_traindatas = True

        # 获取训练和测试数据
        # X_train = self.X_train+self.X_test
        # Y_train = self.Y_train+self.Y_test
        X_train = []
        Y_train = []
        for i in range(3):
            X_train.append(np.concatenate([self.X_train[i], self.X_test[i]], axis=0))
            Y_train.append(np.concatenate([self.Y_train[i], self.Y_test[i]], axis=0))

        Y_test = self.Y_test
        X_test = self.X_test

        X = X_train

        labels = Y_train

        X_data1 = []
        X_data2 = []
        y = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 1000)
        prcs = []  # 用于存储每一折的 precision 曲线
        mean_recall = np.linspace(0, 1, 1000)  # 平均 recall 曲线


        # ————————————————————————————————————————————————下面这一行关注一下
        num = np.arange(1666)  # 这个是circ_mi关联数量*2，也就是正样本数量*2
        # ————————————————————————————————————————————————上面这一行关注一下！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        np.random.shuffle(num)
        for j in range(3):
            X_data1_, X_data2_ = self.transfer_array_format(
                X[j])  # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)

            X_data1_ = np.concatenate((X_data1_, X_data2_), axis=1)  # axis=1 , rowwoise concatenation
            # 为什么拼接？ # 因为每个circRNA和miRNA都有两个特征向量，一个是miRNA的特征向量，一个是circRNA的特征向量

            y_, encoder = self.preprocess_labels(labels[j])  # labels labels_new
            X_data1_ = X_data1_[num]  # 代表随机打乱circRNA和miRNA的特征向量

            # X_data2_ = X_data2_[num]# 代表随机打乱circRNA和miRNA的特征向量
            y_ = Y_train[j]
            y_ = np.array(y_, dtype=np.int32)
            y_ = y_[num]
            t = 0
            X_data1.append(X_data1_)
            # X_data2.append(X_data2_)
            y.append(y_)

        num_cross_val = 5  # 5折交叉验证

        all_performance = []  # 存储每一折的性能指标

        all_prob = {}
        num_classifier = 3
        all_prob[0] = []
        all_prob[1] = []
        all_prob[2] = []
        all_prob[3] = []
        all_averrage = []

        tprs = []

        clf_start_time = time.time()
        for fold in range(num_cross_val):
            # 每一折的3种分类器数据
            train1 = []
            test1 = []
            train_label = []
            test_label = []
            realLabel = []
            trainLabelNew = []
            probaList = []
            probaCoefList = []

            for i in range(3):
                trainTmp = np.array([x for i, x in enumerate(X_data1[i]) if i % num_cross_val != fold])
                testTmp = np.array([x for i, x in enumerate(X_data1[i]) if i % num_cross_val == fold])
                train_labelTmp = np.array([x for i, x in enumerate(y[i]) if i % num_cross_val != fold])
                test_labelTmp = np.array([x for i, x in enumerate(y[i]) if i % num_cross_val == fold])

                # 训练集的特征以及标签
                train1.append(trainTmp)
                test1.append(testTmp)
                # 验证集的特征以及标签
                train_label.append(train_labelTmp)
                test_label.append(test_labelTmp)

            clfName = ''
            # 分类
            # ！！！！！！！！！！分类部分的逻辑：训练9个分类器，然后九个分类器的结果取平均值，作为最后的结果！！！！！！！！！！！！
            # 目前了来看，每个分类器的训练样本不一样啊，因为它的负样本是随机选取的啊。
            for i in range(3):
                real_labelTmp = []
                for val in test_label[i]:
                    if val == 1:
                        real_labelTmp.append(1)
                    else:
                        real_labelTmp.append(0)
                train_label_newTmp = []
                for val in train_label[i]:
                    if val == 1:
                        train_label_newTmp.append(1)
                    else:
                        train_label_newTmp.append(0)
                class_index = 0
                prefilter_train = train1[i]  # 训练集特征数据
                prefilter_test = test1[i]

                # clf = XGBClassifier(n_estimators=self.cat_epoch, max_depth=self.cat_depth)
                clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed,
                                         verbose=0, task_type='CPU')
                clf.fit(prefilter_train, train_label_newTmp)  # ** *Training
                # [:,1] # 取第二列的概率值(正类的概率值)
                ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # **testing

                proba = self.transfer_label_from_prob(ae_y_pred_prob)  #
                probaList.append(proba)
                probaCoefList.append(ae_y_pred_prob)
                realLabel.append(real_labelTmp)  # 加载验证集真实的标签
                trainLabelNew.append(train_label_newTmp)

            # 单独一折求平均
            avgProbCoef = probaCoefList[0]
            for i in range(1, 3):
                tempProb = probaCoefList[i]
                for j in range(len(avgProbCoef)):
                    avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
            for i in range(len(avgProbCoef)):
                avgProbCoef[i] = avgProbCoef[i] / 3

            avgProb = self.transfer_label_from_prob(avgProbCoef)
            acc, precision, sensitivity, specificity, MCC, f1_score, tp, tn, fp, fn = self.calculate_performace(
                len(realLabel[0]), avgProb,
                realLabel[0])

            fpr, tpr, auc_thresholds = roc_curve(realLabel[0], avgProbCoef)
            auc_score = auc(fpr, tpr)

            scipy.io.savemat('raw_DNN', {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score})

            precision1, recall, _ = precision_recall_curve(realLabel[0], avgProbCoef)

            aupr_score = auc(recall, precision1)

            print("AUTO-RF: ACC =", acc,
                  "Precision =", precision,
                  "Recall =", sensitivity,
                  "Specificity =", specificity,
                  "MCC =", MCC,
                  "AUC =", auc_score,
                  "AUPR =", aupr_score,
                  "F1 =", f1_score)

            all_performance.append(
                [acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score, tp, tn, fp, fn])
            t = t + 1  # AUC fold number

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # plt.plot(fpr, tpr, label='%s(AUC = %0.4f)' % (clfName, auc_score))
            # plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
            plt.figure(1)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.8, color=colorlist[t-1],
                     label='fold %d (AUC = %0.4f)' % (t, auc_score))
            MyEnlarge(0, 0.8, 0.2, 0.20, 0.5, 0, 2.5, mean_fpr, tprs[t-1], 1.5, colorlist[t-1])


            recall=recall[::-1]
            precision1 = precision1[::-1]
            recall = np.append(recall, 1.0)
            precision1 = np.append(precision1, 0.0)
            # 绘制当前 AUPR 曲线
            plt.figure(2)
            plt.plot(recall, precision1, lw=1.5, alpha=0.8, color=colorlist[t - 1],
                    label='fold %d (AUPR = %0.4f)' % (t, aupr_score))


            # 绘制当前 AUPR 曲线
            prc_interp = np.interp(mean_recall, recall, precision1)
            prc_interp = np.clip(prc_interp, 0.0, 1.0)
            prc_interp[0] = 1.0
            prcs.append(interp(mean_recall, recall, precision1))

            # 添加子图放大（比如 Recall 0.0~0.2, Precision 0.8~1.0）
            MyEnlarge(0.8, 0.8, 0.2, 0.2, 0.0, 0.0, 2.5, mean_recall, prcs[t - 1], 1.5, colorlist[t - 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)  # one dimensional interpolation
            mean_tpr[0] = 0.0

            plt.xlabel('False positive rate, (1-Specificity)')
            plt.ylabel('True positive rate,(Sensitivity)')
            plt.title('Receiver Operating Characteristic curve: 5-Fold CV')
            # plt.title('Five classification method comparision')

        clf_time = -clf_start_time + time.time();

        print("clf using ", clf_time, 's')

        mean_tpr /= num_cross_val
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        #
        #
        #
        #
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        # plt.plot(mean_fpr, mean_tpr, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % mean_auc)
        plt.figure(1)
        plt.plot(mean_fpr, mean_tpr, color='black',
                 label=r'Mean (AUC = %0.4f)' % (mean_auc),
                 lw=2, alpha=1)
        # MyEnlarge(0, 0.8, 0.15, 0.20, 0.5, 0, 2.5, mean_fpr, mean_tpr, 2, colorlist[5])
        # plt.legend()
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC curve')
        plt.legend(bbox_to_anchor=(0.48, 0.7))

        plt.savefig('ROC-5fold.svg', bbox_inches='tight', pad_inches=0.05)
        plt.savefig('ROC-5fold.tif', dpi=600, bbox_inches='tight', pad_inches=0.05)

        # ======================= 保存 AUPR 图（Figure 2） ==========================
        plt.figure(2)
        mean_precision = np.mean(prcs, axis=0)
        mean_aupr = auc(mean_recall, mean_precision)

        plt.plot(mean_recall, mean_precision, color='black',
                 label=r'Mean (AUPR = %0.4f)' % mean_aupr,
                 lw=2, alpha=1)

        # MyEnlarge(0.8, 0.8, 0.2, 0.2, 0.0, 0.0, 2.5, mean_recall, mean_precision, 2, 'black')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontsize=13)
        plt.ylabel('Precision', fontsize=13)
        plt.title('PR curve')
        plt.legend(bbox_to_anchor=(0.45, 0.7))
        plt.tight_layout()
        plt.savefig('PRC-5fold.svg', bbox_inches='tight', pad_inches=0.05,dpi=1200,)
        # plt.savefig('PRC-5fold.tif', dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.savefig('PRC-5fold.tif')
        # os._exit(0)

        print('*******AUTO-RF*****')
        print('mean performance of rf using raw feature')
        print(np.mean(np.array(all_performance), axis=0))
        Mean_Result = []
        Mean_Result = np.mean(np.array(all_performance), axis=0)
        Std_Result = []
        Std_Result = np.std(np.array(all_performance), axis=0)

        print('---' * 20)
        print('mean_TP=', Mean_Result[8], 'mean_TN=', Mean_Result[9], 'mean_FP=', Mean_Result[10], 'mean_FN=',
              Mean_Result[11])
        print('Mean-Accuracy=', Mean_Result[0], Std_Result[0], '\n Mean-precision=', Mean_Result[1], Std_Result[1])
        print('Mean-Sensitivity=', Mean_Result[2], Std_Result[2], '\n Mean-Specificity=', Mean_Result[3], Std_Result[3])
        print('Mean-MCC=', Mean_Result[4], Std_Result[4], '\n' 'Mean-auc_score=', Mean_Result[5], Std_Result[5])
        print('Mean-Aupr-score=', Mean_Result[6], Std_Result[6], '\n' 'Mean_F1=', Mean_Result[7], Std_Result[7])
        print('---' * 20)
        # 计算完成后保存结果
        metrics_dict = {
            "cat_epoch": self.cat_epoch, "cat_depth": self.cat_depth,
            "准确率": Mean_Result[0], "准确率标准差": Std_Result[0],
            "精确率": Mean_Result[1], "精确率标准差": Std_Result[1],
            "敏感性": Mean_Result[2], "敏感性标准差": Std_Result[2],
            "特异性": Mean_Result[3], "特异性标准差": Std_Result[3],
            "MCC": Mean_Result[4], "MCC标准差": Std_Result[4],
            "AUC": Mean_Result[5], "AUC标准差": Std_Result[5],
            "AUPR": Mean_Result[6], "AUPR标准差": Std_Result[6],
            "F1分数": Mean_Result[7], "F1分数标准差": Std_Result[7],
            "平均TP": Mean_Result[8], "平均TN": Mean_Result[9],
            "平均FP": Mean_Result[10], "平均FN": Mean_Result[11],
            "交叉验证折数": num_cross_val
        }

        self.save_metrics_to_file(metrics_dict, f"{filename}", info)

    def transfer_label_from_prob(self, proba):
        label = [1 if val >= self.threshold else 0 for val in proba]
        return label

    def save_metrics_to_file(self, metrics_dict, filename, info, mode="a"):

        with open(filename, mode, encoding='utf-8') as f:
            # 写入评估信息
            f.write(f"评估信息：{info}\n")
            # 写入时间戳
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 写入所有指标
            for metric_name, metric_value in metrics_dict.items():
                if isinstance(metric_value, (int, float)):
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {metric_value}\n")

            # 添加分隔线
            f.write("-" * 50 + "\n\n")

    def prediction(self, type="circRNA", name="", sequence=""):
        if not self.is_loading_embeddings:
            self.load_embbedings()
        self.is_loading_embeddings = True

        id = -1
        circ_list = self.circ_list
        mi_list = self.mi_list
        for i in range(len(circ_list)):
            if circ_list[i] == name:
                id = i
                break
        # 如果找得到
        if id != -1:
            probaCoefList = []
            for i in range(3):
                # 先加载catboost模型
                clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed,
                                         verbose=0, task_type='GPU')
                clf.load_model(self.path_cat_model + str(i + 1) + '.model')
                # 然后加载每一对ci-mi组合的特征
                X = []
                circRNA_fea_tmp = list(self.circ_embedding[i][id])
                for j in range(self.mi_number):
                    miRAN_fea_tmp = list(self.mi_embedding[i][j])
                    tmp_fea = (circRNA_fea_tmp, miRAN_fea_tmp)
                    X.append(tmp_fea)

                X = np.array(X)

                X_data1, X_data2 = self.transfer_array_format(X)
                X_data1 = np.concatenate((X_data1, X_data2), axis=1)

                # 进行预测
                ae_y_pred_prob = clf.predict_proba(X_data1)[:, 1]
                probaCoefList.append(ae_y_pred_prob)

            # 下面计算结果
            avgProbCoef = probaCoefList[0]
            for i in range(1, 3):
                tempProb = probaCoefList[i]
                for j in range(len(avgProbCoef)):
                    avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
            for i in range(len(avgProbCoef)):
                avgProbCoef[i] = avgProbCoef[i] / 3
                avgProbCoef[i] = 1.0 / (1 + np.exp(-8 * (avgProbCoef[i] - self.threshold)))

            # 列表
            ans_pred = []  # 一个二元组，第一位是概率，第二位是mi的名字
            for i in range(self.mi_number):
                ans_pred.append((mi_list[i], self.mi_seq_list[i], int(10000 * avgProbCoef[i]) / 10000))

            ans_pred = sorted(ans_pred, key=lambda x: x[2], reverse=True)

            print(ans_pred[:10])
            return ans_pred[:10]

        else:
            id = -1
            circ_list = self.circ_list
            mi_list = self.mi_list
            for i in range(len(mi_list)):
                if mi_list[i] == name:
                    id = i
                    break
            # 如果找不到
            if id == -1:
                print('Can not foundthis RNA \n')
                return str('Can not find this RNA')

            print('id:', id)

            probaCoefList = []

            for i in range(3):

                # 先加载catboost模型
                clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed,
                                         verbose=0, task_type='GPU')
                clf.load_model(self.path_cat_model + str(i + 1) + '.model')
                # 然后加载每一对ci-mi组合的特征
                X = []
                miRNA_fea_tmp = list(self.mi_embedding[i][id])
                for j in range(self.circ_number):
                    circRNA_fea_tmp = list(self.circ_embedding[i][j])
                    tmp_fea = (circRNA_fea_tmp, miRNA_fea_tmp)
                    X.append(tmp_fea)

                X = np.array(X)

                X_data1, X_data2 = self.transfer_array_format(X)
                X_data1 = np.concatenate((X_data1, X_data2), axis=1)

                # 进行预测
                ae_y_pred_prob = clf.predict_proba(X_data1)[:, 1]
                probaCoefList.append(ae_y_pred_prob)

            # 下面计算结果
            avgProbCoef = probaCoefList[0]
            for i in range(1, 3):
                tempProb = probaCoefList[i]
                for j in range(len(avgProbCoef)):
                    avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
            for i in range(len(avgProbCoef)):
                avgProbCoef[i] = avgProbCoef[i] / 3
                avgProbCoef[i] = 1.0 / (1 + np.exp(-8 * (avgProbCoef[i] - self.threshold)))

            # 列表
            ans_pred = []  # 一个二元组，第一位是概率，第二位是mi的名字
            for i in range(self.circ_number):
                ans_pred.append((circ_list[i], self.circ_seq_list[i][:10], int(10000 * avgProbCoef[i]) / 10000))

            ans_pred = sorted(ans_pred, key=lambda x: x[2], reverse=True)

            print(ans_pred[:10])
            return ans_pred[:10]


if __name__ == "__main__":
    # torch.set_num_threads(8)
    print(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    cmi = CMI()

    # 先进行embedding操作
    time1 = time.time()
    cmi.embedding()
    time2 = time.time()
    embedding_tiem = time2 - time1
    minutes = int(embedding_tiem // 60)
    seconds = embedding_tiem % 60
    runtime_str = f"embedding程序运行时间：{minutes} 分 {seconds:.2f} 秒\n"
    with open("runtime_log.txt", "a", encoding="utf-8") as f:
        f.write(runtime_str)


    # 然后对模型进行五折交叉验证
    time1 = time.time()
    cmi.cross_validate()
    time2 = time.time()
    embedding_tiem = time2 - time1
    minutes = int(embedding_tiem // 60)
    seconds = embedding_tiem % 60
    runtime_str = f"训练程序运行时间：{minutes} 分 {seconds:.2f} 秒\n"
    with open("runtime_log.txt", "a", encoding="utf-8") as f:
        f.write(runtime_str)



    time1 = time.time()
    # 最后再跑测试集
    cmi.work_on_test_set()
    time2 = time.time()
    embedding_tiem = time2 - time1
    minutes = int(embedding_tiem // 60)
    seconds = embedding_tiem % 60
    runtime_str = f"测试程序运行时间：{minutes} 分 {seconds:.2f} 秒\n"
    with open("runtime_log.txt", "a", encoding="utf-8") as f:
        f.write(runtime_str)

    # cmi.prediction(name="hsa_circ_0000615")
    print(1)
    # cmi.prediction(name="hsa_circ_0000615")

    # .prediction(name="miR-338-3p")

# >hsa_circ_0000615
# CAATGATGTTGTCCACTGGGCATGTACTGACCAATGT
# >miR-338-3p
# UCCAGCAUCAGUGAUUUUGUUG



