# import numpy as np
# import pandas as pd
# import math
# from pandas import DataFrame as df
# #根据circ_mi_association关联矩阵，计算GIP_Similarity
# # 读取数据
#
# association_matrix = pd.read_csv('F:\PyCharmProject\GECCMI\mydataset/association\ori_circRNAs_miRNAs_association.csv',encoding='utf-8')
# association_matrix = df(association_matrix)
# association_matrix.set_index(association_matrix.columns[0], inplace=True)
#
# association_matrix=association_matrix.values
# #print(association_matrix)
# # 332列，181行
# miRNA_number = association_matrix.shape[1]
# circRNA_number =association_matrix.shape[0]
#
# # 计算miRNA之间的相似度
# association_matrix = association_matrix.T
#
# miRNA_similarity = np.zeros([miRNA_number, miRNA_number])  # 332个miRNA之间的相似度，初始化矩阵
#
# width = 0
# print('width is :', width)
# for m in range(miRNA_number):
#     width += np.sum(association_matrix[m]**2)**0.5  # 按定义用二阶范数计算width parameter
#
#
# print('width is :', width)
#
# # 计算association_matrix
# count = 0
# for m in range(miRNA_number):
#     for n in range(miRNA_number):
#         miRNA_similarity[m, n] = math.exp((np.sum((association_matrix[m] - association_matrix[n])**2)**0.5
#                                         * width/miRNA_number) * (-1))  # 计算不同行（disease）之间的二阶范数
#        # if miRNA_similarity[m, n] == 1:
#            # miRNA_similarity[m, n] = 0.8  # 这里是一个大问题，两个向量相同可以说它有一定相关度，可是计算出相关度等于1又不合理，只能定义一个值
#
#
# # 保存结果
#
# result = pd.DataFrame(miRNA_similarity)
# #print(result)
# result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/miRNAs_GIP_simimlarity.csv',encoding='utf-8')
#
# result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/miRNAs_GIP_simimlarity.txt',index=False,header=False,sep='\t')
# #计算circRNA
# association_matrix = association_matrix.T
#
# circRNA_similarity = np.zeros([circRNA_number, circRNA_number])  #
#
# width = 0
# print('width is :', width)
# for m in range(circRNA_number):
#     width += np.sum(association_matrix[m]**2)**0.5  # 按定义用二阶范数计算width parameter
#
#
# print('width is :', width)
#
# # 计算association_matrix
# count = 0
# for m in range(circRNA_number):
#     for n in range(circRNA_number):
#         circRNA_similarity[m, n] = math.exp((np.sum((association_matrix[m] - association_matrix[n])**2)**0.5
#                                         * width/circRNA_number) * (-1))  # 计算不同行（disease）之间的二阶范数
#        # if miRNA_similarity[m, n] == 1:
#            # miRNA_similarity[m, n] = 0.8  # 这里是一个大问题，两个向量相同可以说它有一定相关度，可是计算出相关度等于1又不合理，只能定义一个值
# # 保存结果
# print(circRNA_number)
# result = pd.DataFrame(circRNA_similarity)
# #print(result)
# result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/circRNAs_GIP_simimlarity.csv',encoding='utf-8')
#
# result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/circRNAs_GIP_simimlarity.txt',index=False,header=False,sep='\t')











import numpy as np
import pandas as pd
import math
from pandas import DataFrame as df

# 读取数据
association_matrix = pd.read_csv('F:\PyCharmProject\GECCMI\mydataset/association\ori_circRNAs_miRNAs_association.csv', encoding='utf-8')
association_matrix = df(association_matrix)
association_matrix.set_index(association_matrix.columns[0], inplace=True)
association_matrix = association_matrix.values

miRNA_number = association_matrix.shape[1]
circRNA_number = association_matrix.shape[0]

# 计算miRNA之间的相似度
association_matrix = association_matrix.T

# 计算gamma_c：宽度参数
squared_norms = np.sum(association_matrix ** 2, axis=1)  # 每个向量的二范数平方
gamma = 0.5 / np.mean(squared_norms)  # 根据公式计算gamma

miRNA_similarity = np.zeros([miRNA_number, miRNA_number])

# 计算GIP核相似度矩阵
for m in range(miRNA_number):
    for n in range(miRNA_number):
        dist_sq = np.sum((association_matrix[m] - association_matrix[n]) ** 2)  # 距离的平方
        miRNA_similarity[m, n] = math.exp(-gamma * dist_sq)

# 保存结果
result = pd.DataFrame(miRNA_similarity)
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/miRNAs_GIP_simimlarity.csv', encoding='utf-8')
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/miRNAs_GIP_simimlarity.txt', index=False, header=False, sep='\t')

# 计算circRNA之间的相似度
association_matrix = association_matrix.T

squared_norms = np.sum(association_matrix ** 2, axis=1)
gamma = 0.5 / np.mean(squared_norms)

circRNA_similarity = np.zeros([circRNA_number, circRNA_number])

for m in range(circRNA_number):
    for n in range(circRNA_number):
        dist_sq = np.sum((association_matrix[m] - association_matrix[n]) ** 2)
        circRNA_similarity[m, n] = math.exp(-gamma * dist_sq)

# 保存结果
result = pd.DataFrame(circRNA_similarity)
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/circRNAs_GIP_simimlarity.csv', encoding='utf-8')
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity/circRNAs_GIP_simimlarity.txt', index=False, header=False, sep='\t')
