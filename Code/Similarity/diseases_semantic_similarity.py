import numpy as np
import pandas as pd

# 疾病语义相似度提取


meshid = pd.read_csv('F:\PyCharmProject\GECCMI\dataop\myCMI\circMeSHID.csv', header=0)
disease = meshid['MeSH'].tolist()
id = meshid['MeSHID'].tolist()# 疾病的MeSH ID

meshdis = pd.read_csv('F:\PyCharmProject\GECCMI\dataop\myCMI\circmesh.csv', header=0)
unique_disease = meshdis['MeSH'].tolist()

for i in range(len(disease)):
    disease[i] = {}

print("开始计算每个病的DV")

# 遍历每个疾病的MeSH ID，计算其对应的语义相似度
# 这个循环结束后，disease列表中的每个字典都包含了该疾病的MeSH ID及其对应的语义相似度
for i in range(len(disease)):

    if len(id[i]) > 3:
        disease[i][id[i]] = 1#id[i](疾病的MeSH ID) 作为键（key），将其对应的值（value）赋为 1
        id[i] = id[i][:-4]
        # print(disease[i])
        if len(id[i]) > 3:
            disease[i][id[i]] = round(1 * 0.8, 5)
            id[i] = id[i][:-4]
            # print(disease[i])
            if len(id[i]) > 3:
                disease[i][id[i]] = round(1 * 0.8 * 0.8, 5)
                id[i] = id[i][:-4]
                # print(disease[i])
                if len(id[i]) > 3:
                    disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8, 5)
                    id[i] = id[i][:-4]
                    # print(disease[i])
                    if len(id[i]) > 3:
                        disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                        id[i] = id[i][:-4]
                        # print(disease[i])
                        if len(id[i]) > 3:
                            disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                            id[i] = id[i][:-4]
                            # print(disease[i])
                            if len(id[i]) > 3:
                                disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                id[i] = id[i][:-4]
                                # print(disease[i])
                                if len(id[i]) > 3:
                                    disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                    id[i] = id[i][:-4]
                                    # print(disease[i])
                                else:
                                    disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                    # print(disease[i])
                            else:
                                disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                # print(disease[i])
                        else:
                            disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                            # print(disease[i])
                    else:
                        disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                        # print(disease[i])
                else:
                    disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8, 5)
                    # print(disease[i])
            else:
                disease[i][id[i][:3]] = round(1 * 0.8 * 0.8, 5)
                # print(disease[i])
        else:
            disease[i][id[i][:3]] = round(1 * 0.8, 5)
            # print(disease[i])
    else:
        disease[i][id[i][:3]] = 1
        # print(disease[i])



unique_disease = meshdis['MeSH'].tolist()# 疾病的唯一标识符

disease_name = meshid['MeSH'].tolist()# 疾病名称
unique_disease_name = meshdis['MeSH'].tolist()# 唯一疾病名称

for i in range(len(unique_disease)):
    unique_disease[i] = {}#
    for j in range(len(disease_name)):
        if unique_disease_name[i] == disease_name[j]:
            unique_disease[i].update(disease[j])# 找到对应的疾病名称，将其对应的语义相似度字典添加到唯一疾病字典中
# 此循环结束后，unique_disease列表中的每个字典都包含了该疾病的唯一标识符及其对应的语义相似度（）

similarity = np.zeros([len(unique_disease_name), len(unique_disease_name)])

# 计算疾病之间的语义相似度
for m in range(len(unique_disease_name)):
    for n in range(len(unique_disease_name)):
        # 一个疾病的可能有多个MeSH ID，因此需要将它们的语义相似度进行归一化处理
        denominator = sum(unique_disease[m].values()) + sum(unique_disease[n].values())# 分母为两个疾病的语义相似度之和
        numerator = 0
        for k, v in unique_disease[m].items():
            if k in unique_disease[n].keys():# 如果两个疾病的MeSH ID有交集
                numerator += v + unique_disease[n].get(k)
        similarity[m, n] = round(numerator/denominator, 5)

result = pd.DataFrame(similarity)
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity\circdiseases_semantic_similarity.csv')

result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity\circdiseases_semantic_similarity.txt',index=False,header=False,sep='\t')