import pandas as pd
import numpy as np

dis_ss = pd.read_csv('F:\PyCharmProject\GECCMI\mydataset\similarity\midiseases_semantic_similarity.csv', header=0,index_col=0).values
cd_adjmat = pd.read_csv('F:\PyCharmProject\GECCMI\mydataset\\association\ori_miRNAs_diseases_association.csv', header=0,index_col=0).values

rows = len(cd_adjmat)
result = np.zeros([rows, rows])

for i in range(rows): #遍历每一个micRNA（每一行）
    index_list = []
    for k in range(len(cd_adjmat[1])):#然后看每一列
       if cd_adjmat[i][k] == 1:
           index_list.append(k)  #把这一行为1的元素位置给放到index_list里面
    if len(index_list) == 0:     #如果这个circRNA不与任何疾病关联，则不用去管它
        continue

    for j in range(0, i+1):     #便利前面的每一行（包括自身）
        index_list2 = []
        for k in range(len(cd_adjmat[1])):#然后看每一列
            if cd_adjmat[j][k] == 1:
                index_list2.append(k)
        if len(index_list2) == 0:
            continue
        sum1=0
        sum2=0

        for k1 in range(len(index_list)):
            sum1 = sum1 + max(dis_ss[index_list[k1], index_list2])
        for k2 in range(len(index_list2)):
            sum2 = sum2 + max(dis_ss[index_list, index_list2[k2]])
        result[i, j] = (sum1 + sum2) / (len(index_list) + len(index_list2))
        result[j, i] = result[i, j]

for t in range(rows):
    result[t][t] = 1

result = pd.DataFrame(result)
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity\miRNAs_function_similarity.csv')
result.to_csv('F:\PyCharmProject\GECCMI\mydataset\similarity\miRNAs_function_similarity.txt',index=False,header=False,sep='\t')
