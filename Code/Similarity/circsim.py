import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
'''
可以调整k-mer的大小 67行
以及k-mer的频率加权有无 15行
'''

def kmer_count(sequence, k):
    """ 计算 RNA 序列的 k-mer 频率 """
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)  # 统计频率

# 考虑频率加权
def tanimoto_similarity(seq1, seq2):
    """ 计算 Tanimoto 相似性（不消除重复的 k-mer） """
    # 获取每个 k-mer 的频率
    kmers1 = seq1
    kmers2 = seq2

    # 获取所有可能的 k-mer
    all_kmers = set(kmers1.keys()).union(set(kmers2.keys()))

    # 计算交集和并集的频率
    intersection = sum(min(kmers1.get(kmer, 0), kmers2.get(kmer, 0)) for kmer in all_kmers)
    union = sum(max(kmers1.get(kmer, 0), kmers2.get(kmer, 0)) for kmer in all_kmers)

    # 返回 Tanimoto 相似性
    return intersection / union if union != 0 else 0

# 不考虑频率加权
# def tanimoto_similarity(seq1, seq2):
#     """ 计算 Tanimoto 相似性 """
#     set1, set2 = set(seq1.keys()), set(seq2.keys())
#     intersection = len(set1 & set2)
#     union = len(set1 | set2)
#     return intersection / union if union != 0 else 0

# GIPK 计算
def euclidean_distance(seq1, seq2, kmer_list):
    """ 计算 k-mer 频率向量的欧几里得距离"""
    vec1 = np.array([seq1.get(kmer, 0) for kmer in kmer_list])# k-mer频率向量(存在即可还是频率?)
    vec2 = np.array([seq2.get(kmer, 0) for kmer in kmer_list])
    return np.linalg.norm(vec1 - vec2)


def compute_gamma(distance_matrix, gamma_prime=0.5):
    """ 计算 GIPK 相似性的 gamma 参数，基于归一化范数计算 """
    mean_norm = np.mean(distance_matrix ** 2)  # 计算所有元素的均值
    return gamma_prime / mean_norm if mean_norm != 0 else 1.0  # 防止除 0


def gipk_similarity(dist_matrix, gamma):
    """ 计算 GIPK 相似性矩阵 """
    gamma = np.clip(gamma, 1e-5, 1e5)  # 限制 gamma 避免异常情况
    return np.exp(-gamma * (dist_matrix ** 2))


def combined_similarity(tani_matrix, gip_matrix):
    """ 计算最终相似性矩阵：如果 Tanimoto 为 0，则使用 GIPK """
    combined_matrix = np.where(tani_matrix > 0, tani_matrix, gip_matrix)
    print(tani_matrix)
    np.fill_diagonal(combined_matrix, 1)  # 自相似度设为1
    count_tani_le_zero_all = np.sum(tani_matrix <= 0)
    print(f"整个 Tanimoto 矩阵中 <= 0 的元素数目: {count_tani_le_zero_all}")
    return combined_matrix# 返回最终相似性矩阵

# k-mer 的设定需要实验
def build_similarity_matrix(csv_file, k_circ=2, k_mi=5, output_file="F:\PyCharmProject\GECCMI\mydataset\similarity\circrna_sim_matrix."):
    """ 从 CSV 读取 RNA 序列，计算相似性矩阵，并保存为 CSV """
    df = pd.read_csv(csv_file)

    # 提取 RNA 名称和序列
    rna_names = df["circRNA"].tolist()
    rna_sequences = df["sequence"].tolist()

    # 计算 k-mer 频率
    kmer_dict = {name: kmer_count(seq, k_circ) for name, seq in zip(rna_names, rna_sequences)}

    # 确定完整的 k-mer 词典，保证特征维度一致
    all_kmers = sorted(set(kmer for kmers in kmer_dict.values() for kmer in kmers))

    n = len(rna_names)
    distance_matrix = np.zeros((n, n))
    tanimoto_matrix = np.zeros((n, n))

    # 计算 Tanimoto 相似性和欧几里得距离
    for i, j in combinations(range(n), 2):
        tanimoto_sim = tanimoto_similarity(kmer_dict[rna_names[i]], kmer_dict[rna_names[j]])
        distance = euclidean_distance(kmer_dict[rna_names[i]], kmer_dict[rna_names[j]], all_kmers)

        tanimoto_matrix[i, j] = tanimoto_sim
        tanimoto_matrix[j, i] = tanimoto_sim
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    # 计算 GIPK 相似性
    gamma = compute_gamma(distance_matrix)
    gip_matrix = gipk_similarity(distance_matrix, gamma)

    # 计算最终相似性矩阵
    final_matrix = combined_similarity(tanimoto_matrix, gip_matrix)

    # 保存相似性矩阵
    sim_df = pd.DataFrame(final_matrix, index=rna_names, columns=rna_names)# 用名字索引
    sim_df.to_csv(output_file+"csv")
    sim_df.to_csv(output_file+"txt", index=False,
                  header=False, sep='\t')

    return sim_df


csv_path = "F:\PyCharmProject\GECCMI\dataop\myCMI\circRNA_seq.csv" #序列路径
similarity_matrix = build_similarity_matrix(csv_path)
print(similarity_matrix)
