
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# --------------------------
# 参数设置
# --------------------------
INPUT_FILE = "../mydataset/similarity/circRNAs_GIP_simimlarity.txt"    # 输入的相似度矩阵路径（Tab分隔）
OUTPUT_FILE = "../mydataset/node2vec/circRNAs_GIP_embedding.txt" # 输出的嵌入向量文件路径（CSV）

DIMENSIONS = 256         # 嵌入维度
WALK_LENGTH = 80        # 每次游走长度
NUM_WALKS = 10         # 每个节点的游走次数
P = 1                   # 返回概率
Q = 1                   # 探索概率
WEIGHT_THRESHOLD = 0 # 边权重剪枝阈值（过滤微弱相似度）
# circavg 0.05
# circmax 0.05
# miavg 0.2
# mimax 0.2
# --------------------------
# Step 1. 读取相似度矩阵（Tab分隔的 TXT 文件）
# --------------------------
sim_matrix = pd.read_csv(INPUT_FILE, sep='\t', header=None).values
assert sim_matrix.shape[0] == sim_matrix.shape[1], "输入矩阵必须是方阵！"
n = sim_matrix.shape[0]

# --------------------------
# Step 2. 构建图（无向图）
# --------------------------
G = nx.Graph()
for i in range(n):
    for j in range(n):
        if i != j and sim_matrix[i, j] >= WEIGHT_THRESHOLD:
            G.add_edge(i, j, weight=sim_matrix[i, j])

print(f"图构建完成：节点数 = {G.number_of_nodes()}，边数 = {G.number_of_edges()}")

# --------------------------
# Step 3. 运行 Node2Vec 嵌入
# --------------------------
node2vec = Node2Vec(
    G,
    dimensions=DIMENSIONS,
    walk_length=WALK_LENGTH,
    num_walks=NUM_WALKS,
    p=P,
    q=Q,
    weight_key='weight',
    workers=8,
    seed=SEED
)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# --------------------------
# Step 4. 提取嵌入并保存
# --------------------------
embeddings = np.array([model.wv[str(i)] for i in range(n)])
df_embed = pd.DataFrame(embeddings, index=[f"node_{i}" for i in range(n)])
df_embed.to_csv(OUTPUT_FILE,index=False, header=False, sep='\t')

print(f"嵌入完成，结果保存至：{OUTPUT_FILE}")