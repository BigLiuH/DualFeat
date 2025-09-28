import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import random

# --------------------------
# 固定随机种子
# --------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --------------------------
# 参数设置（保持与原Node2Vec一致）
# --------------------------
INPUT_FILE = "../../mydataset/similarity/circRNAs_GIP_simimlarity.txt"    # 输入相似度矩阵路径
OUTPUT_FILE = "../../mydataset/DeepWalk/circRNAs_GIP_embedding.txt"       # 输出嵌入文件路径

DIMENSIONS = 256        # 嵌入维度
WALK_LENGTH = 80        # 每条随机游走长度
NUM_WALKS = 10          # 每个节点的游走次数
WINDOW_SIZE = 10        # Word2Vec窗口大小
WEIGHT_THRESHOLD = 0    # 边权重剪枝阈值
EPOCHS = 1              # 训练轮数，DeepWalk原始实现中一般为1
MIN_COUNT = 1           # Word2Vec最小词频

# --------------------------
# Step 1. 读取相似度矩阵
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
            G.add_edge(str(i), str(j))  # 注意节点必须用字符串，Word2Vec输入为str

print(f"图构建完成：节点数 = {G.number_of_nodes()}，边数 = {G.number_of_edges()}")

# --------------------------
# Step 3. 实现 DeepWalk 随机游走
# --------------------------
def deepwalk_walk(graph, start_node, walk_length):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk

def simulate_walks(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(deepwalk_walk(graph, node, walk_length))
    return walks

print("开始生成随机游走路径 ...")
walks = simulate_walks(G, NUM_WALKS, WALK_LENGTH)
print(f"生成随机游走路径完成，总计路径数：{len(walks)}")

# --------------------------
# Step 4. 用 Word2Vec 训练嵌入
# --------------------------
print("开始训练 Word2Vec 模型 ...")
model = Word2Vec(
    sentences=walks,
    vector_size=DIMENSIONS,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    sg=1,                  # 使用Skip-Gram模型
    workers=4,
    seed=SEED,
    epochs=EPOCHS
)

# --------------------------
# Step 5. 提取节点嵌入并保存
# --------------------------
embeddings = np.array([model.wv[str(i)] for i in range(n)])
df_embed = pd.DataFrame(embeddings, index=[f"node_{i}" for i in range(n)])
df_embed.to_csv(OUTPUT_FILE, index=False, header=False, sep='\t')

print(f"嵌入完成，结果保存至：{OUTPUT_FILE}")
