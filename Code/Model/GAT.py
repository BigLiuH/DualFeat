import torch
from torch import nn
from torch_geometric.nn.conv import GATConv



# -------------------------- 完整GATModel类 --------------------------
class GATModel(nn.Module):
    def __init__(self, args, mv):
        super(GATModel, self).__init__()
        self.mv = mv
        self.args = args

        # GAT层定义（保持不变）
        self.gat_x1_f = GATConv(self.args.fm, self.args.fm, heads=8, concat=False)# circ（avg）
        self.gat_x2_f = GATConv(self.args.fm, self.args.fm, heads=8, concat=False)# （max）



        self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):
        torch.manual_seed(1)

        # 选择哪一个视图
        if self.mv == 1:
            x_m = data['seq']['data_matrix']#.cuda()
            x_m_f_edge_index = data['seq']['edges']
        elif self.mv == 2:
            x_m = data['gip']['data_matrix']#.cuda()
            x_m_f_edge_index = data['gip']['edges']
        elif self.mv == 3:
            x_m = data['fun']['data_matrix']#.cuda()
            x_m_f_edge_index = data['fun']['edges']
        else:
            raise ValueError("Invalid mv value. It should be 1, 2, or 3.")

        # 构建边索引
        x_m_f_edge_index = torch.tensor(x_m_f_edge_index, dtype=torch.long, device=x_m.device)

        # GAT 第一层
        # x_m_f1 = torch.relu(self.gat_x1_f(x_m.cuda(), x_m_f_edge_index))  # 第一层
        x_m_f1 = torch.relu(self.gat_x1_f(x_m, x_m_f_edge_index))

        # GAT 第二层
        x_m_f2 = torch.relu(self.gat_x2_f(x_m_f1, x_m_f_edge_index))

        # 拼接两层 GAT 输出
        XM = torch.cat((x_m_f1, x_m_f2), dim=1).t()  # 维度: [N, hidden1 + hidden2]
        XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)
        x = self.cnn_x(XM)  # 原XM_channel_attention改为直接使用XM
        x = x.view(self.args.out_channels, self.args.miRNA_number).t()
        # 拆分为 circRNA 和 miRNA 表示（前504为circRNA，后420为miRNA）
        circ_embed = x[:504, :]  # circRNA节点嵌入
        mirna_embed = x[504:, :]  # miRNA节点嵌入

        # 返回相似度矩阵（乘积），以及两个嵌入矩阵
        return circ_embed.mm(mirna_embed.t()), circ_embed, mirna_embed
