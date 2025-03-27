import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                           kernel_size, padding=(kernel_size - 1) // 2))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels,
                                           kernel_size, padding=(kernel_size - 1) // 2))
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.activation(self.conv2(x)))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.activation(x + residual)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers += [TemporalBlock(in_channels, num_channels[i], kernel_size, dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GAT_TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 节点特征编码（关键修改：输入维度适配）
        self.node_encoder = nn.Sequential(
            nn.Linear(5, 32),  # 输入dim=5（对应data_loader中的5个节点特征）
            nn.ReLU(),
            nn.Linear(32, config['model']['gat_hidden_dim'])
        )

        # 2. GAT层（修改：显式使用边特征）
        self.gat_convs = nn.ModuleList()
        in_channels = config['model']['gat_hidden_dim']
        for i in range(config['model']['gat_num_layers']):
            out_channels = config['model']['gat_hidden_dim']
            heads = config['model']['gat_num_heads']
            concat = True if i < config['model']['gat_num_layers'] - 1 else False
            self.gat_convs.append(
                GATConv(
                    in_channels,
                    out_channels // heads,
                    heads=heads,
                    concat=concat,
                    dropout=config['model']['tcn_dropout'],
                    edge_dim=3,  # 显式声明使用3维边特征
                    add_self_loops=False  # 微服务调用图通常不需要自环
                )
            )
            in_channels = out_channels

        # 3. 动态边权重计算（增强版）
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(3, 32),  # 输入：avg_duration, call_count, error_rate
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 4. TCN日志处理器（修改输入维度）
        self.tcn = TCN(
            num_inputs=1,  # 输入dim=1（每个服务的日志事件计数）
            num_channels=config['model']['tcn_num_channels'],
            kernel_size=config['model']['tcn_kernel_size'],
            dropout=config['model']['tcn_dropout']
        )

        # 5. 特征融合（增强版）
        gat_out_dim = config['model']['gat_hidden_dim']
        tcn_out_dim = config['model']['tcn_num_channels'][-1]
        self.fusion = nn.Sequential(
            nn.Linear(gat_out_dim + tcn_out_dim + 2, 256),  # +2来自duration和path_length
            nn.ReLU(),
            nn.Dropout(config['model']['tcn_dropout']),
            nn.LayerNorm(256),
            nn.Linear(256, 128)
        )

        # 6. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['model']['num_classes'])
        )

    def forward(self, data):
        # 1. 节点特征编码
        x = self.node_encoder(data.x)  # [num_nodes, gat_hidden_dim]

        # 2. 动态边权重计算
        edge_weight = self.edge_weight_mlp(data.edge_attr).squeeze()  # [num_edges]

        # 3. GAT处理
        for conv in self.gat_convs[:-1]:
            x = F.elu(conv(x, data.edge_index, edge_attr=data.edge_attr, edge_weight=edge_weight))
            x = F.dropout(x, p=self.config['model']['tcn_dropout'], training=self.training)
        x = self.gat_convs[-1](x, data.edge_index, edge_attr=data.edge_attr, edge_weight=edge_weight)

        # 4. 日志序列处理
        log_features = self.tcn(data.log_features.T)  # [seq_len, num_nodes] -> [seq_len, tcn_out_dim]
        log_features = log_features.mean(dim=0)  # [tcn_out_dim]

        # 5. 全局图特征聚合
        graph_features = torch.cat([
            x.mean(dim=0),  # 全局节点特征均值
            torch.tensor([data.duration.item(), data.path_length.item()]).to(x.device)  # 添加轨迹特征
        ])

        # 6. 特征融合
        combined = torch.cat([graph_features, log_features], dim=-1)
        combined = self.fusion(combined)

        # 7. 分类
        out = self.classifier(combined)
        return F.log_softmax(out, dim=-1)