import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from typing import Optional, Callable


class TraceLogDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """
        参数:
            root: 数据根目录路径
            transform: 可选的数据变换函数
            pre_transform: 可选的预处理函数
        """
        self.root = os.path.abspath(root)
        super().__init__(self.root, transform, pre_transform)

        # 加载原始数据文件
        self.traces = pd.read_csv(osp.join(self.root, 'traces_processed.csv'))
        self.logs = pd.read_csv(osp.join(self.root, 'logs_processed.csv'))
        self.service_graph = nx.read_gexf(osp.join(self.root, 'service_graph.gexf'))

        # 预处理日志数据
        self._preprocess_logs()

        # 构建数据集
        self._preprocess()

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def raw_dir(self) -> str:
        return self.root

    def _preprocess_logs(self):
        """从日志消息中提取处理时间"""
        # 提取耗时（单位：ms）
        self.logs['duration_ms'] = self.logs['message'].str.extract(r'(\d+)ms').astype(float)
        # 提取服务名（如果日志格式统一）
        self.logs['service'] = self.logs['message'].str.extract(r'^(ts-\w+-service)')

    def _build_node_features(self, nodes: list) -> torch.Tensor:
        """构建节点特征矩阵"""
        node_features = []
        for node in nodes:
            attrs = self.service_graph.nodes[node]
            features = [
                attrs.get('avg_cpu', 0.0),  # 平均CPU使用率
                attrs.get('avg_mem', 0.0),  # 平均内存使用
                attrs.get('avg_duration', 0.0),  # 平均处理时间
                attrs.get('error_rate', 0.0),  # 错误率
                attrs.get('degree_centrality', 0.0)  # 节点中心度
            ]
            node_features.append(features)
        return torch.tensor(node_features, dtype=torch.float)

    def _build_edge_attributes(self, edges: list, node_to_idx: dict) -> tuple:
        """构建边索引和边属性"""
        edge_indices = []
        edge_attrs = []

        for u, v, d in edges:
            # 添加边索引
            edge_indices.append([node_to_idx[u], node_to_idx[v]])

            # 构建边属性 [avg_duration, call_count, error_rate]
            edge_attrs.append([
                d.get('avg_duration', 0.0),
                np.log1p(d.get('call_count', 1)),  # 对数变换
                d.get('error_rate', 0.0)
            ])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return edge_index, edge_attr

    def _parse_service_path(self, path_str: str) -> list:
        """解析调用路径字符串"""
        return [node.strip() for node in path_str.split('->') if node.strip()]

    def _get_trace_label(self, trace) -> torch.Tensor:
        """根据anomaly_type列生成标签（关键修改点）"""
        # 将normal标记为0，其他异常类型标记为1
        return torch.tensor(0 if trace.anomaly_type == 'normal' else 1, dtype=torch.long)

    def _preprocess(self):
        """主预处理函数"""
        # 获取所有服务节点
        nodes = list(self.service_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # 1. 构建全局图数据
        x = self._build_node_features(nodes)
        edge_index, edge_attr = self._build_edge_attributes(
            list(self.service_graph.edges(data=True)), node_to_idx)

        # 2. 处理每条trace
        self.processed_data = []
        for trace in self.traces.itertuples():
            # 获取日志序列特征（示例：使用事件计数）
            trace_logs = self.logs[self.logs['trace_id'] == trace.trace_id]
            log_features = torch.zeros(len(nodes))  # 每个服务的日志事件计数

            for _, log in trace_logs.iterrows():
                if log.service in node_to_idx:
                    log_features[node_to_idx[log.service]] += 1

            # 创建Data对象（添加duration和path_length特征）
            data = Data(
                x=x.clone(),  # 节点特征
                edge_index=edge_index,  # 边拓扑
                edge_attr=edge_attr.clone(),  # 边属性
                log_features=log_features.unsqueeze(1),  # 日志特征 [num_nodes, 1]
                y=self._get_trace_label(trace),  # 标签
                path=self._parse_service_path(trace.call_path),  # 调用路径（修改列名）
                duration=torch.tensor([trace.duration_ms / 1000.0]),  # 转换为秒
                path_length=torch.tensor([trace.path_length]),  # 路径长度
                trace_id=trace.trace_id  # 用于调试
            )
            self.processed_data.append(data)

    def len(self) -> int:
        return len(self.processed_data)

    def get(self, idx: int) -> Data:
        return self.processed_data[idx]

    def _validate_data(self):
        """数据验证（可选）"""
        assert len(self.processed_data) > 0
        sample_data = self.get(0)
        assert sample_data.x.dim() == 2
        assert sample_data.edge_index.dim() == 2
        assert sample_data.edge_attr.dim() == 2
        assert sample_data.y.dim() == 0
        print(f"数据验证通过！共 {self.len()} 条轨迹数据")
        print(f"节点特征维度: {sample_data.x.shape}")
        print(f"边特征维度: {sample_data.edge_attr.shape}")