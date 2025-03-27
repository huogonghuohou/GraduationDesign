import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import json
import os

# ================== 配置参数 ==================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TOTAL_TRACES = 8000  # 总调用链数量
LOG_MULTIPLIER = 5  # 日志数量相对于trace的倍数
OUTPUT_DIR = "dataset"  # 输出目录

# 微服务架构定义
SERVICES = [
    'ts-order-service',
    'ts-user-service',
    'ts-auth-service',
    'ts-ticket-service',
    'ts-station-service',
    'ts-payment-service',
    'ts-config-service'
]


# ================== 数据生成核心函数 ==================
def generate_base_traces(num):
    """生成基础调用链数据"""
    traces = []
    for _ in range(num):
        trace_id = f"trace-{random.randint(100000, 999999)}"
        duration = abs(int(np.random.normal(300, 50)))
        timestamp = datetime.now() - timedelta(seconds=random.randint(0, 86400))
        service = random.choice(SERVICES)

        traces.append({
            "trace_id": trace_id,
            "service": service,
            "duration_ms": duration,
            "timestamp": timestamp,
            "status": "success",
            "call_path": generate_call_path(),
            "anomaly_type": "normal"
        })
    return pd.DataFrame(traces)


def generate_call_path():
    """生成随机调用路径"""
    path_length = random.randint(3, min(5, len(SERVICES)))
    return "->".join(random.sample(SERVICES, path_length))


class AnomalyInjector:
    """异常注入系统（6种异常类型）"""

    @staticmethod
    def inject_anomalies(traces_df, anomaly_ratio=0.2):
        total_samples = len(traces_df)
        normal_count = int(total_samples * (1 - anomaly_ratio))
        total_anomalies = total_samples - normal_count

        # 定义异常类型分布
        anomaly_dist = {
            'timeout': int(total_anomalies * 0.25),
            'cascade_failure': int(total_anomalies * 0.20),
            'high_error_rate': int(total_anomalies * 0.15),
            'infinite_loop': int(total_anomalies * 0.10),
            'resource_exhaustion': int(total_anomalies * 0.15),
            'illegal_argument': int(total_anomalies * 0.15)
        }

        # 处理四舍五入误差
        diff = total_anomalies - sum(anomaly_dist.values())
        anomaly_dist['timeout'] += diff

        # 初始化所有样本为正常
        traces_df['anomaly_type'] = 'normal'
        anomaly_indices = []

        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=anomaly_dist['timeout'], replace=False)
        traces_df.loc[selected, 'anomaly_type'] = 'timeout'
        traces_df.loc[selected, 'duration_ms'] *= 10  # 10倍延迟
        anomaly_indices.extend(selected)

        # ============= 2. 注入级联故障 =============
        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=anomaly_dist['cascade_failure'], replace=False)
        traces_df.loc[selected, 'anomaly_type'] = 'cascade_failure'
        traces_df.loc[selected, 'status'] = 'failed'
        # 级联故障的特征路径
        cascade_path = "ts-auth-service->ts-order-service->ts-payment-service"
        traces_df.loc[selected, 'call_path'] = cascade_path
        anomaly_indices.extend(selected)

        # ============= 3. 注入高错误率 =============
        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=anomaly_dist['high_error_rate'], replace=False)
        traces_df.loc[selected, 'anomaly_type'] = 'high_error_rate'
        traces_df.loc[selected, 'status'] = 'failed'
        # 集中在ticket-service
        traces_df.loc[selected, 'service'] = 'ts-ticket-service'
        anomaly_indices.extend(selected)

        # ============= 4. 注入无限循环 =============
        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=anomaly_dist['infinite_loop'], replace=False)
        traces_df.loc[selected, 'anomaly_type'] = 'infinite_loop'
        traces_df.loc[selected, 'duration_ms'] = 30000  # 固定30秒延迟
        # 添加循环特征路径
        loop_path = "ts-user-service->ts-user-service->ts-user-service"
        traces_df.loc[selected, 'call_path'] = loop_path
        anomaly_indices.extend(selected)

        # ============= 5. 注入资源耗尽 =============
        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=anomaly_dist['resource_exhaustion'], replace=False)
        traces_df.loc[selected, 'anomaly_type'] = 'resource_exhaustion'
        traces_df.loc[selected, 'duration_ms'] = traces_df.loc[selected, 'duration_ms'] * 20
        traces_df.loc[selected, 'status'] = 'failed'
        # 添加内存错误特征
        traces_df.loc[selected, 'service'] = 'ts-config-service'
        anomaly_indices.extend(selected)

        # ============= 6. 注入非法参数 =============
        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=anomaly_dist['illegal_argument'], replace=False)
        traces_df.loc[selected, 'anomaly_type'] = 'illegal_argument'
        traces_df.loc[selected, 'status'] = 'failed'
        traces_df.loc[selected, 'duration_ms'] = 100  # 快速失败
        # 参数错误特征路径
        arg_path = "ts-order-service->ts-station-service->ts-ticket-service"
        traces_df.loc[selected, 'call_path'] = arg_path
        anomaly_indices.extend(selected)

        return traces_df


def generate_logs_with_anomalies(traces_df):
    """生成与异常关联的日志数据"""
    logs = []
    log_templates = {
        "normal": "{service} processed request in {duration}ms",
        "timeout": "{service} request timeout after {duration}ms | trace_id={trace_id}",
        "cascade_failure": "{service} failed due to dependency service breakdown | trace_id={trace_id}",
        "high_error_rate": "{service} internal server error (code:5XX) | trace_id={trace_id}",
        "infinite_loop": "{service} potential infinite loop detected | duration={duration}ms",
        "resource_exhaustion": "{service} resource exhausted (memory/CPU) | trace_id={trace_id}",
        "illegal_argument": "{service} received illegal argument: code={error_code}"
    }

    for _, trace in traces_df.iterrows():
        log_count = random.randint(1, 3)
        for _ in range(log_count):
            try:
                message = log_templates[trace['anomaly_type']].format(
                    service=trace['service'],
                    duration=trace['duration_ms'],
                    trace_id=trace['trace_id'],
                    error_code=random.randint(400, 499)  # 用于illegal_argument
                )
            except KeyError:
                message = f"{trace['service']} unknown anomaly occurred"

            logs.append({
                "trace_id": trace['trace_id'],
                "service": trace['service'],
                "timestamp": trace['timestamp'] + timedelta(milliseconds=random.randint(0, trace['duration_ms'])),
                "level": "ERROR" if trace['anomaly_type'] != 'normal' else
                random.choices(["INFO", "WARN"], weights=[0.9, 0.1])[0],
                "message": message,
                "anomaly_type": trace['anomaly_type']
            })
    return pd.DataFrame(logs)


# ================== 特征工程 ==================
def extract_trace_features(traces_df):
    """提取调用链特征"""
    # 调用路径特征
    traces_df['path_length'] = traces_df['call_path'].apply(lambda x: len(x.split('->')))

    # 时间特征
    traces_df['timestamp'] = pd.to_datetime(traces_df['timestamp'])
    traces_df['hour'] = traces_df['timestamp'].dt.hour

    # 服务热点编码
    service_dummies = pd.get_dummies(traces_df['service'], prefix='service')
    traces_df = pd.concat([traces_df, service_dummies], axis=1)

    # 数值特征标准化
    scaler = MinMaxScaler()
    traces_df[['duration_norm', 'path_length_norm']] = scaler.fit_transform(
        traces_df[['duration_ms', 'path_length']])

    return traces_df


def build_service_graph(traces_df):
    """构建服务调用关系图（带完整属性）"""
    G = nx.DiGraph()

    # ============= 1. 节点初始化 =============
    # 预先计算服务级别的统计信息
    service_stats = traces_df.groupby('service').agg({
        'duration_ms': ['mean', 'max', 'min'],
        'status': lambda x: (x == 'failed').mean()  # 错误率
    }).reset_index()
    service_stats.columns = ['service', 'avg_duration', 'max_duration', 'min_duration', 'error_rate']

    # 添加节点及基础属性
    for service in SERVICES:
        stats = service_stats[service_stats['service'] == service].iloc[0] if service in service_stats[
            'service'].values else {
            'avg_duration': 300, 'max_duration': 500, 'min_duration': 100, 'error_rate': 0.01
        }
        G.add_node(service,
                   service_type=("backend" if "ts-" in service else "gateway"),
                   avg_cpu=random.uniform(10, 80),  # 模拟CPU使用率
                   avg_mem=random.uniform(20, 90),  # 模拟内存使用率
                   **stats.to_dict())

    # ============= 2. 边构建 =============
    # 计算边级别的统计信息（修复括号问题）
    edge_records = []
    for _, row in traces_df.iterrows():
        nodes = row['call_path'].split('->')
        for i in range(len(nodes) - 1):
            edge_records.append({
                'source': nodes[i],
                'target': nodes[i + 1],
                'trace_id': row['trace_id'],
                'duration': row['duration_ms'],
                'status': row['status']
            })

    edge_stats = pd.DataFrame(edge_records)

    if not edge_stats.empty:
        edge_agg = edge_stats.groupby(['source', 'target']).agg({
            'duration': ['mean', 'count'],
            'status': lambda x: (x == 'failed').mean()
        }).reset_index()
        edge_agg.columns = ['source', 'target', 'avg_duration', 'call_count', 'error_rate']

        # 添加带属性的边
        for _, edge in edge_agg.iterrows():
            G.add_edge(edge['source'], edge['target'],
                       avg_duration=edge['avg_duration'],
                       call_count=edge['call_count'],
                       error_rate=edge['error_rate'],
                       weight=edge['call_count'] / edge_agg['call_count'].max())  # 归一化权重

    # ============= 3. 图级别特征 =============
    # 计算中心性指标
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = nx.degree_centrality(G).get(node, 0)
        G.nodes[node]['betweenness_centrality'] = nx.betweenness_centrality(G).get(node, 0)
        G.nodes[node]['closeness_centrality'] = nx.closeness_centrality(G).get(node, 0)

    # 添加图级别元数据
    G.graph['created_at'] = datetime.now().isoformat()
    G.graph['total_traces'] = len(traces_df)
    G.graph['avg_duration'] = traces_df['duration_ms'].mean()

    return G


# ================== 主流程 ==================
def main():
    print("🚀 开始生成数据集...")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 生成基础调用链数据
    traces_df = generate_base_traces(TOTAL_TRACES)
    print(f"✅ 生成基础调用链数据完成（共{len(traces_df)}条）")

    # 2. 注入异常（20%比例）
    traces_df = AnomalyInjector.inject_anomalies(traces_df)
    normal_ratio = len(traces_df[traces_df['anomaly_type'] == 'normal']) / len(traces_df)
    print(f"✅ 异常注入完成（正常数据比例：{normal_ratio:.1%}）")

    # 3. 生成关联日志
    logs_df = generate_logs_with_anomalies(traces_df)
    print(f"✅ 生成关联日志完成（共{len(logs_df)}条）")

    # 4. 特征工程
    traces_df = extract_trace_features(traces_df)
    print("✅ 特征工程完成")

    # 5. 构建服务调用图
    service_graph = build_service_graph(traces_df)
    print("✅ 服务调用图构建完成")

    # 6. 保存数据集
    traces_df.to_csv(f"{OUTPUT_DIR}/traces_processed.csv", index=False)
    logs_df.to_csv(f"{OUTPUT_DIR}/logs_processed.csv", index=False)
    nx.write_gexf(service_graph, f"{OUTPUT_DIR}/service_graph.gexf")

    # 保存统计信息
    stats = {
        "total_traces": len(traces_df),
        "normal_ratio": normal_ratio,
        "anomaly_distribution": traces_df['anomaly_type'].value_counts().to_dict(),
        "log_level_distribution": logs_df['level'].value_counts().to_dict()
    }
    with open(f"{OUTPUT_DIR}/dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"""
🎉 数据集生成完成！保存在 {OUTPUT_DIR} 目录下：
  - traces_processed.csv：处理后的调用链数据
  - logs_processed.csv：关联日志数据
  - service_graph.gexf：服务调用关系图
  - dataset_stats.json：数据集统计信息
    """)


if __name__ == "__main__":
    main()