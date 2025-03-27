#不用了
import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# 微服务列表
SERVICES = [
    'ts-order-service',
    'ts-user-service',
    'ts-auth-service',
    'ts-ticket-service',
    'ts-station-service',
    'ts-payment-service',
    'ts-config-service'
]


# 1. 生成基础调用链数据
def generate_base_traces(num):
    traces = []
    for _ in range(num):
        trace_id = f"trace-{random.randint(100000, 999999)}"
        duration = abs(int(np.random.normal(300, 50)))  # 正态分布
        timestamp = datetime.now() - timedelta(seconds=random.randint(0, 86400))
        service = random.choice(SERVICES)

        traces.append({
            "trace_id": trace_id,
            "service": service,
            "duration_ms": duration,
            "timestamp": timestamp.isoformat(),
            "status": "success",  # 初始全部标记为成功
            "call_path": generate_call_path(),
            "anomaly_type": "normal"  # 初始标记为正常
        })
    return pd.DataFrame(traces)


def generate_call_path():
    """生成随机调用路径（3-5个服务）"""
    path_length = random.randint(3, 5)
    return "->".join(random.sample(SERVICES, path_length))


# 2. 异常注入系统（6种异常类型）
class AnomalyInjector:
    @staticmethod
    def inject_timeout(traces_df):
        """异常类型1：服务超时（性能问题）"""
        affected = traces_df.sample(frac=0.15, random_state=42)
        traces_df.loc[affected.index, 'duration_ms'] = affected['duration_ms'] * 10
        traces_df.loc[affected.index, 'anomaly_type'] = 'timeout'
        return traces_df

    @staticmethod
    def inject_cascade_failure(traces_df):
        """异常类型2：级联故障（服务依赖问题）"""
        cascade_services = ['ts-order-service', 'ts-auth-service', 'ts-payment-service']
        affected = traces_df[traces_df['service'].isin(cascade_services)].sample(frac=0.7)
        traces_df.loc[affected.index, 'status'] = 'failed'
        traces_df.loc[affected.index, 'anomaly_type'] = 'cascade_failure'
        return traces_df

    @staticmethod
    def inject_high_error_rate(traces_df):
        """异常类型3：高错误率（服务不稳定）"""
        target_service = 'ts-ticket-service'
        affected = traces_df[traces_df['service'] == target_service].sample(frac=0.6)
        traces_df.loc[affected.index, 'status'] = 'failed'
        traces_df.loc[affected.index, 'anomaly_type'] = 'high_error_rate'
        return traces_df

    @staticmethod
    def inject_infinite_loop(traces_df):
        """异常类型4：无限循环（逻辑错误）"""
        target_service = 'ts-user-service'
        affected = traces_df[traces_df['service'] == target_service].sample(frac=0.1)
        traces_df.loc[affected.index, 'duration_ms'] = 30000  # 固定30秒超长延迟
        traces_df.loc[affected.index, 'anomaly_type'] = 'infinite_loop'
        return traces_df

    @staticmethod
    def inject_resource_exhaustion(traces_df):
        """异常类型5：资源耗尽（基础设施问题）"""
        affected = traces_df.sample(frac=0.05, random_state=42)
        traces_df.loc[affected.index, 'duration_ms'] = affected['duration_ms'] * 20
        traces_df.loc[affected.index, 'status'] = 'failed'
        traces_df.loc[affected.index, 'anomaly_type'] = 'resource_exhaustion'
        return traces_df

    @staticmethod
    def inject_illegal_argument(traces_df):
        """异常类型6：参数错误（业务逻辑问题）"""
        affected = traces_df.sample(frac=0.08, random_state=42)
        traces_df.loc[affected.index, 'status'] = 'failed'
        traces_df.loc[affected.index, 'duration_ms'] = 100  # 快速失败特征
        traces_df.loc[affected.index, 'anomaly_type'] = 'illegal_argument'
        return traces_df


# 3. 日志生成与异常关联
def generate_logs_with_anomalies(traces_df, log_multiplier=5):
    """生成与异常关联的日志数据"""
    logs = []

    # 为每条trace生成关联日志
    for _, trace in traces_df.iterrows():
        base_logs = random.randint(1, 3)  # 每条trace对应1-3条日志

        for _ in range(base_logs):
            log_level = "INFO" if trace['anomaly_type'] == "normal" else random.choice(["ERROR", "WARN"])
            logs.append({
                "trace_id": trace['trace_id'],
                "timestamp": trace['timestamp'],
                "service": trace['service'],
                "level": log_level,
                "message": generate_log_message(trace),
                "anomaly_type": trace['anomaly_type']
            })

    # 补充系统日志（与trace无直接关联）
    sys_logs = int(len(traces_df) * 0.3)
    for _ in range(sys_logs):
        logs.append({
            "trace_id": "system",
            "timestamp": (datetime.now() - timedelta(seconds=random.randint(0, 86400))).isoformat(),
            "service": random.choice(SERVICES),
            "level": random.choices(["INFO", "DEBUG"], weights=[0.8, 0.2])[0],
            "message": "System heartbeat check",
            "anomaly_type": "normal"
        })

    return pd.DataFrame(logs)


def generate_log_message(trace):
    """根据异常类型生成对应的日志消息"""
    anomaly_type = trace['anomaly_type']
    service = trace['service']

    msg_templates = {
        "normal": f"{service} normal operation",
        "timeout": f"{service} request timeout after {trace['duration_ms']}ms",
        "cascade_failure": f"{service} dependency service unavailable",
        "high_error_rate": f"{service} internal server error",
        "infinite_loop": f"{service} possible infinite loop detected",
        "resource_exhaustion": f"{service} database connection pool exhausted",
        "illegal_argument": f"{service} received invalid parameters: {random.randint(100, 999)}"
    }
    return msg_templates.get(anomaly_type, "Unknown event")


# 4. 主流程控制
# 修改主流程控制部分
if __name__ == "__main__":
    # 生成基础数据（2000条，全标记为正常）
    traces_df = generate_base_traces(2000)

    # 计算需要注入的异常数量（20%比例）
    total_samples = len(traces_df)
    normal_count = int(total_samples * 0.8)  # 80%正常数据
    total_anomalies = total_samples - normal_count  # 20%异常数据

    # 按比例分配各异常类型数量（确保总和=total_anomalies）
    anomaly_dist = {
        'timeout': int(total_anomalies * 0.25),
        'cascade_failure': int(total_anomalies * 0.20),
        'high_error_rate': int(total_anomalies * 0.15),
        'infinite_loop': int(total_anomalies * 0.10),
        'resource_exhaustion': int(total_anomalies * 0.15),
        'illegal_argument': int(total_anomalies * 0.15)
    }
    # 处理可能的四舍五入误差
    diff = total_anomalies - sum(anomaly_dist.values())
    anomaly_dist['timeout'] += diff  # 将差额加到第一个异常类型

    # 标记正常样本（先保证足够数量的正常样本）
    traces_df['anomaly_type'] = 'normal'

    # 分层注入异常（确保不重复修改样本）
    anomaly_indices = []
    for anomaly_type, count in anomaly_dist.items():
        # 从未被选中的样本中抽取
        candidates = traces_df.index.difference(anomaly_indices)
        selected = np.random.choice(list(candidates), size=count, replace=False)
        anomaly_indices.extend(selected)

        # 应用对应异常类型
        traces_df.loc[selected, 'anomaly_type'] = anomaly_type
        if anomaly_type == 'timeout':
            traces_df.loc[selected, 'duration_ms'] *= 10
        elif anomaly_type == 'cascade_failure':
            traces_df.loc[selected, 'status'] = 'failed'
        # ...其他异常类型的处理...

    # 生成关联日志
    logs_df = generate_logs_with_anomalies(traces_df)

    # 保存数据集
    traces_df.to_csv("traces_dataset.csv", index=False)
    logs_df.to_csv("logs_dataset.csv", index=False)

    # 打印数据分布报告
    print("=== 数据集分布验证 ===")
    print(f"总调用链数: {len(traces_df)}")
    print(f"正常数据比例: {len(traces_df[traces_df['anomaly_type'] == 'normal']) / len(traces_df):.1%}")
    print("\n异常类型分布:")
    print(traces_df[traces_df['anomaly_type'] != 'normal']['anomaly_type'].value_counts())
    print("\n日志级别分布:")
    print(logs_df['level'].value_counts())