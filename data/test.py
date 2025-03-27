# 读取数据示例
import pandas as pd
traces = pd.read_csv("traces_dataset.csv")
logs = pd.read_csv("logs_dataset.csv")

# 检查异常比例（验证注入效果）
print(traces['status'].value_counts(normalize=True))
print(logs['level'].value_counts(normalize=True))


pd.set_option('display.max_colwidth', 100)  # 设置列显示宽度为100字符
logs = pd.read_csv("dataset/logs_processed.csv")
print(logs[logs['anomaly_type'] == 'resource_exhaustion']['message'].head())