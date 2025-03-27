import networkx as nx
import matplotlib.pyplot as plt

# 1. 读取生成的图文件
graph = nx.read_gexf("service_graph.gexf")

# 2. 简单可视化
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(graph, seed=42)  # 布局算法
nx.draw_networkx(
    graph,
    pos=pos,
    with_labels=True,
    node_color="skyblue",
    edge_color="gray",
    node_size=800,
    font_size=10
)

# 3. 添加标题和显示
plt.title("Microservice Call Graph Visualization", fontsize=14)
plt.axis("off")  # 关闭坐标轴
plt.tight_layout()
plt.show()