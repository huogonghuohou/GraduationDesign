import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import yaml
from typing import Dict, Any
from models.gat_tcn import GAT_TCN
from src.data_loader import TraceLogDataset


def load_config(config_path: str) -> Dict[str, Any]:
    """安全的配置加载函数，包含类型验证"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 类型转换安全处理
    training_config = config['training']
    required_types = {
        'learning_rate': float,
        'weight_decay': float,
        'batch_size': int,
        'epochs': int,
        'early_stopping_patience': int
    }

    for key, dtype in required_types.items():
        try:
            training_config[key] = dtype(training_config[key])
        except (ValueError, KeyError) as e:
            raise ValueError(f"配置参数 '{key}' 类型错误，期望 {dtype.__name__} 类型") from e

    return config


def evaluate(model, loader, device):
    """增强版评估函数"""
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(out[:, 1].exp().cpu().numpy())  # 假设二分类

    metrics = {
        'report': classification_report(y_true, y_pred, output_dict=True),
        'cm': confusion_matrix(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_score),
        'ap': average_precision_score(y_true, y_score),  # PR-AUC
        'accuracy': np.mean(np.array(y_true) == np.array(y_pred))
    }
    return metrics


def train():
    #加载并验证配置
    config = load_config('../configs/model_config.yaml')
    # 1. 初始化配置
    # with open('../configs/model_config.yaml', 'r', encoding='utf-8') as f:
    #     config = yaml.safe_load(f)

    print("\n[配置验证]")
    print(f"学习率: {config['training']['learning_rate']} (类型: {type(config['training']['learning_rate'])})")
    print(f"权重衰减: {config['training']['weight_decay']} (类型: {type(config['training']['weight_decay'])})")


    # 2. 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. 输出目录设置
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"../outputs/results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)

    # 4. 数据加载与验证
    dataset = TraceLogDataset(root='../data/dataset')
    print("\n[数据验证]")
    print(f"总样本数: {len(dataset)}")
    print(f"示例数据维度:")
    sample_data = dataset[0]
    print(f"  - 节点特征: {sample_data.x.shape}")
    print(f"  - 边索引: {sample_data.edge_index.shape}")
    print(f"  - 边属性: {sample_data.edge_attr.shape}")
    print(f"  - 日志特征: {sample_data.log_features.shape}")

    # 数据集划分
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # 类别权重计算（处理样本不平衡）
    train_labels = [data.y.item() for data in train_dataset]
    class_weights = torch.tensor(
        compute_class_weight(
            'balanced',
            classes=np.array([0, 1]),
            y=np.array(train_labels + [0, 1])  # 添加平滑样本
        )[:-2],  # 移除添加的平滑样本
        dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 6. 初始化模型
    model = GAT_TCN(config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # 7. 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size']
    )

    # 类型转换安全处理(冗余代码）
    # training_config = config['training']
    # training_config.update({
    #     'learning_rate': float(training_config['learning_rate']),
    #     'weight_decay': float(training_config['weight_decay']),
    #     'batch_size': int(training_config['batch_size']),
    #     'epochs': int(training_config['epochs']),
    #     'early_stopping_patience': int(training_config['early_stopping_patience'])
    # })

    # 初始化优化器
    # 初始化模型
    model = GAT_TCN(config).to(device)

    # 直接使用已转换的配置值
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],  # 已确保是float
        weight_decay=config['training']['weight_decay']  # 已确保是float
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['training'].get('lr_factor', 0.5),
        patience=config['training'].get('lr_patience', 3),
        verbose=True
    )

    # 8. 训练循环
    best_val_acc = 0.0
    early_stop_counter = 0

    print("\n[开始训练]")
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss, correct = 0, 0

        # 训练阶段
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            correct += out.argmax(dim=1).eq(data.y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # 验证阶段
        val_metrics = evaluate(model, val_loader, device)
        val_loss = 1 - val_metrics['accuracy']  # 使用1-accuracy作为loss
        val_acc = val_metrics['accuracy']

        # 学习率调整
        scheduler.step(val_acc)

        # 模型保存（最佳模型）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"Epoch {epoch + 1}: 保存最佳模型 (Val Acc={val_acc:.4f})")
        else:
            early_stop_counter += 1

        # TensorBoard记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # 打印进度
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # 早停检查
        if early_stop_counter >= config['training']['early_stopping_patience']:
            print(f"早停触发！连续{early_stop_counter}轮验证准确率未提升")
            break

    # 9. 最终测试
    print("\n[测试最佳模型]")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device)

    # 10. 保存结果
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"最佳验证准确率: {checkpoint['val_metrics']['accuracy']:.4f}\n")
        f.write(f"测试准确率: {test_metrics['accuracy']:.4f}\n")
        f.write(f"测试AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"测试AP: {test_metrics['ap']:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(classification_report(
            test_metrics['report']['true'],
            test_metrics['report']['pred'],
            target_names=['normal', 'anomaly']
        ))

    # 保存混淆矩阵图
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_metrics['cm'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    print("\n[训练完成]")
    print(f"结果已保存至: {output_dir}")
    writer.close()


if __name__ == '__main__':
    train()