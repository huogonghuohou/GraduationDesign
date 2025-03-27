import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import os


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(out[:, 1].exp().cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    return report, cm, auc


def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()


def save_results(report, cm, auc, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(str(report))
        f.write(f"\nAUC: {auc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()