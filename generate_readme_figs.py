import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# 读取检测结果
csv_path = 'results/fusion_test_results.csv'
df = pd.read_csv(csv_path)

# 1. 混淆矩阵
cm = confusion_matrix(df['label'], df['pred'])
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
plt.close()

# 2. ROC曲线
fpr, tpr, _ = roc_curve(df['label'], df['score'])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('results/roc_curve.png')
plt.close()

# 3. 分数分布直方图
plt.figure()
sns.histplot(df['score'], bins=30, kde=True)
plt.xlabel('Predicted Score')
plt.title('Score Distribution')
plt.tight_layout()
plt.savefig('results/score_distribution.png')
plt.close()

# 4. PR曲线
precision, recall, _ = precision_recall_curve(df['label'], df['score'])
plt.figure()
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.savefig('results/pr_curve.png')
plt.close()

print('所有图表已生成，保存在results/目录下。') 