import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, accuracy_score
import os
import numpy as np

st.set_page_config(page_title="Fusion Model Evaluation", layout="wide", page_icon="🤖", initial_sidebar_state="expanded")

# 深色科技风主题
st.markdown("""
    <style>
    body, .stApp {background-color: #181c24; color: #e0e6f1;}
    .css-1d391kg {background-color: #23272f;}
    .css-1v0mbdj {background-color: #23272f;}
    .stButton>button {background-color: #23272f; color: #e0e6f1; border-radius: 8px;}
    .stDataFrame {background-color: #23272f; color: #e0e6f1;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 融合模型评测与可视化 (Fusion Model Evaluation)")

# 读取结果csv
csv_path = st.sidebar.text_input("Results CSV Path", "results/fusion_test_results.csv")
if not os.path.exists(csv_path):
    st.warning(f"找不到结果文件: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

# 动态指标展示
col1, col2, col3, col4, col5 = st.columns(5)
acc = (df['label'] == df['pred']).mean()
col1.metric("Accuracy", f"{acc:.4f}")
prec = precision_score(df['label'], df['pred'])
rec = recall_score(df['label'], df['pred'])
f1 = f1_score(df['label'], df['pred'])
auc_score = roc_auc_score(df['label'], df['score'])
col2.metric("Precision", f"{prec:.4f}")
col3.metric("Recall", f"{rec:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("AUC", f"{auc_score:.4f}")

# 混淆矩阵
st.subheader("Confusion Matrix")
cm = confusion_matrix(df['label'], df['pred'])
fig_cm, ax_cm = plt.subplots(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('True')
st.pyplot(fig_cm)

# ROC曲线
st.subheader("ROC Curve Receiver Operating Characteristic")
fpr, tpr, _ = roc_curve(df['label'], df['score'])
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='#00e6e6', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# 详细表格
st.subheader("Detailed Prediction Results Table")
st.dataframe(df, use_container_width=True)

# 动态筛选/下载
st.download_button("Download Results CSV", data=df.to_csv(index=False), file_name="fusion_test_results.csv")

# 展示训练集/测试集样本数
st.sidebar.markdown("**Total Samples:** " + str(len(df)))
if 'label' in df:
    st.sidebar.markdown("**Positive Samples:** " + str(df['label'].sum()))
    st.sidebar.markdown("**Negative Samples:** " + str(len(df) - df['label'].sum()))

# 样本分布条形图
st.subheader("Class Distribution")
if 'label' in df:
    fig_dist, ax_dist = plt.subplots()
    df['label'].value_counts().sort_index().plot(kind='bar', ax=ax_dist, color=['#00e6e6', '#e67e22'])
    ax_dist.set_xticklabels(['Negative (0)', 'Positive (1)'], rotation=0)
    ax_dist.set_ylabel('Count')
    ax_dist.set_xlabel('Class')
    ax_dist.set_title('Class Distribution')
    st.pyplot(fig_dist)

# 预测概率分布直方图
if 'score' in df:
    st.subheader("Score Distribution")
    fig_score, ax_score = plt.subplots()
    ax_score.hist(df['score'], bins=20, color='#00e6e6', alpha=0.7)
    ax_score.set_xlabel('Predicted Score')
    ax_score.set_ylabel('Count')
    ax_score.set_title('Score Distribution')
    st.pyplot(fig_score)

# 错误样本表格
st.subheader("Wrong Predicted Samples")
if 'label' in df and 'pred' in df:
    wrong_df = df[df['label'] != df['pred']]
    st.write("Wrong Samples Count: " + str(len(wrong_df)))
    st.dataframe(wrong_df.head(20), use_container_width=True)

# 主要指标趋势折线图（如有epoch列）
if 'epoch' in df.columns:
    st.subheader("Main Metrics Trend vs. Epoch")
    metric_cols = [col for col in df.columns if col not in ['epoch', 'label', 'pred', 'score']]
    for metric in metric_cols:
        fig_metric, ax_metric = plt.subplots()
        ax_metric.plot(df['epoch'], df[metric], marker='o')
        ax_metric.set_xlabel('Epoch')
        ax_metric.set_ylabel(metric)
        ax_metric.set_title(f'{metric} Trend')
        st.pyplot(fig_metric)

# 置信度最高的前10个预测
st.subheader("Top 10 Predictions with Highest Confidence")
if 'score' in df:
    top10 = df.copy()
    top10['abs_score'] = (top10['score'] - 0.5).abs()
    top10 = top10.sort_values('abs_score', ascending=False).head(10)
    st.dataframe(top10.drop(columns=['abs_score']), use_container_width=True)

# 展示正负样本score均值/方差
if 'score' in df and 'label' in df:
    pos_scores = df[df['label'] == 1]['score']
    neg_scores = df[df['label'] == 0]['score']
    st.sidebar.markdown("**Positive Samples Score Mean:** " + str(pos_scores.mean()))
    st.sidebar.markdown("**Positive Samples Score Variance:** " + str(pos_scores.var()))
    st.sidebar.markdown("**Negative Samples Score Mean:** " + str(neg_scores.mean()))
    st.sidebar.markdown("**Negative Samples Score Variance:** " + str(neg_scores.var()))

# 展示不同阈值下的准确率/召回率/精确率曲线
if 'score' in df and 'label' in df:
    st.subheader("Accuracy/Recall/Precision vs. Threshold")
    thresholds = np.linspace(0, 1, 100)
    accs, precs, recs = [], [], []
    for t in thresholds:
        preds = (df['score'] > t).astype(int)
        accs.append(accuracy_score(df['label'], preds))
        precs.append(precision_score(df['label'], preds, zero_division=0))
        recs.append(recall_score(df['label'], preds, zero_division=0))
    fig_thr, ax_thr = plt.subplots()
    ax_thr.plot(thresholds, accs, label='Accuracy')
    ax_thr.plot(thresholds, precs, label='Precision')
    ax_thr.plot(thresholds, recs, label='Recall')
    ax_thr.set_xlabel('Threshold')
    ax_thr.set_ylabel('Metric')
    ax_thr.set_title('Metrics vs. Threshold')
    ax_thr.legend()
    st.pyplot(fig_thr)

# PR曲线
if 'score' in df and 'label' in df:
    st.subheader("PR Curve (Precision-Recall Curve)")
    precision, recall, _ = precision_recall_curve(df['label'], df['score'])
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, color='#e67e22', lw=2)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    st.pyplot(fig_pr)

# 混淆矩阵详细数值
if 'label' in df and 'pred' in df:
    cm = confusion_matrix(df['label'], df['pred'])
    st.write('Confusion Matrix Detailed Values:')
    st.write(pd.DataFrame(cm, index=['True Negative Samples(0)', 'True Positive Samples(1)'], columns=['Predicted Negative Samples(0)', 'Predicted Positive Samples(1)']))

# 正负样本score分布KDE曲线
if 'score' in df and 'label' in df:
    st.subheader("Positive/Negative Samples Score Distribution KDE Curve")
    fig_kde, ax_kde = plt.subplots()
    try:
        sns.kdeplot(pos_scores, ax=ax_kde, label='Positive Samples', color='#e67e22')
        sns.kdeplot(neg_scores, ax=ax_kde, label='Negative Samples', color='#00e6e6')
        ax_kde.set_xlabel('score')
        ax_kde.set_ylabel('Density')
        ax_kde.legend()
        st.pyplot(fig_kde)
    except Exception as e:
        st.write(f"KDE Plotting Failed: {e}")

# 预测为正/负的样本数量和比例
if 'pred' in df:
    st.sidebar.markdown("**Predicted Positive Samples Count:** " + str(int((df['pred']==1).sum())))
    st.sidebar.markdown("**Predicted Negative Samples Count:** " + str(int((df['pred']==0).sum())))
    st.sidebar.markdown("**Predicted Positive Samples Percentage:** " + str((df['pred']==1).mean()*100) + "%")
    st.sidebar.markdown("**Predicted Negative Samples Percentage:** " + str((df['pred']==0).mean()*100) + "%") 