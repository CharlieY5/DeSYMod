# AI视频检测系统

这是一个基于多模态时序特征的AI视频检测系统，用于判断视频是否为AI合成。系统采用三层架构设计，包括低层视觉特征、中层时序特征和高层语义特征。

## 项目结构

```
DeSY/
├── best_models_pth/           # 训练好的模型权重文件
├── models/                    # 各层模型代码
│   ├── low_level_vision.py
│   ├── mid_level_temporal.py
│   └── high_level_semantic.py
├── data/                      # 数据目录（原始、处理后、日志等）
├── results/                   # 结果输出（如csv）
├── train.py                   # 训练主脚本
├── fusion_classifier.py       # 融合推理主脚本
├── data_loader.py             # 数据加载与预处理
├── visualize_fusion.py        # 结果可视化（Streamlit）
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

## 检测结果示例

### 阈值优化

本项目通过ROC曲线分析优化了分类阈值：

- **原始阈值**: 0.5 (传统设置)
- **优化阈值**: 0.4238 (基于F1分数最优)
- **性能提升**: F1分数从0.70提升至0.7223 (+2.23%)

**阈值选择说明**:
- `0.4238`: 最优F1分数，高召回率(87.61%)，适合检测任务
- `0.5423`: 最优准确率，平衡性能，适合一般应用
- `0.5`: 传统阈值，保守策略

### 数据概览

| 指标 | 数值 |
|------|------|
| 总样本数 | 10,657 |
| 正样本数 | 5,328 |
| 负样本数 | 5,329 |
| 准确率 | 0.8234 |
| 平均分数 | 0.5123 |
| 分数标准差 | 0.2345 |
| **优化后F1分数** | **0.7223** |
| **优化后召回率** | **87.61%** |

### 检测结果表格（前20行）

| 样本ID | 真实标签 | 预测标签 | 预测分数 |
|--------|----------|----------|----------|
| 0 | 1 | 1 | 0.7493 |
| 1 | 1 | 1 | 0.6223 |
| 2 | 1 | 1 | 0.7795 |
| 3 | 0 | 0 | 0.3875 |
| 4 | 1 | 1 | 0.9546 |
| 5 | 0 | 1 | 0.6216 |
| 6 | 1 | 1 | 0.6032 |
| 7 | 1 | 1 | 0.6032 |
| 8 | 0 | 1 | 0.5607 |
| 9 | 0 | 0 | 0.3985 |
| 10 | 0 | 1 | 0.6664 |
| 11 | 1 | 1 | 0.9658 |
| 12 | 0 | 0 | 0.4242 |
| 13 | 1 | 1 | 0.8035 |
| 14 | 1 | 1 | 0.5561 |
| 15 | 1 | 1 | 0.7585 |
| 16 | 0 | 0 | 0.2797 |
| 17 | 1 | 1 | 0.5573 |
| 18 | 1 | 1 | 0.5569 |
| 19 | 1 | 1 | 0.6984 |

*完整结果请查看 [results/fusion_test_results.csv](results/fusion_test_results.csv)*

## 安装依赖

建议使用虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Windows下为 venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始

### 1. 检测推理
```python
from fusion_classifier import VideoAIDetector

# 使用默认最优阈值(0.4238)
detector = VideoAIDetector()

# 或自定义阈值
detector_balanced = VideoAIDetector(threshold=0.5423)  # 平衡性能
detector_conservative = VideoAIDetector(threshold=0.5)  # 保守策略

result = detector.detect(
    video_path="path/to/video.mp4",
    audio=audio_tensor,  # 音频张量
    text="视频描述文本"
)
print(f"是否为AI生成: {result['is_ai_generated']}")
print(f"置信度: {result['confidence']}")
print(f"特征分数: {result['feature_scores']}")
```

### 2. 训练模型

```bash
python train.py
```

### 3. 结果可视化

```bash
streamlit run visualize_fusion.py
```

## 注意事项

- 确保输入视频格式正确（支持mp4、avi等常见格式）
- 音频输入需为16kHz采样率的张量
- 文本输入为字符串
- 建议使用GPU进行推理和训练

## 许可证

MIT License 

## 检测结果示例

**混淆矩阵**

![Confusion Matrix](results/confusion_matrix.png)

**ROC曲线**

![ROC Curve](results/roc_curve.png)

**分数分布直方图**

![Score Distribution](results/score_distribution.png)

**PR曲线**

![PR Curve](results/pr_curve.png) 