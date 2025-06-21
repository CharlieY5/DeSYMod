import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

class VideoPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_keyframes(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        return torch.stack(frames)

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.conv2 = resnet.layer1
        self.conv3 = resnet.layer2
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames = x.size(0), x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        
        # Reshape back to include temporal dimension
        f1 = f1.view(batch_size, num_frames, -1)
        f2 = f2.view(batch_size, num_frames, -1)
        f3 = f3.view(batch_size, num_frames, -1)
        
        return f1, f2, f3

class LowLevelVisionModel(nn.Module):
    def __init__(self):
        super(LowLevelVisionModel, self).__init__()
        # 加载预训练的ResNet-50模型
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 添加新的分类层
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 修改为2个类别（真实/生成）
        )
        
    def forward(self, x):
        # 输入x的形状应该是 [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, c, h, w = x.size()
        
        # 将输入重塑为 [batch_size * num_frames, channels, height, width]
        x = x.view(-1, c, h, w)
        
        # 提取特征
        features = self.features(x)
        features = features.view(batch_size, num_frames, -1)
        
        # 对每一帧的特征取平均
        features = features.mean(dim=1)
        
        # 分类
        output = self.classifier(features)
        return output  # 返回形状为 [batch_size, 2] 的张量

if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    num_frames = 16
    channels = 3
    height = 224
    width = 224
    
    model = LowLevelVisionModel()
    x = torch.randn(batch_size, num_frames, channels, height, width)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")