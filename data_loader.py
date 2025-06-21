import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import json
from torchvision import transforms
import librosa
import requests
import zipfile
import rarfile
import shutil
from tqdm import tqdm
import urllib.request
import ssl
import certifi
import warnings

# 禁用SSL警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

class UCF101Downloader:
    def __init__(self, data_dir, force_extract=False):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.videos_dir = os.path.join(data_dir, 'raw/videos/UCF-101/UCF-101')
        self.annotations_dir = os.path.join(self.raw_dir, 'annotations')
        self.force_extract = force_extract
        
    def _get_all_videos(self):
        """递归获取所有视频文件路径，兼容多级子目录"""
        print(f"[DEBUG] 正在递归查找视频目录: {self.videos_dir}")
        video_files = []
        for root, _, files in os.walk(self.videos_dir):
            for file in files:
                if file.endswith('.avi'):
                    rel_path = os.path.relpath(os.path.join(root, file), self.videos_dir)
                    video_files.append(rel_path)
        print(f"[DEBUG] 共找到 {len(video_files)} 个视频文件")
        if len(video_files) > 0:
            print(f"[DEBUG] 示例视频文件: {video_files[:3]}")
        return video_files
        
    def _create_train_test_splits(self):
        """创建训练测试集划分"""
        print("创建训练测试集划分...")
        all_videos = self._get_all_videos()
        print(f"[DEBUG] all_videos: {all_videos}")
        if not all_videos:
            print(f"[ERROR] 没有找到任何视频文件，当前查找目录: {self.videos_dir}")
            return False
            
        print(f"找到 {len(all_videos)} 个视频文件")
        
        # 创建划分目录
        splits_dir = os.path.join(self.raw_dir, 'UCF101TrainTestSplits-RecognitionTask')
        os.makedirs(splits_dir, exist_ok=True)
        
        # 随机打乱视频列表
        np.random.shuffle(all_videos)
        
        # 划分训练集和测试集（80%训练，20%测试）
        split_idx = int(len(all_videos) * 0.8)
        train_videos = all_videos[:split_idx]
        test_videos = all_videos[split_idx:]
        
        # 保存训练集列表
        train_file = os.path.join(splits_dir, 'UCF101TrainList01.txt')
        with open(train_file, 'w') as f:
            for video in train_videos:
                f.write(f"{video} 0\n")  # 0是类别标签，这里暂时不用
        
        # 保存测试集列表
        test_file = os.path.join(splits_dir, 'UCF101TestList01.txt')
        with open(test_file, 'w') as f:
            for video in test_videos:
                f.write(f"{video} 0\n")  # 0是类别标签，这里暂时不用
                
        print(f"训练集大小: {len(train_videos)}")
        print(f"测试集大小: {len(test_videos)}")
        return True
        
    def _create_annotations(self):
        """创建标注文件"""
        print("创建标注文件...")
        
        # 如果没有划分文件，先创建
        splits_dir = os.path.join(self.raw_dir, 'UCF101TrainTestSplits-RecognitionTask')
        if not os.path.exists(splits_dir):
            if not self._create_train_test_splits():
                return False
        
        # 读取训练集和测试集文件列表
        train_list = []
        test_list = []
        
        # 读取划分文件
        train_file = os.path.join(splits_dir, 'UCF101TrainList01.txt')
        test_file = os.path.join(splits_dir, 'UCF101TestList01.txt')
        
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_list.extend([line.strip().split(' ')[0] for line in f])
                
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_list.extend([line.strip().split(' ')[0] for line in f])
        
        # 创建标注数据
        annotations = {
            'train': [],
            'test': []
        }
        
        # 处理训练集
        for video_path in train_list:
            # 检查视频文件是否存在
            full_path = os.path.join(self.videos_dir, video_path)
            if not os.path.exists(full_path):
                print(f"警告: 训练视频文件不存在: {full_path}")
                continue
                
            # 检查视频是否可以打开
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"警告: 无法打开训练视频文件: {full_path}")
                cap.release()
                continue
            cap.release()
            
            # 随机分配标签（这里只是示例，实际应用中应该使用真实的标签）
            is_ai_generated = np.random.random() > 0.5
            
            annotations['train'].append({
                'video_path': video_path,
                'is_ai_generated': is_ai_generated
            })
            
        # 处理测试集
        for video_path in test_list:
            # 检查视频文件是否存在
            full_path = os.path.join(self.videos_dir, video_path)
            if not os.path.exists(full_path):
                print(f"警告: 测试视频文件不存在: {full_path}")
                continue
                
            # 检查视频是否可以打开
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"警告: 无法打开测试视频文件: {full_path}")
                cap.release()
                continue
            cap.release()
            
            # 随机分配标签（这里只是示例，实际应用中应该使用真实的标签）
            is_ai_generated = np.random.random() > 0.5
            
            annotations['test'].append({
                'video_path': video_path,
                'is_ai_generated': is_ai_generated
            })
        
        # 保存标注文件
        os.makedirs(self.annotations_dir, exist_ok=True)
        annotations_path = os.path.join(self.annotations_dir, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=4)
            
        print(f"标注文件已保存到: {annotations_path}")
        print(f"训练集样本数: {len(annotations['train'])}")
        print(f"测试集样本数: {len(annotations['test'])}")
        
        return True

class VideoDataset(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, num_frames=16):
        self.annotations = json.load(open(annotations_file))
        self.video_dir = video_dir
        self.num_frames = num_frames
        
        # 设置数据增强
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        else:
            self.transform = transform
            
        print(f"加载标注文件: {annotations_file}")
        print(f"训练集样本数: {len(self.annotations['train'])}")
        print(f"测试集样本数: {len(self.annotations['test'])}")
        
        # 验证视频文件路径
        self.valid_samples = []
        for idx, video_info in enumerate(self.annotations['train']):
            video_path = os.path.join(self.video_dir, video_info['video_path'])
            if os.path.exists(video_path):
                self.valid_samples.append(idx)
            else:
                print(f"警告: 视频文件不存在: {video_path}")
        
        print(f"有效样本数: {len(self.valid_samples)}")
        
    def __len__(self):
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
        try:
            # 获取视频信息
            video_info = self.annotations['train'][self.valid_samples[idx]]
            video_path = os.path.join(self.video_dir, video_info['video_path'])
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {video_path}")
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.num_frames:
                raise RuntimeError(f"视频帧数不足: {total_frames} < {self.num_frames}")
            
            # 计算采样间隔
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            # 读取帧
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"无法读取帧 {frame_idx}")
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            # 转换为张量
            frames = torch.stack([self.transform(frame) for frame in frames])
            
            # 获取标签
            label = torch.tensor(1 if video_info['is_ai_generated'] else 0, dtype=torch.long)
            
            return {
                'frames': frames,
                'label': label
            }
            
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {e}")
            # 返回一个空帧序列和标签
            empty_frames = torch.zeros((self.num_frames, 3, 224, 224))
            empty_label = torch.tensor(0, dtype=torch.long)
            return {
                'frames': empty_frames,
                'label': empty_label
            }

def create_dataloader(data_dir, batch_size=32, num_workers=4, pin_memory=True, force_extract=False):
    """创建数据加载器"""
    # 准备数据集
    downloader = UCF101Downloader(data_dir, force_extract)
    if not downloader._create_annotations():
        raise RuntimeError("创建标注文件失败")
        
    # 确保标注文件存在
    annotations_dir = os.path.join(data_dir, 'raw', 'annotations')
    annotations_file = os.path.join(annotations_dir, 'annotations.json')
    
    # 创建数据集
    train_dataset = VideoDataset(
        annotations_file=annotations_file,
        video_dir=os.path.join(data_dir, 'raw/videos/UCF-101/UCF-101'),
        transform=None,  # 使用默认的数据增强
        num_frames=16
    )
    
    val_dataset = VideoDataset(
        annotations_file=annotations_file,
        video_dir=os.path.join(data_dir, 'raw/videos/UCF-101/UCF-101'),
        transform=None,  # 使用默认的数据增强
        num_frames=16
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) 
    
    return train_loader, val_loader 