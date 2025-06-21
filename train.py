import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
from tqdm import tqdm
import argparse
from data_loader import create_dataloader
from models.mid_level_temporal import MidLevelTemporalModel
from models.high_level_semantic import HighLevelSemanticModel
from models.low_level_vision import LowLevelVisionModel

# 设置CUDA优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 设置默认保存路径
DEFAULT_SAVE_DIR = 'best_models_pth'
DEFAULT_LOG_DIR = 'data/logs'
DEFAULT_DATA_DIR = 'data'

def setup_directories():
    """创建必要的目录"""
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

def train_low_level_vision(model, train_loader, val_loader, args):
    """训练低级视觉模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 设置梯度累积步数
    accumulation_steps = 4  # 累积4步再更新
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()  # 清零梯度
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            try:
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device).long()  # 确保标签是Long类型
                
                # 前向传播
                outputs = model(frames)  # [batch_size, 2]
                loss = criterion(outputs, labels)
                
                # 缩放损失
                loss = loss / accumulation_steps
                loss.backward()
                
                # 梯度累积
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                _, predicted = outputs.max(1)  # 获取预测的类别
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # 清理缓存
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"训练批次 {i} 时出错: {e}")
                continue
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    frames = batch['frames'].to(device)
                    labels = batch['label'].to(device).long()  # 确保标签是Long类型
                    
                    outputs = model(frames)  # [batch_size, 2]
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)  # 获取预测的类别
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # 清理缓存
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"验证批次时出错: {e}")
                    continue
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_low_level_vision.pth'))
            print('保存最佳模型')

def train_mid_level_temporal(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    low_level_model = LowLevelVisionModel().to(device)
    low_level_model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    writer = SummaryWriter(os.path.join(args.log_dir, 'mid_level_temporal'))
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)

            # 用低级视觉模型提取特征（不要再展平像素！）
            with torch.no_grad():
                batch_size, num_frames, C, H, W = frames.shape
                frames_reshaped = frames.view(-1, C, H, W)
                features = low_level_model.features(frames_reshaped)  # [B*N, 2048, 1, 1]
                features = features.view(batch_size, num_frames, -1)  # [B, N, 2048]

            optimizer.zero_grad()
            outputs = model(features)  # 只能用features，禁止用frames
            outputs = outputs.mean(dim=(1,2))  # [batch]
            labels = labels.float()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device)

                # 用低级视觉模型提取特征（不要再展平像素！）
                batch_size, num_frames, C, H, W = frames.shape
                frames_reshaped = frames.view(-1, C, H, W)
                features = low_level_model.features(frames_reshaped)  # [B*N, 2048, 1, 1]
                features = features.view(batch_size, num_frames, -1)  # [B, N, 2048]

                outputs = model(features)  # 只能用features，禁止用frames
                outputs = outputs.mean(dim=(1,2))  # [batch]
                labels = labels.float()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.save_dir, 'mid_level_temporal_best.pth'))
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

def train_high_level_semantic(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    from models.low_level_vision import LowLevelVisionModel
    low_level_model = LowLevelVisionModel().to(device)
    low_level_model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    writer = SummaryWriter(os.path.join(args.log_dir, 'high_level_semantic'))
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            # 用低级视觉模型提取特征
            with torch.no_grad():
                batch_size, num_frames, C, H, W = frames.shape
                frames_reshaped = frames.view(-1, C, H, W)
                features = low_level_model.features(frames_reshaped)  # [B*N, 2048, 1, 1]
                features = features.view(batch_size, num_frames, -1)  # [B, N, 2048]
                features = features.mean(dim=1)  # [B, 2048]
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device)
                # 用低级视觉模型提取特征
                batch_size, num_frames, C, H, W = frames.shape
                frames_reshaped = frames.view(-1, C, H, W)
                features = low_level_model.features(frames_reshaped)  # [B*N, 2048, 1, 1]
                features = features.view(batch_size, num_frames, -1)  # [B, N, 2048]
                features = features.mean(dim=1)  # [B, 2048]
                outputs = model(features)
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.save_dir, 'high_level_semantic_best.pth'))
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

def get_default_args():
    """获取默认参数"""
    return {
        'data_dir': DEFAULT_DATA_DIR,
        'save_dir': DEFAULT_SAVE_DIR,
        'log_dir': DEFAULT_LOG_DIR,
        'batch_size': 2,  # 显存优化，减小批量大小
        'epochs': 100,
        'lr': 0.001,
        'num_workers': 2,  # 减少工作进程数
        'pin_memory': True,
        'force_extract': False,
        'learning_rate': 0.001,
        'skip_low_level': False,  # 是否跳过低级视觉模型训练
        'skip_mid_level': False   # 是否跳过中级时序模型训练
    }

def main(args=None):
    """
    主训练函数
    Args:
        args: 可以是argparse.Namespace对象或字典，如果为None则使用默认参数
    """
    from models.low_level_vision import LowLevelVisionModel
    
    if args is None:
        # 使用默认参数
        args = get_default_args()
    
    if isinstance(args, dict):
        # 如果传入的是字典，转换为Namespace
        args = argparse.Namespace(**args)
    
    # 创建必要的目录
    setup_directories()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        force_extract=args.force_extract
    )
    
    if train_loader is None or val_loader is None:
        print("创建数据加载器失败，请检查数据集文件")
        return
    
    # 初始化模型
    print("初始化低级视觉模型...")
    low_level_model = LowLevelVisionModel()
    
    print("初始化中级时序模型...")
    mid_level_model = MidLevelTemporalModel(input_dim=2048)
    
    print("初始化高级语义模型...")
    high_level_model = HighLevelSemanticModel()
    
    low_level_path = os.path.join('best_models_pth', 'best_low_level_vision.pth')
    mid_level_path = os.path.join('best_models_pth', 'mid_level_temporal_best.pth')
    high_level_path = os.path.join('best_models_pth', 'high_level_semantic_best.pth')

    # 如果已存在best_low_level_vision.pth则自动跳过低级视觉模型训练
    if os.path.exists(low_level_path) or getattr(args, 'skip_low_level', False):
        print(f"检测到{low_level_path}，自动跳过低级视觉模型训练。")
    else:
        print("开始训练低级视觉模型...")
        train_low_level_vision(low_level_model, train_loader, val_loader, args)
    
    # 如果已存在mid_level_temporal_best.pth则自动跳过中级时序模型训练
    if os.path.exists(mid_level_path) or getattr(args, 'skip_mid_level', False):
        print(f"检测到{mid_level_path}，自动跳过中级时序模型训练。")
    else:
        print("开始训练中级时序模型...")
        old_epochs = args.epochs
        args.epochs = 30
        train_mid_level_temporal(mid_level_model, train_loader, val_loader, args)
        args.epochs = old_epochs
    
    print("开始训练高级语义模型...")
    # 固定高级模型训练轮数为30
    old_epochs = args.epochs
    args.epochs = 30
    train_high_level_semantic(high_level_model, train_loader, val_loader, args)
    args.epochs = old_epochs

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeSY Training')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                      help='数据目录路径')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                      help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR,
                      help='日志保存目录')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='批次大小（显存优化，建议2或1）')
    parser.add_argument('--epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载器的工作进程数')
    parser.add_argument('--pin_memory', type=bool, default=True,
                      help='是否使用固定内存')
    parser.add_argument('--force_extract', action='store_true',
                      help='是否强制重新解压数据集')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--skip_low_level', action='store_true',
                      help='是否跳过低级视觉模型训练')
    parser.add_argument('--skip_mid_level', action='store_true',
                      help='是否跳过中级时序模型训练')
    return parser.parse_args()

if __name__ == '__main__':
    # 只在直接运行脚本时解析命令行参数
    args = parse_args()
    main(args) 