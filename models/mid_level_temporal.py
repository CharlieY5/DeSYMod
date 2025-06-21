import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerTemporalModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super(TransformerTemporalModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MambaBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

class MambaTemporalModel(nn.Module):
    def __init__(self, input_dim, num_blocks=1):
        super(MambaTemporalModel, self).__init__()
        self.blocks = nn.ModuleList([MambaBlock(input_dim) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class MidLevelTemporalModel(nn.Module):
    def __init__(self, input_dim):
        super(MidLevelTemporalModel, self).__init__()
        self.transformer_model = TransformerTemporalModel(input_dim)
        self.mamba_model = MambaTemporalModel(input_dim)
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x):
        transformer_out = self.transformer_model(x)
        mamba_out = self.mamba_model(x)
        
        # 加权融合
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_output = weights[0] * transformer_out + weights[1] * mamba_out
        
        return fused_output

if __name__ == "__main__":
    # 测试代码
    batch_size = 1
    seq_length = 16
    feature_dim = 2048  # ResNet-50特征维度
    
    model = MidLevelTemporalModel(feature_dim)
    x = torch.randn(batch_size, seq_length, feature_dim)
    output = model.forward(x) 