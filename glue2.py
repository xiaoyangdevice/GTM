
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import warnings
from pathlib import Path
from transformers import AutoTokenizer
from mamba_ssm import Mamba

warnings.filterwarnings('ignore')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 30522
HIDDEN_DIM = 768
NUM_LAYERS = 6
NUM_LAYERS1 = 12
NUM_HEADS = 12

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

# 检查点目录
CHECKPOINT_DIR = Path("./checkpoints")

# 清理显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# 模型定义
# ==============================================================================

class TransformerLayer(nn.Module):
    """Transformer 层（双向注意力，用于分类任务）"""
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, x):
        # Pre-LN + Self-Attention（双向，无因果掩码）
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, need_weights=False)[0]
        x = self.dropout1(x)
        x = residual + x
        
        # Pre-LN + FFN
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = self.dropout3(x)
        x = residual + x
        
        return x


class MambaLayer(nn.Module):
    """单向 Mamba 层"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class BiMambaLayer(nn.Module):
    
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        # 可学习的方向融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # 前向 Mamba
        x_fwd = self.mamba_fwd(x)
        
        # 后向 Mamba（翻转序列，处理后翻回）
        x_bwd = torch.flip(x, dims=[1])
        x_bwd = self.mamba_bwd(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])
        
        # 可学习融合
        alpha = torch.sigmoid(self.fusion_weight)
        x = alpha * x_fwd + (1 - alpha) * x_bwd
        
        x = self.dropout(x)
        return residual + x


class GatedTransMambaBlock(nn.Module):
   
    
    def __init__(self, layer_idx=0, dropout=0.1, use_bidirectional=True):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Transformer 分支
        self.trans = TransformerLayer(
            d_model=HIDDEN_DIM, 
            nhead=NUM_HEADS, 
            dim_feedforward=1024, 
            dropout=dropout
        )
        
        # Mamba 分支（使用双向Mamba）
        if use_bidirectional:
            self.mamba = BiMambaLayer(HIDDEN_DIM, d_state=16, d_conv=4, expand=2, dropout=dropout)
        else:
            self.mamba = MambaLayer(HIDDEN_DIM, d_state=16, d_conv=4, expand=2, dropout=dropout)
        
        # 门控前的归一化
        self.norm_trans = nn.LayerNorm(HIDDEN_DIM)
        self.norm_mamba = nn.LayerNorm(HIDDEN_DIM)
        
        # 多头门控机制
        self.num_gates = 4
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM//4), 
                nn.GELU(), 
                nn.Linear(HIDDEN_DIM//4, 1), 
                nn.Sigmoid()
            ) for _ in range(self.num_gates)
        ])
        self.gate_aggregate = nn.Linear(self.num_gates, 1, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        
        # 可学习的残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        
       
        self.layer_bias = nn.Parameter(torch.tensor([0.3 - 0.03 * layer_idx]))
        
        # 最终归一化
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        residual = x
        
        # Transformer 分支
        x_t = self.trans(self.norm_trans(x))
        
        # Mamba 分支
        x_m = self.mamba(self.norm_mamba(x))
        
        # 多头门控
        gate_weights = torch.cat([g(x) for g in self.gate_heads], dim=-1)
        gate = torch.sigmoid(self.gate_aggregate(gate_weights).squeeze(-1) + self.layer_bias).unsqueeze(-1)
        
        # 自适应混合
        fused = gate * x_t + (1 - gate) * x_m
        
        # 输出投影 + 残差
        out = self.dropout(self.out_proj(fused))
        return self.norm(residual + self.residual_scale * out)


class GatedTransMamba(nn.Module):
   
    # 支持的池化方式
    POOLING_METHODS = ['cls', 'mean', 'max', 'concat']
    
    def __init__(self, num_classes=2, dropout=0.1, use_bidirectional=True, pooling='mean'):
        """
        Args:
            num_classes: 分类数
            dropout: dropout率
            use_bidirectional: 是否使用双向Mamba
            pooling: 池化方式
                - 'cls': 使用CLS token（预训练权重兼容）
                - 'mean': 平均池化（推荐用于分类）
                - 'max': 最大池化
                - 'concat': CLS + mean + max 拼接（更强但维度变大）
        """
        super().__init__()
        
        if pooling not in self.POOLING_METHODS:
            raise ValueError(f"pooling must be one of {self.POOLING_METHODS}, got {pooling}")
        
        self.pooling = pooling
        self.use_bidirectional = use_bidirectional
        
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token（保留用于兼容预训练权重）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList([
            GatedTransMambaBlock(layer_idx=i, dropout=dropout, use_bidirectional=use_bidirectional) 
            for i in range(NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        
        # 根据池化方式设置分类头
        if pooling == 'concat':
            # CLS + mean + max = 3 * HIDDEN_DIM
            self.head = nn.Linear(HIDDEN_DIM * 3, num_classes)
        else:
            self.head = nn.Linear(HIDDEN_DIM, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pe.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers: 
            x = layer(x)
        
        x = self.norm(x)
        
        # 根据池化方式提取表示
        if self.pooling == 'cls':
            # 使用 CLS token（位置0）
            pooled = x[:, 0]
        elif self.pooling == 'mean':
            # 平均池化（排除CLS token）
            pooled = x[:, 1:].mean(dim=1)
        elif self.pooling == 'max':
            # 最大池化（排除CLS token）
            pooled = x[:, 1:].max(dim=1)[0]
        elif self.pooling == 'concat':
            # 拼接：CLS + mean + max
            cls_pool = x[:, 0]
            mean_pool = x[:, 1:].mean(dim=1)
            max_pool = x[:, 1:].max(dim=1)[0]
            pooled = torch.cat([cls_pool, mean_pool, max_pool], dim=-1)
        
        return self.head(pooled)


class AlternateTransMamba(nn.Module):
    """层间交替混合（使用双向Mamba）"""
    
    def __init__(self, num_classes=2, dropout=0.1, use_bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList()
        for i in range(NUM_LAYERS1):
            if i % 2 == 0:
                # Transformer 层
                self.layers.append(
                    TransformerLayer(
                        d_model=HIDDEN_DIM, 
                        nhead=NUM_HEADS, 
                        dim_feedforward=HIDDEN_DIM * 4, 
                        dropout=dropout
                    )
                )
            else:
                # Mamba 层（使用双向）
                if use_bidirectional:
                    self.layers.append(
                        BiMambaLayer(HIDDEN_DIM, d_state=16, d_conv=4, expand=2, dropout=dropout)
                    )
                else:
                    self.layers.append(
                        MambaLayer(HIDDEN_DIM, d_state=16, d_conv=4, expand=2, dropout=dropout)
                    )
        
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


class TransMambaBlock(nn.Module):
   
    
    def __init__(self, layer_idx=0, num_layers=NUM_LAYERS, trans_point=64, dropout=0.1, use_bidirectional=True):
        super().__init__()
        d_model = HIDDEN_DIM
        self.trans_point = trans_point
        self.layer_idx = layer_idx
        
        # Transformer部分（用于前段）- 双向注意力
        self.trans = TransformerLayer(d_model, NUM_HEADS, 1024, dropout)
        self.norm_trans = nn.LayerNorm(d_model)
        
        # Mamba部分（用于后段）- 双向 Mamba
        if use_bidirectional:
            self.mamba = BiMambaLayer(d_model, d_state=16, d_conv=4, expand=2, dropout=dropout)
        else:
            self.mamba = MambaLayer(d_model, d_state=16, d_conv=4, expand=2, dropout=dropout)
        self.norm_mamba = nn.LayerNorm(d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x, return_segment=False):
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # 动态调整trans_point（不能超过序列长度）
        actual_trans_point = min(self.trans_point, seq_len)
        
        # 前段：Transformer（双向注意力）
        if actual_trans_point > 0:
            x_front = self.trans(self.norm_trans(x[:, :actual_trans_point]))
        else:
            x_front = None
        
        # 后段：Mamba（双向）
        if actual_trans_point < seq_len:
            x_back = self.mamba(self.norm_mamba(x[:, actual_trans_point:]))
        else:
            x_back = None
        
        # 拼接
        if x_front is not None and x_back is not None:
            x = torch.cat([x_front, x_back], dim=1)
        elif x_front is not None:
            x = x_front
        else:
            x = x_back
        
        # 输出
        out = self.dropout(self.out_proj(x))
        output = self.norm(residual + out)
        
        if return_segment:
            # 返回分段掩码（用于可视化）
            segment_mask = torch.zeros(seq_len, device=x.device)
            if actual_trans_point > 0:
                segment_mask[:actual_trans_point] = 1.0  # 1.0 表示 Transformer
            return output, segment_mask
        return output


class TransMamba(nn.Module):
  
    
    def __init__(self, num_classes=2, num_layers=NUM_LAYERS, trans_points=None, dropout=0.1, use_bidirectional=True):
        super().__init__()
        d_model = HIDDEN_DIM
        
        if trans_points is None:
            # 分类任务：浅层需要更多 Transformer 来建立全局依赖
            # 深层可以更多 Mamba 来高效处理
            # trans_points 从大到小：[128, 96, 64, 48, 32, 16]
            trans_points = [128, 96, 64, 48, 32, 16]
        
        self.trans_points = trans_points
        
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pe = nn.Embedding(8192, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransMambaBlock(
                layer_idx=i,
                num_layers=num_layers,
                trans_point=trans_points[i] if i < len(trans_points) else 16,
                dropout=dropout,
                use_bidirectional=use_bidirectional
            ) for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pe.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x, return_segments=False):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        segments = []
        for layer in self.layers:
            if return_segments:
                x, segment_mask = layer(x, return_segment=True)
                segments.append(segment_mask)
            else:
                x = layer(x)
        
        logits = self.head(self.norm(x[:, 0]))  # 使用 CLS token
        
        if return_segments:
            return logits, segments
        return logits


class HyenaLayer(nn.Module):
   
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        g = torch.sigmoid(self.gate(x))
        out = self.proj(g * h)
        out = self.dropout(out)
        return residual + out


class Hyena(nn.Module):
   
    
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList([
            HyenaLayer(HIDDEN_DIM, dropout=dropout) for _ in range(NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


class LSTM_Attn(nn.Module):
   
    
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))

        lstm_dropout = dropout if NUM_LAYERS > 1 else 0.0
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM // 2, num_layers=NUM_LAYERS,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.lstm_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.lstm_norm = nn.LayerNorm(HIDDEN_DIM)

        self.attn_norm = nn.LayerNorm(HIDDEN_DIM)
        self.attn = nn.MultiheadAttention(HIDDEN_DIM, NUM_HEADS, batch_first=True, dropout=dropout)

        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)

        # LSTM 分支
        residual = x
        x, _ = self.lstm(x)
        x = self.lstm_proj(x)
        x = self.lstm_norm(x + residual)

        # Attention 分支（双向，无因果掩码）
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + residual

        return self.head(self.norm(x[:, 0]))


class PureTransformer(nn.Module):
    """纯 Transformer"""
    
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList([
            TransformerLayer(HIDDEN_DIM, NUM_HEADS, 1024, dropout=dropout)
            for _ in range(NUM_LAYERS1)
        ])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


class PureMamba(nn.Module):
    """纯 Mamba（使用双向Mamba）"""

    def __init__(self, num_classes=2, dropout=0.1, use_bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.use_bidirectional = use_bidirectional
        if use_bidirectional:
            self.layers = nn.ModuleList([
                BiMambaLayer(HIDDEN_DIM, d_state=16, d_conv=4, expand=2, dropout=dropout)
                for _ in range(NUM_LAYERS1)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaLayer(HIDDEN_DIM, d_state=16, d_conv=4, expand=2, dropout=dropout)
                for _ in range(NUM_LAYERS1)
            ])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


class RetentionLayer(nn.Module):
    

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        gate = torch.sigmoid(self.gate(x))
        out = gate * v + (1 - gate) * k
        out = self.out_proj(out)
        out = self.dropout(out)
        return residual + out


class RetFormer(nn.Module):
    """RetFormer：Retention层 + Transformer层交替堆叠"""

    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))

        self.layers = nn.ModuleList()
        for i in range(NUM_LAYERS1):
            if i % 2 == 0:
                self.layers.append(RetentionLayer(HIDDEN_DIM, dropout=dropout))
            else:
                self.layers.append(TransformerLayer(
                    HIDDEN_DIM, NUM_HEADS, 1024, dropout=dropout
                ))

        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


class GatedMLPLayer(nn.Module):
    

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # 门控MLP部分
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.gate = nn.Linear(d_model, d_model * 4)
        
        # 空间门控：沿序列维度进行信息交换
        # 输入维度是 d_model * 4（与 fc1 输出匹配）
        self.spatial_proj = nn.Linear(d_model * 4, d_model * 4)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # 门控MLP
        h = F.gelu(self.fc1(x))  # [B, T, D*4]
        g = torch.sigmoid(self.gate(x))  # [B, T, D*4]
        out = h * g  # [B, T, D*4]
        
       
        spatial_logits = self.spatial_proj(out)  # [B, T, D*4]
        spatial_weights = torch.softmax(spatial_logits, dim=1)  # [B, T, D*4]
        out = out * spatial_weights  # [B, T, D*4]
        
        out = self.fc2(out)  # [B, T, D]
        out = self.dropout(out)
        return residual + out


class GatedMLP(nn.Module):
    """门控MLP作为对照基线（带空间门控）"""

    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList([
            GatedMLPLayer(HIDDEN_DIM, dropout=dropout) for _ in range(NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


# ==============================================================================
# RWKV 模型（分类任务版本 - 双向）
# ==============================================================================

class RWKVLayer(nn.Module):
   
    
    def __init__(self, d_model, dropout=0.1, bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        # Receptance (R): 决定接受多少信息
        self.receptance_proj = nn.Linear(d_model, d_model)
        # Key (K): 用于计算权重
        self.key_proj = nn.Linear(d_model, d_model)
        # Value (V): 实际内容
        self.value_proj = nn.Linear(d_model, d_model)
        
        # 可学习的衰减参数（每个维度独立）
        self.time_decay = nn.Parameter(torch.randn(d_model) * 0.1 - 3.0)  # 初始化为负值，exp后接近0
        # 时间门控
        self.time_first = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
        # LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.receptance_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.receptance_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x):
       
        residual = x
        x = self.norm(x)
        
        batch_size, seq_len, d_model = x.shape
        
        # 投影
        R = torch.sigmoid(self.receptance_proj(x))  # [B, T, D]
        K = self.key_proj(x)  # [B, T, D]
        V = self.value_proj(x)  # [B, T, D]
        
        if self.bidirectional:
            # 双向版本：使用简化的线性注意力
            # 计算所有位置对之间的交互
            
            # 使用缩放点积注意力的简化版本
            # Q = K (自注意力变体)
            scale = d_model ** 0.5
            
            # 计算注意力分数: [B, T, T]
            attn_scores = torch.bmm(K, K.transpose(1, 2)) / scale
            
            # 添加可学习的位置偏置
            # 时间衰减：距离越远，权重越小
            time_decay = torch.exp(self.time_decay)  # [D]
            # 创建相对位置距离矩阵
            positions = torch.arange(seq_len, device=x.device)
            dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # [T, T]
            # 衰减偏置
            decay_bias = -dist.unsqueeze(-1) * time_decay.view(1, 1, -1)  # [T, T, D]
            decay_bias = decay_bias.mean(-1)  # [T, T] 简化为标量
            
            attn_scores = attn_scores + decay_bias.unsqueeze(0)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # 应用注意力
            out = torch.bmm(attn_weights, V)  # [B, T, D]
        else:
            # 单向版本：因果注意力
            scale = d_model ** 0.5
            attn_scores = torch.bmm(K, K.transpose(1, 2)) / scale
            
            # 因果掩码
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            out = torch.bmm(attn_weights, V)
        
        # 输出：R * out
        out = R * out
        out = self.output_proj(out)
        out = self.dropout(out)
        
        return residual + out


class RWKV(nn.Module):
   
    
    def __init__(self, num_classes=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList([
            RWKVLayer(HIDDEN_DIM, dropout=dropout, bidirectional=bidirectional)
            for _ in range(NUM_LAYERS1)
        ])
        
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))



class RetNetLayer(nn.Module):
   
    
    def __init__(self, d_model, dropout=0.1, bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** 0.5
        self.bidirectional = bidirectional
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 衰减因子 γ（可学习）
        self.gamma_param = nn.Parameter(torch.tensor(2.0))
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x):
        """
        Retention 的双向实现（用于分类任务）
        """
        residual = x
        x = self.norm(x)
        
        batch_size, seq_len, d_model = x.shape
        
        # Q, K, V 投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 数值稳定的衰减因子
        gamma = torch.sigmoid(self.gamma_param)
        
        # 创建相对位置矩阵
        time_indices = torch.arange(seq_len, device=x.device)
        relative_pos = time_indices.unsqueeze(0) - time_indices.unsqueeze(1)
        
        # 衰减偏置
        log_gamma = torch.log(gamma.clamp(min=1e-10))
        decay_bias = relative_pos.abs() * log_gamma
        
        if not self.bidirectional:
            # 单向：使用因果掩码
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            decay_bias = decay_bias.masked_fill(causal_mask == 0, float('-inf'))
        
        # 计算 Q @ K^T / sqrt(d)
        QK = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        # 添加衰减偏置，然后 softmax
        attn_weights = F.softmax(QK + decay_bias.unsqueeze(0), dim=-1)
        
        # 计算 attention @ V
        out = torch.bmm(attn_weights, V)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return residual + out


class RetNet(nn.Module):
    """
    RetNet (Retentive Network) 模型（分类任务版本）
    
    纯 Retention 层堆叠，用于分类任务
    """
    
    def __init__(self, num_classes=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pe = nn.Embedding(8192, HIDDEN_DIM)
        self.emb_dropout = nn.Dropout(dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        
        self.layers = nn.ModuleList([
            RetNetLayer(HIDDEN_DIM, dropout=dropout, bidirectional=bidirectional)
            for _ in range(NUM_LAYERS1)
        ])
        
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=x.device)
        x = torch.cat([cls_tokens, self.emb(x)], dim=1)
        x = x + self.pe(positions)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.head(self.norm(x[:, 0]))


# ==============================================================================
# 工具函数
# ==============================================================================

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_seq_len=128, batch_size=1, device=None):
   
    if device is None:
        device = DEVICE
    
    try:
        # 尝试使用 thop 库
        from thop import profile
        
        # 创建输入
        dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, input_seq_len)).to(device)
        model = model.to(device)
        
        # 计算 FLOPs
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops / 1e9  # 转换为 GFLOPs
    
    except ImportError:
        # thop 库不可用，使用手动估算
        print("  ⚠️ thop 库未安装，使用手动估算 FLOPs")
        return estimate_flops_manual(model, input_seq_len, batch_size)


def estimate_flops_manual(model, input_seq_len=128, batch_size=1):
    
    # 获取模型信息
    params = count_parameters(model)
    
    # 基于模型类型的估算系数
    # Transformer: 约 6 * params * seq_len (注意力计算)
    # Mamba/SSM: 约 2 * params * seq_len (线性递归)
    # 混合模型: 根据比例估算
    
    model_name = model.__class__.__name__
    
    # 序列长度（包含 CLS token）
    seq_len = input_seq_len + 1
    
    # 估算系数
    if 'Transformer' in model_name or 'PureTransformer' in model_name:
        # 纯 Transformer: 注意力是 O(seq_len^2)
        flops_estimate = 6 * params * seq_len + 2 * params * seq_len * seq_len / HIDDEN_DIM
    elif 'Mamba' in model_name and 'Gated' not in model_name:
        # 纯 Mamba: 线性复杂度
        flops_estimate = 2 * params * seq_len
    elif 'RWKV' in model_name:
        # RWKV: 线性注意力，但需要计算衰减矩阵
        flops_estimate = 3 * params * seq_len + seq_len * seq_len * HIDDEN_DIM
    elif 'RetNet' in model_name or 'Ret' in model_name:
        # RetNet: 类似 RWKV
        flops_estimate = 3 * params * seq_len + seq_len * seq_len * HIDDEN_DIM
    elif 'GatedTransMamba' in model_name or 'Ours' in model_name:
        # 混合模型: Transformer + Mamba
        flops_estimate = 4 * params * seq_len + params * seq_len * seq_len / HIDDEN_DIM
    else:
        # 默认估算
        flops_estimate = 3 * params * seq_len
    
    return flops_estimate / 1e9  # 转换为 GFLOPs


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """余弦退火学习率调度器（带预热）"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, opt, crit, scaler, scheduler=None, grad_accum=1):
    """训练一个epoch"""
    model.train()
    total_loss, total_tokens, start_time = 0, 0, time.time()
    opt.zero_grad()
    
    for step, batch in enumerate(loader):
        x, y = batch["input_ids"].to(DEVICE), batch["label"].to(DEVICE)
        total_tokens += x.numel()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = crit(logits, y) / grad_accum
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            loss.backward()
            if (step + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        
        total_loss += loss.item() * grad_accum
    
    elapsed = time.time() - start_time
    max_memory = torch.cuda.max_memory_allocated(DEVICE) / 1e9 if torch.cuda.is_available() else 0
    
    return total_loss / len(loader), max_memory, total_tokens / elapsed


@torch.no_grad()
def evaluate(model, loader):
    """评估模型"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_tokens, total_time = 0, 0, 0
    
    crit = nn.CrossEntropyLoss(reduction='sum')
    
    for batch in loader:
        x, y = batch["input_ids"].to(DEVICE), batch["label"].to(DEVICE)
        total_tokens += x.numel()
        
        start = time.time()
        logits = model(x)
        elapsed = time.time() - start
        total_time += elapsed
        
        loss = crit(logits, y)
        total_loss += loss.item()
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    
    from sklearn.metrics import f1_score, accuracy_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Perplexity
    avg_loss = total_loss / len(all_labels)
    perplexity = math.exp(min(avg_loss, 10))  # 防止溢出
    
    # 推理延迟 (ms/token)
    latency_ms_per_token = (total_time / total_tokens) * 1000
    
    return accuracy, f1, latency_ms_per_token, perplexity


def load_pretrained_weights(model, model_name, checkpoint_dir):
    """
    加载预训练权重
    
    Args:
        model: 模型实例
        model_name: 模型名称（用于查找检查点）
        checkpoint_dir: 检查点目录
    
    Returns:
        (model, loaded): 加载权重后的模型和是否成功加载的标志
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / model_name
    
    if not model_dir.exists():
        print(f"  ⚠️ 检查点目录不存在: {model_dir}")
        return model, False
    
    # 查找最佳检查点
    best_pt = model_dir / "checkpoint_best.pt"
    if not best_pt.exists():
        for pattern in ["best.pt", "model_best.pt", "checkpoint.pt"]:
            alt_path = model_dir / pattern
            if alt_path.exists():
                best_pt = alt_path
                break
    
    if not best_pt.exists():
        print(f"  ⚠️ 未找到检查点文件: {model_dir}")
        return model, False
    
    try:
        # PyTorch 2.6+ 需要 weights_only=False 来加载包含 numpy 数组的检查点
        checkpoint = torch.load(best_pt, map_location=DEVICE, weights_only=False)
        
        # 提取模型权重（检查点可能是字典格式或直接的 state_dict）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  📦 检查点信息: Epoch {checkpoint.get('epoch', 'N/A')}, Best PPL: {checkpoint.get('best_ppl', 'N/A')}")
        else:
            state_dict = checkpoint
        
        # 获取模型当前状态
        model_state = model.state_dict()
        
        # 调试信息：打印键名对比
        pretrained_keys = set(state_dict.keys())
        model_keys = set(model_state.keys())
        
        common_keys = pretrained_keys & model_keys
        only_pretrained = pretrained_keys - model_keys
        only_model = model_keys - pretrained_keys
        
        print(f"  📊 键名统计: 预训练 {len(pretrained_keys)} 个, 模型 {len(model_keys)} 个, 共同 {len(common_keys)} 个")
        
        if only_pretrained:
            print(f"  📤 仅在预训练模型中: {list(only_pretrained)[:5]}...")
        if only_model:
            print(f"  📥 仅在当前模型中: {list(only_model)[:5]}...")
        
        # 过滤并加载权重
        filtered_state = {}
        matched, shape_mismatch = 0, 0
        shape_mismatch_keys = []
        
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                    matched += 1
                else:
                    shape_mismatch += 1
                    shape_mismatch_keys.append(f"{k}: {v.shape} -> {model_state[k].shape}")
        
        # 加载过滤后的权重
        if filtered_state:
            model.load_state_dict(filtered_state, strict=False)
            print(f"  ✅ 加载权重: {matched} 匹配, {shape_mismatch} 形状不匹配")
            if shape_mismatch_keys[:3]:
                print(f"  ⚠️ 形状不匹配示例: {shape_mismatch_keys[:3]}")
        else:
            print(f"  ❌ 没有匹配的权重可加载")
        
        return model, matched > 0
    
    except Exception as e:
        print(f"  ❌ 加载权重失败: {e}")
        import traceback
        traceback.print_exc()
        return model, False
