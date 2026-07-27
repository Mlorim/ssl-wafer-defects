"""
models/hybrid_vit.py

HybridCNN-ViT из статьи "SemiWaferNet: Efficient Semi-Supervised Hybrid
CNN-Transformer Models for Wafer Defect Classification and Segmentation"
(Shi, Liu, Zhou et al., 2026). CNN backbone (stem conv + residual block) для
локальных признаков -> flatten в токены -> компактный ViT-энкодер для
глобального контекста -> class token -> линейная голова.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Стандартный residual block (formula 2 статьи: y = F(x, Wi) + T(x)).
    T(x) — identity, если размерности не меняются, иначе projection shortcut
    (1x1 conv + BN). В HybridCNN-ViT используется один блок 64->128 каналов
    со stride=2 (formula 3).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: Optional[nn.Module]
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class HybridCNNViT(nn.Module):
    """
    forward(x) -> logits [B, num_classes].

    Формулы статьи:
      F1 = MaxPool(ReLU(BN(Conv3x3(X))))                      -> 64 x H/2 x W/2
      F2 = ResBlock(F1, 64->128, stride=2)                     -> 128 x H/4 x W/4
      AdaptiveAvgPool -> Fc (фиксированный spatial размер)
      Flatten в N токенов, линейная проекция в D=128, + pos embedding,
      + class token, dropout(0.5), Transformer encoder (L=4, 8 heads,
      ffn=256, GELU, dropout=0.2), classification head на class token.

    При input_size=32 (как в статье): F2 = 128x8x8, N=64 токена.
    """

    def __init__(
        self,
        num_classes: int = 9,
        input_channels: int = 3,
        input_size: int = 32,
        embed_dim: int = 128,
        token_grid_size: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 256,
        pre_transformer_dropout: float = 0.5,
        transformer_dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.res_block = ResidualBlock(64, 128, stride=2)
        self.token_grid_size = token_grid_size
        self.pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))

        num_tokens = token_grid_size * token_grid_size
        self.token_proj = nn.Linear(128, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pre_transformer_dropout = nn.Dropout(pre_transformer_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        f1 = self.stem(x)
        f2 = self.res_block(f1)
        fc = self.pool(f2)  # [B, 128, grid, grid]

        tokens = fc.flatten(2).transpose(1, 2)  # [B, N, 128]
        tokens = self.token_proj(tokens)  # [B, N, D]
        tokens = tokens + self.pos_embedding

        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, N+1, D]
        tokens = self.pre_transformer_dropout(tokens)

        encoded = self.norm(self.transformer(tokens))
        cls_out = encoded[:, 0]  # class token
        return self.head(cls_out)
