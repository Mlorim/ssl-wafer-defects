"""
models/cbam.py

CBAM-CNN из статьи "CBAM-enhanced lightweight CNN for wafer map defect
classification" — легковесная архитектура для 26x26x3 one-hot wafer map:
2 блока (Conv-ReLU-MaxPool + модифицированный CBAM), затем Flatten -> FC(128,
ReLU) -> FC(num_classes).

Модификации CBAM относительно оригинала (Woo et al., 2018):
  - Channel attention: общий MLP с BatchNorm после каждого аффинного
    преобразования (стабилизация активаций на малых пространственных сетках).
  - Spatial attention: параллельные мультимасштабные свёртки 3x3/5x5/7x7,
    усреднённые, вместо одной свёртки 7x7.
  - Residual skip внутри блока: выход = spatial_attn(channel_attn(x)) + x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Mc(F) = sigma(g(GAP(F)) + g(GMP(F))), g = BN2(W2 . relu(BN1(W1 . s)))."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def _g(self, s: torch.Tensor) -> torch.Tensor:
        return self.bn2(self.fc2(F.relu(self.bn1(self.fc1(s)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        att = torch.sigmoid(self._g(avg) + self._g(mx))
        return att.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Ms(F') = sigma(mean(Conv3x3(Q), Conv5x5(Q), Conv7x7(Q))), Q = [AvgPool_c, MaxPool_c]."""

    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        q = torch.cat([avg, mx], dim=1)
        fused = (self.conv3(q) + self.conv5(q) + self.conv7(q)) / 3
        return torch.sigmoid(fused)


class CBAM(nn.Module):
    """
    X' = Mc(F) ⊙ F, X_tilde = Ms(X') ⊙ X', Y = X_tilde + F (residual skip
    вокруг всего блока внимания — добавлено авторами для стабилизации
    градиентов при сильном дисбалансе классов).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prime = self.channel_attention(x) * x
        x_tilde = self.spatial_attention(x_prime) * x_prime
        return x_tilde + x


class CBAMCNNClassifier(nn.Module):
    """
    Proposed CBAM-CNN. Свёртки без паддинга (valid), поэтому при входе
    26x26 (нативное разрешение WM-811K без ресайза) финальная feature map —
    4x4 (conv1 k=4,s=1 -> 23x23; pool1 k=2,s=2 -> 11x11; conv2 k=3,s=1 -> 9x9;
    pool2 k=2,s=2 -> 4x4), что соответствует расчёту рецептивного поля в статье.

    conv2_channels=72 подобран так, чтобы Flatten(4x4x72) = 1152 совпадало с
    x ∈ R^1152 из статьи; conv1_channels статья явно не специфицирует —
    используется разумный дефолт (32).
    """

    def __init__(
        self,
        num_classes: int = 9,
        input_channels: int = 3,
        conv1_channels: int = 32,
        conv2_channels: int = 72,
        cbam_reduction: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, conv1_channels, kernel_size=4, stride=1)
        self.cbam1 = CBAM(conv1_channels, reduction=cbam_reduction)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1)
        self.cbam2 = CBAM(conv2_channels, reduction=cbam_reduction)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(conv2_channels * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.cbam1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.cbam2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
