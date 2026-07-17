"""
models/heads.py

Classification head и projection head (для SupCon), общие для всех моделей.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Простой линейный классификатор поверх фич backbone."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ProjectionHead(nn.Module):
    """
    MLP projection head для SupCon (Khosla et al. 2020 используют
    2-layer MLP + L2-нормализацию выходного эмбеддинга).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = nn.functional.normalize(z, dim=1)
        return z
