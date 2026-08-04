"""ResNet-18 classifier used by ClimEx (Jeong et al., IEEE Access 2026)."""

import torch
import torch.nn as nn

from .backbones import ResNetBackbone


class ClimExClassifier(nn.Module):
    def __init__(self, num_classes: int = 9, dropout: float = 0.5):
        super().__init__()
        self.backbone = ResNetBackbone(backbone="resnet18", pretrained=False)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.backbone(x)))
