"""
models/classifiers.py

Полные модели, собранные из backbone + head(ы): WaferClassifier (baseline /
mean teacher), WaferContrastiveModel (SupCon / mean teacher + SupCon) и
LatentFusionClassifier (student-модель Wei et al. "Latent Vector Representation").
"""

import torch
import torch.nn as nn

from .backbones import ResNetBackbone, FusionResNetBackbone
from .heads import ClassificationHead, ProjectionHead


class WaferClassifier(nn.Module):
    """
    Backbone + classification head. Используется для supervised baseline,
    как student/teacher модель в Mean Teacher (без SupCon), а также как
    teacher-модель в LatentVectorMethod.

    forward(x) -> logits [B, num_classes]
    """

    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()
        self.backbone = ResNetBackbone(backbone=backbone, pretrained=pretrained)
        self.head = ClassificationHead(self.backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


class WaferContrastiveModel(nn.Module):
    """
    Backbone + classification head + projection head.
    Используется для методов с SupCon loss (SupConMethod, MeanTeacherSupCon).

    forward(x) -> (logits [B, num_classes], embedding [B, proj_dim])
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        proj_dim: int = 128,
        proj_hidden_dim: int = 512,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(backbone=backbone, pretrained=pretrained)
        self.head = ClassificationHead(self.backbone.feat_dim, num_classes)
        self.projection = ProjectionHead(
            self.backbone.feat_dim, hidden_dim=proj_hidden_dim, out_dim=proj_dim
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits = self.head(feat)
        embedding = self.projection(feat)
        return logits, embedding


class LatentFusionClassifier(nn.Module):
    """
    Student-модель из Wei et al. (2024, "Latent Vector Representation"):
    FusionResNetBackbone (ResNet с точкой инъекции VAE latent vector) +
    classification head.

    forward(x, latent) -> logits [B, num_classes].
    latent=None эквивалентно обычному ResNet без fusion.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        latent_dim: int = 128,
        fusion_stage: int = 2,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = FusionResNetBackbone(
            backbone=backbone, latent_dim=latent_dim, fusion_stage=fusion_stage, pretrained=pretrained
        )
        self.head = ClassificationHead(self.backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor, latent: torch.Tensor = None) -> torch.Tensor:
        feat = self.backbone(x, latent=latent)
        return self.head(feat)
