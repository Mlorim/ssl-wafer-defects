"""Lightweight torchvision CNNs from Shin and Yoo, Sensors 2023."""

import torch.nn as nn
from torchvision import models


class EfficientWaferCNN(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_small", num_classes: int = 9):
        super().__init__()
        builders = {
            "resnet18": models.resnet18,
            "efficientnet_v2_s": models.efficientnet_v2_s,
            "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
            "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
            "mobilenet_v2": models.mobilenet_v2,
            "mobilenet_v3_small": models.mobilenet_v3_small,
        }
        if backbone not in builders:
            raise ValueError(f"Unknown efficient CNN backbone: {backbone}")
        self.backbone_name = backbone
        self.network = builders[backbone](weights=None)
        if backbone == "resnet18":
            self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        elif backbone.startswith("shufflenet"):
            self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        else:
            last = self.network.classifier[-1]
            self.network.classifier[-1] = nn.Linear(last.in_features, num_classes)

    def forward(self, x):
        return self.network(x)
