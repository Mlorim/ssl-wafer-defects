"""
models/backbones.py

ResNet backbone (адаптирован под 1-канальные grayscale wafer maps), и
FusionResNetBackbone — вариант с точкой инъекции VAE latent vector после
указанного stage, нужный для метода Wei et al. "Latent Vector Representation".
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# Число каналов feature map после layer1..layer4 у разных семейств ResNet.
_BASIC_BLOCK_STAGE_CHANNELS = [64, 128, 256, 512]        # resnet18, resnet34
_BOTTLENECK_STAGE_CHANNELS = [256, 512, 1024, 2048]      # resnet50, resnet101, resnet152


def _stage_channels(backbone: str):
    if backbone in ("resnet18", "resnet34"):
        return _BASIC_BLOCK_STAGE_CHANNELS
    elif backbone in ("resnet50", "resnet101", "resnet152"):
        return _BOTTLENECK_STAGE_CHANNELS
    else:
        raise ValueError(f"Неизвестный backbone: {backbone}")


class ResNetBackbone(nn.Module):
    """
    ResNet backbone (resnet18/34/50/...). Первый conv слой заменён на
    1-канальный вход (wafer maps не имеют RGB-каналов). Веса ImageNet не
    используются по умолчанию, т.к. распределение wafer map сильно
    отличается от натуральных фото.

    forward(x) -> feature vector [B, feat_dim]
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()
        resnet = getattr(tvm, backbone)(weights="IMAGENET1K_V1" if pretrained else None)

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.feat_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class FusionResNetBackbone(nn.Module):
    """
    ResNet backbone с точкой инъекции VAE latent vector после указанного
    stage (1..4). Wei et al. (2024, "Latent Vector Representation") в
    ablation study показывают, что вставка после stage2 (fusion_stage=2)
    даёт лучший результат из 4 позиций.

    Статья не специфицирует механизм слияния явно — используем latent
    vector, спроецированный маленьким MLP в размерность канала feature map
    данной stage, и складываем поэлементно (broadcast по H,W). Это простой
    дифференцируемый вариант, не меняющий форму feature map, что упрощает
    ablation по stage.

    forward(x, latent=None) -> feature vector [B, feat_dim].
    latent=None эквивалентно обычному ResNet без fusion (полезно для
    сравнения teacher/student без VAE в рамках одного класса).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        latent_dim: int = 128,
        fusion_stage: int = 2,
        pretrained: bool = False,
    ):
        super().__init__()
        if fusion_stage not in (1, 2, 3, 4):
            raise ValueError("fusion_stage должен быть одним из 1, 2, 3, 4")

        resnet = getattr(tvm, backbone)(weights="IMAGENET1K_V1" if pretrained else None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        self.avgpool = resnet.avgpool
        self.feat_dim = resnet.fc.in_features
        self.fusion_stage = fusion_stage

        fusion_channels = _stage_channels(backbone)[fusion_stage - 1]
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, fusion_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels, fusion_channels),
        )

    def forward(self, x: torch.Tensor, latent: torch.Tensor = None) -> torch.Tensor:
        x = self.stem(x)
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if latent is not None and i == self.fusion_stage:
                z = self.latent_proj(latent)          # [B, C]
                z = z.unsqueeze(-1).unsqueeze(-1)       # [B, C, 1, 1]
                x = x + z.expand_as(x)                   # broadcast-add по H, W
        x = self.avgpool(x)
        return torch.flatten(x, 1)
