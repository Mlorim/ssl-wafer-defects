"""ViT-Tiny/16 classifier used in arXiv:2504.02494."""

import torch
import torch.nn as nn


class WaferViTTiny(nn.Module):
    """ImageNet-pretrained ViT-Tiny with the paper's learnable 1->3 adapter."""

    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Для ViT-Tiny установите зависимость `timm`.") from exc

        model_name = (
            "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
            if pretrained else "vit_tiny_patch16_224"
        )
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1)
        with torch.no_grad():
            self.gray_to_rgb.weight.fill_(1.0)
            self.gray_to_rgb.bias.zero_()
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        x = self.gray_to_rgb(x)
        x = (x - self.image_mean) / self.image_std
        return self.vit(x)
