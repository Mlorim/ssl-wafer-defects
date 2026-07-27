"""
models/registry.py

Фабрика моделей по имени метода. Чтобы добавить модель для нового метода,
достаточно зарегистрировать builder здесь — build_model/train.py не меняются.
"""

from typing import Callable, Dict

import torch.nn as nn

from .classifiers import WaferClassifier, WaferContrastiveModel, LatentFusionClassifier
from .cbam import CBAMCNNClassifier
from .hybrid_vit import HybridCNNViT
from .mm_wae import MMWAE
from .efficient_cnn import EfficientWaferCNN


def _build_baseline(num_classes: int, config: dict) -> nn.Module:
    return WaferClassifier(
        num_classes=num_classes,
        backbone=config.get("backbone", "resnet18"),
        pretrained=config.get("pretrained", False),
    )


def _build_contrastive(num_classes: int, config: dict) -> nn.Module:
    return WaferContrastiveModel(
        num_classes=num_classes,
        backbone=config.get("backbone", "resnet18"),
        proj_dim=config.get("projection_dim", 128),
        proj_hidden_dim=config.get("projection_hidden_dim", 512),
        pretrained=config.get("pretrained", False),
    )


def _build_latent_fusion(num_classes: int, config: dict) -> nn.Module:
    return LatentFusionClassifier(
        num_classes=num_classes,
        backbone=config.get("backbone", "resnet50"),
        latent_dim=config.get("latent_dim", 128),
        fusion_stage=config.get("fusion_stage", 2),
        pretrained=config.get("pretrained", False),
    )


def _build_cbam_cnn(num_classes: int, config: dict) -> nn.Module:
    return CBAMCNNClassifier(
        num_classes=num_classes,
        input_channels=config.get("input_channels", 3),
        conv1_channels=config.get("conv1_channels", 32),
        conv2_channels=config.get("conv2_channels", 72),
        cbam_reduction=config.get("cbam_reduction", 8),
        hidden_dim=config.get("hidden_dim", 128),
    )


def _build_hybrid_cnn_vit(num_classes: int, config: dict) -> nn.Module:
    return HybridCNNViT(
        num_classes=num_classes,
        input_channels=config.get("input_channels", 3),
        input_size=config.get("input_size", 32),
        embed_dim=config.get("embed_dim", 128),
        token_grid_size=config.get("token_grid_size", 8),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        ffn_dim=config.get("ffn_dim", 256),
        pre_transformer_dropout=config.get("pre_transformer_dropout", 0.5),
        transformer_dropout=config.get("transformer_dropout", 0.2),
    )


def _build_mm_wae(num_classes: int, config: dict) -> nn.Module:
    return MMWAE(
        num_classes=num_classes,
        input_size=config.get("input_size", 32),
        latent_dim=config.get("latent_dim", 64),
        spatial_dim=config.get("spatial_dim", 512),
        freq_dim=config.get("freq_dim", 64),
        texture_dim=config.get("texture_dim", 128),
        fusion_dim=config.get("fusion_dim", 256),
        fusion_heads=config.get("fusion_heads", 4),
        classifier_hidden_dim=config.get("classifier_hidden_dim", 128),
    )


def _build_efficient_cnn(num_classes: int, config: dict) -> nn.Module:
    return EfficientWaferCNN(
        backbone=config.get("backbone", "mobilenet_v3_small"),
        num_classes=num_classes,
    )


MODEL_REGISTRY: Dict[str, Callable[[int, dict], nn.Module]] = {
    "baseline": _build_baseline,
    "mean_teacher": _build_baseline,
    "supcon": _build_contrastive,
    "mean_teacher_supcon": _build_contrastive,
    "latent_vector_representation": _build_latent_fusion,
    "cbam_cnn": _build_cbam_cnn,
    "hybrid_cnn_vit": _build_hybrid_cnn_vit,
    "mm_wae": _build_mm_wae,
    "efficient_cnn": _build_efficient_cnn,
}


def build_model(method: str, num_classes: int, config: dict) -> nn.Module:
    """
    method: имя из MODEL_REGISTRY (см. также methods.METHOD_REGISTRY).
    config: секция model: конкретного yaml-конфига.
    """
    if method not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестный метод: {method}. Доступны: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[method](num_classes, config)
