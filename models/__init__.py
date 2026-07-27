"""
models/

Модульный пакет с архитектурами: backbone'ы, головы, полные модели-классификаторы
и VAE. Для добавления новой архитектуры под новый метод — добавить файл сюда и
зарегистрировать builder в registry.py.
"""

from .backbones import ResNetBackbone, FusionResNetBackbone
from .heads import ClassificationHead, ProjectionHead
from .classifiers import WaferClassifier, WaferContrastiveModel, LatentFusionClassifier
from .vae import WaferVAE
from .cbam import CBAM, ChannelAttention, SpatialAttention, CBAMCNNClassifier
from .cae import WaferCAE
from .hybrid_vit import ResidualBlock, HybridCNNViT
from .mm_wae import MMWAE, SpatialBranch, FrequencyBranch, TextureBranch, MultimodalFusion, MMWAEDecoder, MultimodalClassifier
from .efficient_cnn import EfficientWaferCNN
from .registry import MODEL_REGISTRY, build_model

__all__ = [
    "ResNetBackbone",
    "FusionResNetBackbone",
    "ClassificationHead",
    "ProjectionHead",
    "WaferClassifier",
    "WaferContrastiveModel",
    "LatentFusionClassifier",
    "WaferVAE",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "CBAMCNNClassifier",
    "WaferCAE",
    "ResidualBlock",
    "HybridCNNViT",
    "MMWAE",
    "SpatialBranch",
    "FrequencyBranch",
    "TextureBranch",
    "MultimodalFusion",
    "MMWAEDecoder",
    "MultimodalClassifier",
    "EfficientWaferCNN",
    "MODEL_REGISTRY",
    "build_model",
]
