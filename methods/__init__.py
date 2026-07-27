"""
methods/

SSL-методы для классификации дефектов на wafer maps. Каждый метод реализует
общий интерфейс SSLMethod (см. base.py): fit(...), evaluate(...),
evaluate_per_class(...), state_dict()/load_state_dict(). Это позволяет
train.py оставаться неизменным при добавлении новых методов — нужно только
добавить файл сюда и зарегистрировать его в registry.py.

Соответствие таблице I из статьи Wei et al. (2024, mean teacher):
    SupervisedBaseline    -> "Resnet (Baseline)"
    MeanTeacher            -> "+ mean teacher"
    SupConMethod            -> "+ SupConLoss"
    MeanTeacherSupCon      -> "+ mean teacher & SupConLoss"

Соответствие таблице II из статьи Wei et al. (2024, latent vector representation):
    SupervisedBaseline (backbone=resnet50) -> "Without VAE"
    LatentVectorMethod                      -> "With VAE"

CBAMCNNMethod — метод из статьи "CBAM-enhanced lightweight CNN for wafer map
defect classification": полностью supervised, обучается на нативном 26x26x3
one-hot представлении, сбалансированном через downsampling + CAE-аугментацию
(см. datasets.prepare_datasets_cbam).

HybridCNNViTMethod — классификационная часть статьи "SemiWaferNet: Efficient
Semi-Supervised Hybrid CNN-Transformer Models for Wafer Defect Classification
and Segmentation": CNN+ViT гибрид с three-stage progressive pseudo-labeling
(MC-Dropout неопределённость + class-adaptive confidence threshold, см.
methods/hybrid_vit.py). Сегментационная часть статьи (ConvoFormer-UNet) не
реализована — другой тип задачи (маски/IoU), вне рамок этого репозитория.
"""

from .base import SSLMethod
from .supervised import SupervisedBaseline
from .mean_teacher import MeanTeacher
from .supcon import SupConMethod
from .mean_teacher_supcon import MeanTeacherSupCon
from .latent_vector import LatentVectorMethod
from .cbam_cnn import CBAMCNNMethod
from .hybrid_vit import HybridCNNViTMethod
from .mm_wae import MMWAEMethod
from .efficient_cnn import EfficientCNNMethod
from .registry import METHOD_REGISTRY, build_method

__all__ = [
    "SSLMethod",
    "SupervisedBaseline",
    "MeanTeacher",
    "SupConMethod",
    "MeanTeacherSupCon",
    "LatentVectorMethod",
    "CBAMCNNMethod",
    "HybridCNNViTMethod",
    "MMWAEMethod",
    "EfficientCNNMethod",
    "METHOD_REGISTRY",
    "build_method",
]
