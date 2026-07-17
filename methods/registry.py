"""
methods/registry.py

Фабрика методов по имени + гиперпараметрам из конфига. build_method получает
ПОЛНЫЙ yaml-конфиг (все секции: dataset/model/method/train/eval/output), т.к.
некоторым методам (LatentVectorMethod) нужны гиперпараметры не только из
method:, но и из model: (backbone/latent_dim/fusion_stage) и train:
(используются через optimizer_factory для нескольких фаз с разными LR).
Обычные методы читают из config только свою секцию method: и игнорируют остальное.
"""

from .supervised import SupervisedBaseline
from .mean_teacher import MeanTeacher
from .supcon import SupConMethod
from .mean_teacher_supcon import MeanTeacherSupCon
from .latent_vector import LatentVectorMethod
from .cbam_cnn import CBAMCNNMethod
from .hybrid_vit import HybridCNNViTMethod
from .mm_wae import MMWAEMethod


METHOD_REGISTRY = {
    "baseline": SupervisedBaseline,
    "mean_teacher": MeanTeacher,
    "supcon": SupConMethod,
    "mean_teacher_supcon": MeanTeacherSupCon,
    "latent_vector_representation": LatentVectorMethod,
    "cbam_cnn": CBAMCNNMethod,
    "hybrid_cnn_vit": HybridCNNViTMethod,
    "mm_wae": MMWAEMethod,
}


def build_method(method_name: str, model, optimizer_factory, device: str, config: dict):
    """Фабрика методов по имени + полному конфигу."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Неизвестный метод: {method_name}. Доступны: {list(METHOD_REGISTRY.keys())}")

    method_cls = METHOD_REGISTRY[method_name]
    method_cfg = config.get("method", {})

    if method_name == "baseline":
        return method_cls(model, optimizer_factory, device=device)
    elif method_name == "mean_teacher":
        return method_cls(
            model,
            optimizer_factory,
            device=device,
            ema_alpha=method_cfg.get("ema_alpha", 0.99),
            consistency_weight=method_cfg.get("consistency_weight", 1.0),
        )
    elif method_name == "supcon":
        return method_cls(
            model,
            optimizer_factory,
            device=device,
            supcon_weight=method_cfg.get("supcon_weight", 1.0),
            supcon_temperature=method_cfg.get("supcon_temperature", 0.07),
        )
    elif method_name == "mean_teacher_supcon":
        return method_cls(
            model,
            optimizer_factory,
            device=device,
            ema_alpha=method_cfg.get("ema_alpha", 0.99),
            consistency_weight=method_cfg.get("consistency_weight", 1.0),
            supcon_weight=method_cfg.get("supcon_weight", 1.0),
            supcon_temperature=method_cfg.get("supcon_temperature", 0.07),
        )
    elif method_name == "latent_vector_representation":
        return method_cls(model, optimizer_factory, device=device, config=config)
    elif method_name == "cbam_cnn":
        return method_cls(
            model, optimizer_factory, device=device,
            max_epochs=method_cfg.get("max_epochs", 30),
            patience=method_cfg.get("patience", 5),
        )
    elif method_name == "hybrid_cnn_vit":
        return method_cls(model, optimizer_factory, device=device, config=config)
    elif method_name == "mm_wae":
        return method_cls(model, optimizer_factory, device=device, config=config)
