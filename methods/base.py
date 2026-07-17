"""
methods/base.py

SSLMethod — общий интерфейс для всех SSL-методов. Метод объявляет, какие
данные ему нужны (requires_unlabeled / requires_contrastive_labeled /
requires_raw_pool), train.py строит loader'ы через DataLoaderFactory на
основе этих флагов, а не через if/elif по имени метода.

fit() — единственная точка входа, которую вызывает train.py. Дефолтная
реализация ниже подходит для методов с одной фазой обучения (train_epoch на
каждую эпоху) — этому соответствуют SupervisedBaseline/MeanTeacher/
SupConMethod/MeanTeacherSupCon. Методы с несколькими фазами (например
LatentVectorMethod: VAE pretrain -> teacher -> pseudo-labeling -> student ->
fine-tune) полностью переопределяют fit(), т.к. их структура обучения не
укладывается в "один train_epoch на эпоху".
"""

import os
from abc import ABC, abstractmethod

import torch


class SSLMethod(ABC):
    requires_unlabeled = False
    requires_contrastive_labeled = False
    requires_raw_pool = False
    supports_pseudo_labeling = False

    def _build_default_loaders(self, data: dict, loader_factory):
        if self.requires_contrastive_labeled:
            labeled_loader = loader_factory.contrastive_loader(data["labeled_images"], data["labeled_labels"])
        else:
            labeled_loader = loader_factory.labeled_loader(
                data["labeled_images"], data["labeled_labels"], transform_mode="weak"
            )

        unlabeled_loader = None
        if self.requires_unlabeled and len(data["unlabeled_images"]) > 0:
            unlabeled_loader = loader_factory.unlabeled_loader(data["unlabeled_images"])

        return labeled_loader, unlabeled_loader

    def build_eval_loader(self, data: dict, loader_factory):
        """
        Строит eval loader для test-набора. Дефолт — обычный WaferDataset
        (grayscale [1,H,W]). Методы с другим представлением данных (например
        CBAMCNNMethod, one-hot [3,H,W]) переопределяют это. train.py вызывает
        этот метод (а не loader_factory напрямую) для финальной оценки после
        fit(), чтобы не завязываться на конкретный формат данных метода.
        """
        return loader_factory.eval_loader(data["test_images"], data["test_labels"])

    def fit(self, data: dict, loader_factory, epochs: int, eval_every: int, checkpoint_path: str, log_fn=print) -> dict:
        """
        Дефолтная (однофазная) реализация: строит loader'ы один раз, затем
        train_epoch на каждую эпоху, eval каждые eval_every эпох, сохраняет
        best checkpoint по F1. Возвращает {"best_f1": float, "best_metrics": dict}.
        """
        labeled_loader, unlabeled_loader = self._build_default_loaders(data, loader_factory)
        test_loader = self.build_eval_loader(data, loader_factory)

        best_f1 = 0.0
        best_metrics = None
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(labeled_loader, unlabeled_loader)
            stats_str = ", ".join(f"{k}={v:.4f}" for k, v in train_stats.items())
            log_fn(f"Epoch {epoch}/{epochs} | {stats_str}")

            if epoch % eval_every == 0 or epoch == epochs:
                metrics = self.evaluate(test_loader)
                log_fn(
                    f"  [Eval] Acc={metrics['accuracy']*100:.2f}% "
                    f"Prec={metrics['precision']*100:.2f}% "
                    f"Recall={metrics['recall']*100:.2f}% "
                    f"F1={metrics['f1']*100:.2f}%"
                )
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_metrics = metrics
                    torch.save(self.state_dict(), checkpoint_path)

        return {"best_f1": best_f1, "best_metrics": best_metrics}

    @abstractmethod
    def evaluate(self, loader) -> dict: ...

    @abstractmethod
    def evaluate_per_class(self, loader): ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: dict): ...
