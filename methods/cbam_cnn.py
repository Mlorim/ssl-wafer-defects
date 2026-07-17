"""
methods/cbam_cnn.py

CBAM-CNN метод из статьи "CBAM-enhanced lightweight CNN for wafer map defect
classification". Полностью supervised: обучается только на labeled 26x26x3
one-hot данных (без unlabeled, без pseudo-labeling/consistency), поэтому по
шагу обучения не отличается от SupervisedBaseline (CE loss, один forward pass
за шаг) — переопределяются loader'ы (one-hot тензоры [H,W,3] требуют другого
Dataset-класса, чем grayscale [H,W] у остальных методов) и fit() (добавлен
early stopping по отдельному validation-сплиту для подбора оптимального числа
эпох вместо фиксированного количества из train.epochs конфига).
"""

import os

import torch

from .supervised import SupervisedBaseline


class CBAMCNNMethod(SupervisedBaseline):
    def __init__(
        self,
        model,
        optimizer_factory,
        device: str = "cuda",
        max_epochs: int = 30,
        patience: int = 5,
    ):
        super().__init__(model, optimizer_factory, device)
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimal_epochs = None

    def _build_default_loaders(self, data: dict, loader_factory):
        labeled_loader = loader_factory.onehot_loader(
            data["labeled_images"], data["labeled_labels"], transform_mode="weak"
        )
        return labeled_loader, None

    def build_eval_loader(self, data: dict, loader_factory):
        return loader_factory.onehot_eval_loader(data["test_images"], data["test_labels"])

    def build_val_loader(self, data: dict, loader_factory):
        return loader_factory.onehot_eval_loader(data["val_images"], data["val_labels"])

    def fit(
        self, data: dict, loader_factory, epochs: int, eval_every: int, checkpoint_path: str, log_fn=print
    ) -> dict:
        """
        Переопределяет однофазный fit() базового класса (SSLMethod): вместо
        фиксированного числа эпох (epochs, из train.epochs конфига — здесь
        игнорируется) использует early stopping по отдельному validation-
        сплиту (data["val_images"]/["val_labels"], см.
        datasets.prepare_datasets_cbam): останавливается, если val F1 не
        улучшается self.patience эпох подряд, но не превышает self.max_epochs.
        "Оптимальное" число эпох — эпоха, на которой был достигнут лучший val
        F1 (сохраняется в self.optimal_epochs и в возвращаемом dict).

        Checkpoint сохраняется по val F1 (не test — test используется только
        для финального отчёта, не участвует в подборе числа эпох, чтобы
        избежать утечки информации из test в выбор модели).
        """
        labeled_loader, _ = self._build_default_loaders(data, loader_factory)
        val_loader = self.build_val_loader(data, loader_factory)
        test_loader = self.build_eval_loader(data, loader_factory)

        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_f1 = 0.0
        best_test_metrics = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_stats = self.train_epoch(labeled_loader)
            stats_str = ", ".join(f"{k}={v:.4f}" for k, v in train_stats.items())
            log_fn(f"Epoch {epoch}/{self.max_epochs} | {stats_str}")

            val_metrics = self.evaluate(val_loader)
            log_fn(
                f"  [Val] Acc={val_metrics['accuracy']*100:.2f}% "
                f"Prec={val_metrics['precision']*100:.2f}% "
                f"Recall={val_metrics['recall']*100:.2f}% "
                f"F1={val_metrics['f1']*100:.2f}%"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_test_metrics = self.evaluate(test_loader)
                torch.save(self.state_dict(), checkpoint_path)
                log_fn(f"  [Val] Новый лучший результат, checkpoint сохранён (epoch {epoch})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                log_fn(
                    f"Early stopping: val F1 не улучшался {self.patience} эпох подряд "
                    f"(оптимальное число эпох: {best_epoch})"
                )
                break

        self.optimal_epochs = best_epoch
        log_fn(f"Оптимальное число эпох (по val F1={best_val_f1*100:.2f}%): {best_epoch}")

        return {
            "best_f1": best_test_metrics["f1"] if best_test_metrics else 0.0,
            "best_metrics": best_test_metrics,
            "optimal_epochs": best_epoch,
        }
