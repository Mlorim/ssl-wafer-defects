"""Leakage-safe supervised reproduction of arXiv:2504.02494."""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import WM38KDataset
from evaluate import compute_metrics
from .base import SSLMethod


class ViTTinyMethod(SSLMethod):
    def __init__(self, model, optimizer_factory, device: str, config: dict):
        self.model = model.to(device)
        self.optimizer_factory = optimizer_factory
        self.device = device
        self.config = config

    def _loader(self, data, split, shuffle=False):
        dataset = WM38KDataset(
            data["images"], data[f"{split}_indices"], data["all_labels"],
            train=split == "train",
            image_size=self.config["dataset"].get("image_size", 224),
            rotation_degrees=self.config["dataset"].get("rotation_degrees", 180),
            zoom_scale=tuple(self.config["dataset"].get("zoom_scale", [0.85, 1.0])),
        )
        return DataLoader(
            dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=self.device == "cuda",
        )

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        for images, labels in loader:
            logits = self.model(images.to(self.device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
        return y_true, y_pred

    def fit(self, data, loader_factory, epochs, eval_every, checkpoint_path, log_fn=print):
        del loader_factory
        train_loader = self._loader(data, "train", shuffle=True)
        val_loader = self._loader(data, "val")
        optimizer = self.optimizer_factory(self.model.parameters())
        best_f1, best_metrics, best_epoch = -1.0, None, 0
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % eval_every and epoch != epochs:
                continue
            y_true, y_pred = self._predict(val_loader)
            metrics = compute_metrics(y_true, y_pred)
            log_fn(
                f"Epoch {epoch}/{epochs} loss={total_loss/max(1, len(train_loader)):.4f} "
                f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}"
            )
            if metrics["f1"] > best_f1:
                best_f1, best_metrics, best_epoch = metrics["f1"], metrics, epoch
                torch.save(self.state_dict(), checkpoint_path)

        return {
            "best_f1": best_f1,
            "best_metrics": best_metrics,
            "optimal_epochs": best_epoch,
        }

    def build_eval_loader(self, data, loader_factory):
        del loader_factory
        return self._loader(data, "test")

    def evaluate_per_class(self, loader):
        return self._predict(loader)

    def evaluate(self, loader):
        return compute_metrics(*self._predict(loader))

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)
