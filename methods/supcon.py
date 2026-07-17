"""
methods/supcon.py

SupConMethod: ResNet + SupCon loss поверх labeled данных (без Mean Teacher).
Соответствует строке "+ SupConLoss" в Table I. Модель должна быть
WaferContrastiveModel (forward -> logits, embedding).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import compute_metrics
from losses import supcon_loss
from .base import SSLMethod


class SupConMethod(SSLMethod):
    requires_contrastive_labeled = True

    def __init__(
        self,
        model,
        optimizer_factory,
        device: str = "cuda",
        supcon_weight: float = 1.0,
        supcon_temperature: float = 0.07,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer_factory(self.model.parameters())
        self.device = device
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader] = None) -> dict:
        """
        labeled_loader здесь ожидается ContrastiveWaferDataset-based
        (возвращает view1, view2, label), чтобы иметь два view на объект
        для positive pairs в SupCon.
        """
        self.model.train()
        total_loss, total_cls_loss, total_supcon_loss = 0.0, 0.0, 0.0
        n_batches = 0

        for view1, view2, labels in labeled_loader:
            view1, view2, labels = view1.to(self.device), view2.to(self.device), labels.to(self.device)

            logits1, emb1 = self.model(view1)
            logits2, emb2 = self.model(view2)

            cls_loss = F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)
            cls_loss = cls_loss / 2

            embeddings = torch.cat([emb1, emb2], dim=0)
            labels_doubled = torch.cat([labels, labels], dim=0)
            sc_loss = supcon_loss(embeddings, labels_doubled, temperature=self.supcon_temperature)

            loss = cls_loss + self.supcon_weight * sc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_supcon_loss += sc_loss.item()
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "cls_loss": total_cls_loss / max(n_batches, 1),
            "supcon_loss": total_supcon_loss / max(n_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.device)
            logits, _ = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        return compute_metrics(all_labels, all_preds)

    @torch.no_grad()
    def evaluate_per_class(self, loader: DataLoader):
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.device)
            logits, _ = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        return all_labels, all_preds

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
