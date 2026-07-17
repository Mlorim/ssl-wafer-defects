"""
methods/supervised.py

SupervisedBaseline — обычный supervised ResNet, обучается только на labeled
данных. Соответствует строке "Resnet (Baseline)" в Table I статьи Wei et al.
(mean teacher) и строке "Without VAE" в Table II статьи Wei et al. (latent
vector representation) — используется как общий baseline для обеих статей.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import compute_metrics
from .base import SSLMethod


class SupervisedBaseline(SSLMethod):
    def __init__(self, model: nn.Module, optimizer_factory, device: str = "cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer_factory(self.model.parameters())
        self.device = device

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader] = None) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in labeled_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.device)
            logits = self.model(images)
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
            logits = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        return all_labels, all_preds

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
