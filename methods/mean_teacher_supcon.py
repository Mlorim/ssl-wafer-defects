"""
methods/mean_teacher_supcon.py

MeanTeacherSupCon: полный метод из статьи Wei et al. (2024, mean teacher):
Mean Teacher + SupCon loss. L = L_classification + L_consistency + L_supcontrast
(формула 3). Соответствует строке "+ mean teacher & SupConLoss" в Table I —
лучший результат статьи (Acc 84.63%, F1 83.40%). Модель должна быть
WaferContrastiveModel (forward -> logits, embedding).
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import compute_metrics
from losses import consistency_loss, supcon_loss, ema_update
from .base import SSLMethod


class MeanTeacherSupCon(SSLMethod):
    requires_unlabeled = True
    requires_contrastive_labeled = True

    def __init__(
        self,
        model,
        optimizer_factory,
        device: str = "cuda",
        ema_alpha: float = 0.99,
        consistency_weight: float = 1.0,
        supcon_weight: float = 1.0,
        supcon_temperature: float = 0.07,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer_factory(self.model.parameters())
        self.device = device
        self.teacher = copy.deepcopy(self.model).to(device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.ema_alpha = ema_alpha
        self.consistency_weight = consistency_weight
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader] = None) -> dict:
        """
        labeled_loader: ContrastiveWaferDataset (view1, view2, label) — для CE + SupCon
        unlabeled_loader: UnlabeledWaferDataset (weak, strong) — для consistency loss
        """
        self.model.train()
        self.teacher.train()

        total_loss = total_cls = total_cons = total_sc = 0.0
        n_batches = 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
        n_steps = len(labeled_loader)

        for _ in range(n_steps):
            try:
                view1, view2, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                view1, view2, labels = next(labeled_iter)

            view1, view2, labels = view1.to(self.device), view2.to(self.device), labels.to(self.device)

            logits1, emb1 = self.model(view1)
            logits2, emb2 = self.model(view2)

            cls_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2

            embeddings = torch.cat([emb1, emb2], dim=0)
            labels_doubled = torch.cat([labels, labels], dim=0)
            sc_loss = supcon_loss(embeddings, labels_doubled, temperature=self.supcon_temperature)

            cons_loss = torch.tensor(0.0, device=self.device)
            if unlabeled_iter is not None:
                try:
                    weak_imgs, strong_imgs = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    weak_imgs, strong_imgs = next(unlabeled_iter)

                weak_imgs = weak_imgs.to(self.device)
                strong_imgs = strong_imgs.to(self.device)

                with torch.no_grad():
                    teacher_logits, _ = self.teacher(weak_imgs)
                student_unlabeled_logits, _ = self.model(strong_imgs)
                cons_loss = consistency_loss(student_unlabeled_logits, teacher_logits)

            loss = cls_loss + self.consistency_weight * cons_loss + self.supcon_weight * sc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ema_update(self.model, self.teacher, self.ema_alpha)

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_cons += cons_loss.item() if torch.is_tensor(cons_loss) else cons_loss
            total_sc += sc_loss.item()
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "cls_loss": total_cls / max(n_batches, 1),
            "consistency_loss": total_cons / max(n_batches, 1),
            "supcon_loss": total_sc / max(n_batches, 1),
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
        return {"model": self.model.state_dict(), "teacher": self.teacher.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.teacher.load_state_dict(state["teacher"])
