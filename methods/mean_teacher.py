"""
methods/mean_teacher.py

MeanTeacher: student обучается на labeled (CE loss) + unlabeled (consistency
loss против teacher), teacher обновляется через EMA. Соответствует строке
"+ mean teacher" в Table I статьи Wei et al. (2024, mean teacher paper).
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses import consistency_loss, ema_update
from .supervised import SupervisedBaseline


class MeanTeacher(SupervisedBaseline):
    requires_unlabeled = True

    def __init__(
        self,
        model,
        optimizer_factory,
        device: str = "cuda",
        ema_alpha: float = 0.99,
        consistency_weight: float = 1.0,
    ):
        super().__init__(model, optimizer_factory, device)
        self.teacher = copy.deepcopy(self.model).to(device)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.ema_alpha = ema_alpha
        self.consistency_weight = consistency_weight

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader] = None) -> dict:
        self.model.train()
        self.teacher.train()  # BN должен считать статистику, но без градиентов

        total_loss, total_cls_loss, total_cons_loss = 0.0, 0.0, 0.0
        n_batches = 0

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        n_steps = len(labeled_loader)
        for _ in range(n_steps):
            try:
                images, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                images, labels = next(labeled_iter)

            images, labels = images.to(self.device), labels.to(self.device)

            student_logits = self.model(images)
            cls_loss = F.cross_entropy(student_logits, labels)

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
                    teacher_logits = self.teacher(weak_imgs)
                student_unlabeled_logits = self.model(strong_imgs)
                cons_loss = consistency_loss(student_unlabeled_logits, teacher_logits)

            loss = cls_loss + self.consistency_weight * cons_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ema_update(self.model, self.teacher, self.ema_alpha)

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_cons_loss += cons_loss.item() if torch.is_tensor(cons_loss) else cons_loss
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "cls_loss": total_cls_loss / max(n_batches, 1),
            "consistency_loss": total_cons_loss / max(n_batches, 1),
        }

    def state_dict(self):
        return {"model": self.model.state_dict(), "teacher": self.teacher.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.teacher.load_state_dict(state["teacher"])
