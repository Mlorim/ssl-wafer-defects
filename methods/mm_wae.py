"""
methods/mm_wae.py

MM-WAE — "MM-WAE: Multimodal Wasserstein Autoencoders for Semi-Supervised
Wafer Map Defect Recognition" (Zhang, Sun, Liu, Zhang, 2026). Однофазный
метод (в отличие от LatentVectorMethod/HybridCNNViTMethod — здесь нет
отдельных стадий teacher/student/pseudo-labeling): train_epoch чередует
labeled и unlabeled мини-батчи.

Labeled батчи: все 4 loss-компонента (L_recon + λ_mmd·L_mmd + λ_cls·L_cls +
λ_cons·L_cons). Unlabeled батчи: только L_recon + λ_mmd·L_mmd (раздел 3.5
статьи: "without relying on potentially erroneous pseudo-labels").
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import compute_metrics
from losses import mmd_loss, class_frequency_weights, modality_consistency_loss
from .base import SSLMethod


class MMWAEMethod(SSLMethod):
    requires_unlabeled = True

    def __init__(self, model, optimizer_factory, device: str = "cuda", config: dict = None):
        config = config or {}
        model_cfg = config.get("model", {})
        method_cfg = config.get("method", {})

        self.model = model.to(device)
        self.optimizer = optimizer_factory(self.model.parameters())
        self.device = device

        self.num_classes = model_cfg.get("num_classes", 9)

        self.lambda_mmd = method_cfg.get("lambda_mmd", 1.0)
        self.lambda_cls = method_cfg.get("lambda_cls", 1.0)
        self.lambda_cons = method_cfg.get("lambda_cons", 1.0)
        self.class_alpha = method_cfg.get("class_alpha", 1.5)
        self.mmd_c = method_cfg.get("mmd_c", None)
        self.grad_clip_norm = method_cfg.get("grad_clip_norm", 1.0)

    def _build_default_loaders(self, data: dict, loader_factory):
        """
        Unlabeled ветка MM-WAE использует одиночные (не weak/strong пары, как
        у Mean Teacher) изображения — только для реконструкции и MMD, без
        консистентности. Переиспользуем raw_pool_loader (VAEDataset), уже
        существующий для VAE-претрейна LatentVectorMethod.
        """
        labeled_loader = loader_factory.labeled_loader(
            data["labeled_images"], data["labeled_labels"], transform_mode="weak"
        )
        unlabeled_loader = None
        if len(data["unlabeled_images"]) > 0:
            unlabeled_loader = loader_factory.raw_pool_loader(data["unlabeled_images"])
        return labeled_loader, unlabeled_loader

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader] = None) -> dict:
        self.model.train()
        total_loss = total_recon = total_mmd = total_cls = total_cons = 0.0
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
            logits, recon, z, gate_weights, branch_weights = self.model(images)

            z_prior = torch.randn_like(z)
            recon_loss = F.l1_loss(recon, images)
            mmd = mmd_loss(z, z_prior, c=self.mmd_c)

            weights = class_frequency_weights(labels, self.num_classes, alpha=self.class_alpha).to(self.device)
            cls_loss = F.cross_entropy(logits, labels, weight=weights)
            cons_loss = modality_consistency_loss(gate_weights, branch_weights)

            loss = recon_loss + self.lambda_mmd * mmd + self.lambda_cls * cls_loss + self.lambda_cons * cons_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_mmd += mmd.item()
            total_cls += cls_loss.item()
            total_cons += cons_loss.item()
            n_batches += 1

            if unlabeled_iter is not None:
                try:
                    u_images = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    u_images = next(unlabeled_iter)

                u_images = u_images.to(self.device)
                _, u_recon, u_z, _, _ = self.model(u_images)
                u_z_prior = torch.randn_like(u_z)

                u_recon_loss = F.l1_loss(u_recon, u_images)
                u_mmd = mmd_loss(u_z, u_z_prior, c=self.mmd_c)
                u_loss = u_recon_loss + self.lambda_mmd * u_mmd

                self.optimizer.zero_grad()
                u_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

        return {
            "loss": total_loss / max(n_batches, 1),
            "recon_loss": total_recon / max(n_batches, 1),
            "mmd_loss": total_mmd / max(n_batches, 1),
            "cls_loss": total_cls / max(n_batches, 1),
            "cons_loss": total_cons / max(n_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            logits, *_ = self.model(images)
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
            logits, *_ = self.model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        return all_labels, all_preds

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
