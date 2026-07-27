"""
methods/hybrid_vit.py

HybridCNN-ViT — three-stage progressive pseudo-labeling метод из статьи
"SemiWaferNet: Efficient Semi-Supervised Hybrid CNN-Transformer Models for
Wafer Defect Classification and Segmentation" (Shi, Liu, Zhou et al., 2026).
Реализована только классификационная часть (HybridCNN-ViT) — сегментационная
модель статьи (ConvoFormer-UNet) вне рамок этого репозитория (другой тип
задачи: маски вместо меток класса, IoU вместо accuracy/F1).

Stage 1: supervised warm-up на labeled данных.
Stage 2: MC-Dropout псевдо-разметка unlabeled пула teacher'ом из Stage 1
         (class-adaptive confidence threshold + entropy/MI uncertainty
         filtering, формулы 5-13 статьи), обучение на labeled + принятые
         pseudo-labeled сэмплы.
Stage 3: teacher — модель из Stage 2, псевдо-метки полностью пересчитываются
         заново (не объединяются со Stage 2's набором, а заменяют его),
         финальное обучение на labeled + обновлённый pseudo-labeled набор.

Weighted cross-entropy (формула 14, w_c = 1/sqrt(n_c)) используется на каждом
этапе, веса пересчитываются из фактического состава обучающего набора этого
этапа (не за батч — иначе оценка n_c была бы слишком шумной на маленьких батчах).
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import compute_metrics
from losses import inverse_sqrt_class_weights
from .base import SSLMethod


def _enable_mc_dropout(model: nn.Module):
    """
    model.eval() отключает BatchNorm/Dropout как обычно, затем Dropout-слои
    переводятся обратно в train() — MC-Dropout (Gal & Ghahramani, 2016)
    сэмплирует стохастичность только через dropout, не трогая BatchNorm
    running stats (статья, раздел 2.2: "so that stochastic forward sampling
    can be performed without changing the deterministic training objective").
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class HybridCNNViTMethod(SSLMethod):
    requires_unlabeled = True

    def __init__(self, model, optimizer_factory, device: str = "cuda", config: dict = None):
        config = config or {}
        model_cfg = config.get("model", {})
        method_cfg = config.get("method", {})

        self.device = device
        self.optimizer_factory = optimizer_factory
        self.model = model.to(device)
        self.num_classes = model_cfg.get("num_classes", 9)

        self.stage1_epochs = method_cfg.get("stage1_epochs", 15)
        self.stage2_epochs = method_cfg.get("stage2_epochs", 15)
        self.stage3_epochs = method_cfg.get("stage3_epochs", 20)
        self.stage1_lr = method_cfg.get("stage1_lr")
        self.stage2_lr = method_cfg.get("stage2_lr")
        self.stage3_lr = method_cfg.get("stage3_lr")
        self.mc_samples = method_cfg.get("mc_samples", 20)
        self.mc_chunk_size = method_cfg.get("mc_chunk_size", 4)
        self.tau_base = method_cfg.get("tau_base", 0.94)
        self.alpha = method_cfg.get("alpha", 0.08)
        self.beta = method_cfg.get("beta", 0.02)
        self.eps_entropy = method_cfg.get("eps_entropy", 0.08)
        self.eps_mi = method_cfg.get("eps_mi", 0.12)
        self.stage_diagnostics = []

    def build_eval_loader(self, data: dict, loader_factory):
        return loader_factory.onehot_eval_loader(data["test_images"], data["test_labels"])

    def build_val_loader(self, data: dict, loader_factory):
        return loader_factory.onehot_eval_loader(data["val_images"], data["val_labels"])

    # ---- MC-Dropout инференс: mean prob, predictive entropy, mutual information ----

    @torch.no_grad()
    def _mc_dropout_infer(self, loader: DataLoader):
        _enable_mc_dropout(self.model)
        all_mean_probs, all_entropy, all_mi = [], [], []

        for images in loader:
            images = images.to(self.device)
            sample_chunks = []
            remaining = self.mc_samples
            while remaining:
                chunk_size = min(self.mc_chunk_size, remaining)
                repeated = images.repeat((chunk_size, 1, 1, 1))
                chunk_probs = F.softmax(self.model(repeated), dim=1)
                sample_chunks.append(
                    chunk_probs.reshape(chunk_size, images.shape[0], -1)
                )
                remaining -= chunk_size
            probs_samples = torch.cat(sample_chunks, dim=0)  # [M, B, C]
            mean_probs = probs_samples.mean(dim=0)  # [B, C], формула (5)
            entropy = -(mean_probs * torch.log(mean_probs.clamp(min=1e-12))).sum(dim=1)  # формула (11)
            sample_entropy = -(probs_samples * torch.log(probs_samples.clamp(min=1e-12))).sum(dim=2)  # [M, B]
            mi = entropy - sample_entropy.mean(dim=0)  # формула (12)

            all_mean_probs.append(mean_probs.cpu())
            all_entropy.append(entropy.cpu())
            all_mi.append(mi.cpu())

        self.model.eval()
        return (
            torch.cat(all_mean_probs).numpy(),
            torch.cat(all_entropy).numpy(),
            torch.cat(all_mi).numpy(),
        )

    def _select_pseudo_labels(self, unlabeled_images: np.ndarray, loader_factory, log_fn, stage: int):
        """
        Class-adaptive confidence threshold (формула 10) + entropy/MI hard gates
        (формула 13). ŷ(x)=argmax mean prob, q(x)=max mean prob (формула 6).
        """
        if len(unlabeled_images) == 0:
            return unlabeled_images, np.array([], dtype=np.int64)

        loader = loader_factory.onehot_unlabeled_loader(unlabeled_images)
        mean_probs, entropy, mi = self._mc_dropout_infer(loader)

        y_hat = mean_probs.argmax(axis=1)
        q = mean_probs.max(axis=1)

        class_term = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            cls_mask = y_hat == cls
            if not cls_mask.any():
                continue
            mu_c = q[cls_mask].mean()
            sigma_c = q[cls_mask].std()
            class_term[cls] = self.alpha * (sigma_c / mu_c) if mu_c > 0 else 0.0

        threshold = self.tau_base + class_term[y_hat] + self.beta * (1 - entropy)
        threshold = np.clip(threshold, 0.0, 1.0)

        accept_mask = (q >= threshold) & (entropy < self.eps_entropy) & (mi < self.eps_mi)
        accept_rate = accept_mask.mean() if len(accept_mask) else 0.0
        accepted_counts = np.bincount(y_hat[accept_mask], minlength=self.num_classes)
        candidate_counts = np.bincount(y_hat, minlength=self.num_classes)
        diagnostics = {
            "stage": stage,
            "accepted": int(accept_mask.sum()),
            "total": int(len(accept_mask)),
            "accept_rate": float(accept_rate),
            "candidate_counts": candidate_counts.tolist(),
            "accepted_counts": accepted_counts.tolist(),
            "mean_confidence": float(q.mean()),
            "mean_accepted_confidence": (
                float(q[accept_mask].mean()) if accept_mask.any() else None
            ),
            "mean_entropy": float(entropy.mean()),
            "mean_mi": float(mi.mean()),
        }
        self.stage_diagnostics.append(diagnostics)
        log_fn(
            f"[Stage {stage}] pseudo-labeling: accepted {int(accept_mask.sum())}/{len(accept_mask)} "
            f"({accept_rate * 100:.2f}%), per class={accepted_counts.tolist()}, "
            f"mean confidence={diagnostics['mean_confidence']:.4f}"
        )
        return unlabeled_images[accept_mask], y_hat[accept_mask]

    # ---- Обучение одного этапа (supervised warm-up ИЛИ labeled+pseudo) ----

    def _train_stage(
        self, images: np.ndarray, labels: np.ndarray, val_loader, epochs: int, eval_every: int,
        best_f1: float, best_metrics, log_fn, loader_factory, stage_name: str,
        lr_override=None,
    ):
        loader = loader_factory.onehot_loader(images, labels, transform_mode="weak")
        # веса считаются один раз из состава ВСЕГО обучающего набора этапа
        # (формула 14), а не за батч — иначе оценка n_c будет слишком шумной
        class_weights = inverse_sqrt_class_weights(
            torch.as_tensor(labels, dtype=torch.long), self.num_classes
        ).to(self.device)

        optimizer = self.optimizer_factory(
            self.model.parameters(), lr_override=lr_override
        )
        stage_best_f1 = -1.0
        stage_best_metrics = None
        stage_best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            for batch_images, batch_labels in loader:
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
                logits = self.model(batch_images)
                loss = F.cross_entropy(logits, batch_labels, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            log_fn(f"[{stage_name}] Epoch {epoch}/{epochs} | loss={total_loss / max(n_batches, 1):.4f}")

            if epoch % eval_every == 0 or epoch == epochs:
                metrics = self.evaluate(val_loader)
                log_fn(
                    f"  [Val] Acc={metrics['accuracy']*100:.2f}% "
                    f"Prec={metrics['precision']*100:.2f}% "
                    f"Recall={metrics['recall']*100:.2f}% "
                    f"F1={metrics['f1']*100:.2f}%"
                )
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_metrics = metrics
                if metrics["f1"] > stage_best_f1:
                    stage_best_f1 = metrics["f1"]
                    stage_best_metrics = metrics
                    # The next stage must use the best teacher from this stage,
                    # not merely the weights left by its final epoch.
                    stage_best_state = copy.deepcopy(self.state_dict())

        if stage_best_state is None:
            raise RuntimeError(f"{stage_name}: validation was never evaluated")
        self.load_state_dict(stage_best_state)
        log_fn(
            f"[{stage_name}] restored best teacher | "
            f"Val F1={stage_best_metrics['f1']*100:.2f}%"
        )
        return best_f1, best_metrics, stage_best_state

    # ---- fit(): 3-этапный цикл ----

    def fit(self, data: dict, loader_factory, epochs: int, eval_every: int, checkpoint_path: str, log_fn=print) -> dict:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # выбор чекпоинтов — по validation-сплиту (как в статье, секция 4.1);
        # test НЕ участвует в model selection, иначе это утечка test в обучение
        val_loader = self.build_val_loader(data, loader_factory)
        best_f1, best_metrics = 0.0, None
        global_best_state = None
        self.stage_diagnostics = []

        # Stage 1: supervised warm-up на Dl
        previous_best_f1 = best_f1
        best_f1, best_metrics, stage_state = self._train_stage(
            data["labeled_images"], data["labeled_labels"], val_loader,
            epochs=self.stage1_epochs, eval_every=eval_every,
            best_f1=best_f1, best_metrics=best_metrics, log_fn=log_fn,
            loader_factory=loader_factory, stage_name="Stage 1 (supervised)",
            lr_override=self.stage1_lr,
        )
        if best_f1 > previous_best_f1:
            global_best_state = copy.deepcopy(stage_state)
        self.load_state_dict(global_best_state)
        torch.save(global_best_state, checkpoint_path)

        # Stage 2: MC-Dropout псевдо-разметка teacher'ом из Stage 1, Dl ∪ Dpseudo(2)
        pseudo_images, pseudo_labels = self._select_pseudo_labels(
            data["unlabeled_images"], loader_factory, log_fn, stage=2
        )
        stage2_images = np.concatenate([data["labeled_images"], pseudo_images]) if len(pseudo_images) else data["labeled_images"]
        stage2_labels = np.concatenate([data["labeled_labels"], pseudo_labels]) if len(pseudo_images) else data["labeled_labels"]
        previous_best_f1 = best_f1
        best_f1, best_metrics, stage_state = self._train_stage(
            stage2_images, stage2_labels, val_loader,
            epochs=self.stage2_epochs, eval_every=eval_every,
            best_f1=best_f1, best_metrics=best_metrics, log_fn=log_fn,
            loader_factory=loader_factory, stage_name="Stage 2 (+ pseudo-labels)",
            lr_override=self.stage2_lr,
        )
        if best_f1 > previous_best_f1:
            global_best_state = copy.deepcopy(stage_state)
        else:
            log_fn("[Stage 2] no validation improvement; rolling back teacher")
        self.load_state_dict(global_best_state)
        torch.save(global_best_state, checkpoint_path)

        # Stage 3: teacher — модель после Stage 2; псевдо-метки пересчитываются
        # заново (заменяют Stage 2's набор, а не объединяются с ним), Dl ∪ Dpseudo(3)
        pseudo_images, pseudo_labels = self._select_pseudo_labels(
            data["unlabeled_images"], loader_factory, log_fn, stage=3
        )
        stage3_images = np.concatenate([data["labeled_images"], pseudo_images]) if len(pseudo_images) else data["labeled_images"]
        stage3_labels = np.concatenate([data["labeled_labels"], pseudo_labels]) if len(pseudo_images) else data["labeled_labels"]
        previous_best_f1 = best_f1
        best_f1, best_metrics, stage_state = self._train_stage(
            stage3_images, stage3_labels, val_loader,
            epochs=self.stage3_epochs, eval_every=eval_every,
            best_f1=best_f1, best_metrics=best_metrics, log_fn=log_fn,
            loader_factory=loader_factory, stage_name="Stage 3 (refreshed pseudo-labels)",
            lr_override=self.stage3_lr,
        )
        if best_f1 > previous_best_f1:
            global_best_state = copy.deepcopy(stage_state)
        else:
            log_fn("[Stage 3] no validation improvement; keeping previous best model")

        if global_best_state is None:
            global_best_state = copy.deepcopy(stage_state)
        self.load_state_dict(global_best_state)
        torch.save(global_best_state, checkpoint_path)

        return {
            "best_f1": best_f1,
            "best_metrics": best_metrics,
            "pseudo_label_diagnostics": self.stage_diagnostics,
        }

    # ---- eval / checkpoint interface ----

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
