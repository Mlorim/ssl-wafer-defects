"""
methods/latent_vector.py

Wei, Zhao, Zheng, Zeng (2024) — "Wafer Map Defect Patterns Semi-Supervised
Classification Using Latent Vector Representation". Метод состоит из 5 фаз
(в отличие от однофазных train_epoch методов из mean_teacher.py и др.),
поэтому полностью переопределяет SSLMethod.fit():

  1) VAE pretrain: обучаем WaferVAE на всём пуле изображений (labeled +
     unlabeled, без лейблов) — латентный вектор Z (mu(x)) служит глобальным
     представлением распределения дефекта.
  2) Teacher pretrain: WaferClassifier (тот же backbone, без VAE fusion)
     обучается только на labeled данных.
  3) Pseudo-labeling: teacher размечает unlabeled пул, оставляем сэмплы с
     confidence > confidence_threshold, берём top-K (target_per_class) по
     confidence для каждого класса.
  4) Student train: LatentFusionClassifier (ResNet + латентный вектор,
     вплавленный после fusion_stage) обучается на labeled + pseudo-labeled,
     latent = vae.encode_latent(image) (VAE заморожен на этой и следующей фазе).
  5) Fine-tune: student дообучается на исходном labeled-only наборе (ниже LR),
     чтобы скорректировать возможный шум от псевдо-разметки.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import compute_metrics
from losses import vae_loss
from models import WaferVAE, WaferClassifier
from .base import SSLMethod


class LatentVectorMethod(SSLMethod):
    requires_unlabeled = True
    requires_raw_pool = True
    supports_pseudo_labeling = True

    def __init__(self, model, optimizer_factory, device: str = "cuda", config: dict = None):
        config = config or {}
        model_cfg = config.get("model", {})
        method_cfg = config.get("method", {})

        self.device = device
        self.optimizer_factory = optimizer_factory
        self.num_classes = model_cfg.get("num_classes", 9)

        # student — модель, построенная build_model() для этого метода (LatentFusionClassifier)
        self.student = model.to(device)

        # teacher и VAE метод строит сам — им нужны архитектуры, для которых
        # нет отдельной записи в MODEL_REGISTRY (см. план, раздел B.5):
        # build_model() возвращает только "основную"/чекпоинтируемую модель
        # (student), а методы с несколькими под-моделями достраивают
        # остальные внутри себя из той же секции model: конфига.
        self.teacher = WaferClassifier(
            num_classes=self.num_classes,
            backbone=model_cfg.get("backbone", "resnet50"),
            pretrained=model_cfg.get("pretrained", False),
        ).to(device)

        self.vae = WaferVAE(
            latent_dim=model_cfg.get("latent_dim", 128),
            hidden_channels=tuple(model_cfg.get("vae_hidden_channels", [32, 64, 128])),
        ).to(device)

        self.confidence_threshold = method_cfg.get("confidence_threshold", 0.9)
        self.target_per_class = method_cfg.get("target_per_class", 2000)
        self.vae_epochs = method_cfg.get("vae_epochs", 30)
        self.vae_lr = method_cfg.get("vae_lr", 0.001)
        self.vae_kl_weight = method_cfg.get("vae_kl_weight", 1.0)
        self.teacher_epochs = method_cfg.get("teacher_epochs", 30)
        self.student_epochs_override = method_cfg.get("student_epochs")
        self.finetune_epochs = method_cfg.get("finetune_epochs", 20)
        self.finetune_lr = method_cfg.get("finetune_lr", 0.0002)

    # ---- Фаза 1: VAE pretrain -------------------------------------------------

    def _pretrain_vae(self, loader: DataLoader, epochs: int, log_fn):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.vae_lr)
        self.vae.train()
        for epoch in range(1, epochs + 1):
            total, total_recon, total_kl, n_batches = 0.0, 0.0, 0.0, 0
            for images in loader:
                images = images.to(self.device)
                recon, mu, logvar = self.vae(images)
                loss, parts = vae_loss(recon, images, mu, logvar, kl_weight=self.vae_kl_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item()
                total_recon += parts["recon_loss"]
                total_kl += parts["kl_loss"]
                n_batches += 1
            log_fn(
                f"[VAE pretrain] Epoch {epoch}/{epochs} | "
                f"loss={total / max(n_batches, 1):.4f} "
                f"recon={total_recon / max(n_batches, 1):.4f} "
                f"kl={total_kl / max(n_batches, 1):.4f}"
            )
        self.vae.eval()

    # ---- Фаза 2: teacher pretrain ----------------------------------------------

    def _train_teacher(self, loader: DataLoader, epochs: int, log_fn):
        optimizer = self.optimizer_factory(self.teacher.parameters())
        self.teacher.train()
        for epoch in range(1, epochs + 1):
            total_loss, n_batches = 0.0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.teacher(images)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            log_fn(f"[Teacher pretrain] Epoch {epoch}/{epochs} | loss={total_loss / max(n_batches, 1):.4f}")
        self.teacher.eval()

    # ---- Фаза 3: pseudo-labeling ------------------------------------------------

    @torch.no_grad()
    def _pseudo_label(self, unlabeled_images: np.ndarray, loader_factory, log_fn):
        if len(unlabeled_images) == 0:
            return unlabeled_images, np.array([], dtype=np.int64)

        loader = loader_factory.unlabeled_eval_loader(unlabeled_images)
        self.teacher.eval()

        all_conf, all_pred = [], []
        for images in loader:
            images = images.to(self.device)
            logits = self.teacher(images)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            all_conf.extend(conf.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

        all_conf = np.array(all_conf)
        all_pred = np.array(all_pred)

        above_threshold = np.where(all_conf > self.confidence_threshold)[0]
        selected_idx = []
        for cls in range(self.num_classes):
            cls_idx = above_threshold[all_pred[above_threshold] == cls]
            cls_idx = cls_idx[np.argsort(-all_conf[cls_idx])]  # по убыванию confidence
            selected_idx.append(cls_idx[: self.target_per_class])
        selected_idx = np.concatenate(selected_idx) if selected_idx else np.array([], dtype=int)

        log_fn(
            f"[Pseudo-labeling] {len(selected_idx)}/{len(unlabeled_images)} сэмплов прошли "
            f"confidence>{self.confidence_threshold} и top-{self.target_per_class}/класс"
        )
        return unlabeled_images[selected_idx], all_pred[selected_idx]

    # ---- Фазы 4/5: student train / fine-tune (общий цикл) ------------------------

    def _train_classifier_phase(
        self, loader, test_loader, epochs, eval_every, checkpoint_path,
        optimizer, best_f1, best_metrics, log_fn, phase_name,
    ):
        for epoch in range(1, epochs + 1):
            self.student.train()
            total_loss, n_batches = 0.0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    latent = self.vae.encode_latent(images)
                logits = self.student(images, latent=latent)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            log_fn(f"[{phase_name}] Epoch {epoch}/{epochs} | loss={total_loss / max(n_batches, 1):.4f}")

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

        return best_f1, best_metrics

    # ---- fit(): полный 5-фазный цикл --------------------------------------------

    def fit(self, data: dict, loader_factory, epochs: int, eval_every: int, checkpoint_path: str, log_fn=print) -> dict:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        student_epochs = self.student_epochs_override or epochs

        # Phase 1: VAE pretrain на всём пуле изображений (labeled + unlabeled)
        raw_pool = np.concatenate([data["labeled_images"], data["unlabeled_images"]])
        vae_loader = loader_factory.raw_pool_loader(raw_pool)
        self._pretrain_vae(vae_loader, epochs=self.vae_epochs, log_fn=log_fn)

        # Phase 2: teacher pretrain на labeled
        teacher_loader = loader_factory.labeled_loader(
            data["labeled_images"], data["labeled_labels"], transform_mode="weak"
        )
        self._train_teacher(teacher_loader, epochs=self.teacher_epochs, log_fn=log_fn)

        # Phase 3: pseudo-labeling unlabeled пула
        pseudo_images, pseudo_labels = self._pseudo_label(data["unlabeled_images"], loader_factory, log_fn)

        # Phase 4: student train на labeled + pseudo-labeled, с VAE latent fusion
        if len(pseudo_images) > 0:
            combined_images = np.concatenate([data["labeled_images"], pseudo_images])
            combined_labels = np.concatenate([data["labeled_labels"], pseudo_labels])
        else:
            combined_images, combined_labels = data["labeled_images"], data["labeled_labels"]

        student_loader = loader_factory.labeled_loader(combined_images, combined_labels, transform_mode="weak")
        test_loader = loader_factory.eval_loader(data["test_images"], data["test_labels"])

        student_optimizer = self.optimizer_factory(self.student.parameters())
        best_f1, best_metrics = self._train_classifier_phase(
            student_loader, test_loader, epochs=student_epochs, eval_every=eval_every,
            checkpoint_path=checkpoint_path, optimizer=student_optimizer,
            best_f1=0.0, best_metrics=None, log_fn=log_fn, phase_name="Student train",
        )

        # Phase 5: fine-tune на исходном labeled-only наборе (ниже LR)
        finetune_loader = loader_factory.labeled_loader(
            data["labeled_images"], data["labeled_labels"], transform_mode="weak"
        )
        finetune_optimizer = self.optimizer_factory(self.student.parameters(), lr_override=self.finetune_lr)
        best_f1, best_metrics = self._train_classifier_phase(
            finetune_loader, test_loader, epochs=self.finetune_epochs, eval_every=eval_every,
            checkpoint_path=checkpoint_path, optimizer=finetune_optimizer,
            best_f1=best_f1, best_metrics=best_metrics, log_fn=log_fn, phase_name="Fine-tune",
        )

        return {"best_f1": best_f1, "best_metrics": best_metrics}

    # ---- eval / checkpoint interface ---------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.student.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            latent = self.vae.encode_latent(images)
            logits = self.student(images, latent=latent)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        return compute_metrics(all_labels, all_preds)

    @torch.no_grad()
    def evaluate_per_class(self, loader: DataLoader):
        self.student.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            latent = self.vae.encode_latent(images)
            logits = self.student(images, latent=latent)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        return all_labels, all_preds

    def state_dict(self):
        return {
            "vae": self.vae.state_dict(),
            "teacher": self.teacher.state_dict(),
            "student": self.student.state_dict(),
        }

    def load_state_dict(self, state):
        self.vae.load_state_dict(state["vae"])
        self.teacher.load_state_dict(state["teacher"])
        self.student.load_state_dict(state["student"])
