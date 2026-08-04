"""ClimEx: class-balanced dynamic-threshold consistency training."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import ClimExWaferDataset
from evaluate import compute_metrics
from .base import SSLMethod


class ClimExMethod(SSLMethod):
    requires_unlabeled = True

    def __init__(self, model, optimizer_factory, device: str, config: dict):
        self.model = model.to(device)
        self.optimizer = optimizer_factory(self.model.parameters())
        self.device = device
        self.config = config
        cfg = config["method"]
        self.num_classes = config["model"].get("num_classes", 9)
        self.majority_class = cfg.get("majority_class_index", 8)
        self.fixed_threshold = float(cfg.get("majority_threshold", 0.95))
        self.threshold_momentum = float(cfg.get("threshold_momentum", 0.8))
        self.thresholds = torch.full(
            (self.num_classes,), self.fixed_threshold, device=device
        )
        self.thresholds[self.majority_class] = self.fixed_threshold
        self.iteration = 0

    def _dataset(self, data, indices, mode):
        cfg = self.config["dataset"]
        return ClimExWaferDataset(
            data["wafer_maps"],
            data["all_labels"],
            indices,
            mode=mode,
            image_size=cfg.get("image_size", 96),
            rotation_degrees=cfg.get("rotation_degrees", 180.0),
            noise_std=cfg.get("noise_std", 0.1),
            cutout_scale=cfg.get("cutout_scale", [0.08, 0.20]),
        )

    def _loader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=self.device == "cuda",
        )

    @staticmethod
    def _next(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _update_thresholds(self, confidence_windows):
        majority = confidence_windows[self.majority_class]
        if not majority:
            return
        majority_values = np.concatenate(majority)
        rho = float(np.mean(majority_values >= self.fixed_threshold))
        previous = self.thresholds.detach().cpu().numpy()
        updated = previous.copy()
        for cls in range(self.num_classes):
            if cls == self.majority_class or not confidence_windows[cls]:
                continue
            values = np.concatenate(confidence_windows[cls])
            if len(values) == 0:
                continue
            count = max(1, int(np.ceil(rho * len(values))))
            raw_threshold = np.sort(values)[::-1][count - 1]
            updated[cls] = (
                (1.0 - self.threshold_momentum) * raw_threshold
                + self.threshold_momentum * previous[cls]
            )
        updated[self.majority_class] = self.fixed_threshold
        self.thresholds = torch.tensor(updated, dtype=torch.float32, device=self.device)

    def fit(
        self,
        data,
        loader_factory,
        epochs,
        eval_every,
        checkpoint_path,
        log_fn=print,
    ):
        cfg = self.config["method"]
        total_iterations = int(cfg.get("iterations", 51200))
        update_every = int(cfg.get("threshold_update_every", 512))
        eval_every_iterations = int(cfg.get("eval_every_iterations", 512))
        labeled_loader = self._loader(
            self._dataset(data, data["labeled_indices"], "train"),
            shuffle=True,
            drop_last=True,
        )
        unlabeled_loader = self._loader(
            self._dataset(data, data["unlabeled_indices"], "unlabeled"),
            shuffle=True,
            drop_last=True,
        )
        val_loader = self._loader(
            self._dataset(data, data["val_indices"], "eval")
        )
        labeled_iter, unlabeled_iter = iter(labeled_loader), iter(unlabeled_loader)
        confidence_windows = [[] for _ in range(self.num_classes)]
        best_f1, best_metrics = -1.0, None
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        for step in range(1, total_iterations + 1):
            (labeled_images, labels), labeled_iter = self._next(
                labeled_iter, labeled_loader
            )
            (weak_images, strong_images), unlabeled_iter = self._next(
                unlabeled_iter, unlabeled_loader
            )
            labeled_images = labeled_images.to(self.device)
            labels = labels.to(self.device)
            weak_images = weak_images.to(self.device)
            strong_images = strong_images.to(self.device)

            self.model.train()
            supervised_loss = F.cross_entropy(self.model(labeled_images), labels)
            with torch.no_grad():
                weak_probabilities = F.softmax(self.model(weak_images), dim=1)
                confidence, pseudo_labels = weak_probabilities.max(dim=1)
                accepted = confidence > self.thresholds[pseudo_labels]
                confidence_cpu = confidence.detach().cpu().numpy()
                pseudo_cpu = pseudo_labels.detach().cpu().numpy()
                for cls in range(self.num_classes):
                    class_values = confidence_cpu[pseudo_cpu == cls]
                    if len(class_values):
                        confidence_windows[cls].append(class_values)

            strong_logits = self.model(strong_images)
            per_sample = F.cross_entropy(
                strong_logits, pseudo_labels, reduction="none"
            )
            unsupervised_loss = (per_sample * accepted.float()).mean()
            loss = supervised_loss + unsupervised_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iteration = step

            if step % update_every == 0:
                self._update_thresholds(confidence_windows)
                confidence_windows = [[] for _ in range(self.num_classes)]

            if step % eval_every_iterations == 0 or step == total_iterations:
                metrics = self.evaluate(val_loader)
                threshold_text = ",".join(f"{x:.3f}" for x in self.thresholds.tolist())
                log_fn(
                    f"Iteration {step}/{total_iterations} | loss={loss.item():.4f} "
                    f"Ls={supervised_loss.item():.4f} Lu={unsupervised_loss.item():.4f} "
                    f"accept={accepted.float().mean().item():.3f} | "
                    f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f} | "
                    f"tau=[{threshold_text}]"
                )
                if metrics["f1"] > best_f1:
                    best_f1, best_metrics = metrics["f1"], metrics
                    torch.save(self.state_dict(), checkpoint_path)

        return {
            "best_f1": best_f1,
            "best_metrics": best_metrics,
            "iterations_trained": total_iterations,
        }

    def build_eval_loader(self, data, loader_factory):
        return self._loader(self._dataset(data, data["test_indices"], "eval"))

    @torch.no_grad()
    def evaluate_per_class(self, loader):
        self.model.eval()
        all_labels, all_predictions = [], []
        for images, labels in loader:
            predictions = self.model(images.to(self.device)).argmax(dim=1).cpu()
            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())
        return all_labels, all_predictions

    def evaluate(self, loader):
        labels, predictions = self.evaluate_per_class(loader)
        return compute_metrics(labels, predictions)

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "thresholds": self.thresholds.detach().cpu(),
            "iteration": self.iteration,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.thresholds = state.get(
            "thresholds", torch.full((self.num_classes,), self.fixed_threshold)
        ).to(self.device)
        self.iteration = state.get("iteration", 0)
