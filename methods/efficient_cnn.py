"""Leakage-safe reproduction of Shin and Yoo, Sensors 2023."""

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from datasets import EfficientWaferDataset
from evaluate import compute_metrics
from .base import SSLMethod


class EfficientCNNMethod(SSLMethod):
    def __init__(self, model, optimizer_factory, device: str, config: dict):
        self.prototype = model.cpu()
        self.optimizer_factory = optimizer_factory
        self.device = device
        self.config = config
        self.fold_states = []
        self.fold_metrics = []

    def _limited_stratified(self, indices, labels, maximum, seed):
        indices = np.asarray(indices, dtype=np.int64)
        if maximum is None or len(indices) <= maximum:
            return indices
        rng = np.random.RandomState(seed)
        chosen = []
        class_labels = labels[indices]
        for cls in np.unique(class_labels):
            cls_idx = indices[class_labels == cls]
            quota = max(1, int(round(maximum * len(cls_idx) / len(indices))))
            chosen.extend(rng.choice(cls_idx, min(quota, len(cls_idx)), replace=False))
        chosen = np.asarray(chosen, dtype=np.int64)
        if len(chosen) > maximum:
            chosen = rng.choice(chosen, maximum, replace=False)
        return chosen

    def _oversample_train(self, indices, labels, seed):
        cfg = self.config["method"]
        rng = np.random.RandomState(seed)
        target = int(cfg.get("target_per_defect_class", 10000))
        cap = cfg.get("max_train_per_class")
        output = []
        local_labels = labels[indices]
        for cls in np.unique(labels):
            cls_idx = np.asarray(indices)[local_labels == cls]
            desired = len(cls_idx) if int(cls) == 8 else max(len(cls_idx), target)
            if cap is not None:
                desired = min(desired, int(cap))
            if desired <= len(cls_idx):
                sampled = rng.choice(cls_idx, desired, replace=False)
            else:
                sampled = np.concatenate(
                    [cls_idx, rng.choice(cls_idx, desired - len(cls_idx), replace=True)]
                )
            output.extend(sampled.tolist())
        rng.shuffle(output)
        return np.asarray(output, dtype=np.int64)

    def _dataset(self, data, indices, train):
        cfg = self.config["dataset"]
        return EfficientWaferDataset(
            data["wafer_maps"],
            data["all_labels"],
            indices,
            train=train,
            image_size=cfg.get("image_size", 224),
            rotation_degrees=cfg.get("rotation_degrees", 180),
            transform_probability=cfg.get("transform_probability", 0.5),
        )

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=self.device == "cuda",
        )

    @torch.no_grad()
    def _predict_model(self, model, loader):
        model.eval()
        probabilities, labels = [], []
        for images, targets in loader:
            probabilities.append(F.softmax(model(images.to(self.device)), dim=1).cpu())
            labels.append(targets)
        return torch.cat(probabilities), torch.cat(labels).numpy()

    def fit(self, data, loader_factory, epochs, eval_every, checkpoint_path, log_fn=print):
        labels = data["all_labels"]
        development = data["development_indices"]
        splitter = StratifiedKFold(
            n_splits=self.config["method"].get("num_folds", 4),
            shuffle=True,
            random_state=self.config["dataset"]["seed"],
        )
        splits = list(splitter.split(development, labels[development]))
        max_folds = min(self.config["method"].get("max_folds", len(splits)), len(splits))
        patience = self.config["method"].get("patience", 10)
        self.fold_states, self.fold_metrics = [], []
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        for fold, (train_pos, val_pos) in enumerate(splits[:max_folds], start=1):
            seed = self.config["dataset"]["seed"] + fold
            train_idx = self._oversample_train(development[train_pos], labels, seed)
            val_idx = self._limited_stratified(
                development[val_pos], labels,
                self.config["method"].get("max_eval_samples"), seed,
            )
            train_loader = self._loader(self._dataset(data, train_idx, True), shuffle=True)
            val_loader = self._loader(self._dataset(data, val_idx, False))
            model = copy.deepcopy(self.prototype).to(self.device)
            optimizer = self.optimizer_factory(model.parameters())
            best_f1, best_state, stale = -1.0, None, 0
            log_fn(f"Fold {fold}/{max_folds}: train={len(train_idx)}, val={len(val_idx)}")

            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                for images, targets in train_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(images), targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % eval_every != 0 and epoch != epochs:
                    continue
                probs, y_true = self._predict_model(model, val_loader)
                metrics = compute_metrics(y_true, probs.argmax(1).numpy())
                log_fn(
                    f"  Epoch {epoch}/{epochs} loss={total_loss/max(len(train_loader),1):.4f} "
                    f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}"
                )
                if metrics["f1"] > best_f1:
                    best_f1, stale = metrics["f1"], 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    stale += 1
                    if stale >= patience:
                        break

            self.fold_states.append(best_state)
            self.fold_metrics.append({"fold": fold, "best_f1": best_f1})
            torch.save(self.state_dict(), checkpoint_path)
            del model, optimizer
            if self.device == "mps":
                torch.mps.empty_cache()

        mean_f1 = float(np.mean([item["best_f1"] for item in self.fold_metrics]))
        return {"best_f1": mean_f1, "best_metrics": {"f1": mean_f1}}

    def build_eval_loader(self, data, loader_factory):
        indices = self._limited_stratified(
            data["test_indices"],
            data["all_labels"],
            self.config["method"].get("max_test_samples"),
            self.config["dataset"]["seed"] + 1000,
        )
        return self._loader(self._dataset(data, indices, False))

    @torch.no_grad()
    def evaluate_per_class(self, loader):
        if not self.fold_states:
            raise RuntimeError("No trained fold checkpoints")
        summed, y_true = None, None
        for state in self.fold_states:
            model = copy.deepcopy(self.prototype).to(self.device)
            model.load_state_dict(state)
            probabilities, labels = self._predict_model(model, loader)
            summed = probabilities if summed is None else summed + probabilities
            y_true = labels
            del model
        return y_true.tolist(), (summed / len(self.fold_states)).argmax(1).tolist()

    def evaluate(self, loader):
        y_true, y_pred = self.evaluate_per_class(loader)
        return compute_metrics(y_true, y_pred)

    def state_dict(self):
        return {"fold_models": self.fold_states, "fold_metrics": self.fold_metrics}

    def load_state_dict(self, state):
        self.fold_states = state["fold_models"]
        self.fold_metrics = state.get("fold_metrics", [])
